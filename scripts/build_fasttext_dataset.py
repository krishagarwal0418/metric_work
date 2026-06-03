from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LABELS = {
    "prompt_injection",
    "jailbreak",
    "toxicity",
    "hate",
    "sexual",
    "violence",
    "self_harm",
}
SAFE_LABEL = "safe"


@dataclass(frozen=True)
class Row:
    text: str
    labels: frozenset[str]
    source: str
    source_split: str
    channel_source: str = "user_input"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    dataset_id: str | None = None
    config: str | None = None
    splits: tuple[str, ...] = ("train",)
    gated: bool = False
    url: str | None = None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="rogue_prompt_injections",
        kind="rogue_csv",
        dataset_id="rogue-security/prompt-injections-benchmark",
        url="https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark/resolve/main/test.csv",
        splits=("test",),
        gated=True,
    ),
    SourceSpec(
        name="deepset_prompt_injections",
        kind="hf_prompt_binary",
        dataset_id="deepset/prompt-injections",
        splits=("train",),
    ),
    SourceSpec(
        name="lakera_gandalf",
        kind="hf_lakera_gandalf",
        dataset_id="Lakera/gandalf-rct-attack-categories",
        splits=("train",),
    ),
    SourceSpec(
        name="cyberec_prompt_injection",
        kind="hf_prompt_binary",
        dataset_id="cyberec/Prompt-injection-dataset",
        splits=("train",),
    ),
    SourceSpec(
        name="qualifire_safety",
        kind="hf_safety_multiclass",
        dataset_id="qualifire/safety-benchmark",
        splits=("train", "test"),
        gated=True,
    ),
    SourceSpec(
        name="toxigen",
        kind="hf_toxigen",
        dataset_id="toxigen/toxigen-data",
        splits=("train",),
    ),
    SourceSpec(
        name="open_moderator",
        kind="hf_safety_multiclass",
        dataset_id="TeichAI/open-moderator-v1",
        splits=("train",),
    ),
    SourceSpec(
        name="ifmain_text_moderation_multilingual",
        kind="hf_moderation_scores",
        dataset_id="ifmain/text-moderation-02-multilingual",
        splits=("train",),
    ),
    SourceSpec(
        name="zysec_harmful_behaviors",
        kind="hf_moderation_scores",
        dataset_id="ZySec-AI/harmful_behaviors",
        splits=("train",),
    ),
    SourceSpec(
        name="quantaspark_cortyx_safety",
        kind="hf_safety_multiclass",
        dataset_id="QuantaSparkLabs/cortyx-safety-dataset",
        splits=("train",),
    ),
)

PRESETS = {
    # Good first training corpus: direct prompt-injection data, moderation-score
    # datasets with category labels, and SQuAD hard-negative benign questions.
    # Excludes weaker/unclear schema sources unless explicitly requested.
    "quality": (
        "rogue_prompt_injections",
        "deepset_prompt_injections",
        "lakera_gandalf",
        "cyberec_prompt_injection",
        "toxigen",
        "ifmain_text_moderation_multilingual",
        "quantaspark_cortyx_safety",
        "squad_v2_safe",
    ),
    "prompt": (
        "rogue_prompt_injections",
        "deepset_prompt_injections",
        "lakera_gandalf",
        "cyberec_prompt_injection",
        "squad_v2_safe",
    ),
    "safety": (
        "toxigen",
        "ifmain_text_moderation_multilingual",
        "quantaspark_cortyx_safety",
    ),
}

SOURCES = (
    *SOURCES,
    SourceSpec(
        name="squad_v2_safe",
        kind="hf_safe_text",
        dataset_id="squad_v2",
        splits=("train", "validation"),
    ),
)

TEXT_KEYS = (
    "text",
    "prompt",
    "instruction",
    "input",
    "content",
    "message",
    "user_input",
    "question",
    "sentence",
)
LABEL_KEYS = (
    "label",
    "labels",
    "category",
    "categories",
    "class",
    "target",
    "is_prompt_injection",
    "is_jailbreak",
    "jailbreak",
    "attack",
    "is_attack",
    "toxicity",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-label fastText dataset for observation labels.")
    parser.add_argument("--output-dir", default="data/fasttext_corpus")
    parser.add_argument("--sources", default="all", help="Comma-separated source names, or 'all'.")
    parser.add_argument(
        "--preset",
        default="",
        choices=sorted(PRESETS),
        help="Curated source preset. Overrides --sources.",
    )
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=0, help="Optional balancing cap after dedupe.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--include-gated", action="store_true", help="Include gated sources such as Rogue.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    if args.list_sources:
        print(json.dumps({"sources": [spec.name for spec in SOURCES], "presets": PRESETS}, indent=2, sort_keys=True))
        return

    selected = _select_sources(",".join(PRESETS[args.preset]) if args.preset else args.sources)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    failures: list[dict[str, str]] = []
    source_stats: dict[str, dict[str, Any]] = {}
    for spec in selected:
        if spec.gated and not args.include_gated:
            source_stats[spec.name] = {"status": "skipped_gated", "rows_mapped": 0}
            continue
        try:
            source_rows = list(_load_source(spec, args.hf_token))
        except Exception as exc:  # noqa: BLE001 - keep builder resilient across external datasets.
            failures.append({"source": spec.name, "error": str(exc)})
            source_stats[spec.name] = {"status": "failed", "rows_mapped": 0, "error": str(exc)}
            continue
        if args.limit_per_source:
            source_rows = source_rows[: args.limit_per_source]
        source_stats[spec.name] = {
            "status": "included",
            "rows_mapped": len(source_rows),
            "label_counts": dict(_label_counts(source_rows)),
        }
        rows.extend(source_rows)

    rows, conflict_count = _dedupe_and_drop_conflicts(rows)
    if args.max_per_label:
        rows = _cap_per_label(rows, args.max_per_label, args.seed)
    splits = _split_rows(rows, args.train_ratio, args.valid_ratio, args.seed)

    for split, split_rows in splits.items():
        _write_jsonl(out_dir / f"{split}.jsonl", split_rows)
        _write_fasttext(out_dir / f"{split}.txt", split_rows)

    manifest = {
        "labels": sorted(LABELS),
        "safe_label": SAFE_LABEL,
        "sources_requested": [spec.name for spec in selected],
        "sources_included": sorted({row.source for row in rows}),
        "failures": failures,
        "rows_total": len(rows),
        "conflicting_duplicates_dropped": conflict_count,
        "split_counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "label_counts": dict(_label_counts(rows)),
        "source_counts": dict(Counter(row.source for row in rows)),
        "source_stats": source_stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _select_sources(names: str) -> list[SourceSpec]:
    if names == "all":
        return list(SOURCES)
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    by_name = {spec.name: spec for spec in SOURCES}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise SystemExit(f"Unknown sources: {missing}. Known: {sorted(by_name)}")
    return [by_name[name] for name in wanted]


def _load_source(spec: SourceSpec, token: str) -> Iterable[Row]:
    if spec.kind == "rogue_csv":
        yield from _load_rogue_csv(spec, token)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install data dependencies first: pip install -e '.[data]'") from exc

    load_kwargs: dict[str, Any] = {}
    if token:
        load_kwargs["token"] = token
    if spec.config:
        dataset = load_dataset(spec.dataset_id, spec.config, **load_kwargs)
    else:
        dataset = load_dataset(spec.dataset_id, **load_kwargs)

    for split in spec.splits:
        if split not in dataset:
            continue
        for item in dataset[split]:
            row = _map_hf_item(spec, split, item)
            if row is not None:
                yield row


def _load_rogue_csv(spec: SourceSpec, token: str) -> Iterable[Row]:
    if not spec.url:
        return
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(spec.url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            lines = response.read().decode("utf-8").splitlines()
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError("Rogue is gated. Accept terms on Hugging Face and pass HF_TOKEN.") from exc
        raise
    for item in csv.DictReader(lines):
        text = _clean_text(item.get("text"))
        if not text:
            continue
        raw_label = str(item.get("label", "")).lower()
        labels = frozenset({"prompt_injection", "jailbreak"} if raw_label == "jailbreak" else {SAFE_LABEL})
        yield Row(text=text, labels=labels, source=spec.name, source_split="test")


def _map_hf_item(spec: SourceSpec, split: str, item: dict[str, Any]) -> Row | None:
    text = _extract_text(item)
    if not text:
        return None

    if spec.kind in {"hf_prompt_binary", "hf_lakera_gandalf"}:
        labels = _map_prompt_labels(item, spec.kind)
    elif spec.kind == "hf_safety_multiclass":
        labels = _map_safety_labels(item)
    elif spec.kind == "hf_toxigen":
        labels = _map_toxigen_labels(item)
    elif spec.kind == "hf_moderation_scores":
        labels = _map_moderation_score_labels(item)
    elif spec.kind == "hf_safe_text":
        labels = frozenset({SAFE_LABEL})
    else:
        labels = frozenset()

    if not labels:
        return None
    return Row(text=text, labels=labels, source=spec.name, source_split=split)


def _extract_text(item: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = item.get(key)
        text = _clean_text(value)
        if text:
            return text
    # Some datasets store chat messages as lists/dicts. Keep this conservative.
    for value in item.values():
        if isinstance(value, list):
            parts = []
            for entry in value:
                if isinstance(entry, dict):
                    parts.append(str(entry.get("content", "")))
                elif isinstance(entry, str):
                    parts.append(entry)
            text = _clean_text("\n".join(parts))
            if text:
                return text
    return ""


def _map_prompt_labels(item: dict[str, Any], kind: str) -> frozenset[str]:
    values = _label_values(item)
    joined = " ".join(values).lower()
    binary = _binary_value(item, ("label", "target", "is_prompt_injection", "is_jailbreak", "jailbreak", "attack", "is_attack"))
    if binary is True:
        return frozenset({"prompt_injection", "jailbreak"})
    if binary is False:
        return frozenset({SAFE_LABEL})
    if kind == "hf_lakera_gandalf":
        if _is_safe_label(joined):
            return frozenset({SAFE_LABEL})
        return frozenset({"prompt_injection", "jailbreak"})
    if any(token in joined for token in ("injection", "jailbreak", "attack", "malicious", "unsafe", "1", "true", "yes")):
        return frozenset({"prompt_injection", "jailbreak"})
    if _is_safe_label(joined):
        return frozenset({SAFE_LABEL})
    return frozenset()


def _map_safety_labels(item: dict[str, Any]) -> frozenset[str]:
    joined = " ".join(_label_values(item)).lower()
    labels: set[str] = set()
    if _is_truthy(item.get("Dangerous Content")):
        labels.add("violence")
    if _is_truthy(item.get("Harassment")):
        labels.add("toxicity")
    if _is_truthy(item.get("Sexually Explicit Information")):
        labels.add("sexual")
    if _is_truthy(item.get("Hate Speech")):
        labels.add("hate")
    if _is_truthy(item.get("Safe")) and not labels:
        labels.add(SAFE_LABEL)
    if _is_safe_label(joined):
        labels.add(SAFE_LABEL)
    if any(x in joined for x in ("toxic", "toxicity", "harassment", "insult", "abuse", "obscene")):
        labels.add("toxicity")
    if any(x in joined for x in ("hate", "hateful", "identity_attack", "racist", "slur")):
        labels.add("hate")
    if any(x in joined for x in ("sexual", "sex", "porn", "erotic", "explicit")):
        labels.add("sexual")
    if any(x in joined for x in ("violence", "violent", "threat", "weapon", "bomb", "kill")):
        labels.add("violence")
    if any(x in joined for x in ("self_harm", "self-harm", "suicide", "self harm")):
        labels.add("self_harm")
    return frozenset(labels)


def _map_toxigen_labels(item: dict[str, Any]) -> frozenset[str]:
    values = _label_values(item)
    joined = " ".join(values).lower()
    labels: set[str] = set()
    toxicity_scores = [
        _float_value(item.get("toxicity_human")),
        _float_value(item.get("toxicity_ai")),
        _float_value(item.get("intent")),
    ]
    if any(score is not None and score >= 3.0 for score in toxicity_scores):
        labels.update({"toxicity", "hate"})
    if str(item.get("lewd", "")).lower().strip() not in {"", "doesn't contain lewd content", "none"}:
        labels.add("sexual")
    if any(x in joined for x in ("hate", "toxic", "1", "true")):
        labels.update({"toxicity", "hate"})
    elif _is_safe_label(joined) or "0" in joined:
        labels.add(SAFE_LABEL)
    return frozenset(labels)


def _map_moderation_score_labels(item: dict[str, Any]) -> frozenset[str]:
    labels: set[str] = set()
    categories = item.get("categories")
    scores = item.get("category_scores")
    if isinstance(item.get("moderation"), dict):
        moderation = item["moderation"]
        categories = categories or moderation.get("categories")
        scores = scores or moderation.get("category_scores")
        if moderation.get("results"):
            labels.update(_labels_from_category_dict(moderation.get("results"), threshold=None))
    if isinstance(item.get("openai_moderation"), dict):
        moderation = item["openai_moderation"]
        categories = categories or moderation.get("categories")
        scores = scores or moderation.get("category_scores")
        if moderation.get("results"):
            labels.update(_labels_from_category_dict(moderation.get("results"), threshold=None))
    if item.get("results"):
        labels.update(_labels_from_category_dict(item.get("results"), threshold=None))
    labels.update(_labels_from_category_dict(categories, threshold=None))
    labels.update(_labels_from_category_dict(scores, threshold=0.5))
    if _is_truthy(item.get("flagged")) is False and not labels:
        labels.add(SAFE_LABEL)
    return frozenset(labels)


def _labels_from_category_dict(value: Any, threshold: float | None) -> set[str]:
    labels: set[str] = set()
    if isinstance(value, list) and value and isinstance(value[0], dict):
        # OpenAI moderation response: results=[{categories:{...}, category_scores:{...}}]
        for entry in value:
            labels.update(_labels_from_category_dict(entry.get("categories"), threshold=None))
            labels.update(_labels_from_category_dict(entry.get("category_scores"), threshold=0.5))
        return labels
    if not isinstance(value, dict):
        return labels
    for raw_name, raw_score in value.items():
        name = str(raw_name).lower().replace("/", "_").replace("-", "_")
        if threshold is None:
            active = _is_truthy(raw_score)
        else:
            score = _float_value(raw_score)
            active = score is not None and score >= threshold
        if not active:
            continue
        if name.startswith("harassment"):
            labels.add("toxicity")
        elif name.startswith("hate"):
            labels.add("hate")
        elif name.startswith("sexual"):
            labels.add("sexual")
        elif name.startswith("violence"):
            labels.add("violence")
        elif name.startswith("self_harm"):
            labels.add("self_harm")
    return labels


def _label_values(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in LABEL_KEYS:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, list):
            values.extend(str(v) for v in value)
        elif isinstance(value, dict):
            values.extend(f"{k}:{v}" for k, v in value.items())
        else:
            values.append(str(value))
    return values


def _is_safe_label(value: str) -> bool:
    value = value.lower().strip()
    return value in {"0", "false", "safe", "benign", "clean", "normal", "ok", "none"} or "benign" in value


def _binary_value(item: dict[str, Any], keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, bool):
            return value
        text = str(value).lower().strip()
        if text in {"1", "true", "yes", "attack", "jailbreak", "injection", "malicious"}:
            return True
        if text in {"0", "false", "no", "benign", "safe", "normal"}:
            return False
    return None


def _is_truthy(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    text = str(value).lower().strip()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _float_value(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_and_drop_conflicts(rows: list[Row]) -> tuple[list[Row], int]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[_hash_text(row.text)].append(row)

    merged: list[Row] = []
    conflicts = 0
    for group in grouped.values():
        unsafe: set[str] = set()
        saw_safe = False
        for row in group:
            if SAFE_LABEL in row.labels:
                saw_safe = True
            unsafe.update(label for label in row.labels if label != SAFE_LABEL)
        if saw_safe and unsafe:
            conflicts += len(group)
            continue
        first = group[0]
        labels = frozenset(unsafe or {SAFE_LABEL})
        sources = "+".join(sorted({row.source for row in group}))
        splits = "+".join(sorted({row.source_split for row in group}))
        merged.append(Row(text=first.text, labels=labels, source=sources, source_split=splits))
    return merged, conflicts


def _cap_per_label(rows: list[Row], max_per_label: int, seed: int) -> list[Row]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    counts: Counter[str] = Counter()
    kept: list[Row] = []
    for row in shuffled:
        primary = sorted(label for label in row.labels if label != SAFE_LABEL) or [SAFE_LABEL]
        if all(counts[label] >= max_per_label for label in primary):
            continue
        kept.append(row)
        for label in primary:
            counts[label] += 1
    return kept


def _split_rows(rows: list[Row], train_ratio: float, valid_ratio: float, seed: int) -> dict[str, list[Row]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    valid_end = train_end + int(len(shuffled) * valid_ratio)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def _write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "text": row.text,
                        "labels": sorted(row.labels),
                        "source": row.source,
                        "source_split": row.source_split,
                        "channel_source": row.channel_source,
                        "text_hash": _hash_text(row.text),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_fasttext(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            labels = sorted(row.labels)
            prefix = " ".join(f"__label__{label}" for label in labels)
            text = row.text.replace("\n", " ")
            f.write(f"{prefix} {text}\n")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()


def _label_counts(rows: list[Row]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.labels)
    return counts


if __name__ == "__main__":
    sys.exit(main())
