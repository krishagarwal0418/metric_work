from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Any

from observation_labeler import ObservationLabeler
from observation_labeler.model_backends import build_signals_fallback_registry

DATASET_ID = "rogue-security/prompt-injections-benchmark"
CSV_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/test.csv"
POSITIVE_LABELS = {"prompt_injection", "jailbreak"}


def _download_csv(path: Path, token: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(CSV_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise SystemExit(
                "Dataset file access failed. This Hugging Face dataset is gated.\n"
                "Accept the dataset terms in your Hugging Face account, then run:\n"
                "  export HF_TOKEN=<your_token>\n"
                "  python scripts/run_rogue_benchmark.py\n"
            ) from exc
        raise


def _safe_div(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(round((len(ordered) - 1) * pct), len(ordered) - 1)
    return round(ordered[idx], 3)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _risk_bucket(risk_score: int) -> str:
    if risk_score >= 75:
        return "75-100"
    if risk_score >= 50:
        return "50-74"
    if risk_score >= 25:
        return "25-49"
    return "0-24"


def _update_error_observability(
    bucket: dict[str, Counter[str]],
    *,
    labels: list[str],
    detector_sources: list[str],
    fallbacks_used: list[str],
    confidence: str,
    risk_score: int,
) -> None:
    bucket["labels"].update(labels or ["none"])
    bucket["detector_sources"].update(detector_sources or ["none"])
    bucket["fallbacks_used"].update(fallbacks_used or ["none"])
    bucket["confidence"].update([confidence])
    bucket["risk_buckets"].update([_risk_bucket(risk_score)])


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    confusion = report["confusion"]
    latency = report["latency_ms"]
    throughput = report["throughput"]
    lines = [
        "# Rogue Security Prompt Injection Benchmark",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Rows evaluated: `{report['rows']}`",
        "",
        "## Classification Metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']}`",
        f"- Precision: `{metrics['precision']}`",
        f"- Recall: `{metrics['recall']}`",
        f"- F1: `{metrics['f1']}`",
        f"- Specificity: `{metrics['specificity']}`",
        f"- False positive rate: `{metrics['false_positive_rate']}`",
        f"- False negative rate: `{metrics['false_negative_rate']}`",
        f"- Negative predictive value: `{metrics['negative_predictive_value']}`",
        f"- Balanced accuracy: `{metrics['balanced_accuracy']}`",
        f"- Matthews correlation coefficient: `{metrics['mcc']}`",
        "",
        "## Confusion Matrix",
        "",
        f"- TP: `{confusion['tp']}`",
        f"- FP: `{confusion['fp']}`",
        f"- TN: `{confusion['tn']}`",
        f"- FN: `{confusion['fn']}`",
        "",
        "## Latency",
        "",
        f"- Average latency: `{latency['avg']}` ms",
        f"- p50 latency: `{latency['p50']}` ms",
        f"- p95 latency: `{latency['p95']}` ms",
        f"- p99 latency: `{latency['p99']}` ms",
        f"- Min latency: `{latency['min']}` ms",
        f"- Max latency: `{latency['max']}` ms",
        "",
        "## Throughput",
        "",
        f"- Actual measured throughput: `{throughput['actual_rows_per_second']}` rows/sec",
        f"- Theoretical single-worker throughput from avg latency: `{throughput['theoretical_single_worker_rows_per_second']}` rows/sec",
        f"- CPU count estimate: `{throughput['cpu_count']}`",
        f"- Theoretical CPU-parallel throughput estimate: `{throughput['theoretical_cpu_parallel_rows_per_second']}` rows/sec",
        "",
        "## Label Distribution",
        "",
        "Expected labels:",
        "```json",
        json.dumps(report["expected_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "Predicted labels:",
        "```json",
        json.dumps(report["predicted_label_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Detector Observability",
        "",
        "Detector source counts:",
        "```json",
        json.dumps(report["detector_source_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "Fallback route counts:",
        "```json",
        json.dumps(report["fallback_route_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "Model version counts:",
        "```json",
        json.dumps(report["model_version_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Error Observability",
        "",
        "False positives:",
        "```json",
        json.dumps(report["error_observability"]["false_positives"], indent=2, sort_keys=True),
        "```",
        "",
        "False negatives:",
        "```json",
        json.dumps(report["error_observability"]["false_negatives"], indent=2, sort_keys=True),
        "```",
        "",
        "## Misses",
        "",
        "Misses are reported by row and text hash only; raw prompt text is not written.",
        "```json",
        json.dumps(report["misses_without_raw_text"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data)
    if args.download or not data_path.exists():
        _download_csv(data_path, os.environ.get("HF_TOKEN"))

    fallback_registry = None
    if args.models_dir:
        enabled_fallbacks = None
        if args.fallbacks:
            enabled_fallbacks = {name.strip() for name in args.fallbacks.split(",") if name.strip()}
        fallback_registry = build_signals_fallback_registry(
            models_dir=args.models_dir,
            signals_repo=args.signals_repo,
            enabled_fallbacks=enabled_fallbacks,
        )
    labeler = ObservationLabeler(
        fasttext_model_path=args.fasttext_model,
        fallback_registry=fallback_registry,
        forced_fallbacks=[name.strip() for name in args.force_fallbacks.split(",") if name.strip()]
        if args.force_fallbacks
        else None,
    )
    started = time.perf_counter()

    total = 0
    tp = fp = tn = fn = 0
    expected_counts: Counter[str] = Counter()
    predicted_label_counts: Counter[str] = Counter()
    detector_source_counts: Counter[str] = Counter()
    fallback_route_counts: Counter[str] = Counter()
    model_version_counts: Counter[str] = Counter()
    false_positive_observability = {
        "labels": Counter(),
        "detector_sources": Counter(),
        "fallbacks_used": Counter(),
        "confidence": Counter(),
        "risk_buckets": Counter(),
    }
    false_negative_observability = {
        "labels": Counter(),
        "detector_sources": Counter(),
        "fallbacks_used": Counter(),
        "confidence": Counter(),
        "risk_buckets": Counter(),
    }
    misses: list[dict[str, Any]] = []
    latencies: list[float] = []

    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"Expected CSV columns text,label; got {reader.fieldnames}")
        for row in reader:
            if args.limit and total >= args.limit:
                break
            text = row["text"]
            expected = row["label"].strip().lower()
            result = labeler.classify_text(text, source=args.source)
            predicted_positive = bool(POSITIVE_LABELS.intersection(result.labels))
            expected_positive = expected == "jailbreak"

            total += 1
            expected_counts[expected] += 1
            predicted_label_counts.update(result.labels or ["none"])
            detector_source_counts.update(result.detector_sources or ["none"])
            fallback_route_counts.update(result.fallbacks_used or ["none"])
            for name, version in result.model_versions.items():
                model_version_counts[f"{name}={version}"] += 1
            latencies.append(result.latency_ms)

            if expected_positive and predicted_positive:
                tp += 1
            elif not expected_positive and predicted_positive:
                fp += 1
                _update_error_observability(
                    false_positive_observability,
                    labels=result.labels,
                    detector_sources=result.detector_sources,
                    fallbacks_used=result.fallbacks_used,
                    confidence=result.confidence,
                    risk_score=result.risk_score,
                )
            elif not expected_positive and not predicted_positive:
                tn += 1
            else:
                fn += 1
                _update_error_observability(
                    false_negative_observability,
                    labels=result.labels,
                    detector_sources=result.detector_sources,
                    fallbacks_used=result.fallbacks_used,
                    confidence=result.confidence,
                    risk_score=result.risk_score,
                )

            if (expected_positive != predicted_positive) and len(misses) < args.keep_misses:
                misses.append(
                    {
                        "row": total - 1,
                        "text_hash": _text_hash(text),
                        "expected": expected,
                        "predicted_positive": predicted_positive,
                        "labels": result.labels,
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                    }
                )
            if args.progress_every and total % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rows_per_second = total / elapsed if elapsed else 0.0
                print(
                    json.dumps(
                        {
                            "progress_rows": total,
                            "tp": tp,
                            "fp": fp,
                            "tn": tn,
                            "fn": fn,
                            "rows_per_second": round(rows_per_second, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    total_runtime_ms = round((time.perf_counter() - started) * 1000, 3)
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = round((2 * precision * recall / (precision + recall)), 4) if precision + recall else 0.0
    specificity = _safe_div(tn, tn + fp)
    false_positive_rate = _safe_div(fp, fp + tn)
    false_negative_rate = _safe_div(fn, fn + tp)
    negative_predictive_value = _safe_div(tn, tn + fn)
    balanced_accuracy = round((recall + specificity) / 2, 4) if total else 0.0
    mcc_den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = round(((tp * tn) - (fp * fn)) / sqrt(mcc_den), 4) if mcc_den else 0.0

    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    actual_rows_per_second = round(total / (total_runtime_ms / 1000), 3) if total_runtime_ms else 0.0
    theoretical_single = round(1000 / avg_latency, 3) if avg_latency else 0.0
    cpu_count = os.cpu_count() or 1

    report = {
        "dataset": DATASET_ID,
        "channel_source": args.source,
        "rows": total,
        "expected_counts": dict(expected_counts),
        "predicted_label_counts": dict(predicted_label_counts),
        "detector_source_counts": dict(detector_source_counts),
        "fallback_route_counts": dict(fallback_route_counts),
        "model_version_counts": dict(model_version_counts),
        "positive_labels": sorted(POSITIVE_LABELS),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "error_observability": {
            "false_positives": {
                key: dict(counter) for key, counter in false_positive_observability.items()
            },
            "false_negatives": {
                key: dict(counter) for key, counter in false_negative_observability.items()
            },
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "negative_predictive_value": negative_predictive_value,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "total_runtime_ms": total_runtime_ms,
        },
        "latency_ms": {
            "avg": avg_latency,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "throughput": {
            "actual_rows_per_second": actual_rows_per_second,
            "theoretical_single_worker_rows_per_second": theoretical_single,
            "cpu_count": cpu_count,
            "theoretical_cpu_parallel_rows_per_second": round(theoretical_single * cpu_count, 3),
        },
        "misses_without_raw_text": misses,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report, Path(args.markdown_output))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run ObservationLabeler on {DATASET_ID}.")
    parser.add_argument("--data", default="data/rogue_prompt_injections_test.csv")
    parser.add_argument("--output", default="reports/rogue_prompt_injections_report.json")
    parser.add_argument("--markdown-output", default="reports/rogue_prompt_injections_report.md")
    parser.add_argument("--download", action="store_true", help="Force re-download dataset CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for a smoke run.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N rows.")
    parser.add_argument("--keep-misses", type=int, default=50)
    parser.add_argument(
        "--source",
        default="user_input",
        choices=["user_input", "retrieved_context", "tool_output", "llm_output", "system_prompt"],
        help="Observation channel policy to use for label promotion.",
    )
    parser.add_argument("--fasttext-model", default="", help="Path to fasttext_safety.bin.")
    parser.add_argument("--models-dir", default="", help="Directory containing ONNX fallback models.")
    parser.add_argument(
        "--fallbacks",
        default="",
        help="Comma-separated fallback names to enable, for example fallback_a,fallback_c. Empty enables all.",
    )
    parser.add_argument(
        "--force-fallbacks",
        default="",
        help="Comma-separated fallback names to run on every valid row. When set, fastText routing is skipped.",
    )
    parser.add_argument("--signals-repo", default="/home/krish-agarwal/oxygen/signals")
    args = parser.parse_args()

    report = run(args)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(json.dumps(report["latency_ms"], indent=2, sort_keys=True))
    print(json.dumps(report["throughput"], indent=2, sort_keys=True))
    print(json.dumps(report["confusion"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
