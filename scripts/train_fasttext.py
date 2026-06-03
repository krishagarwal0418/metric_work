from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a fastText observation classifier.")
    parser.add_argument("--train", default="data/fasttext_quality_v3/train.txt")
    parser.add_argument("--valid", default="data/fasttext_quality_v3/valid.txt")
    parser.add_argument("--test", default="data/fasttext_quality_v3/test.txt")
    parser.add_argument("--output", default="fasttext_observation_v0_1.bin")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--word-ngrams", type=int, default=3)
    parser.add_argument("--minn", type=int, default=3)
    parser.add_argument("--maxn", type=int, default=6)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--bucket", type=int, default=2_000_000)
    parser.add_argument("--thread", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--loss", default="ova", choices=["ova", "softmax", "hs", "ns"])
    args = parser.parse_args()

    import fasttext

    model = fasttext.train_supervised(
        input=args.train,
        lr=args.lr,
        epoch=args.epoch,
        wordNgrams=args.word_ngrams,
        minn=args.minn,
        maxn=args.maxn,
        dim=args.dim,
        bucket=args.bucket,
        loss=args.loss,
        thread=args.thread,
    )
    model.save_model(args.output)

    metrics = {
        "params": vars(args),
        "fasttext_builtin": {
            "valid": _builtin_test(model, args.valid),
            "test": _builtin_test(model, args.test),
        },
        "threshold_metrics": {
            "valid": _threshold_eval(model, args.valid, args.threshold),
            "test": _threshold_eval(model, args.test, args.threshold),
        },
    }
    metrics_path = Path(args.metrics_output or f"{args.output}.metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _builtin_test(model: Any, path: str) -> dict[str, float]:
    n, precision, recall = model.test(path, k=-1)
    return {"rows": n, "precision": precision, "recall": recall}


def _threshold_eval(model: Any, path: str, threshold: float) -> dict[str, Any]:
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()
    exact_match = 0
    rows = 0

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            expected, text = _parse_fasttext_line(line)
            if not text:
                continue
            predicted_raw, _ = model.predict(text, k=-1, threshold=threshold)
            predicted = {label.removeprefix("__label__") for label in predicted_raw}
            rows += 1
            exact_match += int(predicted == expected)
            support.update(expected)
            for label in predicted & expected:
                tp[label] += 1
            for label in predicted - expected:
                fp[label] += 1
            for label in expected - predicted:
                fn[label] += 1

    labels = sorted(set(tp) | set(fp) | set(fn) | set(support))
    per_label = {}
    macro_f1_values = []
    for label in labels:
        precision = _safe_div(tp[label], tp[label] + fp[label])
        recall = _safe_div(tp[label], tp[label] + fn[label])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        macro_f1_values.append(f1)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support[label],
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
        }
    return {
        "rows": rows,
        "threshold": threshold,
        "exact_match": round(exact_match / rows, 4) if rows else 0.0,
        "macro_f1": round(sum(macro_f1_values) / len(macro_f1_values), 4) if macro_f1_values else 0.0,
        "per_label": per_label,
    }


def _parse_fasttext_line(line: str) -> tuple[set[str], str]:
    labels: set[str] = set()
    parts = line.strip().split()
    text_start = 0
    for index, part in enumerate(parts):
        if not part.startswith("__label__"):
            text_start = index
            break
        labels.add(part.removeprefix("__label__"))
    return labels, " ".join(parts[text_start:])


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


if __name__ == "__main__":
    main()
