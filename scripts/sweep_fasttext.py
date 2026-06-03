from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from train_fasttext import threshold_eval

LABEL_WEIGHTS = {
    "prompt_injection": 1.5,
    "jailbreak": 1.3,
    "self_harm": 1.4,
    "violence": 1.2,
    "hate": 1.0,
    "sexual": 1.0,
    "toxicity": 0.8,
    "safe": 0.8,
}

CONFIGS = [
    {
        "name": "balanced_200d",
        "lr": 0.4,
        "epoch": 30,
        "wordNgrams": 3,
        "minn": 3,
        "maxn": 6,
        "dim": 200,
        "bucket": 2_000_000,
        "loss": "ova",
    },
    {
        "name": "strong_300d",
        "lr": 0.3,
        "epoch": 50,
        "wordNgrams": 4,
        "minn": 2,
        "maxn": 6,
        "dim": 300,
        "bucket": 5_000_000,
        "loss": "ova",
    },
    {
        "name": "recall_300d",
        "lr": 0.5,
        "epoch": 40,
        "wordNgrams": 4,
        "minn": 2,
        "maxn": 7,
        "dim": 300,
        "bucket": 5_000_000,
        "loss": "ova",
    },
    {
        "name": "compact_200d",
        "lr": 0.5,
        "epoch": 25,
        "wordNgrams": 3,
        "minn": 3,
        "maxn": 5,
        "dim": 200,
        "bucket": 2_000_000,
        "loss": "ova",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep fastText production configs.")
    parser.add_argument("--train", default="data/fasttext_quality_v3/train.txt")
    parser.add_argument("--valid", default="data/fasttext_quality_v3/valid.txt")
    parser.add_argument("--test", default="data/fasttext_quality_v3/test.txt")
    parser.add_argument("--output-dir", default="models/fasttext_sweep")
    parser.add_argument("--best-output", default="fasttext_observation_best.bin")
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.50,0.60")
    parser.add_argument("--thread", type=int, default=8)
    parser.add_argument("--configs", default="", help="Comma-separated config names. Empty runs all.")
    args = parser.parse_args()

    import fasttext

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    configs = _selected_configs(args.configs)
    runs = []

    for config in configs:
        started = time.perf_counter()
        model_path = out_dir / f"{config['name']}.bin"
        print(f"training {config['name']} -> {model_path}", flush=True)
        model = fasttext.train_supervised(
            input=args.train,
            lr=config["lr"],
            epoch=config["epoch"],
            wordNgrams=config["wordNgrams"],
            minn=config["minn"],
            maxn=config["maxn"],
            dim=config["dim"],
            bucket=config["bucket"],
            loss=config["loss"],
            thread=args.thread,
        )
        model.save_model(str(model_path))

        threshold_results = []
        for threshold in thresholds:
            valid = threshold_eval(model, args.valid, threshold)
            score = production_score(valid)
            threshold_results.append({"threshold": threshold, "score": score, "valid": valid})
        best_threshold_result = max(threshold_results, key=lambda item: item["score"])
        test = threshold_eval(model, args.test, best_threshold_result["threshold"])
        run = {
            "config": config,
            "model_path": str(model_path),
            "seconds": round(time.perf_counter() - started, 3),
            "best_threshold": best_threshold_result["threshold"],
            "best_valid_score": best_threshold_result["score"],
            "valid": best_threshold_result["valid"],
            "test": test,
            "all_thresholds": threshold_results,
        }
        runs.append(run)
        (out_dir / f"{config['name']}.metrics.json").write_text(
            json.dumps(run, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "name": config["name"],
                    "best_threshold": run["best_threshold"],
                    "best_valid_score": run["best_valid_score"],
                    "valid_macro_f1": run["valid"]["macro_f1"],
                    "test_macro_f1": run["test"]["macro_f1"],
                    "seconds": run["seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ranked = sorted(runs, key=lambda item: item["best_valid_score"], reverse=True)
    report = {"ranked": ranked}
    (out_dir / "sweep_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if ranked:
        shutil.copyfile(ranked[0]["model_path"], args.best_output)
        print(json.dumps({"best_model": ranked[0]["model_path"], "copied_to": args.best_output}, sort_keys=True))


def _selected_configs(names: str) -> list[dict[str, Any]]:
    if not names:
        return CONFIGS
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    by_name = {config["name"]: config for config in CONFIGS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise SystemExit(f"Unknown configs: {missing}. Known: {sorted(by_name)}")
    return [by_name[name] for name in wanted]


def production_score(metrics: dict[str, Any]) -> float:
    per_label = metrics["per_label"]
    weighted_f1 = 0.0
    weighted_recall = 0.0
    weight_total = 0.0
    for label, stats in per_label.items():
        weight = LABEL_WEIGHTS.get(label, 1.0)
        weight_total += weight
        weighted_f1 += weight * stats["f1"]
        weighted_recall += weight * stats["recall"]
    if not weight_total:
        return 0.0
    macro_f1 = metrics["macro_f1"]
    weighted_f1 /= weight_total
    weighted_recall /= weight_total
    exact_match = metrics["exact_match"]
    return round((0.45 * weighted_f1) + (0.35 * weighted_recall) + (0.15 * macro_f1) + (0.05 * exact_match), 6)


if __name__ == "__main__":
    main()
