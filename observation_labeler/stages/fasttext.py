from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from observation_labeler.types import CategoryScore, NormalizedText

_LABEL_MAP = {
    "__label__prompt_injection": "prompt_injection",
    "__label__jailbreak": "jailbreak",
    "__label__toxicity": "toxicity",
    "__label__hate": "hate",
    "__label__sexual": "sexual",
    "__label__violence": "violence",
    "__label__self_harm": "self_harm",
}


@dataclass
class FastTextClassifier:
    model_path: str | None = None

    def __post_init__(self) -> None:
        self.model = None
        self.model_version = "fasttext:absent"
        if not self.model_path:
            return
        path = Path(self.model_path)
        if not path.exists():
            return
        try:
            import fasttext

            self.model = fasttext.load_model(str(path))
            self.model_version = f"fasttext:{path.stat().st_mtime_ns}"
        except Exception:
            self.model = None

    def predict(self, normalized: NormalizedText) -> list[CategoryScore]:
        if self.model is None:
            return []
        labels, probs = self.model.predict(normalized.classification_text, k=5)
        scores: list[CategoryScore] = []
        for raw_label, prob in zip(labels, probs, strict=False):
            label = _LABEL_MAP.get(raw_label, raw_label.removeprefix("__label__"))
            score = float(prob)
            if score <= 0:
                continue
            scores.append(
                CategoryScore(
                    label=label,
                    score=min(score, 1.0),
                    confidence="high" if score >= 0.75 else "medium" if score >= 0.4 else "low",
                    sources=["fasttext"],
                )
            )
        return scores
