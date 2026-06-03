from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from observation_labeler.types import CategoryScore, NormalizedText


class FallbackModel(Protocol):
    model_version: str

    def predict(self, text: str) -> dict[str, float]:
        ...


@dataclass
class FallbackRegistry:
    models: dict[str, FallbackModel] = field(default_factory=dict)

    def run(self, fallback_names: list[str], normalized: NormalizedText) -> tuple[list[CategoryScore], dict[str, str]]:
        scores: list[CategoryScore] = []
        versions: dict[str, str] = {}
        for name in fallback_names:
            model = self.models.get(name)
            if model is None:
                versions[name] = "absent"
                continue
            versions[name] = model.model_version
            for label, score in model.predict(normalized.normalized).items():
                scores.append(
                    CategoryScore(
                        label=label,
                        score=max(0.0, min(float(score), 1.0)),
                        confidence="high" if score >= 0.75 else "medium" if score >= 0.4 else "low",
                        sources=[name],
                    )
                )
        return scores, versions
