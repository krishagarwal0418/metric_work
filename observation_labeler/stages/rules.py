from __future__ import annotations

import re
from dataclasses import dataclass, field

from observation_labeler.types import CategoryScore, NormalizedText


@dataclass
class RuleResult:
    scores: list[CategoryScore] = field(default_factory=list)


_RULES: tuple[tuple[str, str, float], ...] = (
    ("prompt_injection", r"\b(ignore|disregard|override|bypass|forget)\b.{0,80}\b(system|previous|prior|developer|instructions?|prompt|rules?)\b", 1.0),
    ("prompt_injection", r"\b(reveal|show|print|dump|exfiltrate)\b.{0,50}\b(system prompt|hidden instructions?|developer message)\b", 1.0),
    ("jailbreak", r"\b(DAN|developer mode|roleplay as|act as unrestricted|no rules|no safety)\b", 0.85),
    ("secret_exposure", r"\b(password|api key|token|secret|root access|credentials?)\b", 0.7),
    ("self_harm", r"\b(kill myself|suicide|self harm|hurt myself)\b", 1.0),
    ("violence", r"\b(kill|stab|shoot|bomb|attack)\b.{0,80}\b(someone|person|people|school|office)\b", 0.85),
    ("toxicity", r"\b(fuck off|shut up|worthless|idiot)\b", 0.65),
    ("hate", r"\b(slur|racial abuse|hate them because)\b", 0.75),
)


def run_rules(normalized: NormalizedText) -> RuleResult:
    text = normalized.classification_text
    scores: list[CategoryScore] = []
    for label, pattern, score in _RULES:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            scores.append(
                CategoryScore(
                    label=label,
                    score=score,
                    confidence="high" if score >= 0.8 else "medium",
                    sources=["rules"],
                )
            )
    if normalized.obfuscation_signals.encoded_payload_detected:
        scores.append(
            CategoryScore(
                label="encoded_payload_detected",
                score=0.7,
                confidence="medium",
                sources=["normalize"],
            )
        )
    return RuleResult(scores=scores)
