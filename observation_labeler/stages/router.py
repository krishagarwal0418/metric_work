from __future__ import annotations

from observation_labeler.types import CategoryScore, NormalizedText


def route_fallbacks(scores: list[CategoryScore], normalized: NormalizedText) -> list[str]:
    by_label = {score.label: score.score for score in scores}
    fallbacks: set[str] = set()

    if (
        by_label.get("prompt_injection", 0) >= 0.35
        or normalized.obfuscation_signals.homoglyph_hits > 0
        or normalized.obfuscation_signals.zero_width_count > 0
        or normalized.obfuscation_signals.encoded_payload_detected
    ):
        fallbacks.add("fallback_a")
    if by_label.get("jailbreak", 0) >= 0.25:
        fallbacks.add("fallback_c")
    if any(by_label.get(label, 0) >= 0.25 for label in ("toxicity", "hate", "sexual", "violence")):
        fallbacks.add("fallback_b")
    if any(by_label.get(label, 0) >= 0.25 for label in ("sexual", "violence", "self_harm")):
        fallbacks.add("fallback_d")

    return sorted(fallbacks)
