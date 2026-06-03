from __future__ import annotations

from observation_labeler.types import CategoryScore, ChannelSource

_WEIGHTS = {
    "prompt_injection": 1.0,
    "jailbreak": 1.0,
    "self_harm": 1.0,
    "secret_exposure": 0.9,
    "violence": 0.9,
    "hate": 0.9,
    "sexual": 0.8,
    "sexual_minors": 1.0,
    "toxicity": 0.7,
    "unsafe_pattern": 0.7,
    "encoded_payload_detected": 0.5,
    "validation_failed": 0.5,
}

_DEFAULT_LABEL_THRESHOLD = 0.25
_BASE_THRESHOLDS = {
    "prompt_injection": 0.25,
    "jailbreak": 0.25,
    "secret_exposure": 0.45,
    "toxicity": 0.55,
    "hate": 0.60,
    "sexual": 0.65,
    "sexual_minors": 0.25,
    "violence": 0.65,
    "self_harm": 0.45,
    "unsafe_pattern": 0.55,
    "encoded_payload_detected": 0.50,
    "validation_failed": 0.50,
}
_CHANNEL_LABEL_THRESHOLDS: dict[ChannelSource, dict[str, float]] = {
    # User input is the action channel: lower thresholds preserve attack recall.
    "user_input": {
        **_BASE_THRESHOLDS,
    },
    # System prompts often contain policy text, attack examples, and refusal
    # guidance. Preserve detector scores, but require very strong evidence
    # before promoting attack-looking text to a visible channel label.
    "system_prompt": {
        "prompt_injection": 1.01,
        "jailbreak": 1.01,
        "secret_exposure": 0.80,
        "toxicity": 0.90,
        "hate": 0.90,
        "sexual": 0.90,
        "sexual_minors": 0.50,
        "violence": 0.90,
        "self_harm": 0.85,
        "unsafe_pattern": 0.85,
        "encoded_payload_detected": 0.50,
        "validation_failed": 0.50,
    },
    # Retrieved/tool/model channels are untrusted content surfaces, but they are
    # not the same as a direct user action. Keep them between user and system.
    "retrieved_context": {
        "prompt_injection": 0.50,
        "jailbreak": 0.60,
        "secret_exposure": 0.60,
        "toxicity": 0.65,
        "hate": 0.70,
        "sexual": 0.75,
        "sexual_minors": 0.30,
        "violence": 0.75,
        "self_harm": 0.60,
        "unsafe_pattern": 0.65,
        "encoded_payload_detected": 0.50,
        "validation_failed": 0.50,
    },
    "tool_output": {
        "prompt_injection": 0.50,
        "jailbreak": 0.60,
        "secret_exposure": 0.60,
        "toxicity": 0.65,
        "hate": 0.70,
        "sexual": 0.75,
        "sexual_minors": 0.30,
        "violence": 0.75,
        "self_harm": 0.60,
        "unsafe_pattern": 0.65,
        "encoded_payload_detected": 0.50,
        "validation_failed": 0.50,
    },
    "llm_output": {
        "prompt_injection": 0.70,
        "jailbreak": 0.70,
        "secret_exposure": 0.65,
        "toxicity": 0.70,
        "hate": 0.75,
        "sexual": 0.80,
        "sexual_minors": 0.30,
        "violence": 0.80,
        "self_harm": 0.65,
        "unsafe_pattern": 0.70,
        "encoded_payload_detected": 0.50,
        "validation_failed": 0.50,
    },
}


def _label_threshold(label: str, channel_source: ChannelSource | None) -> float:
    if channel_source is None:
        return _BASE_THRESHOLDS.get(label, _DEFAULT_LABEL_THRESHOLD)
    return _CHANNEL_LABEL_THRESHOLDS.get(channel_source, {}).get(
        label,
        _BASE_THRESHOLDS.get(label, _DEFAULT_LABEL_THRESHOLD),
    )


def aggregate_scores(
    scores: list[CategoryScore],
    *,
    channel_source: ChannelSource | None = None,
) -> tuple[list[CategoryScore], int, str, list[str]]:
    merged: dict[str, CategoryScore] = {}
    for score in scores:
        existing = merged.get(score.label)
        if existing is None:
            merged[score.label] = score
            continue
        if score.score > existing.score:
            existing.score = score.score
            existing.confidence = score.confidence
        existing.sources = sorted(set(existing.sources) | set(score.sources))

    ordered = sorted(merged.values(), key=lambda item: (-item.score, item.label))
    risk_items = [
        item
        for item in ordered
        if item.label in _WEIGHTS and item.score >= _label_threshold(item.label, channel_source)
    ]
    labels = [item.label for item in risk_items]
    risk = 0
    if risk_items:
        risk = round(100 * max(item.score * _WEIGHTS[item.label] for item in risk_items))
    confidence = "high" if any(item.confidence == "high" for item in risk_items) else (
        "medium" if any(item.confidence == "medium" for item in risk_items) else "low"
    )
    return ordered, min(risk, 100), confidence, labels
