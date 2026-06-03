from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ChannelSource = Literal[
    "user_input",
    "retrieved_context",
    "tool_output",
    "llm_output",
    "system_prompt",
]

LABELS = {
    "prompt_injection",
    "jailbreak",
    "toxicity",
    "hate",
    "sexual",
    "sexual_minors",
    "violence",
    "self_harm",
    "unsafe_pattern",
    "secret_exposure",
    "validation_failed",
    "encoded_payload_detected",
}


class ObservationChannel(BaseModel):
    source: ChannelSource
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObfuscationSignals(BaseModel):
    zero_width_count: int = 0
    homoglyph_hits: int = 0
    had_html_entity: bool = False
    had_url_escape: bool = False
    encoded_payload_detected: bool = False


class NormalizedText(BaseModel):
    original: str
    normalized: str
    classification_text: str
    normalization_was_material: bool = False
    obfuscation_signals: ObfuscationSignals = Field(default_factory=ObfuscationSignals)


class CategoryScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: Literal["low", "medium", "high"] = "low"
    sources: list[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    channel_source: ChannelSource
    labels: list[str] = Field(default_factory=list)
    category_scores: list[CategoryScore] = Field(default_factory=list)
    risk_score: int = Field(default=0, ge=0, le=100)
    confidence: Literal["low", "medium", "high"] = "low"
    detector_sources: list[str] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    validation_error: str | None = None
    normalized_text_hash: str | None = None
    latency_ms: float = 0.0


class ObservationResult(BaseModel):
    labels: list[str] = Field(default_factory=list)
    risk_score: int = Field(default=0, ge=0, le=100)
    confidence: Literal["low", "medium", "high"] = "low"
    channel_results: list[ClassificationResult] = Field(default_factory=list)
    latency_ms: float = 0.0
