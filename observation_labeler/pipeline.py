from __future__ import annotations

import hashlib
import time

from observation_labeler.stages.aggregate import aggregate_scores
from observation_labeler.stages.fallbacks import FallbackRegistry
from observation_labeler.stages.fasttext import FastTextClassifier
from observation_labeler.stages.normalize import normalize_channel
from observation_labeler.stages.router import route_fallbacks
from observation_labeler.stages.rules import run_rules
from observation_labeler.stages.validate import ValidationConfig, validate_channel
from observation_labeler.types import (
    CategoryScore,
    ChannelSource,
    ClassificationResult,
    ObservationChannel,
    ObservationResult,
)


class ObservationLabeler:
    def __init__(
        self,
        *,
        validation_config: ValidationConfig | None = None,
        fasttext_model_path: str | None = None,
        fallback_registry: FallbackRegistry | None = None,
        forced_fallbacks: list[str] | None = None,
    ) -> None:
        self.validation_config = validation_config or ValidationConfig()
        self.fasttext = FastTextClassifier(fasttext_model_path)
        self.fallbacks = fallback_registry or FallbackRegistry()
        self.forced_fallbacks = forced_fallbacks

    def classify_text(
        self,
        content: str,
        *,
        source: ChannelSource = "user_input",
    ) -> ClassificationResult:
        return self.classify_channel(ObservationChannel(source=source, content=content))

    def classify_observation(self, channels: list[ObservationChannel]) -> ObservationResult:
        started = time.perf_counter()
        results = [self.classify_channel(channel) for channel in channels]
        all_scores = [score for result in results for score in result.category_scores]
        _, risk_score, confidence, labels = aggregate_scores(all_scores)
        return ObservationResult(
            labels=labels,
            risk_score=risk_score,
            confidence=confidence,
            channel_results=results,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def classify_channel(self, channel: ObservationChannel) -> ClassificationResult:
        started = time.perf_counter()
        detector_sources: list[str] = []
        model_versions: dict[str, str] = {}

        validation_error = validate_channel(channel, self.validation_config)
        if validation_error is not None:
            score = CategoryScore(
                label="validation_failed",
                score=1.0,
                confidence="high",
                sources=["validator"],
            )
            return ClassificationResult(
                channel_source=channel.source,
                labels=["validation_failed"],
                category_scores=[score],
                risk_score=50,
                confidence="high",
                detector_sources=["validator"],
                validation_error=validation_error.reason,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        normalized = normalize_channel(channel.content)
        normalized_hash = hashlib.sha256(normalized.normalized.encode("utf-8")).hexdigest()

        rule_result = run_rules(normalized)
        scores = list(rule_result.scores)
        if rule_result.scores:
            detector_sources.append("rules")

        if self.forced_fallbacks is None:
            ft_scores = self.fasttext.predict(normalized)
            scores.extend(ft_scores)
            model_versions["fasttext"] = self.fasttext.model_version
            if ft_scores:
                detector_sources.append("fasttext")

        fallback_names = self.forced_fallbacks or route_fallbacks(scores, normalized)
        fallback_scores, fallback_versions = self.fallbacks.run(fallback_names, normalized)
        scores.extend(fallback_scores)
        model_versions.update(fallback_versions)
        if fallback_scores:
            for score in fallback_scores:
                detector_sources.extend(score.sources)
        fallbacks_used = sorted(
            name for name, version in fallback_versions.items() if version != "absent"
        )

        category_scores, risk_score, confidence, labels = aggregate_scores(
            scores,
            channel_source=channel.source,
        )

        return ClassificationResult(
            channel_source=channel.source,
            labels=labels,
            category_scores=category_scores,
            risk_score=risk_score,
            confidence=confidence,
            detector_sources=sorted(set(detector_sources)),
            fallbacks_used=fallbacks_used,
            model_versions=model_versions,
            normalized_text_hash=normalized_hash,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
