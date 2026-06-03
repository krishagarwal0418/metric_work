from __future__ import annotations

from dataclasses import dataclass

from observation_labeler.types import ObservationChannel


@dataclass(frozen=True)
class ValidationConfig:
    max_bytes_user: int = 65536
    max_bytes_context: int = 262144
    max_tokens: int = 4096

    def max_bytes_for(self, source: str) -> int:
        return self.max_bytes_user if source == "user_input" else self.max_bytes_context


@dataclass(frozen=True)
class ValidationError:
    reason: str


def validate_channel(channel: ObservationChannel, config: ValidationConfig) -> ValidationError | None:
    if not channel.content or not channel.content.strip():
        return ValidationError("empty_content")
    try:
        encoded = channel.content.encode("utf-8")
    except UnicodeEncodeError:
        return ValidationError("non_utf8")
    if len(encoded) > config.max_bytes_for(channel.source):
        return ValidationError("oversize_bytes")
    if len(channel.content) // 4 > config.max_tokens:
        return ValidationError("oversize_tokens")
    return None
