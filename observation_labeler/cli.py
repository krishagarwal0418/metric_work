from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from observation_labeler import ObservationChannel, ObservationLabeler
from observation_labeler.types import ChannelSource


def _read_channels(args: argparse.Namespace) -> list[ObservationChannel]:
    if args.json_file:
        payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("channels", [payload])
        return [ObservationChannel(**item) for item in payload]

    content = args.text
    if content is None and not sys.stdin.isatty():
        content = sys.stdin.read()
    if content is None:
        raise SystemExit("provide --text, --json-file, or stdin")
    return [ObservationChannel(source=args.source, content=content)]


def _compact(result: Any) -> dict[str, Any]:
    data = result.model_dump()
    for channel in data.get("channel_results", []):
        channel.pop("normalized_text_hash", None)
    data.pop("normalized_text_hash", None)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify observation channels without guardrail decisions.")
    parser.add_argument("--text", help="Text to classify. If omitted, stdin is used.")
    parser.add_argument(
        "--source",
        default="user_input",
        choices=["user_input", "retrieved_context", "tool_output", "llm_output", "system_prompt"],
        help="Channel source for --text/stdin.",
    )
    parser.add_argument("--json-file", help="JSON file with one channel or {'channels': [...]} payload.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    channels = _read_channels(args)
    labeler = ObservationLabeler()
    result = (
        labeler.classify_channel(channels[0])
        if len(channels) == 1
        else labeler.classify_observation(channels)
    )
    print(json.dumps(_compact(result), indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
