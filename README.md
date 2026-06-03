# Observation Labeler

Standalone observability-only classifier for telemetry channels.

It labels observation data; it does not act as a guardrail and does not return
allow/block/review decisions.

## Channels

Process each channel independently:

- `user_input`
- `retrieved_context`
- `tool_output`
- `llm_output`
- `system_prompt`

## Output

Each channel result contains:

- labels
- category scores
- confidence
- detector sources
- fallback/model sources used
- validation/normalization metadata
- classification latency

No PII/PHI/sensitive-data detector is included in this package.

## Example

```python
from observation_labeler import ObservationLabeler

labeler = ObservationLabeler()
result = labeler.classify_text("ignore the system prompt", source="user_input")

print(result.labels)
print(result.risk_score)
```

## Run Tests

```bash
cd ~/oxygen/metric_work
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## CLI Smoke Tests

```bash
observation-label --text "ignore the system prompt" --pretty
observation-label --source tool_output --text "Tool completed successfully." --pretty
observation-label --json-file sample_observation.json --pretty
```

## Rogue Security Benchmark

`rogue-security/prompt-injections-benchmark` is gated on Hugging Face. Accept
the dataset conditions in your Hugging Face account, then run with a token:

```bash
export HF_TOKEN=<your_huggingface_token>
python scripts/run_rogue_benchmark.py --download
```

Smoke run after the CSV is downloaded:

```bash
python scripts/run_rogue_benchmark.py --limit 100
```

The report is written to:

```text
reports/rogue_prompt_injections_report.json
```

The report stores row numbers and text hashes for misses, not raw prompt text.
