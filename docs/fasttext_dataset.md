# FastText Observation Dataset

This repo now has a Colab-friendly dataset builder for training a stronger
fastText layer across the observation labels:

- `prompt_injection`
- `jailbreak`
- `toxicity`
- `hate`
- `sexual`
- `violence`
- `self_harm`
- `safe`

The fastText model should be used as a router and signal contributor, not as
the only final action labeler.

## Sources

The builder currently supports:

- `rogue_prompt_injections`
  - Dataset: `rogue-security/prompt-injections-benchmark`
  - Labels: `jailbreak`, `benign`
  - Notes: gated, 5k rows, useful but benign class contains jailbreak-shaped roleplay.

- `deepset_prompt_injections`
  - Dataset: `deepset/prompt-injections`
  - Labels: binary prompt injection.

- `lakera_gandalf`
  - Dataset: `Lakera/gandalf-rct-attack-categories`
  - Labels: password-extraction / Gandalf attack categories mapped to prompt injection + jailbreak.

- `cyberec_prompt_injection`
  - Dataset: `cyberec/Prompt-injection-dataset`
  - Labels: binary prompt-injection style labels.

- `qualifire_safety`
  - Dataset: `qualifire/safety-benchmark`
  - Labels mapped from dangerous content, harassment, sexual explicit, hate, safe.
  - Notes: gated.

- `toxigen`
  - Dataset: `toxigen/toxigen-data`
  - Labels mapped to toxicity/hate/sexual signals.

- `open_moderator`
  - Dataset: `TeichAI/open-moderator-v1`
  - Labels mapped from moderation categories.

- `ifmain_text_moderation_multilingual`
  - Dataset: `ifmain/text-moderation-02-multilingual`
  - Labels mapped from OpenAI-style moderation categories/scores.

- `zysec_harmful_behaviors`
  - Dataset: `ZySec-AI/harmful_behaviors`
  - Labels mapped from OpenAI-style moderation categories/scores.

- `quantaspark_cortyx_safety`
  - Dataset: `QuantaSparkLabs/cortyx-safety-dataset`
  - Labels mapped from safety category names.

- `sivasothy_self_harm`
  - Dataset: `sivasothy-Tharsi/self-harm-detection`
  - Labels mapped to `self_harm` / `safe`.

- `enguard_prompt_moderation`
  - Dataset: `enguard/multi-lingual-prompt-moderation`
  - Labels mapped from moderation category names.

- `ucb_measuring_hate_speech`
  - Dataset: `ucberkeley-dlab/measuring-hate-speech`
  - Labels mapped from `hate_speech_score`.

## Colab Setup

```bash
git clone <your-metric-work-repo-url> metric_work
cd metric_work
pip install -e ".[data,fasttext]"
```

If you want gated sources such as Rogue/Qualifire, log in or set a token:

```bash
export HF_TOKEN=...
```

Do not hardcode the token into notebooks committed to git.

## Build Dataset

Small smoke build:

```bash
python scripts/build_fasttext_dataset.py \
  --sources deepset_prompt_injections,toxigen \
  --limit-per-source 1000 \
  --output-dir data/fasttext_smoke
```

Quality build including gated sources:

```bash
python scripts/build_fasttext_dataset.py \
  --preset quality \
  --include-gated \
  --max-per-source 75000 \
  --output-dir data/fasttext_corpus
```

Balanced build for first training attempts:

```bash
python scripts/build_fasttext_dataset.py \
  --preset quality \
  --include-gated \
  --max-per-source 75000 \
  --max-per-label 75000 \
  --output-dir data/fasttext_quality_v3
```

Avoid `--sources all` for training. It is useful for debugging source adapters,
but it can include weak schema mappings. The `quality` preset keeps direct
prompt-injection sources, moderation-score sources, and hard-negative safe
question data.

Outputs:

- `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `train.txt`, `valid.txt`, `test.txt`
- `manifest.json`

The `.txt` files are fastText format:

```text
__label__prompt_injection __label__jailbreak ignore previous instructions...
__label__safe what is the payment policy?
```

## Train FastText

Baseline:

```bash
python scripts/train_fasttext.py \
  --train data/fasttext_quality_v3/train.txt \
  --valid data/fasttext_quality_v3/valid.txt \
  --test data/fasttext_quality_v3/test.txt \
  --output fasttext_observation_v0_1.bin \
  --lr 0.4 \
  --epoch 30 \
  --word-ngrams 3 \
  --minn 3 \
  --maxn 6 \
  --dim 200 \
  --bucket 2000000 \
  --loss ova \
  --threshold 0.35 \
  --thread 8
```

Recommended sweep:

```text
loss: ova
epoch: 15, 25, 40
lr: 0.2, 0.4, 0.7
wordNgrams: 2, 3, 4
dim: 100, 200, 300
minn/maxn: 2-5, 3-6
```

For our architecture, optimize fastText for:

- high recall as a router
- reasonable per-category separation
- calibrated scores for routing

Do not optimize it as the only final action labeler.

## Evaluation Notes

Evaluate separately by channel:

- `user_input`: action recall matters.
- `system_prompt`: attack-looking text may be policy/example text, so it should stay as signal/reference.
- `retrieved_context`: indirect injection and context contamination.
- `tool_output`: tool-originated unsafe output.
- `llm_output`: model-generated unsafe output.

Keep raw source metadata in `manifest.json` and JSONL rows so errors can be
analyzed by dataset source.
