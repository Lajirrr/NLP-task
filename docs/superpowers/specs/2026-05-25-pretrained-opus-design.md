# Pretrained OPUS-MT Integration Design

Date: 2026-05-25

## Goal

Add an optional pretrained English-to-Chinese translation path while preserving the existing from-scratch Transformer pipeline.

## Selected Approach

Use `Helsinki-NLP/opus-mt-en-zh` as the first pretrained model. It is small enough for quick local experiments and directly targets English-to-Chinese translation.

The existing scripts remain unchanged in behavior:

- `code/train.py`
- `code/evaluate.py`
- `code/translate.py`

The pretrained workflow is added through separate entry points:

- `code/pretrained_translate.py`
- `code/pretrained_evaluate.py`
- `code/pretrained_finetune.py`

## Data Flow

The project dataset is tab-separated and already tokenized. For pretrained models, source English tokens are converted back to natural English text:

```text
do n't underestimate my power .
```

becomes:

```text
don't underestimate my power.
```

Chinese target tokens are detokenized by direct concatenation, matching the current project behavior.

Evaluation computes project-local character-level BLEU so results are broadly comparable to the current `--target-level char` runs. Pretrained evaluation may optionally report teacher-forced loss when labels are available.

## Checkpoints

Fine-tuned pretrained checkpoints are saved with Hugging Face `save_pretrained()` into a separate directory:

```text
checkpoints/pretrained-opus-en-zh/
```

This avoids overwriting existing checkpoints such as:

- `checkpoints/best.pt`
- `checkpoints/char/best.pt`

## Dependencies

Add the required Hugging Face runtime dependencies to `requirements.txt`:

- `transformers`
- `sentencepiece`
- `sacremoses`

`transformers` is already installed in the current conda `base` environment, but `sentencepiece` and `sacremoses` are missing and are required by Marian/OPUS tokenizers.

## Verification

Verification should cover:

- Unit tests for dataset detokenization and character BLEU input construction.
- Python compilation for all code scripts.
- A direct pretrained translation smoke test.
- A bounded pretrained evaluation smoke test using a small `--limit`.

