# SentencePiece Target And Validation BLEU Design

Date: 2026-06-07

## Goal

Improve the from-scratch Transformer path without using pretrained translations by adding a SentencePiece Chinese target mode and optional validation BLEU checkpoint selection.

This keeps the project educational and self-contained while borrowing two mature NMT practices from OpenNMT/fairseq-style workflows:

- subword target modeling instead of pure word or character targets
- selecting checkpoints by translation quality, not only teacher-forced loss

## Scope

Included:

- Add `target-level subword`.
- Train a Chinese target-side SentencePiece model from the local dataset only.
- Store enough tokenizer metadata in checkpoints for evaluation and translation.
- Add optional validation BLEU decoding during training.
- Allow best checkpoint selection by either validation loss or validation BLEU.
- Keep existing `word` and `char` modes compatible.

Excluded:

- OPUS or other pretrained teacher distillation.
- Source-side SentencePiece in the first pass.
- Replacing the current PyTorch `nn.Transformer` model.
- Changing the dataset files under `data/`.

## SentencePiece Target Mode

The existing data lines keep their current format:

```text
english tokens<TAB>chinese tokens
```

For `target-level subword`, Chinese target tokens are detokenized by removing spaces between the provided Chinese tokens. SentencePiece then encodes the resulting Chinese string into target pieces.

Example:

```text
CN_TOKEN_1 CN_TOKEN_2 CN_TOKEN_3 CN_PUNCT
```

becomes:

```text
CN_TOKEN_1CN_TOKEN_2CN_TOKEN_3CN_PUNCT
```

and is encoded into SentencePiece pieces such as:

```text
SP_PIECE_1 SP_PIECE_2 SP_PIECE_3
```

Exact pieces depend on the trained SentencePiece model and vocabulary size.

## Tokenizer Files

Add a small utility script for target-side SentencePiece training:

```text
code/train_sentencepiece.py
```

Default output location:

```text
tokenizers/spm-target-vocab1200.model
tokenizers/spm-target-vocab1200.vocab
```

The script reads local splits only. The default training input should use `training.txt`; an optional `--include-valid-test` flag may include validation and test text only for controlled ablation, but the recommended command must avoid test leakage.

Recommended first command:

```powershell
python code\train_sentencepiece.py --data-dir data --vocab-size 1200 --model-prefix tokenizers\spm-target-vocab1200
```

## Dataset Loading

Extend `code/dataset.py`:

- Add `TARGET_LEVEL_SUBWORD = "subword"`.
- Add a lightweight SentencePiece tokenizer wrapper.
- For `target-level subword`, require `--target-spm-model`.
- Build target vocab from the SentencePiece model:
  - `<PAD>` id remains `0`.
  - `<BOS>` id remains `1`.
  - `<EOS>` id remains `2`.
  - `<UNK>` id remains `3`.
  - SentencePiece pieces are shifted after the four special ids if needed.

The project should not depend on SentencePiece ids matching current special-token ids. The dataset layer owns the mapping between SentencePiece piece ids and project ids.

## Checkpoint Metadata

Subword checkpoints store:

- `target_level = "subword"`
- `target_spm_model`
- `tgt_vocab`
- `int2word_tgt`
- model config and existing training options

`evaluate.py` and `translate.py` must load these fields automatically, so a subword checkpoint can be evaluated without retyping tokenizer arguments.

## Decoding And Output

For `word` and `char`, output behavior stays unchanged.

For `subword`:

1. Convert predicted ids to SentencePiece pieces.
2. Stop at `<EOS>`.
3. Remove special tokens.
4. Decode pieces with the loaded SentencePiece model.

This should preserve Chinese punctuation and reduce the awkward repetition patterns seen with pure character decoding.

## Validation BLEU Selection

Extend `code/train.py` with optional BLEU-based checkpointing:

```powershell
--select-metric loss|bleu
--valid-decode-limit 500
--valid-beam-size 1
--valid-length-penalty 1.0
--valid-no-repeat-ngram-size 0
```

Default behavior remains current loss-based selection:

```text
--select-metric loss
```

When `--select-metric bleu` is enabled:

1. Run normal validation loss each epoch.
2. Decode validation examples with the configured quick BLEU settings.
3. Save `best.pt` when validation BLEU improves.
4. Still print validation loss, perplexity, and BLEU.
5. Store the selected metric and score in checkpoint metadata.

For speed, the first pass should use greedy validation BLEU:

```powershell
--valid-beam-size 1 --valid-decode-limit 500
```

The validation set has only 500 examples, so this remains practical after encoder-memory decode caching.

## Recommended Experiment

Train the target tokenizer:

```powershell
python code\train_sentencepiece.py --data-dir data --vocab-size 1200 --model-prefix tokenizers\spm-target-vocab1200
```

Train the model:

```powershell
python code\train.py --target-level subword --target-spm-model tokenizers\spm-target-vocab1200.model --epochs 60 --batch-size 64 --device cuda --checkpoint-dir checkpoints\spm1200-bleu --label-smoothing 0.1 --scheduler noam --warmup-steps 4000 --keep-best-checkpoints 8 --select-metric bleu --valid-decode-limit 500 --valid-beam-size 1
```

Average kept checkpoints:

```powershell
python code\average_checkpoints.py --inputs checkpoints\spm1200-bleu\best_epoch_*.pt --output checkpoints\spm1200-bleu\averaged.pt
```

Evaluate:

```powershell
python code\evaluate.py --checkpoint checkpoints\spm1200-bleu\best.pt --device cuda --beam-size 4 --length-penalty 1.5 --no-repeat-ngram-size 2
python code\evaluate.py --checkpoint checkpoints\spm1200-bleu\averaged.pt --device cuda --beam-size 4 --length-penalty 1.5 --no-repeat-ngram-size 2
```

## Expected Effect

Subword targets may improve semantic consistency over character decoding while keeping unknown-token risk low. Validation BLEU selection should reduce cases where teacher-forced loss improves but actual translation quality does not.

Expected full-test BLEU target for the first pass:

- Conservative: `0.35` to `0.37`, matching the current ensemble range with a single cleaner model.
- Optimistic: `0.38` or slightly higher if validation BLEU selection chooses better checkpoints than loss selection.

Reaching `0.4` remains possible but not guaranteed without source-side subwords, data augmentation, or distillation.

## Testing

Add unit tests for:

- SentencePiece target training writes model and vocab files on a tiny corpus.
- `target-level subword` encodes and decodes Chinese target text through project ids.
- Existing `word` and `char` dataset behavior remains unchanged.
- `train.py` argument parsing accepts BLEU selection flags.
- Checkpoint metadata preserves subword tokenizer path.
- `translate.py` and `evaluate.py` load subword checkpoints without requiring explicit tokenizer arguments.

Run:

```powershell
python -m unittest discover -s tests -v
python -m py_compile code\dataset.py code\train.py code\evaluate.py code\translate.py code\train_sentencepiece.py
```
