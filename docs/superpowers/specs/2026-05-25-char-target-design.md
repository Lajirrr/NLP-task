# Character-Level Chinese Target Design

Date: 2026-05-25

## Goal

Add an optional character-level Chinese target mode to reduce target-side `<UNK>` errors while keeping the existing word-level Transformer workflow compatible.

## Design

English source processing remains unchanged and continues to use `data/word2int_en.json`.

Chinese target processing gains two modes:

- `word`: current behavior; target tokens come from whitespace-separated Chinese words and `data/word2int_cn.json`.
- `char`: remove spaces between the existing Chinese target tokens and split the resulting Chinese sentence into characters.

Example:

```text
他 是 老师 。
```

becomes:

```text
他 是 老 师 。
```

The character vocabulary is built from the dataset splits in `data/training.txt`, `data/validation.txt`, and `data/testing.txt`, with special tokens `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>` at the same ids as the existing vocabularies.

## Checkpoint Compatibility

Existing word-level checkpoints continue to load normally.

New char-level checkpoints store:

- `target_level`
- `tgt_vocab`
- `int2word_tgt`

`evaluate.py` and `translate.py` load these fields from the checkpoint and automatically apply the correct target tokenization.

## Commands

Train a character-level target model:

```powershell
python code\train.py --target-level char --epochs 50 --batch-size 64 --device cuda --checkpoint-dir checkpoints\char
```

Evaluate it:

```powershell
python code\evaluate.py --checkpoint checkpoints\char\best.pt --device cuda --beam-size 4
```

Translate:

```powershell
python code\translate.py --checkpoint checkpoints\char\best.pt --device cuda --beam-size 4 --text "tom is a student ."
```

## Expected Effect

Character-level output should greatly reduce Chinese target `<UNK>` occurrences. It may not fully solve semantic errors caused by the small 18k-pair training set, but it should improve readability for low-frequency Chinese words.
