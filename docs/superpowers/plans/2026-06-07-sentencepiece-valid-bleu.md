# SentencePiece Validation BLEU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure from-scratch SentencePiece Chinese target mode and optional validation BLEU checkpoint selection.

**Architecture:** Keep the existing PyTorch Transformer and English source vocabulary. Add a target-side SentencePiece utility and dataset wrapper, persist tokenizer metadata in checkpoints, teach evaluation/translation to decode subword outputs, and let training select best checkpoints by either validation loss or validation BLEU.

**Tech Stack:** Python in conda `base`, PyTorch, SentencePiece, stdlib `unittest`, existing lightweight BLEU in `code/text_utils.py`.

---

## File Structure

- Create: `code/train_sentencepiece.py`
  - Trains a Chinese target-side SentencePiece model from local parallel splits.
  - Exposes helpers for tests: `detokenize_target_tokens`, `iter_target_texts`, `train_target_sentencepiece`.
- Modify: `code/dataset.py`
  - Adds `TARGET_LEVEL_SUBWORD`, `SentencePieceTargetTokenizer`, subword vocabulary loading, and dataset encoding support.
- Modify: `code/train.py`
  - Adds CLI flags `--target-spm-model`, `--select-metric`, `--valid-decode-limit`, `--valid-beam-size`, `--valid-length-penalty`, `--valid-no-repeat-ngram-size`.
  - Saves tokenizer and metric metadata in checkpoints.
  - Computes optional validation BLEU and chooses best checkpoint by BLEU when requested.
- Modify: `code/translate.py`
  - Loads subword tokenizer metadata from checkpoints and decodes predicted subword pieces to Chinese text.
- Modify: `code/evaluate.py`
  - Builds subword datasets from checkpoint tokenizer metadata and renders subword examples with SentencePiece decode.
- Modify: `README.md`
  - Adds the recommended SentencePiece training, model training, averaging, and evaluation commands.
- Test: `tests/test_sentencepiece_training.py`
  - New tests for tokenizer training helper.
- Modify: `tests/test_dataset.py`
  - Subword dataset encoding/decoding tests.
- Modify: `tests/test_training_utils.py`
  - CLI and checkpoint metadata tests.
- Modify: `tests/test_checkpoint_loading.py`
  - Subword checkpoint loading and translation decode tests.
- Modify: `tests/test_evaluate.py`
  - Subword example rendering and BLEU selection argument tests if not covered elsewhere.

---

## Task 1: Target SentencePiece Training Utility

**Files:**
- Create: `code/train_sentencepiece.py`
- Create: `tests/test_sentencepiece_training.py`

- [ ] **Step 1: Write failing tests for target text extraction and model training**

Create `tests/test_sentencepiece_training.py`:

```python
import tempfile
import unittest
from pathlib import Path


class SentencePieceTrainingTest(unittest.TestCase):
    def test_detokenize_target_tokens_removes_spaces(self):
        from code.train_sentencepiece import detokenize_target_tokens

        self.assertEqual(detokenize_target_tokens(["CN1", "CN2", "."]), "CN1CN2.")

    def test_iter_target_texts_reads_training_split_by_default(self):
        from code.train_sentencepiece import iter_target_texts

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "training.txt").write_text(
                "hello .\tCN1 CN2 .\n", encoding="utf-8"
            )
            (data_dir / "validation.txt").write_text(
                "bye .\tCN3 CN4 .\n", encoding="utf-8"
            )
            (data_dir / "testing.txt").write_text(
                "thanks .\tCN5 CN5 .\n", encoding="utf-8"
            )

            texts = list(iter_target_texts(data_dir, include_valid_test=False))

        self.assertEqual(texts, ["CN1CN2."])

    def test_train_target_sentencepiece_writes_model_and_vocab(self):
        from code.train_sentencepiece import train_target_sentencepiece

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            lines = [
                "hello .\tCN1 CN2 .\n",
                "i am tom .\tCN3 CN4 CN5 .\n",
                "thanks .\tCN6 CN6 CN1 .\n",
                "good morning .\tCN7 CN8 CN2 .\n",
            ]
            (data_dir / "training.txt").write_text("".join(lines), encoding="utf-8")
            model_prefix = Path(tmpdir) / "spm-target"

            model_path = train_target_sentencepiece(
                data_dir=data_dir,
                model_prefix=model_prefix,
                vocab_size=32,
                include_valid_test=False,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(model_path.with_suffix(".vocab").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python -m unittest tests.test_sentencepiece_training -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'code.train_sentencepiece'`.

- [ ] **Step 3: Implement `code/train_sentencepiece.py`**

Create `code/train_sentencepiece.py`:

```python
import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sentencepiece as spm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install sentencepiece in the active Python environment."
    ) from exc

from code.config import DATA_DIR
from code.dataset import parse_parallel_line


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train target-side SentencePiece.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-prefix", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=1200)
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--include-valid-test", action="store_true")
    return parser.parse_args(argv)


def detokenize_target_tokens(tokens: Iterable[str]) -> str:
    return "".join(token for token in tokens if token and not token.isspace())


def iter_target_texts(data_dir: Path, include_valid_test: bool = False):
    split_names = ["training.txt"]
    if include_valid_test:
        split_names.extend(["validation.txt", "testing.txt"])
    for split_name in split_names:
        split_path = data_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        with split_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    _, tgt_tokens = parse_parallel_line(line)
                except ValueError as exc:
                    raise ValueError(f"{split_path}:{line_number}: {exc}") from exc
                text = detokenize_target_tokens(tgt_tokens)
                if text:
                    yield text


def train_target_sentencepiece(
    data_dir: Path,
    model_prefix: Path,
    vocab_size: int = 1200,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    include_valid_test: bool = False,
) -> Path:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    texts = list(iter_target_texts(data_dir, include_valid_test=include_valid_test))
    if not texts:
        raise ValueError(f"No target texts found under {data_dir}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        input_path = Path(tmp.name)
        for text in texts:
            tmp.write(text + "\n")
    try:
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            hard_vocab_limit=False,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
        )
    finally:
        input_path.unlink(missing_ok=True)
    return model_prefix.with_suffix(".model")


def main():
    args = parse_args()
    model_path = train_target_sentencepiece(
        data_dir=args.data_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        include_valid_test=args.include_valid_test,
    )
    print(f"Saved SentencePiece model to {model_path}")
    print(f"Saved SentencePiece vocab to {model_path.with_suffix('.vocab')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_sentencepiece_training -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add code\train_sentencepiece.py tests\test_sentencepiece_training.py
git commit -m "Add target SentencePiece training utility"
```

---

## Task 2: Subword Dataset And Vocabulary Support

**Files:**
- Modify: `code/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Add to `tests/test_dataset.py`:

```python
    def test_sentencepiece_target_tokenizer_maps_piece_ids_after_specials(self):
        from code.dataset import SentencePieceTargetTokenizer

        class FakeProcessor:
            def get_piece_size(self):
                return 3

            def id_to_piece(self, index):
                return ["<unk>", "CN1", "CN2"][index]

            def encode(self, text, out_type=str):
                self.assertEqual(text, "CN1CN2.")
                return ["CN1", "CN2"]

            def decode(self, pieces):
                return "".join(piece for piece in pieces if piece != "<unk>")

        tokenizer = SentencePieceTargetTokenizer.from_processor(FakeProcessor())

        self.assertEqual(tokenizer.vocab["<PAD>"], PAD_ID)
        self.assertEqual(tokenizer.vocab["<BOS>"], BOS_ID)
        self.assertEqual(tokenizer.vocab["<EOS>"], EOS_ID)
        self.assertEqual(tokenizer.vocab["<UNK>"], UNK_ID)
        self.assertEqual(tokenizer.vocab["CN1"], 5)
        self.assertEqual(tokenizer.encode_tokens(["CN1", "CN2", "."]), [5, 6])
        self.assertEqual(tokenizer.decode_ids([5, 6]), "CN1CN2")

    def test_translation_dataset_supports_subword_targets(self):
        from code.dataset import SentencePieceTargetTokenizer

        class FakeProcessor:
            def get_piece_size(self):
                return 3

            def id_to_piece(self, index):
                return ["<unk>", "CN1", "CN2"][index]

            def encode(self, text, out_type=str):
                return ["CN1", "CN2"]

            def decode(self, pieces):
                return "".join(pieces)

        tokenizer = SentencePieceTargetTokenizer.from_processor(FakeProcessor())

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "training.txt"
            split_path.write_text("hello .\tCN1 CN2 .\n", encoding="utf-8")
            src_vocab = {
                "<PAD>": PAD_ID,
                "<BOS>": BOS_ID,
                "<EOS>": EOS_ID,
                "<UNK>": UNK_ID,
                "hello": 4,
                ".": 5,
            }
            dataset = TranslationDataset(
                split_path,
                src_vocab,
                tokenizer.vocab,
                target_level="subword",
                target_tokenizer=tokenizer,
            )

        self.assertEqual(dataset[0]["tgt_tokens"], ["CN1", "CN2"])
        self.assertEqual(dataset[0]["tgt"], [5, 6])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_dataset -v
```

Expected: FAIL because `SentencePieceTargetTokenizer` and `target_level="subword"` do not exist yet.

- [ ] **Step 3: Implement subword support in `code/dataset.py`**

Patch `code/dataset.py`:

```python
TARGET_LEVEL_SUBWORD = "subword"
TARGET_LEVELS = {TARGET_LEVEL_WORD, TARGET_LEVEL_CHAR, TARGET_LEVEL_SUBWORD}
```

Add imports:

```python
try:
    import sentencepiece as spm
except ImportError:
    spm = None
```

Add class:

```python
class SentencePieceTargetTokenizer:
    def __init__(self, model_path: Path | None = None, processor=None):
        if processor is None:
            if spm is None:
                raise RuntimeError("sentencepiece is required for target-level subword.")
            if model_path is None:
                raise ValueError("model_path is required when processor is not provided.")
            processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.model_path = Path(model_path) if model_path is not None else None
        self.processor = processor
        self.vocab, self.int2word = self._build_project_vocab()

    @classmethod
    def from_processor(cls, processor):
        return cls(processor=processor)

    def _build_project_vocab(self):
        vocab = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }
        for piece_id in range(self.processor.get_piece_size()):
            piece = self.processor.id_to_piece(piece_id)
            if piece not in vocab:
                vocab[piece] = len(vocab)
        return vocab, {index: token for token, index in vocab.items()}

    def encode_tokens(self, tgt_tokens: Iterable[str]) -> List[int]:
        text = "".join(token for token in tgt_tokens if token and not token.isspace())
        pieces = self.processor.encode(text, out_type=str)
        return [self.vocab.get(piece, UNK_ID) for piece in pieces]

    def decode_ids(self, ids: Iterable[int]) -> str:
        pieces = []
        for index in ids:
            token = self.int2word.get(int(index), UNK_TOKEN)
            if token == EOS_TOKEN:
                break
            if token in {PAD_TOKEN, BOS_TOKEN, UNK_TOKEN}:
                continue
            pieces.append(token)
        return self.processor.decode(pieces)
```

Update `normalize_target_tokens`:

```python
    if target_level == TARGET_LEVEL_WORD:
        return tokens
    if target_level == TARGET_LEVEL_CHAR:
        return [char for char in "".join(tokens) if not char.isspace()]
    return tokens
```

Update `load_vocabularies` signature and behavior:

```python
def load_vocabularies(
    data_dir: Path = DATA_DIR,
    target_level: str = TARGET_LEVEL_WORD,
    target_spm_model: Path | None = None,
) -> Tuple[Vocab, ReverseVocab, Vocab, ReverseVocab]:
    ...
    if target_level == TARGET_LEVEL_CHAR:
        tgt_vocab, int2word_tgt = build_char_vocab_from_splits(data_dir)
    elif target_level == TARGET_LEVEL_SUBWORD:
        if target_spm_model is None:
            raise ValueError("target_spm_model is required for target-level subword.")
        tokenizer = SentencePieceTargetTokenizer(target_spm_model)
        tgt_vocab, int2word_tgt = tokenizer.vocab, tokenizer.int2word
    else:
        ...
```

Update `TranslationDataset.__init__` signature and encoding:

```python
def __init__(..., target_level: str = TARGET_LEVEL_WORD, target_tokenizer=None):
    ...
    if target_level == TARGET_LEVEL_SUBWORD and target_tokenizer is None:
        raise ValueError("target_tokenizer is required for target-level subword.")
    ...
    if target_level == TARGET_LEVEL_SUBWORD:
        normalized_tgt_tokens = target_tokenizer.processor.encode(
            "".join(tgt_tokens), out_type=str
        )
        target_ids = [tgt_vocab.get(token, UNK_ID) for token in normalized_tgt_tokens]
    else:
        normalized_tgt_tokens = normalize_target_tokens(tgt_tokens, target_level=target_level)
        target_ids = encode_tokens(normalized_tgt_tokens, tgt_vocab, add_eos=False)
```

Store `normalized_tgt_tokens` in `"tgt_tokens"` and `target_ids` in `"tgt"`.

- [ ] **Step 4: Run dataset tests**

Run:

```powershell
python -m unittest tests.test_dataset -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add code\dataset.py tests\test_dataset.py
git commit -m "Add subword target dataset support"
```

---

## Task 3: Subword Checkpoint Loading, Translation, And Evaluation Output

**Files:**
- Modify: `code/translate.py`
- Modify: `code/evaluate.py`
- Modify: `tests/test_checkpoint_loading.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing checkpoint and rendering tests**

Add to `tests/test_checkpoint_loading.py`:

```python
    def test_subword_checkpoint_restores_tokenizer_metadata(self):
        from code.config import DATA_DIR, TransformerConfig
        from code.dataset import load_vocabularies
        from code.train import build_model, save_checkpoint
        from code.translate import load_checkpoint_model

        src_vocab, _, _, _ = load_vocabularies(DATA_DIR)
        tgt_vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "CN1": 4}
        int2word_tgt = {index: token for token, index in tgt_vocab.items()}
        config = TransformerConfig(
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model = build_model(config, len(src_vocab), len(tgt_vocab))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "subword.pt"
            spm_path = Path(tmpdir) / "target.model"
            spm_path.write_bytes(b"fake")
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                epoch=0,
                valid_loss=0.0,
                target_level="subword",
                tgt_vocab=tgt_vocab,
                int2word_tgt=int2word_tgt,
                target_spm_model=spm_path,
            )
            loaded_model, _, _, loaded_tgt_vocab, loaded_int2word_tgt = load_checkpoint_model(
                checkpoint_path, DATA_DIR, torch.device("cpu")
            )

        self.assertEqual(getattr(loaded_model, "target_level"), "subword")
        self.assertEqual(getattr(loaded_model, "target_spm_model"), str(spm_path))
        self.assertEqual(loaded_tgt_vocab, tgt_vocab)
        self.assertEqual(loaded_int2word_tgt[4], "CN1")
```

Add to `tests/test_evaluate.py`:

```python
    def test_render_prediction_uses_subword_tokenizer_when_available(self):
        from code.evaluate import render_target_tokens

        class FakeTokenizer:
            def decode_ids(self, ids):
                self.ids = list(ids)
                return "CN1CN2"

        text = render_target_tokens([4, 5, 2], {4: "CN1", 5: "CN2", 2: "<EOS>"}, FakeTokenizer())

        self.assertEqual(text, "CN1CN2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_checkpoint_loading tests.test_evaluate -v
```

Expected: FAIL because checkpoint metadata and `render_target_tokens` are not implemented.

- [ ] **Step 3: Extend checkpoint save/load**

Modify `code/train.py` `save_checkpoint` signature:

```python
def save_checkpoint(..., target_spm_model: Path | str | None = None, selected_metric: str | None = None, selected_score: float | None = None):
```

Inside `save_checkpoint`:

```python
if target_spm_model is not None:
    checkpoint["target_spm_model"] = str(target_spm_model)
if selected_metric is not None:
    checkpoint["selected_metric"] = selected_metric
if selected_score is not None:
    checkpoint["selected_score"] = selected_score
```

Modify `code/translate.py` imports:

```python
from code.dataset import TARGET_LEVEL_SUBWORD, TARGET_LEVEL_WORD, SentencePieceTargetTokenizer, load_vocabularies
```

Modify `load_checkpoint_model`:

```python
target_spm_model = checkpoint.get("target_spm_model")
...
model.target_spm_model = target_spm_model
model.target_tokenizer = (
    SentencePieceTargetTokenizer(Path(target_spm_model))
    if target_level == TARGET_LEVEL_SUBWORD and target_spm_model
    else None
)
```

When no checkpoint vocab is present, call:

```python
load_vocabularies(data_dir, target_level=target_level, target_spm_model=target_spm_model)
```

- [ ] **Step 4: Add rendering helpers**

In `code/translate.py`, add:

```python
def ids_to_text(ids: Iterable[int], int2word: dict, target_tokenizer=None) -> str:
    if target_tokenizer is not None:
        return target_tokenizer.decode_ids(ids)
    return detokenize_chinese(ids_to_tokens(ids, int2word))
```

Update `translate_text`:

```python
pred_tokens = ids_to_tokens(pred_ids, int2word_cn)
translation = ids_to_text(pred_ids, int2word_cn, getattr(model, "target_tokenizer", None))
return src_tokens, pred_tokens, translation
```

In `code/evaluate.py`, add:

```python
def render_target_tokens(ids_or_tokens, int2word_cn=None, target_tokenizer=None):
    if target_tokenizer is not None and int2word_cn is not None:
        return target_tokenizer.decode_ids(ids_or_tokens)
    return detokenize_chinese(ids_or_tokens)
```

Update `evaluate_bleu_and_examples` so predictions remain token lists for `corpus_bleu`, but examples use the tokenizer for subword output:

```python
target_tokenizer = getattr(model, "target_tokenizer", None)
...
pred_text = (
    target_tokenizer.decode_ids(pred_ids)
    if target_tokenizer is not None
    else detokenize_chinese(pred_tokens)
)
ref_text = (
    target_tokenizer.processor.decode(example["tgt_tokens"])
    if target_tokenizer is not None
    else detokenize_chinese(example["tgt_tokens"])
)
```

When creating `TranslationDataset` in `evaluate.py`, pass:

```python
target_tokenizer=getattr(model, "target_tokenizer", None)
```

- [ ] **Step 5: Run checkpoint/evaluate tests**

Run:

```powershell
python -m unittest tests.test_checkpoint_loading tests.test_evaluate -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code\translate.py code\evaluate.py code\train.py tests\test_checkpoint_loading.py tests\test_evaluate.py
git commit -m "Load and render subword checkpoints"
```

---

## Task 4: Validation BLEU Checkpoint Selection

**Files:**
- Modify: `code/train.py`
- Modify: `tests/test_training_utils.py`

- [ ] **Step 1: Write failing training utility tests**

Add to `tests/test_training_utils.py`:

```python
    def test_parse_args_accepts_subword_and_bleu_selection(self):
        from code.train import parse_args

        args = parse_args(
            [
                "--target-level",
                "subword",
                "--target-spm-model",
                "tokenizers/spm-target-vocab1200.model",
                "--select-metric",
                "bleu",
                "--valid-decode-limit",
                "500",
                "--valid-beam-size",
                "4",
                "--valid-length-penalty",
                "1.5",
                "--valid-no-repeat-ngram-size",
                "2",
            ]
        )

        self.assertEqual(args.target_level, "subword")
        self.assertEqual(str(args.target_spm_model), "tokenizers\\spm-target-vocab1200.model")
        self.assertEqual(args.select_metric, "bleu")
        self.assertEqual(args.valid_decode_limit, 500)
        self.assertEqual(args.valid_beam_size, 4)
        self.assertEqual(args.valid_length_penalty, 1.5)
        self.assertEqual(args.valid_no_repeat_ngram_size, 2)

    def test_is_improved_metric_handles_loss_and_bleu(self):
        from code.train import is_improved_metric

        self.assertTrue(is_improved_metric("loss", current=1.0, best=1.5))
        self.assertFalse(is_improved_metric("loss", current=2.0, best=1.5))
        self.assertTrue(is_improved_metric("bleu", current=0.30, best=0.20))
        self.assertFalse(is_improved_metric("bleu", current=0.10, best=0.20))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_prune_best_checkpoints_can_keep_highest_bleu_scores(self):
        from code.train import prune_best_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            records = []
            for epoch, score in [(1, 0.2), (2, 0.3), (3, 0.1)]:
                path = Path(tmpdir) / f"best_epoch_{epoch:02d}.pt"
                path.write_text(str(score), encoding="utf-8")
                records.append({"path": path, "score": score})

            kept = prune_best_checkpoints(records, keep=2, mode="max", score_key="score")

            self.assertEqual([item["path"].name for item in kept], ["best_epoch_02.pt", "best_epoch_01.pt"])
            self.assertFalse((Path(tmpdir) / "best_epoch_03.pt").exists())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_training_utils -v
```

Expected: FAIL because CLI flags and metric helpers do not exist.

- [ ] **Step 3: Add CLI and metric helpers**

Modify `code/train.py` imports:

```python
from code.dataset import TARGET_LEVEL_SUBWORD, SentencePieceTargetTokenizer, ...
from code.evaluate import evaluate_bleu_and_examples
```

To avoid a circular import with `evaluate.py` importing `move_batch_to_device`, put the validation BLEU helper in `train.py` instead of importing `evaluate.py` if a circular import appears:

```python
from code.text_utils import corpus_bleu
from code.translate import decode_sequence, ids_to_tokens
```

Add CLI flags:

```python
parser.add_argument("--target-spm-model", type=Path, default=None)
parser.add_argument("--select-metric", choices=["loss", "bleu"], default="loss")
parser.add_argument("--valid-decode-limit", type=int, default=None)
parser.add_argument("--valid-beam-size", type=int, default=1)
parser.add_argument("--valid-length-penalty", type=float, default=1.0)
parser.add_argument("--valid-no-repeat-ngram-size", type=int, default=0)
```

Add helpers:

```python
def is_improved_metric(metric: str, current: float, best: float) -> bool:
    if metric == "loss":
        return current < best
    if metric == "bleu":
        return current > best
    raise ValueError(f"Unsupported select metric: {metric}")
```

Update `prune_best_checkpoints` signature:

```python
def prune_best_checkpoints(records: list[dict], keep: int, mode: str = "min", score_key: str = "valid_loss") -> list[dict]:
    reverse = mode == "max"
    sorted_records = sorted(records, key=lambda item: (item[score_key], str(item["path"])), reverse=reverse)
```

When `mode="max"`, break ties deterministically by sorting manually:

```python
if mode == "max":
    sorted_records = sorted(records, key=lambda item: (-item[score_key], str(item["path"])))
else:
    sorted_records = sorted(records, key=lambda item: (item[score_key], str(item["path"])))
```

- [ ] **Step 4: Wire subword tokenizer into training datasets**

In `main()`:

```python
target_tokenizer = None
if args.target_level == TARGET_LEVEL_SUBWORD:
    if args.target_spm_model is None:
        raise ValueError("--target-spm-model is required when --target-level subword")
    target_tokenizer = SentencePieceTargetTokenizer(args.target_spm_model)

src_vocab, _, tgt_vocab, int2word_tgt = load_vocabularies(
    config.data_dir,
    target_level=args.target_level,
    target_spm_model=args.target_spm_model,
)
```

Pass `target_tokenizer=target_tokenizer` to both train and validation datasets.

Add to `training_options`:

```python
"target_spm_model": str(args.target_spm_model) if args.target_spm_model else None,
"select_metric": args.select_metric,
"valid_decode_limit": args.valid_decode_limit,
"valid_beam_size": args.valid_beam_size,
"valid_length_penalty": args.valid_length_penalty,
"valid_no_repeat_ngram_size": args.valid_no_repeat_ngram_size,
```

- [ ] **Step 5: Add validation BLEU helper and selection**

Add helper in `code/train.py`:

```python
def evaluate_validation_bleu(
    model,
    dataset,
    int2word_tgt,
    device,
    max_len,
    beam_size=1,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    decode_limit=None,
):
    from code.text_utils import corpus_bleu
    from code.translate import decode_sequence, ids_to_tokens

    predictions = []
    references = []
    examples = dataset.examples
    if decode_limit is not None:
        examples = examples[: max(decode_limit, 0)]
    model.eval()
    for example in examples:
        pred_ids = decode_sequence(
            model,
            example["src"],
            device,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
            suppress_unk=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        predictions.append(ids_to_tokens(pred_ids, int2word_tgt))
        references.append(example["tgt_tokens"])
    return corpus_bleu(predictions, references)
```

In epoch loop:

```python
valid_bleu = None
selected_score = valid_loss
if args.select_metric == "bleu":
    valid_bleu = evaluate_validation_bleu(...)
    selected_score = valid_bleu
...
metric_text = f" | valid bleu {valid_bleu:.4f}" if valid_bleu is not None else ""
print(f"Epoch ... | valid ppl {valid_ppl:.2f}{metric_text}")
```

Initialize:

```python
best_score = float("inf") if args.select_metric == "loss" else float("-inf")
```

Replace `if valid_loss < best_valid_loss:` with:

```python
if is_improved_metric(args.select_metric, selected_score, best_score):
    best_score = selected_score
```

Call `save_checkpoint(..., target_spm_model=args.target_spm_model, selected_metric=args.select_metric, selected_score=selected_score)`.

When recording best epoch checkpoints:

```python
record = {"path": epoch_path, "valid_loss": valid_loss, "score": selected_score}
best_checkpoint_records = prune_best_checkpoints(
    best_checkpoint_records,
    keep=args.keep_best_checkpoints,
    mode="max" if args.select_metric == "bleu" else "min",
    score_key="score" if args.select_metric == "bleu" else "valid_loss",
)
```

- [ ] **Step 6: Run training utility tests**

Run:

```powershell
python -m unittest tests.test_training_utils -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add code\train.py tests\test_training_utils.py
git commit -m "Select checkpoints by validation BLEU"
```

---

## Task 5: Documentation, Smoke Tests, And First Experiment

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README commands**

Add a section near the char-enhanced commands:

````markdown
## SentencePiece target mode

Train a target-side SentencePiece model from the local training split:

```powershell
python code\train_sentencepiece.py --data-dir data --vocab-size 1200 --model-prefix tokenizers\spm-target-vocab1200
```

Train a from-scratch Transformer with validation BLEU checkpoint selection:

```powershell
python code\train.py --target-level subword --target-spm-model tokenizers\spm-target-vocab1200.model --epochs 60 --batch-size 64 --device cuda --checkpoint-dir checkpoints\spm1200-bleu --label-smoothing 0.1 --scheduler noam --warmup-steps 4000 --keep-best-checkpoints 8 --select-metric bleu --valid-decode-limit 500 --valid-beam-size 1
```

Evaluate:

```powershell
python code\evaluate.py --checkpoint checkpoints\spm1200-bleu\best.pt --device cuda --beam-size 4 --length-penalty 1.5 --no-repeat-ngram-size 2
```
````

- [ ] **Step 2: Run full unit and compile verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m py_compile code\dataset.py code\train.py code\evaluate.py code\translate.py code\train_sentencepiece.py
```

Expected: all tests PASS and compile exits `0`.

- [ ] **Step 3: Train a tiny tokenizer and one-epoch CUDA smoke model**

Run:

```powershell
python code\train_sentencepiece.py --data-dir data --vocab-size 1200 --model-prefix tokenizers\spm-target-vocab1200
python code\train.py --target-level subword --target-spm-model tokenizers\spm-target-vocab1200.model --epochs 1 --batch-size 64 --device cuda --checkpoint-dir checkpoints\spm1200-smoke --label-smoothing 0.1 --scheduler noam --warmup-steps 400 --keep-best-checkpoints 2 --select-metric bleu --valid-decode-limit 20 --valid-beam-size 1
python code\evaluate.py --checkpoint checkpoints\spm1200-smoke\best.pt --device cuda --loss-only
```

Expected:

- tokenizer files exist under `tokenizers/`
- one-epoch train saves `checkpoints\spm1200-smoke\best.pt`
- loss-only evaluation prints test loss/perplexity and `Target level: subword`

- [ ] **Step 4: Commit docs**

```powershell
git add README.md
git commit -m "Document SentencePiece target workflow"
```

- [ ] **Step 5: Run the real experiment only after smoke passes**

Run:

```powershell
python code\train.py --target-level subword --target-spm-model tokenizers\spm-target-vocab1200.model --epochs 60 --batch-size 64 --device cuda --checkpoint-dir checkpoints\spm1200-bleu --label-smoothing 0.1 --scheduler noam --warmup-steps 4000 --keep-best-checkpoints 8 --select-metric bleu --valid-decode-limit 500 --valid-beam-size 1
python code\average_checkpoints.py --inputs checkpoints\spm1200-bleu\best_epoch_*.pt --output checkpoints\spm1200-bleu\averaged.pt
python code\evaluate.py --checkpoint checkpoints\spm1200-bleu\best.pt --device cuda --beam-size 4 --length-penalty 1.5 --no-repeat-ngram-size 2 --decode-limit 200 --num-examples 5
python code\evaluate.py --checkpoint checkpoints\spm1200-bleu\averaged.pt --device cuda --beam-size 4 --length-penalty 1.5 --no-repeat-ngram-size 2 --decode-limit 200 --num-examples 5
```

Expected:

- Compare 200-example BLEU against current best single/ensemble references.
- Run full test only for the better of `best.pt` and `averaged.pt`.

---

## Self-Review

- Spec coverage:
  - SentencePiece training utility: Task 1.
  - Subword target mode and vocabulary mapping: Task 2.
  - Checkpoint tokenizer metadata and loading: Task 3.
  - Validation BLEU checkpoint selection: Task 4.
  - Documentation, smoke verification, and experiment commands: Task 5.
- Completion scan:
  - No unfinished markers or vague fill-in steps remain.
- Type consistency:
  - `target_spm_model`, `SentencePieceTargetTokenizer`, `target_tokenizer`, `selected_metric`, and `selected_score` are consistently named across tasks.
  - `prune_best_checkpoints` defaults preserve old loss behavior while adding explicit `mode` and `score_key` for BLEU.
