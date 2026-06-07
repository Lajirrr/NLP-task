# Transformer Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch `nn.Transformer` English-to-Chinese translation project with training, evaluation, BLEU, checkpointing, and single-sentence inference.

**Architecture:** The code is organized into focused modules under `code/`: configuration, text utilities, dataset loading, model definition, training, evaluation, and inference. Lightweight unit tests cover tokenization, vocabulary conversion, BLEU, dataset collation, and model tensor shape where dependencies are available.

**Tech Stack:** Python 3.9+, PyTorch, tqdm, stdlib unittest.

---

## File Structure

- Create `code/__init__.py`: mark `code` as an importable project package.
- Create `code/config.py`: paths, special token constants, dataclass defaults.
- Create `code/text_utils.py`: tokenizer, special-token filtering, Chinese detokenization, corpus BLEU.
- Create `code/dataset.py`: vocab loading, sentence-pair parsing, dataset class, collate function.
- Create `code/model.py`: positional encoding and `TransformerTranslator`.
- Create `code/train.py`: argument parsing, train/validation loop, checkpoint save.
- Create `code/evaluate.py`: checkpoint load, loss/perplexity/BLEU/example output.
- Create `code/translate.py`: checkpoint load and greedy single-sentence decoding.
- Create `tests/test_text_utils.py`: unit tests for tokenizer, detokenization, BLEU.
- Create `tests/test_dataset.py`: unit tests for parsing and collation.
- Create `tests/test_model.py`: unit test for model output shape, skipped if torch is absent.
- Create `requirements.txt`: runtime dependencies.
- Create `README.md`: Chinese usage documentation.

## Task 1: Text Utilities

**Files:**
- Create: `tests/test_text_utils.py`
- Create: `code/__init__.py`
- Create: `code/text_utils.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest

from code.text_utils import (
    corpus_bleu,
    detokenize_chinese,
    simple_english_tokenize,
)


class TextUtilsTest(unittest.TestCase):
    def test_simple_english_tokenize_lowercases_and_splits_punctuation(self):
        self.assertEqual(
            simple_english_tokenize("Tom is a student."),
            ["tom", "is", "a", "student", "."],
        )

    def test_detokenize_chinese_removes_special_tokens_and_joins_words(self):
        self.assertEqual(
            detokenize_chinese(["<BOS>", "汤姆", "是", "学生", "。", "<EOS>", "<PAD>"]),
            "汤姆是学生。",
        )

    def test_corpus_bleu_returns_one_for_exact_match(self):
        score = corpus_bleu(
            predictions=[["汤姆", "是", "学生", "。"]],
            references=[["汤姆", "是", "学生", "。"]],
        )
        self.assertAlmostEqual(score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m unittest tests.test_text_utils -v`

Expected: fail with `ModuleNotFoundError` or missing functions because `code.text_utils` does not exist yet.

- [ ] **Step 3: Implement text utilities**

```python
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>"}
_PUNCT_RE = re.compile(r"([,.!?;:()\"'])")


def simple_english_tokenize(text: str) -> List[str]:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [token for token in text.split() if token]


def strip_special_tokens(tokens: Iterable[str]) -> List[str]:
    result = []
    for token in tokens:
        if token in SPECIAL_TOKENS:
            if token == "<EOS>":
                break
            continue
        result.append(token)
    return result


def detokenize_chinese(tokens: Iterable[str]) -> str:
    return "".join(strip_special_tokens(tokens))


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
    smooth: float = 1e-9,
) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0

    pred_len = sum(len(pred) for pred in predictions)
    ref_len = sum(len(ref) for ref in references)
    if pred_len == 0:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_counts = _ngram_counts(pred, n)
            ref_counts = _ngram_counts(ref, n)
            clipped += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            total += sum(pred_counts.values())
        precision = clipped / total if total else smooth
        log_precisions.append(math.log(max(precision, smooth)))

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return brevity_penalty * math.exp(sum(log_precisions) / max_n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m unittest tests.test_text_utils -v`

Expected: three tests pass.

## Task 2: Dataset Loading and Collation

**Files:**
- Create: `tests/test_dataset.py`
- Create: `code/config.py`
- Create: `code/dataset.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest


try:
    import torch
except ImportError:
    torch = None

from code.config import PAD_ID
from code.dataset import TranslationDataset, collate_translation_batch, encode_tokens, parse_parallel_line


class DatasetTest(unittest.TestCase):
    def test_parse_parallel_line_splits_source_and_target_tokens(self):
        src, tgt = parse_parallel_line("he is a teacher .\t他 是 老师 。")
        self.assertEqual(src, ["he", "is", "a", "teacher", "."])
        self.assertEqual(tgt, ["他", "是", "老师", "。"])

    def test_encode_tokens_maps_unknown_to_unk_and_appends_eos(self):
        vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "he": 4}
        self.assertEqual(encode_tokens(["he", "missing"], vocab, add_eos=True), [4, 3, 2])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_collate_translation_batch_pads_and_builds_decoder_sequences(self):
        batch = [
            {"src": [4, 5, 2], "tgt": [6, 7]},
            {"src": [4, 2], "tgt": [6]},
        ]
        collated = collate_translation_batch(batch)
        self.assertEqual(collated["src_ids"].tolist(), [[4, 5, 2], [4, 2, PAD_ID]])
        self.assertEqual(collated["tgt_input_ids"].tolist(), [[1, 6, 7], [1, 6, PAD_ID]])
        self.assertEqual(collated["tgt_output_ids"].tolist(), [[6, 7, 2], [6, 2, PAD_ID]])
        self.assertEqual(collated["src_padding_mask"].tolist(), [[False, False, False], [False, False, True]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m unittest tests.test_dataset -v`

Expected: fail because `code.config` and `code.dataset` do not exist.

- [ ] **Step 3: Implement config and dataset modules**

Implement constants and helpers for loading JSON vocabularies, parsing tab-separated sentence pairs, encoding tokens, `TranslationDataset`, and `collate_translation_batch`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m unittest tests.test_dataset -v`

Expected: parser and encoding tests pass; collation test passes if torch is installed or is skipped if torch is missing.

## Task 3: Model Definition

**Files:**
- Create: `tests/test_model.py`
- Create: `code/model.py`

- [ ] **Step 1: Write failing test**

```python
import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTest(unittest.TestCase):
    def test_transformer_translator_returns_target_vocab_logits(self):
        from code.model import TransformerTranslator

        model = TransformerTranslator(
            src_vocab_size=20,
            tgt_vocab_size=30,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        src_ids = torch.tensor([[4, 5, 2], [4, 2, 0]])
        tgt_ids = torch.tensor([[1, 6], [1, 7]])
        logits = model(src_ids, tgt_ids)
        self.assertEqual(tuple(logits.shape), (2, 2, 30))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m unittest tests.test_model -v`

Expected: fail if torch is installed because `code.model` is missing; skip if torch is missing.

- [ ] **Step 3: Implement model**

Implement sinusoidal `PositionalEncoding` and `TransformerTranslator` using `batch_first=True`, source padding masks, target padding masks, and a causal target mask.

- [ ] **Step 4: Run model test**

Run: `py -m unittest tests.test_model -v`

Expected: shape test passes if torch is installed or is skipped if torch is missing.

## Task 4: Training, Evaluation, and Translation Scripts

**Files:**
- Create: `code/train.py`
- Create: `code/evaluate.py`
- Create: `code/translate.py`

- [ ] **Step 1: Implement scripts**

Add CLI scripts that load config, create datasets/loaders, build the model, run training/evaluation/greedy decoding, and handle checkpoint files.

- [ ] **Step 2: Syntax-check scripts**

Run: `py -m py_compile code/train.py code/evaluate.py code/translate.py`

Expected: exit code 0.

## Task 5: Documentation and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: Add requirements**

```text
torch
tqdm
```

- [ ] **Step 2: Add README**

Document setup, training, evaluation, translation, CUDA notes, data format, and expected outputs in Chinese.

- [ ] **Step 3: Run all available verification**

Run:

```powershell
py -m unittest discover -s tests -v
py -m py_compile code/config.py code/text_utils.py code/dataset.py code/model.py code/train.py code/evaluate.py code/translate.py
```

Expected: unit tests pass or torch-dependent tests skip when torch is not installed; py_compile exits 0.
