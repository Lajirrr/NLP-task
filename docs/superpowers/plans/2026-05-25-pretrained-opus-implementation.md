# Pretrained OPUS-MT Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate OPUS-MT pretrained translation workflow for direct inference, evaluation, and fine-tuning without changing the existing from-scratch Transformer workflow.

**Architecture:** Create `code/pretrained_utils.py` for dataset text conversion, Hugging Face loading, batching, and BLEU token preparation. Add three thin CLI scripts for translate, evaluate, and fine-tune, each using separate output directories so existing checkpoints are preserved.

**Tech Stack:** Python 3.12 in conda `base`, PyTorch 2.5.0, Transformers 4.53.3, Marian/OPUS-MT, stdlib unittest.

---

## Tasks

### Task 1: Utility Tests

**Files:**
- Create: `tests/test_pretrained_utils.py`
- Create: `code/pretrained_utils.py`

- [ ] Write failing tests for `detokenize_english_tokens`, `chinese_text_to_chars`, and `load_pretrained_parallel_examples`.
- [ ] Run `python -m unittest tests.test_pretrained_utils -v` and verify failure because `code.pretrained_utils` does not exist.
- [ ] Implement the utility functions.
- [ ] Re-run the same test command and verify it passes.

### Task 2: Pretrained CLI Scripts

**Files:**
- Create: `code/pretrained_translate.py`
- Create: `code/pretrained_evaluate.py`
- Create: `code/pretrained_finetune.py`

- [ ] Add direct translation CLI using `AutoTokenizer` and `AutoModelForSeq2SeqLM`.
- [ ] Add bounded evaluation CLI with `--limit`, character-level BLEU, optional loss, and examples.
- [ ] Add manual fine-tuning CLI that saves best Hugging Face checkpoint to `checkpoints/pretrained-opus-en-zh`.
- [ ] Run `python -m py_compile` on all new scripts.

### Task 3: Dependencies And Docs

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

- [ ] Add `transformers`, `sentencepiece`, and `sacremoses` to requirements.
- [ ] Document direct pretrained translation, bounded evaluation, and fine-tuning commands.
- [ ] Explain that existing from-scratch checkpoints remain separate.

### Task 4: Verification

**Files:**
- No additional file edits.

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m py_compile code/config.py code/text_utils.py code/dataset.py code/model.py code/train.py code/evaluate.py code/translate.py code/pretrained_utils.py code/pretrained_translate.py code/pretrained_evaluate.py code/pretrained_finetune.py`.
- [ ] Install missing runtime dependencies in conda `base` if needed.
- [ ] Run direct translation smoke test with `python code/pretrained_translate.py --text "tom is a student ." --device auto --num-beams 4`.
- [ ] Run bounded evaluation smoke test with `python code/pretrained_evaluate.py --limit 5 --device auto --num-beams 2`.
- [ ] Commit the implementation.

