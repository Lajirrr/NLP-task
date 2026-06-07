# Lightweight Transformer Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the from-scratch Transformer training and decoding path without touching pretrained-model files or existing checkpoints.

**Architecture:** Add optional training controls in `code/train.py`, repetition-aware decoding in `code/translate.py`, evaluation pass-through in `code/evaluate.py`, and a separate `code/average_checkpoints.py` utility. Defaults preserve the current behavior unless the new flags are enabled.

**Tech Stack:** Python 3.12 in conda `base`, PyTorch 2.5.0, stdlib unittest.

---

## Tasks

### Task 1: Training Enhancements

**Files:**
- Modify: `code/train.py`
- Test: `tests/test_training_utils.py`

- [ ] Write failing tests for Noam learning-rate scaling and checkpoint retention ordering.
- [ ] Implement `NoamLRScheduler`, CLI flags `--label-smoothing`, `--scheduler`, `--warmup-steps`, and `--keep-best-checkpoints`.
- [ ] Store these options in checkpoint metadata.
- [ ] Run `python -m unittest tests.test_training_utils -v`.

### Task 2: Checkpoint Averaging

**Files:**
- Create: `code/average_checkpoints.py`
- Test: `tests/test_average_checkpoints.py`

- [ ] Write failing tests for averaging tensor state dicts and preserving metadata from the first checkpoint.
- [ ] Implement CLI utility with `--inputs` and `--output`.
- [ ] Run `python -m unittest tests.test_average_checkpoints -v`.

### Task 3: Repetition-Aware Decoding

**Files:**
- Modify: `code/translate.py`
- Modify: `code/evaluate.py`
- Test: `tests/test_checkpoint_loading.py`

- [ ] Write failing tests for no-repeat ngram blocking.
- [ ] Add `--no-repeat-ngram-size` to translation and evaluation scripts.
- [ ] Apply blocking in greedy and beam decoding before selecting next tokens.
- [ ] Run `python -m unittest tests.test_checkpoint_loading -v`.

### Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Document recommended enhanced char-level training, checkpoint averaging, and evaluation commands.
- [ ] Run full unittest discovery.
- [ ] Run `py_compile` on all code files.
- [ ] Run a one-epoch CUDA smoke train with the enhanced flags.
