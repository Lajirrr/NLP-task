# Character-Level Chinese Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--target-level char` mode for Chinese character-level target training, evaluation, and translation.

**Architecture:** Extend `dataset.py` with target token normalization and dynamic char vocab construction. Store target mode and target vocab in checkpoints so `evaluate.py` and `translate.py` can infer the correct output mode automatically.

**Tech Stack:** Python 3.12 in conda `base`, PyTorch 2.5.0, stdlib unittest.

---

## Tasks

- [ ] Add dataset tests for char target tokenization, char vocab construction, and char-mode dataset examples.
- [ ] Implement `target_level` helpers in `code/dataset.py`.
- [ ] Add checkpoint tests for storing/loading target vocab metadata.
- [ ] Update `code/train.py` to accept `--target-level` and save target metadata.
- [ ] Update `code/evaluate.py` to build datasets with the checkpoint target level.
- [ ] Update `code/translate.py` to load checkpoint target metadata.
- [ ] Update README with char-level commands.
- [ ] Run unittest, py_compile, and a char-level CUDA smoke train.
