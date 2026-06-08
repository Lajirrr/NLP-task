# Cleaned Data Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Back up the original dataset, remove all `@@` rows from active splits, train three cleaned-data character-level Transformer models, evaluate the cleaned ensemble, and add a backend-only demo model profile switch.

**Architecture:** Add a small dataset-cleaning utility with byte-preserving filtering and tests. Extend the demo server with profile-based checkpoint resolution while keeping explicit checkpoint overrides. Execute the data-cleaning and training workflow as reproducible local commands that save cleaned checkpoints and logs beside the existing original-data artifacts.

**Tech Stack:** Python 3.12 in conda `base`, PyTorch, Python standard library, pytest/unittest, PowerShell process control for long training runs.

---

## File Structure

- Create `code/clean_dataset.py`: byte-preserving `@@` row filtering, split backup, active split overwrite, CLI stats output.
- Create `tests/test_clean_dataset.py`: fast tests for filtering, backup creation, retained row order, and backup collision failure.
- Modify `code/demo_server.py`: add `--model-profile original|clean`, keep `--checkpoint` as explicit override, and fail fast if selected checkpoints are missing.
- Modify `tests/test_demo_server.py`: add model profile resolution tests without loading checkpoints.
- Modify active data files after tests pass:
  - `data/training.txt`
  - `data/validation.txt`
  - `data/testing.txt`
- Create tracked backup files:
  - `data/original_with_atat/training.txt`
  - `data/original_with_atat/validation.txt`
  - `data/original_with_atat/testing.txt`
- Local-only generated artifacts, intentionally ignored by git:
  - `checkpoints/char-clean-enhanced/`
  - `checkpoints/char-clean-adam98-e80/`
  - `checkpoints/char-clean-tied256-e60/`
  - `logs/train-char-clean-*.log`
  - `logs/eval-clean-ensemble3-beam4-full.*.log`

## Task 1: Dataset Cleaning Utility

**Files:**
- Create: `tests/test_clean_dataset.py`
- Create: `code/clean_dataset.py`

- [ ] **Step 1: Write failing tests for byte-preserving `@@` row removal**

Create `tests/test_clean_dataset.py` with:

```python
import tempfile
import unittest
from pathlib import Path


class CleanDatasetTest(unittest.TestCase):
    def test_filter_atat_rows_preserves_retained_bytes_and_order(self):
        from code.clean_dataset import filter_atat_rows

        lines = [
            b"keep one\\tbao yi\\r\\n",
            b"drop re@@ move\\tshan chu\\n",
            "keep 中文\\t保留\\n".encode("utf-8"),
            b"drop second@@\\tshan\\n",
        ]

        kept, removed = filter_atat_rows(lines)

        self.assertEqual(
            kept,
            [
                b"keep one\\tbao yi\\r\\n",
                "keep 中文\\t保留\\n".encode("utf-8"),
            ],
        )
        self.assertEqual(removed, 2)

    def test_clean_dataset_splits_creates_backup_and_overwrites_active_splits(self):
        from code.clean_dataset import clean_dataset_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            backup_dir = data_dir / "original_with_atat"
            data_dir.mkdir()
            Path(data_dir, "training.txt").write_bytes(b"a@@ b\\tc\\nkeep\\tline\\n")
            Path(data_dir, "validation.txt").write_bytes(b"valid\\tok\\n")
            Path(data_dir, "testing.txt").write_bytes(b"bad@@\\trow\\nfinal\\tok\\n")

            stats = clean_dataset_splits(data_dir, backup_dir)

            self.assertEqual(Path(data_dir, "training.txt").read_bytes(), b"keep\\tline\\n")
            self.assertEqual(Path(data_dir, "validation.txt").read_bytes(), b"valid\\tok\\n")
            self.assertEqual(Path(data_dir, "testing.txt").read_bytes(), b"final\\tok\\n")
            self.assertEqual(Path(backup_dir, "training.txt").read_bytes(), b"a@@ b\\tc\\nkeep\\tline\\n")
            self.assertEqual(Path(backup_dir, "validation.txt").read_bytes(), b"valid\\tok\\n")
            self.assertEqual(Path(backup_dir, "testing.txt").read_bytes(), b"bad@@\\trow\\nfinal\\tok\\n")
            self.assertEqual(
                [(item.split_name, item.total, item.removed, item.kept) for item in stats],
                [
                    ("training.txt", 2, 1, 1),
                    ("validation.txt", 1, 0, 1),
                    ("testing.txt", 2, 1, 1),
                ],
            )

    def test_clean_dataset_splits_refuses_existing_backup_directory(self):
        from code.clean_dataset import clean_dataset_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            backup_dir = data_dir / "original_with_atat"
            data_dir.mkdir()
            backup_dir.mkdir()
            Path(data_dir, "training.txt").write_text("a\\tb\\n", encoding="utf-8")
            Path(data_dir, "validation.txt").write_text("a\\tb\\n", encoding="utf-8")
            Path(data_dir, "testing.txt").write_text("a\\tb\\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                clean_dataset_splits(data_dir, backup_dir)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm they fail because the cleaning module is missing**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python -m pytest tests\test_clean_dataset.py -v
```

Expected: 3 failures with `ModuleNotFoundError: No module named 'code.clean_dataset'`.

- [ ] **Step 3: Implement the cleaning utility**

Create `code/clean_dataset.py` with:

```python
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_NAMES = ("training.txt", "validation.txt", "testing.txt")
ATAT_MARKER = b"@@"


@dataclass(frozen=True)
class SplitCleanStats:
    split_name: str
    total: int
    removed: int
    kept: int


def filter_atat_rows(lines: Iterable[bytes]) -> tuple[list[bytes], int]:
    kept = []
    removed = 0
    for line in lines:
        if ATAT_MARKER in line:
            removed += 1
        else:
            kept.append(line)
    return kept, removed


def read_split_lines(path: Path) -> list[bytes]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return path.read_bytes().splitlines(keepends=True)


def clean_dataset_splits(
    data_dir: Path = DATA_DIR,
    backup_dir: Path | None = None,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> list[SplitCleanStats]:
    data_dir = Path(data_dir)
    backup_dir = Path(backup_dir) if backup_dir is not None else data_dir / "original_with_atat"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")

    original_contents = {}
    for split_name in split_names:
        split_path = data_dir / split_name
        original_contents[split_name] = read_split_lines(split_path)

    backup_dir.mkdir(parents=True)
    stats = []
    for split_name in split_names:
        lines = original_contents[split_name]
        kept, removed = filter_atat_rows(lines)
        (backup_dir / split_name).write_bytes(b"".join(lines))
        (data_dir / split_name).write_bytes(b"".join(kept))
        stats.append(
            SplitCleanStats(
                split_name=split_name,
                total=len(lines),
                removed=removed,
                kept=len(kept),
            )
        )
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Remove dataset rows containing @@.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--backup-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stats = clean_dataset_splits(args.data_dir, args.backup_dir)
    print("split,total,removed,kept")
    for item in stats:
        print(f"{item.split_name},{item.total},{item.removed},{item.kept}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and confirm the cleaning utility passes**

Run:

```powershell
python -m pytest tests\test_clean_dataset.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit the cleaning utility**

Run:

```powershell
git add code\clean_dataset.py tests\test_clean_dataset.py
git commit -m "Add dataset cleaning utility"
```

Expected: commit succeeds.

## Task 2: Demo Server Model Profiles

**Files:**
- Modify: `tests/test_demo_server.py`
- Modify: `code/demo_server.py`

- [ ] **Step 1: Add failing tests for backend-only profile checkpoint resolution**

Append these tests inside `DemoServerTest` in `tests/test_demo_server.py`:

```python
    def test_parse_args_defaults_to_original_model_profile(self):
        from code.demo_server import parse_args, resolve_checkpoint_paths

        args = parse_args([])
        checkpoints = resolve_checkpoint_paths(args)

        self.assertEqual(args.model_profile, "original")
        self.assertIn("char-enhanced", str(checkpoints[0]))
        self.assertIn("char-adam98-e80", str(checkpoints[1]))
        self.assertIn("char-tied256-e60", str(checkpoints[2]))

    def test_parse_args_resolves_clean_model_profile(self):
        from code.demo_server import parse_args, resolve_checkpoint_paths

        args = parse_args(["--model-profile", "clean"])
        checkpoints = resolve_checkpoint_paths(args)

        self.assertEqual(args.model_profile, "clean")
        self.assertIn("char-clean-enhanced", str(checkpoints[0]))
        self.assertIn("char-clean-adam98-e80", str(checkpoints[1]))
        self.assertIn("char-clean-tied256-e60", str(checkpoints[2]))

    def test_explicit_checkpoint_overrides_model_profile(self):
        from pathlib import Path

        from code.demo_server import parse_args, resolve_checkpoint_paths

        args = parse_args(
            [
                "--model-profile",
                "clean",
                "--checkpoint",
                "checkpoints/custom-a.pt",
                "checkpoints/custom-b.pt",
            ]
        )

        self.assertEqual(
            resolve_checkpoint_paths(args),
            [Path("checkpoints/custom-a.pt"), Path("checkpoints/custom-b.pt")],
        )
```

- [ ] **Step 2: Run profile tests and confirm they fail before implementation**

Run:

```powershell
python -m pytest tests\test_demo_server.py -v
```

Expected: the new tests fail because `model_profile` and `resolve_checkpoint_paths` do not exist.

- [ ] **Step 3: Implement profile resolution**

Modify the top checkpoint constants and parser section in `code/demo_server.py` to:

```python
ORIGINAL_CHECKPOINTS = [
    PROJECT_ROOT / "checkpoints" / "char-enhanced" / "averaged.pt",
    PROJECT_ROOT / "checkpoints" / "char-adam98-e80" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "char-tied256-e60" / "averaged.pt",
]
CLEAN_CHECKPOINTS = [
    PROJECT_ROOT / "checkpoints" / "char-clean-enhanced" / "averaged.pt",
    PROJECT_ROOT / "checkpoints" / "char-clean-adam98-e80" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "char-clean-tied256-e60" / "averaged.pt",
]
MODEL_PROFILES = {
    "original": ORIGINAL_CHECKPOINTS,
    "clean": CLEAN_CHECKPOINTS,
}
DEFAULT_CHECKPOINTS = ORIGINAL_CHECKPOINTS
FRONTEND_DIR = PROJECT_ROOT / "demo_frontend"
```

Modify `parse_args` so `--checkpoint` defaults to `None` and add `--model-profile`:

```python
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILES),
        default="original",
        help="Backend-only model selection. Explicit --checkpoint values override this.",
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None)
```

Add this helper before `build_translator_from_args`:

```python
def resolve_checkpoint_paths(args) -> list[Path]:
    if args.checkpoint:
        return list(args.checkpoint)
    return list(MODEL_PROFILES[args.model_profile])
```

Modify `build_translator_from_args`:

```python
def build_translator_from_args(args):
    return DemoTranslator(
        checkpoint_paths=resolve_checkpoint_paths(args),
        data_dir=args.data_dir,
        device_name=args.device,
        max_len=args.max_len,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        allow_unk=args.allow_unk,
    )
```

Modify the checkpoint existence loop in `main`:

```python
    checkpoint_paths = resolve_checkpoint_paths(args)
    for checkpoint in checkpoint_paths:
        if not checkpoint.exists():
            raise SystemExit(
                f"Checkpoint not found for model profile {args.model_profile!r}: {checkpoint}"
            )
```

- [ ] **Step 4: Run profile and API tests**

Run:

```powershell
python -m pytest tests\test_demo_server.py -v
```

Expected: all demo server tests pass.

- [ ] **Step 5: Commit demo profile support**

Run:

```powershell
git add code\demo_server.py tests\test_demo_server.py
git commit -m "Add demo server model profiles"
```

Expected: commit succeeds.

## Task 3: Backup Original Data And Clean Active Splits

**Files:**
- Modify: `data/training.txt`
- Modify: `data/validation.txt`
- Modify: `data/testing.txt`
- Create: `data/original_with_atat/training.txt`
- Create: `data/original_with_atat/validation.txt`
- Create: `data/original_with_atat/testing.txt`

- [ ] **Step 1: Stop the currently running demo server to free GPU memory**

Run:

```powershell
$connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
foreach ($connection in $connections) {
    Stop-Process -Id $connection.OwningProcess -Force
}
```

Expected: no Python process remains listening on local port `8000`.

- [ ] **Step 2: Run the cleaning utility**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python code\clean_dataset.py --data-dir data --backup-dir data\original_with_atat
```

Expected stdout:

```text
split,total,removed,kept
training.txt,18000,2996,15004
validation.txt,500,79,421
testing.txt,2636,424,2212
```

- [ ] **Step 3: Verify active cleaned split counts and absence of `@@`**

Run:

```powershell
@(
  'training.txt',
  'validation.txt',
  'testing.txt'
) | ForEach-Object {
  $path = Join-Path 'data' $_
  $total = (Get-Content -Path $path).Count
  $with = (Select-String -Path $path -Pattern '@@').Count
  [PSCustomObject]@{ Split = $_; Total = $total; WithAtAt = $with }
} | Format-Table -AutoSize
```

Expected:

```text
Split          Total WithAtAt
-----          ----- --------
training.txt   15004        0
validation.txt   421        0
testing.txt     2212        0
```

- [ ] **Step 4: Verify backup split counts still match the original dataset**

Run:

```powershell
@(
  'training.txt',
  'validation.txt',
  'testing.txt'
) | ForEach-Object {
  $path = Join-Path 'data\original_with_atat' $_
  $total = (Get-Content -Path $path).Count
  $with = (Select-String -Path $path -Pattern '@@').Count
  [PSCustomObject]@{ Split = $_; Total = $total; WithAtAt = $with }
} | Format-Table -AutoSize
```

Expected:

```text
Split          Total WithAtAt
-----          ----- --------
training.txt   18000     2996
validation.txt   500       79
testing.txt     2636      424
```

- [ ] **Step 5: Run focused data and loader tests on cleaned active data**

Run:

```powershell
python -m pytest tests\test_clean_dataset.py tests\test_dataset.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit data backup and cleaned active splits**

Run:

```powershell
git add data\training.txt data\validation.txt data\testing.txt data\original_with_atat\training.txt data\original_with_atat\validation.txt data\original_with_atat\testing.txt
git commit -m "Clean dataset rows containing atat markers"
```

Expected: commit succeeds.

## Task 4: Train Three Cleaned-Data Models

**Files:**
- Local generated artifacts only:
  - `checkpoints/char-clean-enhanced/`
  - `checkpoints/char-clean-adam98-e80/`
  - `checkpoints/char-clean-tied256-e60/`
  - `logs/train-char-clean-enhanced.*.log`
  - `logs/train-char-clean-adam98-e80.*.log`
  - `logs/train-char-clean-tied256-e60.*.log`

- [ ] **Step 1: Confirm CUDA is available**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected: first line is `True` and second line names the NVIDIA GPU.

- [ ] **Step 2: Train `char-clean-enhanced`**

Run:

```powershell
$python = (Get-Command python).Source
$proc = Start-Process -FilePath $python `
  -ArgumentList @(
    "code\train.py",
    "--target-level", "char",
    "--epochs", "80",
    "--batch-size", "64",
    "--device", "cuda",
    "--checkpoint-dir", "checkpoints\char-clean-enhanced",
    "--label-smoothing", "0.1",
    "--scheduler", "noam",
    "--warmup-steps", "4000",
    "--keep-best-checkpoints", "5"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "logs\train-char-clean-enhanced.out.log" `
  -RedirectStandardError "logs\train-char-clean-enhanced.err.log" `
  -WindowStyle Hidden `
  -PassThru `
  -Wait
if ($proc.ExitCode -ne 0) { throw "char-clean-enhanced training failed with exit code $($proc.ExitCode)" }
```

Expected: exit code `0`, `checkpoints\char-clean-enhanced\best.pt` exists, and best epoch files exist.

- [ ] **Step 3: Average retained `char-clean-enhanced` best checkpoints**

Run:

```powershell
python code\average_checkpoints.py --inputs checkpoints\char-clean-enhanced\best_epoch_*.pt --output checkpoints\char-clean-enhanced\averaged.pt
```

Expected: `checkpoints\char-clean-enhanced\averaged.pt` exists.

- [ ] **Step 4: Train `char-clean-adam98-e80`**

Run:

```powershell
$python = (Get-Command python).Source
$proc = Start-Process -FilePath $python `
  -ArgumentList @(
    "code\train.py",
    "--target-level", "char",
    "--epochs", "80",
    "--batch-size", "64",
    "--device", "cuda",
    "--checkpoint-dir", "checkpoints\char-clean-adam98-e80",
    "--label-smoothing", "0.1",
    "--scheduler", "noam",
    "--warmup-steps", "4000",
    "--keep-best-checkpoints", "8",
    "--adam-beta2", "0.98"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "logs\train-char-clean-adam98-e80.out.log" `
  -RedirectStandardError "logs\train-char-clean-adam98-e80.err.log" `
  -WindowStyle Hidden `
  -PassThru `
  -Wait
if ($proc.ExitCode -ne 0) { throw "char-clean-adam98-e80 training failed with exit code $($proc.ExitCode)" }
```

Expected: exit code `0`, `checkpoints\char-clean-adam98-e80\best.pt` exists, and best epoch files exist.

- [ ] **Step 5: Average retained `char-clean-adam98-e80` best checkpoints**

Run:

```powershell
python code\average_checkpoints.py --inputs checkpoints\char-clean-adam98-e80\best_epoch_*.pt --output checkpoints\char-clean-adam98-e80\averaged.pt
```

Expected: `checkpoints\char-clean-adam98-e80\averaged.pt` exists. The final cleaned ensemble will still use `best.pt` for this branch to mirror the current original-data ensemble selection.

- [ ] **Step 6: Train `char-clean-tied256-e60`**

Run:

```powershell
$python = (Get-Command python).Source
$proc = Start-Process -FilePath $python `
  -ArgumentList @(
    "code\train.py",
    "--target-level", "char",
    "--epochs", "60",
    "--batch-size", "64",
    "--device", "cuda",
    "--checkpoint-dir", "checkpoints\char-clean-tied256-e60",
    "--label-smoothing", "0.1",
    "--scheduler", "noam",
    "--warmup-steps", "4000",
    "--keep-best-checkpoints", "8",
    "--tie-target-embeddings"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "logs\train-char-clean-tied256-e60.out.log" `
  -RedirectStandardError "logs\train-char-clean-tied256-e60.err.log" `
  -WindowStyle Hidden `
  -PassThru `
  -Wait
if ($proc.ExitCode -ne 0) { throw "char-clean-tied256-e60 training failed with exit code $($proc.ExitCode)" }
```

Expected: exit code `0`, `checkpoints\char-clean-tied256-e60\best.pt` exists, and best epoch files exist.

- [ ] **Step 7: Average retained `char-clean-tied256-e60` best checkpoints**

Run:

```powershell
python code\average_checkpoints.py --inputs checkpoints\char-clean-tied256-e60\best_epoch_*.pt --output checkpoints\char-clean-tied256-e60\averaged.pt
```

Expected: `checkpoints\char-clean-tied256-e60\averaged.pt` exists.

- [ ] **Step 8: Verify cleaned checkpoint files exist**

Run:

```powershell
@(
  'checkpoints\char-clean-enhanced\averaged.pt',
  'checkpoints\char-clean-adam98-e80\best.pt',
  'checkpoints\char-clean-adam98-e80\averaged.pt',
  'checkpoints\char-clean-tied256-e60\averaged.pt'
) | ForEach-Object {
  [PSCustomObject]@{ Path = $_; Exists = Test-Path $_ }
} | Format-Table -AutoSize
```

Expected: all `Exists` values are `True`.

## Task 5: Evaluate Cleaned Ensemble And Verify Demo Profiles

**Files:**
- Local generated artifacts only:
  - `logs/eval-clean-ensemble3-beam4-full.out.log`
  - `logs/eval-clean-ensemble3-beam4-full.err.log`

- [ ] **Step 1: Run cleaned ensemble full evaluation**

Run:

```powershell
$python = (Get-Command python).Source
$proc = Start-Process -FilePath $python `
  -ArgumentList @(
    "code\evaluate.py",
    "--checkpoint",
    "checkpoints\char-clean-enhanced\averaged.pt",
    "checkpoints\char-clean-adam98-e80\best.pt",
    "checkpoints\char-clean-tied256-e60\averaged.pt",
    "--device", "cuda",
    "--beam-size", "4",
    "--length-penalty", "1.5",
    "--no-repeat-ngram-size", "2"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "logs\eval-clean-ensemble3-beam4-full.out.log" `
  -RedirectStandardError "logs\eval-clean-ensemble3-beam4-full.err.log" `
  -WindowStyle Hidden `
  -PassThru `
  -Wait
if ($proc.ExitCode -ne 0) { throw "cleaned ensemble evaluation failed with exit code $($proc.ExitCode)" }
```

Expected: exit code `0`.

- [ ] **Step 2: Inspect cleaned evaluation metrics**

Run:

```powershell
Get-Content logs\eval-clean-ensemble3-beam4-full.out.log
```

Expected: output includes `Test loss`, `Test perplexity`, `Target level: char`, `Corpus BLEU`, and examples.

- [ ] **Step 3: Run full focused automated tests**

Run:

```powershell
python -m pytest tests\test_clean_dataset.py tests\test_demo_server.py tests\test_evaluate.py tests\test_checkpoint_loading.py tests\test_dataset.py -v
```

Expected: all selected tests pass.

- [ ] **Step 4: Start demo server with the original profile**

Run:

```powershell
$python = (Get-Command python).Source
$proc = Start-Process -FilePath $python `
  -ArgumentList @(
    "-u",
    "code\demo_server.py",
    "--model-profile", "original",
    "--device", "cuda",
    "--port", "8000"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "logs\demo-server-original.out.log" `
  -RedirectStandardError "logs\demo-server-original.err.log" `
  -WindowStyle Hidden `
  -PassThru
Start-Sleep -Seconds 8
Get-Content logs\demo-server-original.out.log -Tail 20
```

Expected: log includes `Translation demo running at http://127.0.0.1:8000`.

- [ ] **Step 5: Verify original profile API translation**

Run:

```powershell
@'
import json
import urllib.request

body = json.dumps({"text": "tom is a student ."}).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:8000/api/translate",
    data=body,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(json.dumps(payload, ensure_ascii=True))
'@ | python -
```

Expected: output contains a non-empty `translation`.

- [ ] **Step 6: Stop original profile server**

Run:

```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force
}
```

Expected: no process remains listening on port `8000`.

- [ ] **Step 7: Start demo server with the clean profile**

Run:

```powershell
$python = (Get-Command python).Source
$proc = Start-Process -FilePath $python `
  -ArgumentList @(
    "-u",
    "code\demo_server.py",
    "--model-profile", "clean",
    "--device", "cuda",
    "--port", "8000"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "logs\demo-server-clean.out.log" `
  -RedirectStandardError "logs\demo-server-clean.err.log" `
  -WindowStyle Hidden `
  -PassThru
Start-Sleep -Seconds 8
Get-Content logs\demo-server-clean.out.log -Tail 20
```

Expected: log includes `Translation demo running at http://127.0.0.1:8000`.

- [ ] **Step 8: Verify clean profile API translation**

Run:

```powershell
@'
import json
import urllib.request

body = json.dumps({"text": "tom is a student ."}).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:8000/api/translate",
    data=body,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(json.dumps(payload, ensure_ascii=True))
'@ | python -
```

Expected: output contains a non-empty `translation`.

- [ ] **Step 9: Verify frontend still has no model selector**

Open:

```text
http://127.0.0.1:8000
```

Manual checks:

- translator UI loads
- there is no visible original/clean model selector
- input panel remains above result panel
- translating `tom is a student .` updates the result panel

## Task 6: Final Report And Branch Completion

**Files:**
- Existing committed code and data
- Local checkpoint and log artifacts

- [ ] **Step 1: Summarize cleaned metrics and comparison context**

Read:

```powershell
Get-Content logs\eval-clean-ensemble3-beam4-full.out.log
Get-Content logs\eval-ensemble3-beam4-full.out.log
```

Report:

- cleaned test row count: `2212`
- original test row count: `2636`
- cleaned corpus BLEU
- original full-test corpus BLEU from existing log
- note that the two BLEU values use different test distributions because cleaned test removes `@@` rows

- [ ] **Step 2: Run final verification before completion**

Run:

```powershell
python -m pytest tests\test_clean_dataset.py tests\test_demo_server.py tests\test_evaluate.py tests\test_checkpoint_loading.py tests\test_dataset.py -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Check final git status**

Run:

```powershell
git status --short --branch
```

Expected: tracked source/data changes are committed; local ignored checkpoints and logs may exist but should not appear in git status.

## Self-Review

Spec coverage:

- Original data backup: Task 3.
- Remove active split rows containing `@@`: Task 3.
- Expected row-count validation: Task 3.
- Three cleaned checkpoint directories: Task 4.
- Best checkpoint averaging: Task 4.
- Cleaned ensemble evaluation: Task 5.
- Backend-only model switch with original default: Task 2 and Task 5.
- Explicit checkpoint override: Task 2.
- Frontend unchanged and no selector: Task 5.

Placeholder scan:

- The plan contains no vague implementation steps or missing command bodies.
- All code-changing steps include concrete code or exact edit targets.
- Long-running training and evaluation commands include exact arguments, log paths, and success checks.

Type and interface consistency:

- `clean_dataset_splits` returns `list[SplitCleanStats]`, and tests inspect `split_name`, `total`, `removed`, and `kept`.
- `filter_atat_rows` accepts byte lines and returns retained byte lines plus removed count.
- `parse_args` exposes `model_profile` and optional `checkpoint`.
- `resolve_checkpoint_paths(args)` is used by tests, `build_translator_from_args`, and `main`.
