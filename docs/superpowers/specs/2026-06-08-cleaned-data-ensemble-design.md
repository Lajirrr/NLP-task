# Cleaned Data Ensemble Design

Date: 2026-06-08

## Goal

Create a second, side-by-side Transformer translation model family trained on dataset splits with all `@@` rows removed. Preserve the original dataset before cleaning, train three cleaned-data character-level models, evaluate the cleaned ensemble, and add a backend-only switch in the demo server so the frontend stays unchanged.

## Current Dataset Audit

The current split files contain many English-side `@@` BPE marker fragments. The requested cleaning rule is to delete the entire parallel sentence pair if the line contains `@@`.

Observed counts before cleaning:

| Split | Current rows | Rows with `@@` | Rows after deletion |
| --- | ---: | ---: | ---: |
| `training.txt` | 18000 | 2996 | 15004 |
| `validation.txt` | 500 | 79 | 421 |
| `testing.txt` | 2636 | 424 | 2212 |

The original data must remain available as a local copy.

## Data Layout

Before modifying active split files, copy the current data to:

```text
data/original_with_atat/
  training.txt
  validation.txt
  testing.txt
```

Then overwrite the active split files with cleaned versions:

```text
data/training.txt
data/validation.txt
data/testing.txt
```

Each cleaned file keeps the original order of retained lines and removes only lines containing the literal substring `@@`.

Vocabulary files stay in `data/` and are not regenerated for English word-level source tokens. This keeps source ids compatible with the existing vocabulary. Character-level Chinese target vocabularies are already stored in each checkpoint, so cleaned checkpoints will carry their own target metadata.

## Training Plan

Train three cleaned-data character-level models in new checkpoint directories:

```text
checkpoints/char-clean-enhanced/
checkpoints/char-clean-adam98-e80/
checkpoints/char-clean-tied256-e60/
```

The directories are new and must not overwrite the existing original-data checkpoints:

```text
checkpoints/char-enhanced/
checkpoints/char-adam98-e80/
checkpoints/char-tied256-e60/
```

The cleaned model family should mirror the current original-data ensemble family as closely as practical:

- `char-clean-enhanced`: character target, label smoothing `0.1`, Noam scheduler, warmup `4000`, keep best checkpoints, average the retained best checkpoints.
- `char-clean-adam98-e80`: character target, label smoothing `0.1`, Noam scheduler, Adam beta2 `0.98`, 80 epochs, keep best checkpoints, average the retained best checkpoints.
- `char-clean-tied256-e60`: character target, tied target embeddings, 60 epochs, keep best checkpoints, average the retained best checkpoints.

Run training from the conda `base` environment and stop the currently running demo server before training to free GPU memory.

## Evaluation

Evaluate the cleaned ensemble against the cleaned `data/testing.txt` split:

```text
checkpoints/char-clean-enhanced/averaged.pt
checkpoints/char-clean-adam98-e80/best.pt
checkpoints/char-clean-tied256-e60/averaged.pt
```

Use the same decode settings as the current original-data ensemble:

- beam size: `4`
- length penalty: `1.5`
- no-repeat ngram size: `2`
- device: `cuda`

Save logs without overwriting existing logs:

```text
logs/eval-clean-ensemble3-beam4-full.out.log
logs/eval-clean-ensemble3-beam4-full.err.log
```

For comparison, the existing original-data full-test log remains:

```text
logs/eval-ensemble3-beam4-full.out.log
```

The cleaned result should report test loss, perplexity, target level, corpus BLEU, and example translations. Because the cleaned test set excludes all `@@` examples, the cleaned BLEU is not directly apples-to-apples with the old full test; it is the quality estimate for the cleaned dataset distribution.

## Demo Backend Switch

Do not expose model selection in the frontend UI.

Add a backend-only `--model-profile` argument to `code/demo_server.py`:

```powershell
python code\demo_server.py --model-profile original --device cuda
python code\demo_server.py --model-profile clean --device cuda
```

`original` remains the default and uses the existing original-data ensemble:

```text
checkpoints/char-enhanced/averaged.pt
checkpoints/char-adam98-e80/best.pt
checkpoints/char-tied256-e60/averaged.pt
```

`clean` uses the cleaned-data ensemble:

```text
checkpoints/char-clean-enhanced/averaged.pt
checkpoints/char-clean-adam98-e80/best.pt
checkpoints/char-clean-tied256-e60/averaged.pt
```

The existing `--checkpoint` argument should remain available as an explicit override for custom experiments. If `--checkpoint` is provided, it takes precedence over `--model-profile`.

## Error Handling

Data cleaning must be conservative:

- fail if any source split file is missing
- fail if the backup directory already exists with different content unless explicitly handled in the implementation plan
- validate row counts after cleaning
- preserve UTF-8 text exactly for retained rows

Training and evaluation should write separate stdout/stderr logs for each long-running command.

The demo server should fail fast with a clear message if `--model-profile clean` is selected before cleaned checkpoints exist.

## Testing

Automated tests should cover:

- filtering removes lines containing `@@`
- retained lines preserve order and content
- demo server resolves default `original` profile checkpoints
- demo server resolves `clean` profile checkpoints
- explicit `--checkpoint` overrides profile defaults

Manual verification should cover:

- original dataset backup exists
- active split row counts match expected cleaned counts
- no active split file contains `@@`
- cleaned checkpoints exist in new directories
- cleaned ensemble evaluation log reports metrics
- demo server starts with `--model-profile original`
- demo server starts with `--model-profile clean`
- frontend UI remains unchanged and does not expose model selection

## Out of Scope

- changing the frontend visual design
- deleting original checkpoints or logs
- changing vocabulary generation strategy
- training pretrained Hugging Face models
- comparing cleaned and original models on a newly constructed common subset unless requested later
