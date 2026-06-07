# Transformer English-to-Chinese Translation System Design

Date: 2026-05-24

## Goal

Build a course/lab style machine translation project in `NLP/code` using PyTorch `nn.Transformer`. The system translates tokenized or plain English sentences into Chinese, for example:

- Input: `tom is a student .`
- Output: `汤姆是个学生。`

The project should be clear enough for learning and reporting: it will include training, validation, testing, single-sentence inference, checkpointing, and usage documentation.

## Data Source

Use only the dataset under `NLP/data`, which is the renamed `cmn-eng-simple` dataset.

Expected files:

- `data/training.txt`: 18,000 sentence pairs
- `data/validation.txt`: 500 sentence pairs
- `data/testing.txt`: 2,636 sentence pairs
- `data/word2int_en.json`
- `data/int2word_en.json`
- `data/word2int_cn.json`
- `data/int2word_cn.json`

Each dataset line has the format:

```text
english tokens<TAB>chinese tokens
```

Example:

```text
he is a teacher .    他 是 老师 。
```

The existing preprocessing scripts in `data/preprocess` remain unchanged.

## Implementation Approach

Use a standard PyTorch implementation built around `torch.nn.Transformer`.

Rejected alternatives:

- Hand-written multi-head attention: useful for paper-level understanding, but too much code for the first stable version.
- HuggingFace or torchtext pipeline: faster to prototype, but hides the Transformer training details and does not naturally fit the existing vocab files.

The selected approach keeps the model logic explicit while relying on PyTorch for the core Transformer block.

## Project Layout

Create these files:

```text
code/
  config.py
  dataset.py
  model.py
  train.py
  evaluate.py
  translate.py
requirements.txt
README.md
```

Responsibilities:

- `config.py`: default paths, token ids, and hyperparameters.
- `dataset.py`: load vocab JSON files, read sentence pairs, convert tokens to ids, pad batches, and produce masks.
- `model.py`: define positional encoding and the Transformer translation model.
- `train.py`: train on `training.txt`, validate on `validation.txt`, and save the best checkpoint.
- `evaluate.py`: compute test loss, perplexity, corpus BLEU, and print translation examples.
- `translate.py`: load a checkpoint and translate one English sentence.
- `requirements.txt`: list runtime dependencies.
- `README.md`: explain installation, training, evaluation, and inference commands.

## Data Flow

For each source English sentence:

1. Split into tokens.
2. Convert tokens with `word2int_en.json`.
3. Unknown tokens map to `<UNK>`.
4. Append `<EOS>`.
5. Pad within each batch using `<PAD>`.

For each target Chinese sentence:

1. Split into tokens.
2. Convert tokens with `word2int_cn.json`.
3. Unknown tokens map to `<UNK>`.
4. Build decoder input as `<BOS> + target_ids`.
5. Build decoder label as `target_ids + <EOS>`.
6. Pad within each batch using `<PAD>`.

The collate function returns:

- `src_ids`
- `tgt_input_ids`
- `tgt_output_ids`
- source padding mask
- target padding mask

Training uses teacher forcing. Inference uses greedy decoding from `<BOS>` until `<EOS>` or a maximum output length is reached.

For output text, remove `<PAD>`, `<BOS>`, and `<EOS>`, then concatenate Chinese tokens directly. The source-side English BPE marker `@@` does not need to be restored in Chinese output.

## Tokenization for Inference

`translate.py` should accept both already-tokenized input and simple plain English.

Examples:

```powershell
py code/translate.py --checkpoint checkpoints/best.pt --text "tom is a student ."
py code/translate.py --checkpoint checkpoints/best.pt --text "Tom is a student."
```

The inference tokenizer will:

- lowercase English input
- separate common punctuation such as `.`, `?`, `!`, and `,`
- split on spaces

This tokenizer is intentionally simple because the provided dataset is already tokenized and vocabulary-driven.

## Model

Default model:

- `d_model = 256`
- `nhead = 4`
- encoder layers = 3
- decoder layers = 3
- feed-forward dimension = 512
- dropout = 0.1

Architecture:

1. Source token embedding.
2. Target token embedding.
3. Sinusoidal positional encoding.
4. PyTorch `nn.Transformer` encoder-decoder.
5. Linear projection from decoder hidden states to Chinese vocabulary logits.

Embedding outputs are multiplied by `sqrt(d_model)` before positional encoding.

## Training

Default hardware target: NVIDIA GPU with CUDA.

`train.py` behavior:

- use CUDA by default
- fail with a clear message if CUDA is requested but unavailable
- allow an optional CPU override for debugging
- train for a configurable number of epochs
- print train loss, validation loss, and validation perplexity each epoch
- save the best validation checkpoint to `checkpoints/best.pt`

Default training settings:

- optimizer: Adam
- learning rate: `1e-4`
- batch size: 64
- epochs: 20
- loss: `CrossEntropyLoss(ignore_index=<PAD>)`
- gradient clipping: enabled with max norm 1.0

Expected command:

```powershell
py code/train.py --epochs 20 --batch-size 64
```

## Evaluation

`evaluate.py` loads a checkpoint and reports:

- test loss
- test perplexity
- corpus BLEU
- several translation examples

BLEU will be implemented locally as a lightweight corpus BLEU calculation with modified n-gram precision and brevity penalty. This avoids requiring NLTK for the first version.

Expected command:

```powershell
py code/evaluate.py --checkpoint checkpoints/best.pt --num-examples 10
```

## Single-Sentence Translation

`translate.py` loads a checkpoint and prints one translated sentence.

Expected command:

```powershell
py code/translate.py --checkpoint checkpoints/best.pt --text "tom is a student ."
```

## Dependencies

Runtime dependencies:

- `torch`
- `tqdm`

The current Windows environment exposes Python through `py`, while `python` points to the Microsoft Store alias. Documentation should therefore use `py` commands.

## Error Handling

The scripts should provide clear errors for:

- missing dataset files
- missing vocab files
- missing checkpoint
- CUDA requested but unavailable
- malformed dataset lines without a tab separator

Unknown words should not crash inference; they should map to `<UNK>`.

## Testing and Verification

Before considering the implementation complete:

1. Import-check the new modules.
2. Run a tiny smoke training pass using a small batch/epoch setting if `torch` is installed.
3. Verify that `evaluate.py` and `translate.py` can load the produced checkpoint.
4. If `torch` is not installed locally, document that verification was limited to syntax checks or explain the missing dependency.

## Out of Scope

This first version will not implement:

- beam search
- attention visualization
- subword training
- preprocessing regeneration
- web UI
- HuggingFace model export

These can be added later after the basic Transformer system is working.
