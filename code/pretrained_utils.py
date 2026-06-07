from pathlib import Path
from typing import Iterable, List

import torch

from .dataset import parse_parallel_line
from .text_utils import corpus_bleu, detokenize_chinese


DEFAULT_PRETRAINED_MODEL = "Helsinki-NLP/opus-mt-en-zh"
DEFAULT_PRETRAINED_BASE_DIR = Path("checkpoints") / "pretrained-opus-base"
DEFAULT_PRETRAINED_CHECKPOINT_DIR = Path("checkpoints") / "pretrained-opus-en-zh"


def detokenize_english_tokens(tokens: Iterable[str]) -> str:
    text = " ".join(tokens).strip()
    replacements = {
        " n't": "n't",
        " 's": "'s",
        " 'm": "'m",
        " 're": "'re",
        " 've": "'ve",
        " 'd": "'d",
        " 'll": "'ll",
        " ,": ",",
        " .": ".",
        " !": "!",
        " ?": "?",
        " ;": ";",
        " :": ":",
        " )": ")",
        "( ": "(",
        ' " ': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def chinese_text_to_chars(text: str) -> List[str]:
    return [char for char in text if not char.isspace()]


def load_pretrained_parallel_examples(split_path: Path, limit: int | None = None) -> List[dict]:
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    examples = []
    with split_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                src_tokens, tgt_tokens = parse_parallel_line(line)
            except ValueError as exc:
                raise ValueError(f"{split_path}:{line_number}: {exc}") from exc

            tgt_text = detokenize_chinese(tgt_tokens)
            examples.append(
                {
                    "src_tokens": src_tokens,
                    "tgt_tokens": tgt_tokens,
                    "src_text": detokenize_english_tokens(src_tokens),
                    "tgt_text": tgt_text,
                    "tgt_chars": chinese_text_to_chars(tgt_text),
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def select_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Install CUDA PyTorch or run with --device cpu."
        )
    return torch.device(requested)


def resolve_pretrained_model_name_or_path(
    model_name_or_path: str | None,
    local_base_dir: Path = DEFAULT_PRETRAINED_BASE_DIR,
) -> str:
    if model_name_or_path:
        return model_name_or_path
    if local_base_dir.exists():
        return str(local_base_dir)
    return DEFAULT_PRETRAINED_MODEL


def load_pretrained_model_and_tokenizer(model_name_or_path: str, device):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install requirements first: python -m pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return model, tokenizer


def generate_translations(
    model,
    tokenizer,
    texts: List[str],
    device,
    batch_size: int = 16,
    max_new_tokens: int = 80,
    num_beams: int = 4,
) -> List[str]:
    translations = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        translations.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    return translations


def compute_char_bleu(predictions: List[str], references: List[str]) -> float:
    pred_chars = [chinese_text_to_chars(prediction) for prediction in predictions]
    ref_chars = [chinese_text_to_chars(reference) for reference in references]
    return corpus_bleu(pred_chars, ref_chars)


def tokenize_translation_batch(
    tokenizer,
    examples: List[dict],
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> dict:
    src_texts = [example["src_text"] for example in examples]
    tgt_texts = [example["tgt_text"] for example in examples]
    model_inputs = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )
    labels = tokenizer(
        text_target=tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_target_length,
    )["input_ids"]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        labels = labels.masked_fill(labels.eq(pad_token_id), -100)
    model_inputs["labels"] = labels
    return model_inputs
