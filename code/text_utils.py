import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>"}
_PUNCT_RE = re.compile(r"([,.!?;:()\"])")
_CONTRACTION_RE = re.compile(r"\b(\w+)('ll|'re|'ve|'d|'m|'s)\b")


def simple_english_tokenize(text: str) -> List[str]:
    """Lowercase English text and split common punctuation as separate tokens."""
    text = text.strip().lower()
    text = re.sub(r"\b(\w+)n't\b", r"\1 n't", text)
    text = _CONTRACTION_RE.sub(r"\1 \2", text)
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
            clipped += sum(
                min(count, ref_counts[gram]) for gram, count in pred_counts.items()
            )
            total += sum(pred_counts.values())
        precision = clipped / total if total else smooth
        log_precisions.append(math.log(max(precision, smooth)))

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return brevity_penalty * math.exp(sum(log_precisions) / max_n)
