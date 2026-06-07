import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import (
    BOS_ID,
    BOS_TOKEN,
    DATA_DIR,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)


Vocab = Dict[str, int]
ReverseVocab = Dict[int, str]
TARGET_LEVEL_WORD = "word"
TARGET_LEVEL_CHAR = "char"
TARGET_LEVELS = {TARGET_LEVEL_WORD, TARGET_LEVEL_CHAR}
DATA_SPLIT_NAMES = ("training.txt", "validation.txt", "testing.txt")


def load_vocab(path: Path) -> Vocab:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(token): int(index) for token, index in data.items()}


def load_reverse_vocab(path: Path) -> ReverseVocab:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(index): str(token) for index, token in data.items()}


def normalize_target_tokens(
    tgt_tokens: Iterable[str], target_level: str = TARGET_LEVEL_WORD
) -> List[str]:
    if target_level not in TARGET_LEVELS:
        raise ValueError(
            f"Unsupported target_level {target_level!r}; expected one of {sorted(TARGET_LEVELS)}"
        )
    tokens = list(tgt_tokens)
    if target_level == TARGET_LEVEL_WORD:
        return tokens
    return [char for char in "".join(tokens) if not char.isspace()]


def build_char_vocab_from_splits(
    data_dir: Path = DATA_DIR, split_names: Sequence[str] = DATA_SPLIT_NAMES
) -> Tuple[Vocab, ReverseVocab]:
    counter = Counter()
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
                counter.update(
                    normalize_target_tokens(tgt_tokens, target_level=TARGET_LEVEL_CHAR)
                )

    vocab = {
        PAD_TOKEN: PAD_ID,
        BOS_TOKEN: BOS_ID,
        EOS_TOKEN: EOS_ID,
        UNK_TOKEN: UNK_ID,
    }
    for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab, {index: token for token, index in vocab.items()}


def load_vocabularies(
    data_dir: Path = DATA_DIR, target_level: str = TARGET_LEVEL_WORD
) -> Tuple[Vocab, ReverseVocab, Vocab, ReverseVocab]:
    if target_level not in TARGET_LEVELS:
        raise ValueError(
            f"Unsupported target_level {target_level!r}; expected one of {sorted(TARGET_LEVELS)}"
        )
    if target_level == TARGET_LEVEL_CHAR:
        tgt_vocab, int2word_tgt = build_char_vocab_from_splits(data_dir)
    else:
        tgt_vocab = load_vocab(data_dir / "word2int_cn.json")
        int2word_tgt = load_reverse_vocab(data_dir / "int2word_cn.json")
    return (
        load_vocab(data_dir / "word2int_en.json"),
        load_reverse_vocab(data_dir / "int2word_en.json"),
        tgt_vocab,
        int2word_tgt,
    )


def parse_parallel_line(line: str) -> Tuple[List[str], List[str]]:
    line = line.lstrip("\ufeff")
    if "\t" not in line:
        raise ValueError(f"Malformed dataset line without tab separator: {line!r}")
    src_text, tgt_text = line.rstrip("\n\r").split("\t", 1)
    src_tokens = [token for token in src_text.strip().split() if token]
    tgt_tokens = [token for token in tgt_text.strip().split() if token]
    if not src_tokens or not tgt_tokens:
        raise ValueError(f"Malformed dataset line with empty side: {line!r}")
    return src_tokens, tgt_tokens


def encode_tokens(tokens: Iterable[str], vocab: Vocab, add_eos: bool = False) -> List[int]:
    ids = [vocab.get(token, UNK_ID) for token in tokens]
    if add_eos:
        ids.append(EOS_ID)
    return ids


class TranslationDataset:
    def __init__(
        self,
        split_path: Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        target_level: str = TARGET_LEVEL_WORD,
    ):
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        if target_level not in TARGET_LEVELS:
            raise ValueError(
                f"Unsupported target_level {target_level!r}; expected one of {sorted(TARGET_LEVELS)}"
            )
        self.examples = []
        with split_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    src_tokens, tgt_tokens = parse_parallel_line(line)
                except ValueError as exc:
                    raise ValueError(f"{split_path}:{line_number}: {exc}") from exc
                tgt_tokens = normalize_target_tokens(tgt_tokens, target_level=target_level)
                self.examples.append(
                    {
                        "src": encode_tokens(src_tokens, src_vocab, add_eos=True),
                        "tgt": encode_tokens(tgt_tokens, tgt_vocab, add_eos=False),
                        "src_tokens": src_tokens,
                        "tgt_tokens": tgt_tokens,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def _pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int = PAD_ID):
    import torch

    max_len = max(len(sequence) for sequence in sequences)
    padded = [
        list(sequence) + [pad_id] * (max_len - len(sequence)) for sequence in sequences
    ]
    return torch.tensor(padded, dtype=torch.long)


def collate_translation_batch(batch: Sequence[dict]) -> dict:
    src_ids = _pad_sequences([item["src"] for item in batch])
    tgt_input_ids = _pad_sequences([[BOS_ID] + list(item["tgt"]) for item in batch])
    tgt_output_ids = _pad_sequences([list(item["tgt"]) + [EOS_ID] for item in batch])

    return {
        "src_ids": src_ids,
        "tgt_input_ids": tgt_input_ids,
        "tgt_output_ids": tgt_output_ids,
        "src_padding_mask": src_ids.eq(PAD_ID),
        "tgt_padding_mask": tgt_input_ids.eq(PAD_ID),
        "src_tokens": [item.get("src_tokens", []) for item in batch],
        "tgt_tokens": [item.get("tgt_tokens", []) for item in batch],
    }
