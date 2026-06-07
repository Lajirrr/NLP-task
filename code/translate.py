import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements first: py -m pip install -r requirements.txt"
    ) from exc

from code.config import BOS_ID, DATA_DIR, EOS_ID, PAD_ID, UNK_ID, TransformerConfig
from code.dataset import TARGET_LEVEL_WORD, load_vocabularies
from code.model import TransformerTranslator
from code.text_utils import detokenize_chinese, simple_english_tokenize, strip_special_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Translate one English sentence to Chinese.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--allow-unk", action="store_true")
    parser.add_argument("--show-tokens", action="store_true")
    return parser.parse_args()


def select_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Install CUDA PyTorch or run with --device cpu."
        )
    return torch.device(requested)


def encode_source_text(text: str, src_vocab: dict) -> Tuple[List[str], List[int]]:
    tokens = simple_english_tokenize(text)
    ids = [src_vocab.get(token, UNK_ID) for token in tokens] + [EOS_ID]
    return tokens, ids


def ids_to_tokens(ids: Iterable[int], int2word: dict) -> List[str]:
    tokens = []
    for index in ids:
        token = int2word.get(int(index), "<UNK>")
        if token == "<EOS>":
            break
        if token in {"<PAD>", "<BOS>"}:
            continue
        tokens.append(token)
    return strip_special_tokens(tokens)


def load_checkpoint_model(checkpoint_path: Path, data_dir: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    src_vocab, int2word_en, _, _ = load_vocabularies(
        data_dir, target_level=TARGET_LEVEL_WORD
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    target_level = checkpoint.get("target_level", TARGET_LEVEL_WORD)
    if "tgt_vocab" in checkpoint and "int2word_tgt" in checkpoint:
        tgt_vocab = {
            str(token): int(index)
            for token, index in checkpoint["tgt_vocab"].items()
        }
        int2word_cn = {
            int(index): str(token)
            for index, token in checkpoint["int2word_tgt"].items()
        }
    else:
        _, _, tgt_vocab, int2word_cn = load_vocabularies(
            data_dir, target_level=target_level
        )
    model_config = checkpoint.get("model_config", {})
    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        **model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.target_level = target_level
    model.tgt_vocab = tgt_vocab
    model.int2word_tgt = int2word_cn
    model.eval()
    return model, src_vocab, int2word_en, tgt_vocab, int2word_cn


def blocked_ngram_tokens(
    generated_ids: Iterable[int] | None, no_repeat_ngram_size: int = 0
) -> set[int]:
    if (
        generated_ids is None
        or no_repeat_ngram_size <= 0
        or len(generated_ids) < no_repeat_ngram_size - 1
    ):
        return set()
    generated = [int(index) for index in generated_ids]
    prefix_length = no_repeat_ngram_size - 1
    current_prefix = tuple(generated[-prefix_length:]) if prefix_length else tuple()
    blocked = set()
    for start in range(0, len(generated) - no_repeat_ngram_size + 1):
        ngram = tuple(generated[start : start + no_repeat_ngram_size])
        if ngram[:-1] == current_prefix:
            blocked.add(int(ngram[-1]))
    return blocked


def apply_decode_constraints(
    logits,
    suppress_unk: bool = True,
    generated_ids: Iterable[int] | None = None,
    no_repeat_ngram_size: int = 0,
):
    if suppress_unk:
        logits[UNK_ID] = -1e9
    for token_id in blocked_ngram_tokens(generated_ids, no_repeat_ngram_size):
        if 0 <= token_id < logits.size(0):
            logits[token_id] = -1e9
    return logits


def normalize_beam_score(sequence_length: int, log_prob: float, length_penalty: float) -> float:
    length = max(sequence_length, 1)
    if length_penalty <= 0:
        return log_prob
    return log_prob / (((5 + length) / 6) ** length_penalty)


def expand_memory_for_beams(memory, beam_count: int):
    if isinstance(memory, tuple):
        return tuple(expand_memory_for_beams(item, beam_count) for item in memory)
    if isinstance(memory, list):
        return [expand_memory_for_beams(item, beam_count) for item in memory]
    return memory.expand(beam_count, -1, -1)


def greedy_decode(
    model,
    src_ids: List[int],
    device,
    max_len: int = 64,
    suppress_unk: bool = True,
    no_repeat_ngram_size: int = 0,
) -> List[int]:
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src.eq(PAD_ID)
    generated = [BOS_ID]

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)
    for _ in range(max_len):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_padding_mask = tgt.eq(PAD_ID)
        with torch.no_grad():
            logits = model.decode_from_memory(
                tgt,
                memory,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
        next_logits = apply_decode_constraints(
            logits[0, -1],
            suppress_unk=suppress_unk,
            generated_ids=generated[1:],
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        next_id = int(next_logits.argmax(dim=-1).item())
        generated.append(next_id)
        if next_id == EOS_ID:
            break
    return generated[1:]


def beam_decode(
    model,
    src_ids: List[int],
    device,
    max_len: int = 64,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    suppress_unk: bool = True,
    no_repeat_ngram_size: int = 0,
) -> List[int]:
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src.eq(PAD_ID)
    beams = [([BOS_ID], 0.0)]
    finished = []

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)
    for _ in range(max_len):
        candidates = []
        active_beams = []
        for sequence, score in beams:
            if sequence[-1] == EOS_ID:
                finished.append((sequence, score))
                continue
            active_beams.append((sequence, score))

        if active_beams:
            tgt = torch.tensor(
                [sequence for sequence, _ in active_beams],
                dtype=torch.long,
                device=device,
            )
            tgt_padding_mask = tgt.eq(PAD_ID)
            beam_memory = expand_memory_for_beams(memory, tgt.size(0))
            beam_src_padding_mask = src_padding_mask.expand(tgt.size(0), -1)
            with torch.no_grad():
                logits = model.decode_from_memory(
                    tgt,
                    beam_memory,
                    src_padding_mask=beam_src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                )

        for beam_index, (sequence, score) in enumerate(active_beams):
            next_logits = apply_decode_constraints(
                logits[beam_index, -1],
                suppress_unk=suppress_unk,
                generated_ids=sequence[1:],
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            log_probs = torch.log_softmax(next_logits, dim=-1)
            values, indices = torch.topk(log_probs, beam_size)
            for value, index in zip(values.tolist(), indices.tolist()):
                candidates.append((sequence + [int(index)], score + float(value)))

        if not candidates:
            break
        beams = sorted(
            candidates,
            key=lambda item: normalize_beam_score(
                sequence_length=len(item[0]) - 1,
                log_prob=item[1],
                length_penalty=length_penalty,
            ),
            reverse=True,
        )[:beam_size]
        if all(sequence[-1] == EOS_ID for sequence, _ in beams):
            finished.extend(beams)
            break

    best_sequence, _ = max(
        finished or beams,
        key=lambda item: normalize_beam_score(
            sequence_length=len(item[0]) - 1,
            log_prob=item[1],
            length_penalty=length_penalty,
        ),
    )
    return best_sequence[1:]


def decode_sequence(
    model,
    src_ids: List[int],
    device,
    max_len: int = 64,
    beam_size: int = 1,
    length_penalty: float = 0.6,
    suppress_unk: bool = True,
    no_repeat_ngram_size: int = 0,
) -> List[int]:
    if beam_size <= 1:
        return greedy_decode(
            model,
            src_ids,
            device,
            max_len=max_len,
            suppress_unk=suppress_unk,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return beam_decode(
        model,
        src_ids,
        device,
        max_len=max_len,
        beam_size=beam_size,
        length_penalty=length_penalty,
        suppress_unk=suppress_unk,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


def translate_text(
    text: str,
    model,
    src_vocab: dict,
    int2word_cn: dict,
    device,
    max_len: int = 64,
    beam_size: int = 1,
    length_penalty: float = 0.6,
    suppress_unk: bool = True,
    no_repeat_ngram_size: int = 0,
) -> Tuple[List[str], List[str], str]:
    src_tokens, src_ids = encode_source_text(text, src_vocab)
    pred_ids = decode_sequence(
        model,
        src_ids,
        device,
        max_len=max_len,
        beam_size=beam_size,
        length_penalty=length_penalty,
        suppress_unk=suppress_unk,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    pred_tokens = ids_to_tokens(pred_ids, int2word_cn)
    return src_tokens, pred_tokens, detokenize_chinese(pred_tokens)


def main():
    args = parse_args()
    device = select_device(args.device)
    model, src_vocab, _, _, int2word_cn = load_checkpoint_model(
        args.checkpoint, args.data_dir, device
    )
    src_tokens, pred_tokens, translation = translate_text(
        args.text,
        model,
        src_vocab,
        int2word_cn,
        device,
        max_len=args.max_len,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        suppress_unk=not args.allow_unk,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    if args.show_tokens:
        print("Source tokens:", " ".join(src_tokens))
        print("Prediction tokens:", " ".join(pred_tokens))
    print(translation)


if __name__ == "__main__":
    main()
