import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements first: py -m pip install -r requirements.txt"
    ) from exc

from code.config import DATA_DIR, PAD_ID
from code.dataset import (
    TARGET_LEVEL_WORD,
    TranslationDataset,
    collate_translation_batch,
)
from code.text_utils import corpus_bleu, detokenize_chinese
from code.train import move_batch_to_device
from code.translate import decode_sequence, ids_to_tokens, load_checkpoint_model, select_device


class TransformerEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("TransformerEnsemble needs at least one model.")
        self.models = nn.ModuleList(models)
        self.target_level = getattr(models[0], "target_level", TARGET_LEVEL_WORD)

    def forward(
        self,
        src_ids,
        tgt_input_ids,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        memory = self.encode(src_ids, src_padding_mask)
        return self.decode_from_memory(
            tgt_input_ids,
            memory,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    def encode(self, src_ids, src_padding_mask=None):
        return [model.encode(src_ids, src_padding_mask) for model in self.models]

    def decode_from_memory(
        self,
        tgt_input_ids,
        memory,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        log_probs = []
        for model, model_memory in zip(self.models, memory):
            logits = model.decode_from_memory(
                tgt_input_ids,
                model_memory,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
            log_probs.append(torch.log_softmax(logits, dim=-1))
        return torch.logsumexp(torch.stack(log_probs), dim=0) - math.log(
            len(log_probs)
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer translator.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--allow-unk", action="store_true")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument(
        "--decode-limit",
        type=int,
        default=None,
        help="Decode only the first N test examples for quick BLEU/example checks.",
    )
    parser.add_argument(
        "--translations-output",
        type=Path,
        default=None,
        help="Write every decoded test translation to a UTF-8 text file.",
    )
    return parser.parse_args(argv)


def load_checkpoint_model_or_ensemble(checkpoint_paths: list[Path], data_dir: Path, device):
    loaded = [
        load_checkpoint_model(checkpoint_path, data_dir, device)
        for checkpoint_path in checkpoint_paths
    ]
    first_model, src_vocab, int2word_en, tgt_vocab, int2word_cn = loaded[0]
    models = [first_model]
    target_level = getattr(first_model, "target_level", TARGET_LEVEL_WORD)

    for model, other_src_vocab, _, other_tgt_vocab, other_int2word_cn in loaded[1:]:
        other_target_level = getattr(model, "target_level", TARGET_LEVEL_WORD)
        if other_target_level != target_level:
            raise ValueError("All ensemble checkpoints must use the same target level.")
        if other_src_vocab != src_vocab or other_tgt_vocab != tgt_vocab:
            raise ValueError("All ensemble checkpoints must use the same vocabularies.")
        if other_int2word_cn != int2word_cn:
            raise ValueError("All ensemble checkpoints must use the same target id mapping.")
        models.append(model)

    if len(models) == 1:
        return first_model, src_vocab, int2word_en, tgt_vocab, int2word_cn

    ensemble = TransformerEnsemble(models).to(device)
    ensemble.target_level = target_level
    ensemble.eval()
    return ensemble, src_vocab, int2word_en, tgt_vocab, int2word_cn


def evaluate_loss(model, dataloader, device) -> float:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="test loss", leave=False):
            batch = move_batch_to_device(batch, device)
            logits = model(
                batch["src_ids"],
                batch["tgt_input_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                batch["tgt_output_ids"].reshape(-1),
            )
            token_count = batch["tgt_output_ids"].ne(PAD_ID).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def evaluate_bleu_and_examples(
    model,
    dataset: TranslationDataset,
    int2word_cn: dict,
    device,
    max_len: int,
    num_examples: int,
    beam_size: int = 1,
    length_penalty: float = 0.6,
    suppress_unk: bool = True,
    no_repeat_ngram_size: int = 0,
    decode_limit: int | None = None,
    translations_output: Path | None = None,
    output_metrics: dict[str, str] | None = None,
):
    predictions = []
    references = []
    examples = []
    translation_rows = []

    examples_to_decode = dataset.examples
    if decode_limit is not None:
        examples_to_decode = examples_to_decode[: max(decode_limit, 0)]

    for index, example in enumerate(tqdm(examples_to_decode, desc="decode", leave=False)):
        pred_ids = decode_sequence(
            model,
            example["src"],
            device,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
            suppress_unk=suppress_unk,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        pred_tokens = ids_to_tokens(pred_ids, int2word_cn)
        src_sentence = " ".join(example["src_tokens"])
        ref_sentence = detokenize_chinese(example["tgt_tokens"])
        pred_sentence = detokenize_chinese(pred_tokens)
        predictions.append(pred_tokens)
        references.append(example["tgt_tokens"])
        translation_rows.append(
            {
                "index": index + 1,
                "src": src_sentence,
                "ref": ref_sentence,
                "pred": pred_sentence,
            }
        )
        if len(examples) < num_examples:
            examples.append(
                {
                    "src": src_sentence,
                    "ref": ref_sentence,
                    "pred": pred_sentence,
                }
            )

    bleu = corpus_bleu(predictions, references)
    if translations_output is not None:
        metrics = dict(output_metrics or {})
        metrics["Corpus BLEU"] = f"{bleu:.4f}"
        if decode_limit is not None:
            metrics["Decoded examples"] = (
                f"{min(max(decode_limit, 0), len(dataset))}/{len(dataset)}"
            )
        write_translations_output(translations_output, translation_rows, metrics=metrics)

    return bleu, examples


def write_translations_output(
    output_path: Path,
    translations: list[dict[str, object]],
    metrics: dict[str, str] | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Transformer translation predictions\n")
        handle.write("# Encoding: UTF-8\n")
        if metrics:
            for key, value in metrics.items():
                handle.write(f"{key}: {value}\n")
        handle.write("\n")

        for row in translations:
            handle.write(f"[{row['index']}]\n")
            handle.write(f"EN : {row['src']}\n")
            handle.write(f"REF: {row['ref']}\n")
            handle.write(f"PRED: {row['pred']}\n")
            handle.write("\n")


def main():
    args = parse_args()
    device = select_device(args.device)
    model, src_vocab, _, tgt_vocab, int2word_cn = load_checkpoint_model_or_ensemble(
        args.checkpoint, args.data_dir, device
    )
    target_level = getattr(model, "target_level", TARGET_LEVEL_WORD)
    test_dataset = TranslationDataset(
        args.data_dir / "testing.txt", src_vocab, tgt_vocab, target_level=target_level
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
    )

    test_loss = evaluate_loss(model, test_loader, device)
    test_ppl = math.exp(min(test_loss, 20))
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_ppl:.2f}")
    print(f"Target level: {target_level}")

    if args.loss_only:
        print("Decode skipped: loss-only")
        return

    bleu, examples = evaluate_bleu_and_examples(
        model,
        test_dataset,
        int2word_cn,
        device,
        args.max_len,
        args.num_examples,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        suppress_unk=not args.allow_unk,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        decode_limit=args.decode_limit,
        translations_output=args.translations_output,
        output_metrics={
            "Test loss": f"{test_loss:.4f}",
            "Test perplexity": f"{test_ppl:.2f}",
            "Target level": target_level,
        },
    )

    print(f"Corpus BLEU: {bleu:.4f}")
    if args.decode_limit is not None:
        print(f"Decoded examples: {min(max(args.decode_limit, 0), len(test_dataset))}/{len(test_dataset)}")
    if args.translations_output is not None:
        print(f"Translations saved to: {args.translations_output}")
    print()
    print("Examples:")
    for idx, example in enumerate(examples, start=1):
        print(f"[{idx}] EN : {example['src']}")
        print(f"    REF: {example['ref']}")
        print(f"    PRED: {example['pred']}")


if __name__ == "__main__":
    main()
