import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements first: python -m pip install -r requirements.txt"
    ) from exc

from code.config import DATA_DIR
from code.pretrained_utils import (
    compute_char_bleu,
    generate_translations,
    load_pretrained_model_and_tokenizer,
    load_pretrained_parallel_examples,
    resolve_pretrained_model_name_or_path,
    select_device,
    tokenize_translation_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained translation model.")
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="HF model id or local checkpoint. Defaults to checkpoints/pretrained-opus-base when present.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="testing.txt")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path to write aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Optional path to write per-example predictions as JSONL.",
    )
    return parser.parse_args()


def move_batch_to_device(batch: dict, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate_loss(
    model,
    tokenizer,
    examples: list[dict],
    device,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> float:
    if not examples:
        return float("nan")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(examples), batch_size), desc="loss", leave=False):
            batch_examples = examples[start : start + batch_size]
            batch = tokenize_translation_batch(
                tokenizer,
                batch_examples,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
            )
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            token_count = batch["labels"].ne(-100).sum().item()
            total_loss += outputs.loss.item() * token_count
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def main():
    args = parse_args()
    device = select_device(args.device)
    split_path = args.data_dir / args.split
    examples = load_pretrained_parallel_examples(split_path, limit=args.limit)
    model_name_or_path = resolve_pretrained_model_name_or_path(args.model_name_or_path)
    model, tokenizer = load_pretrained_model_and_tokenizer(model_name_or_path, device)

    loss = evaluate_loss(
        model,
        tokenizer,
        examples,
        device,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    predictions = generate_translations(
        model,
        tokenizer,
        [example["src_text"] for example in examples],
        device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    references = [example["tgt_text"] for example in examples]
    bleu = compute_char_bleu(predictions, references)
    metrics = {
        "examples_evaluated": len(examples),
        "teacher_forced_loss": loss,
        "teacher_forced_perplexity": math.exp(min(loss, 20)),
        "character_bleu": bleu,
        "model_name_or_path": str(model_name_or_path),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "limit": args.limit,
        "device": str(device),
    }

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as handle:
            for index, (example, prediction) in enumerate(zip(examples, predictions), start=1):
                record = {
                    "index": index,
                    "src_text": example["src_text"],
                    "reference": example["tgt_text"],
                    "prediction": prediction,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Examples evaluated: {len(examples)}")
    print(f"Teacher-forced loss: {loss:.4f}")
    print(f"Teacher-forced perplexity: {math.exp(min(loss, 20)):.2f}")
    print(f"Character BLEU: {bleu:.4f}")
    if args.metrics_output is not None:
        print(f"Metrics saved to: {args.metrics_output}")
    if args.predictions_output is not None:
        print(f"Predictions saved to: {args.predictions_output}")
    print()
    print("Examples:")
    for index, (example, prediction) in enumerate(
        zip(examples[: args.num_examples], predictions[: args.num_examples]), start=1
    ):
        print(f"[{index}] EN : {example['src_text']}")
        print(f"    REF: {example['tgt_text']}")
        print(f"    PRED: {prediction}")


if __name__ == "__main__":
    main()
