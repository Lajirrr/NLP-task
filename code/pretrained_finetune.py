import argparse
import json
import math
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements first: python -m pip install -r requirements.txt"
    ) from exc

from code.config import DATA_DIR
from code.pretrained_utils import (
    DEFAULT_PRETRAINED_CHECKPOINT_DIR,
    load_pretrained_model_and_tokenizer,
    load_pretrained_parallel_examples,
    resolve_pretrained_model_name_or_path,
    select_device,
    tokenize_translation_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune OPUS-MT on the local dataset.")
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="HF model id or local checkpoint. Defaults to checkpoints/pretrained-opus-base when present.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRETRAINED_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_collate_fn(tokenizer, max_source_length: int, max_target_length: int):
    def collate_fn(examples):
        return tokenize_translation_batch(
            tokenizer,
            examples,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

    return collate_fn


def run_epoch(model, dataloader, optimizer, device, grad_clip: float = 1.0) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="train" if is_training else "valid", leave=False):
        batch = move_batch_to_device(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        token_count = batch["labels"].ne(-100).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def save_best_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    epoch: int,
    valid_loss: float,
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "epoch": epoch,
        "valid_loss": valid_loss,
        "model_name_or_path": args.model_name_or_path,
        "data_dir": str(args.data_dir),
        "train_examples": args.limit_train,
        "valid_examples": args.limit_valid,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    train_examples = load_pretrained_parallel_examples(
        args.data_dir / "training.txt", limit=args.limit_train
    )
    valid_examples = load_pretrained_parallel_examples(
        args.data_dir / "validation.txt", limit=args.limit_valid
    )
    model_name_or_path = resolve_pretrained_model_name_or_path(args.model_name_or_path)
    args.model_name_or_path = model_name_or_path
    model, tokenizer = load_pretrained_model_and_tokenizer(model_name_or_path, device)
    collate_fn = build_collate_fn(
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_examples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_valid_loss = float("inf")

    print(f"Device: {device}")
    print(f"Model: {model_name_or_path}")
    print(f"Train examples: {len(train_examples)}, valid examples: {len(valid_examples)}")
    print(f"Output dir: {args.output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.grad_clip)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, None, device)
        valid_ppl = math.exp(min(valid_loss, 20))
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"valid loss {valid_loss:.4f} | valid ppl {valid_ppl:.2f}"
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_best_checkpoint(args.output_dir, model, tokenizer, epoch, valid_loss, args)
            print(f"Saved best pretrained checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
