import argparse
import math
import random
import sys
from dataclasses import asdict
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

from code.config import CHECKPOINT_DIR, DATA_DIR, PAD_ID, TransformerConfig
from code.dataset import (
    TARGET_LEVEL_WORD,
    TARGET_LEVELS,
    TranslationDataset,
    collate_translation_batch,
    load_vocabularies,
)
from code.model import TransformerTranslator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Transformer translator.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tie-target-embeddings", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "noam"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--keep-best-checkpoints", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--target-level",
        choices=sorted(TARGET_LEVELS),
        default=TARGET_LEVEL_WORD,
        help="Chinese target granularity: word keeps existing vocab, char builds a character vocab.",
    )
    return parser.parse_args(argv)


def select_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Install CUDA PyTorch or run with --device cpu."
        )
    return torch.device(requested)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: TransformerConfig, src_vocab_size: int, tgt_vocab_size: int):
    return TransformerTranslator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        tie_target_embeddings=config.tie_target_embeddings,
    )


def compute_noam_lr(
    step: int, d_model: int, warmup_steps: int = 4000, factor: float = 1.0
) -> float:
    step = max(step, 1)
    warmup_steps = max(warmup_steps, 1)
    return factor * (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))


class NoamLRScheduler:
    def __init__(
        self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = compute_noam_lr(
            self.step_num,
            d_model=self.d_model,
            warmup_steps=self.warmup_steps,
            factor=self.factor,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def move_batch_to_device(batch: dict, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip: float = 1.0,
    scheduler=None,
):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="train" if is_training else "valid", leave=False):
        batch = move_batch_to_device(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(
            batch["src_ids"],
            batch["tgt_input_ids"],
            batch["src_padding_mask"],
            batch["tgt_padding_mask"],
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_output_ids"].reshape(-1))

        if is_training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        token_count = batch["tgt_output_ids"].ne(PAD_ID).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def prune_best_checkpoints(records: list[dict], keep: int) -> list[dict]:
    if keep <= 0:
        return []
    sorted_records = sorted(records, key=lambda item: (item["valid_loss"], str(item["path"])))
    kept = sorted_records[:keep]
    kept_paths = {record["path"] for record in kept}
    for record in sorted_records[keep:]:
        path = record["path"]
        if path not in kept_paths and path.exists():
            path.unlink()
    return kept


def save_checkpoint(
    path: Path,
    model,
    config: TransformerConfig,
    epoch: int,
    valid_loss: float,
    target_level: str = TARGET_LEVEL_WORD,
    tgt_vocab: dict | None = None,
    int2word_tgt: dict | None = None,
    training_options: dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {
        "d_model": config.d_model,
        "nhead": config.nhead,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "tie_target_embeddings": config.tie_target_embeddings,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "train_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "epoch": epoch,
        "valid_loss": valid_loss,
        "target_level": target_level,
    }
    if tgt_vocab is not None:
        checkpoint["tgt_vocab"] = {str(token): int(index) for token, index in tgt_vocab.items()}
    if int2word_tgt is not None:
        checkpoint["int2word_tgt"] = {
            int(index): str(token) for index, token in int2word_tgt.items()
        }
    if training_options is not None:
        checkpoint["training_options"] = dict(training_options)
    torch.save(
        checkpoint,
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    config = TransformerConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        tie_target_embeddings=args.tie_target_embeddings,
    )
    src_vocab, _, tgt_vocab, int2word_tgt = load_vocabularies(
        config.data_dir, target_level=args.target_level
    )
    train_dataset = TranslationDataset(
        config.data_dir / "training.txt",
        src_vocab,
        tgt_vocab,
        target_level=args.target_level,
    )
    valid_dataset = TranslationDataset(
        config.data_dir / "validation.txt",
        src_vocab,
        tgt_vocab,
        target_level=args.target_level,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_translation_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_translation_batch,
    )

    model = build_model(config, len(src_vocab), len(tgt_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID, label_smoothing=args.label_smoothing
    )
    optimizer_lr = 0.0 if args.scheduler == "noam" else config.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    scheduler = None
    if args.scheduler == "noam":
        scheduler = NoamLRScheduler(
            optimizer,
            d_model=config.d_model,
            warmup_steps=args.warmup_steps,
            factor=1.0,
        )
    best_valid_loss = float("inf")
    best_path = config.checkpoint_dir / "best.pt"
    best_checkpoint_records = []
    training_options = {
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "keep_best_checkpoints": args.keep_best_checkpoints,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
    }

    print(f"Device: {device}")
    print(f"Target level: {args.target_level}")
    print(
        f"Label smoothing: {args.label_smoothing}, scheduler: {args.scheduler}, "
        f"warmup steps: {args.warmup_steps}"
    )
    print(
        f"Model: d_model={config.d_model}, nhead={config.nhead}, "
        f"layers={config.num_encoder_layers}/{config.num_decoder_layers}, "
        f"ffn={config.dim_feedforward}, dropout={config.dropout}, "
        f"tie_target_embeddings={config.tie_target_embeddings}"
    )
    print(f"Source vocab size: {len(src_vocab)}, target vocab size: {len(tgt_vocab)}")
    print(f"Train examples: {len(train_dataset)}, valid examples: {len(valid_dataset)}")

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.grad_clip,
            scheduler=scheduler,
        )
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, criterion, None, device)
        valid_ppl = math.exp(min(valid_loss, 20))
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"valid loss {valid_loss:.4f} | valid ppl {valid_ppl:.2f}"
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                best_path,
                model,
                config,
                epoch,
                valid_loss,
                target_level=args.target_level,
                tgt_vocab=tgt_vocab,
                int2word_tgt=int2word_tgt,
                training_options=training_options,
            )
            if args.keep_best_checkpoints > 0:
                epoch_path = config.checkpoint_dir / f"best_epoch_{epoch:02d}.pt"
                save_checkpoint(
                    epoch_path,
                    model,
                    config,
                    epoch,
                    valid_loss,
                    target_level=args.target_level,
                    tgt_vocab=tgt_vocab,
                    int2word_tgt=int2word_tgt,
                    training_options=training_options,
                )
                best_checkpoint_records.append(
                    {"path": epoch_path, "valid_loss": valid_loss}
                )
                best_checkpoint_records = prune_best_checkpoints(
                    best_checkpoint_records, keep=args.keep_best_checkpoints
                )
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
