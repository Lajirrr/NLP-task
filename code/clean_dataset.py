import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_NAMES = ("training.txt", "validation.txt", "testing.txt")
ATAT_MARKER = b"@@"


@dataclass(frozen=True)
class SplitCleanStats:
    split_name: str
    total: int
    removed: int
    kept: int


def filter_atat_rows(lines: Iterable[bytes]) -> tuple[list[bytes], int]:
    kept = []
    removed = 0
    for line in lines:
        if ATAT_MARKER in line:
            removed += 1
        else:
            kept.append(line)
    return kept, removed


def read_split_lines(path: Path) -> list[bytes]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return path.read_bytes().splitlines(keepends=True)


def clean_dataset_splits(
    data_dir: Path = DATA_DIR,
    backup_dir: Path | None = None,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> list[SplitCleanStats]:
    data_dir = Path(data_dir)
    backup_dir = Path(backup_dir) if backup_dir is not None else data_dir / "original_with_atat"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")

    original_contents = {}
    for split_name in split_names:
        split_path = data_dir / split_name
        original_contents[split_name] = read_split_lines(split_path)

    backup_dir.mkdir(parents=True)
    stats = []
    for split_name in split_names:
        lines = original_contents[split_name]
        kept, removed = filter_atat_rows(lines)
        (backup_dir / split_name).write_bytes(b"".join(lines))
        (data_dir / split_name).write_bytes(b"".join(kept))
        stats.append(
            SplitCleanStats(
                split_name=split_name,
                total=len(lines),
                removed=removed,
                kept=len(kept),
            )
        )
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Remove dataset rows containing @@.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--backup-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stats = clean_dataset_splits(args.data_dir, args.backup_dir)
    print("split,total,removed,kept")
    for item in stats:
        print(f"{item.split_name},{item.total},{item.removed},{item.kept}")


if __name__ == "__main__":
    main()
