import argparse
import glob
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements first: python -m pip install -r requirements.txt"
    ) from exc


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    expanded = []
    for item in inputs:
        matches = sorted(glob.glob(str(item)))
        if matches:
            expanded.extend(Path(match) for match in matches)
        else:
            expanded.append(item)
    return expanded


def average_state_dicts(state_dicts: list[dict]) -> dict:
    if not state_dicts:
        raise ValueError("No state dicts to average")
    averaged = {}
    first = state_dicts[0]
    for key, first_value in first.items():
        values = [state_dict[key] for state_dict in state_dicts]
        if torch.is_tensor(first_value) and first_value.is_floating_point():
            stacked = torch.stack([value.detach().cpu() for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0)
        else:
            averaged[key] = first_value.detach().cpu() if torch.is_tensor(first_value) else first_value
    return averaged


def average_checkpoints(inputs: list[Path], output: Path):
    inputs = _expand_inputs(inputs)
    if not inputs:
        raise ValueError("At least one checkpoint input is required")
    checkpoints = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoints.append(torch.load(path, map_location="cpu", weights_only=True))

    result = dict(checkpoints[0])
    result["model_state_dict"] = average_state_dicts(
        [checkpoint["model_state_dict"] for checkpoint in checkpoints]
    )
    result["epoch"] = "averaged"
    result["valid_loss"] = None
    result["averaged_from"] = [str(path) for path in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Average Transformer checkpoints.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = average_checkpoints(args.inputs, args.output)
    print(f"Averaged {len(result['averaged_from'])} checkpoints into {args.output}")


if __name__ == "__main__":
    main()
