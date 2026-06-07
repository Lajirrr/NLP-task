import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.pretrained_utils import (
    detokenize_english_tokens,
    generate_translations,
    load_pretrained_model_and_tokenizer,
    resolve_pretrained_model_name_or_path,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate with a pretrained HF model.")
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="HF model id or local checkpoint. Defaults to checkpoints/pretrained-opus-base when present.",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--keep-tokenized-text",
        action="store_true",
        help="Use --text exactly as provided instead of detokenizing project-style English tokens.",
    )
    return parser.parse_args()


def prepare_input_text(text: str, keep_tokenized_text: bool = False) -> str:
    if keep_tokenized_text:
        return text
    return detokenize_english_tokens(text.split())


def main():
    args = parse_args()
    device = select_device(args.device)
    model_name_or_path = resolve_pretrained_model_name_or_path(args.model_name_or_path)
    model, tokenizer = load_pretrained_model_and_tokenizer(model_name_or_path, device)
    source_text = prepare_input_text(args.text, keep_tokenized_text=args.keep_tokenized_text)
    translation = generate_translations(
        model,
        tokenizer,
        [source_text],
        device,
        batch_size=1,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )[0]
    print(translation)


if __name__ == "__main__":
    main()
