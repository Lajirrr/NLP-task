import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.config import DATA_DIR
from code.evaluate import load_checkpoint_model_or_ensemble
from code.translate import select_device, translate_text


ORIGINAL_CHECKPOINTS = [
    PROJECT_ROOT / "checkpoints" / "char-enhanced" / "averaged.pt",
    PROJECT_ROOT / "checkpoints" / "char-adam98-e80" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "char-tied256-e60" / "averaged.pt",
]
CLEAN_CHECKPOINTS = [
    PROJECT_ROOT / "checkpoints" / "char-clean-enhanced" / "averaged.pt",
    PROJECT_ROOT / "checkpoints" / "char-clean-adam98-e80" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "char-clean-tied256-e60" / "averaged.pt",
]
MODEL_PROFILES = {
    "original": ORIGINAL_CHECKPOINTS,
    "clean": CLEAN_CHECKPOINTS,
}
DEFAULT_CHECKPOINTS = ORIGINAL_CHECKPOINTS
FRONTEND_DIR = PROJECT_ROOT / "demo_frontend"


class DemoTranslator:
    def __init__(
        self,
        checkpoint_paths: list[Path],
        data_dir: Path,
        device_name: str,
        max_len: int,
        beam_size: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        allow_unk: bool,
    ):
        self.device = select_device(device_name)
        self.max_len = max_len
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.allow_unk = allow_unk
        self.model, self.src_vocab, _, _, self.int2word_cn = load_checkpoint_model_or_ensemble(
            checkpoint_paths,
            data_dir,
            self.device,
        )

    def translate(self, text: str) -> dict[str, Any]:
        source_tokens, prediction_tokens, translation = translate_text(
            text,
            self.model,
            self.src_vocab,
            self.int2word_cn,
            self.device,
            max_len=self.max_len,
            beam_size=self.beam_size,
            length_penalty=self.length_penalty,
            suppress_unk=not self.allow_unk,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        return {
            "translation": translation,
            "source_tokens": source_tokens,
            "prediction_tokens": prediction_tokens,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the local translation demo server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--frontend-dir", type=Path, default=FRONTEND_DIR)
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILES),
        default="original",
        help="Backend-only model selection. Explicit --checkpoint values override this.",
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.5)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--allow-unk", action="store_true")
    return parser.parse_args(argv)


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]):
    body = json_bytes(payload)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def safe_static_path(frontend_dir: Path, request_path: str) -> Path | None:
    route_path = unquote(request_path.split("?", 1)[0])
    if route_path == "/":
        route_path = "/index.html"

    candidate = (frontend_dir / route_path.lstrip("/")).resolve()
    try:
        candidate.relative_to(frontend_dir.resolve())
    except ValueError:
        return None
    return candidate


def create_request_handler(frontend_dir: Path, translator):
    frontend_dir = frontend_dir.resolve()

    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "TransformerDemo/1.0"

        def do_GET(self):
            static_path = safe_static_path(frontend_dir, self.path)
            if static_path is None or not static_path.exists() or static_path.is_dir():
                self.send_error(404, "Not found")
                return

            content = static_path.read_bytes()
            content_type = (
                mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
            )
            if content_type.startswith("text/") or static_path.suffix in {".js", ".css"}:
                content_type = f"{content_type}; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/api/translate":
                send_json(self, 404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                send_json(self, 400, {"error": "Invalid JSON request."})
                return

            text = str(payload.get("text", "")).strip()
            if not text:
                send_json(self, 400, {"error": "Please enter an English sentence."})
                return

            try:
                send_json(self, 200, translator.translate(text))
            except Exception as exc:
                print(f"Translation failed: {exc}", file=sys.stderr)
                send_json(self, 500, {"error": "Translation failed. Please try again."})

        def do_PUT(self):
            send_json(self, 405, {"error": "Method not allowed."})

        def do_DELETE(self):
            send_json(self, 405, {"error": "Method not allowed."})

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

    return DemoRequestHandler


def resolve_checkpoint_paths(args) -> list[Path]:
    if args.checkpoint:
        return list(args.checkpoint)
    return list(MODEL_PROFILES[args.model_profile])


def build_translator_from_args(args):
    return DemoTranslator(
        checkpoint_paths=resolve_checkpoint_paths(args),
        data_dir=args.data_dir,
        device_name=args.device,
        max_len=args.max_len,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        allow_unk=args.allow_unk,
    )


def main(argv=None):
    args = parse_args(argv)
    if not args.frontend_dir.exists():
        raise SystemExit(f"Frontend directory not found: {args.frontend_dir}")
    checkpoint_paths = resolve_checkpoint_paths(args)
    for checkpoint in checkpoint_paths:
        if not checkpoint.exists():
            raise SystemExit(
                f"Checkpoint not found for model profile {args.model_profile!r}: {checkpoint}"
            )

    print("Loading translation model...")
    translator = build_translator_from_args(args)
    handler_class = create_request_handler(args.frontend_dir, translator)
    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Translation demo running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping translation demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
