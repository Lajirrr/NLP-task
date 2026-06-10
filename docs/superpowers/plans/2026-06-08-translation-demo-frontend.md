# Translation Demo Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, video-friendly single-sentence English-to-Chinese translation web demo backed by the restored character-level Transformer ensemble.

**Architecture:** Add a dependency-free Python `http.server` backend that loads the ensemble once, serves static frontend files, and exposes `POST /api/translate`. Add a plain HTML/CSS/JS frontend using the selected clean translation-card layout with input above output and stable loading/error states.

**Tech Stack:** Python standard library `http.server`, existing PyTorch translation modules, plain HTML, CSS, and JavaScript.

---

## File Structure

- Create `code/demo_server.py`: argument parsing, model-backed `DemoTranslator`, JSON response helpers, static file serving, `POST /api/translate`, and server startup.
- Create `demo_frontend/index.html`: semantic translator shell with language row, input panel, action button, and result panel.
- Create `demo_frontend/styles.css`: selected clean A visual style, responsive layout, loading/error/success states, stable dimensions.
- Create `demo_frontend/app.js`: submit handling, fetch call, button state, keyboard shortcut, and result rendering.
- Create `tests/test_demo_server.py`: fast server tests using a stub translator and a temporary static directory.

## Task 1: Demo Server API Tests

**Files:**
- Create: `tests/test_demo_server.py`
- Later create: `code/demo_server.py`

- [ ] **Step 1: Write failing tests for JSON API and static serving**

Create `tests/test_demo_server.py` with:

```python
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path


class StubTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return {
            "translation": "汤姆是个学生。",
            "source_tokens": ["tom", "is", "a", "student", "."],
            "prediction_tokens": ["汤", "姆", "是", "个", "学", "生", "。"],
        }


class DemoServerTest(unittest.TestCase):
    def run_server(self, frontend_dir, translator):
        from code.demo_server import create_request_handler

        handler_class = create_request_handler(
            frontend_dir=Path(frontend_dir),
            translator=translator,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def post_json(self, url, payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)

    def test_translate_endpoint_returns_utf8_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "index.html").write_text("demo", encoding="utf-8")
            translator = StubTranslator()
            base_url = self.run_server(tmpdir, translator)

            status, payload = self.post_json(
                f"{base_url}/api/translate",
                {"text": "tom is a student ."},
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload["translation"], "汤姆是个学生。")
        self.assertEqual(payload["source_tokens"], ["tom", "is", "a", "student", "."])
        self.assertEqual(translator.calls, ["tom is a student ."])

    def test_translate_endpoint_rejects_blank_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "index.html").write_text("demo", encoding="utf-8")
            base_url = self.run_server(tmpdir, StubTranslator())

            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post_json(f"{base_url}/api/translate", {"text": "   "})

        self.assertEqual(caught.exception.code, 400)
        payload = json.loads(caught.exception.read().decode("utf-8"))
        self.assertEqual(payload["error"], "Please enter an English sentence.")

    def test_index_serves_frontend_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "index.html").write_text("你好 demo", encoding="utf-8")
            base_url = self.run_server(tmpdir, StubTranslator())

            with urllib.request.urlopen(f"{base_url}/", timeout=5) as response:
                body = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("你好 demo", body)
        self.assertEqual(response.headers.get_content_charset(), "utf-8")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm they fail because the server module is missing**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python -m pytest tests\test_demo_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.demo_server'`.

## Task 2: Demo Server Implementation

**Files:**
- Create: `code/demo_server.py`
- Test: `tests/test_demo_server.py`

- [ ] **Step 1: Implement the local server and model-backed translator**

Create `code/demo_server.py` with:

```python
import argparse
import json
import mimetypes
import sys
from functools import partial
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


DEFAULT_CHECKPOINTS = [
    PROJECT_ROOT / "checkpoints" / "char-enhanced" / "averaged.pt",
    PROJECT_ROOT / "checkpoints" / "char-adam98-e80" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "char-tied256-e60" / "averaged.pt",
]
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
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
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
            content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
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


def build_translator_from_args(args):
    return DemoTranslator(
        checkpoint_paths=args.checkpoint,
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
    for checkpoint in args.checkpoint:
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")

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
```

- [ ] **Step 2: Run server tests**

Run:

```powershell
python -m pytest tests\test_demo_server.py -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit server tests and implementation**

Run:

```powershell
git add code\demo_server.py tests\test_demo_server.py
git commit -m "Add local translation demo server"
```

Expected: commit succeeds.

## Task 3: Clean Translation Card Frontend

**Files:**
- Create: `demo_frontend/index.html`
- Create: `demo_frontend/styles.css`
- Create: `demo_frontend/app.js`

- [ ] **Step 1: Create semantic HTML shell**

Create `demo_frontend/index.html` with:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Transformer Translation Demo</title>
    <link rel="stylesheet" href="/styles.css">
  </head>
  <body>
    <main class="page-shell">
      <section class="translator" aria-label="English to Chinese translator">
        <header class="language-row">
          <div class="language-pill">English</div>
          <div class="direction-mark" aria-hidden="true">↓</div>
          <div class="language-pill">中文（简体）</div>
        </header>

        <section class="panel input-panel">
          <div class="panel-heading">
            <label for="sourceText">输入文本</label>
            <button id="translateButton" class="translate-button" type="button">
              翻译
            </button>
          </div>
          <textarea
            id="sourceText"
            class="source-input"
            rows="4"
            spellcheck="false"
            placeholder="Tom is a student."
          ></textarea>
        </section>

        <div class="between-panels" aria-hidden="true">
          <span>↓</span>
        </div>

        <section class="panel output-panel" aria-live="polite">
          <div class="panel-heading">
            <span>翻译结果</span>
            <span id="statusText" class="status-text">等待输入</span>
          </div>
          <p id="translationResult" class="translation-result">
            输入英文句子后，模型翻译会显示在这里。
          </p>
        </section>
      </section>
    </main>
    <script src="/app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Create responsive clean-card CSS**

Create `demo_frontend/styles.css` with:

```css
:root {
  color-scheme: light;
  --page: #f4f1ed;
  --surface: #ffffff;
  --surface-soft: #fbfaf8;
  --border: #dedbd6;
  --text: #4f5358;
  --muted: #7c8289;
  --accent: #3f6f6a;
  --accent-strong: #28514d;
  --danger: #9b3030;
  --shadow: 0 24px 70px rgba(47, 51, 56, 0.12);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--page);
  color: var(--text);
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
}

button,
textarea {
  font: inherit;
}

.page-shell {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px 18px;
}

.translator {
  width: min(920px, 100%);
  background: rgba(255, 255, 255, 0.56);
  border-radius: 22px;
  padding: 22px;
  box-shadow: var(--shadow);
}

.language-row {
  display: grid;
  grid-template-columns: 1fr 42px 1fr;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}

.language-pill {
  height: 42px;
  display: flex;
  align-items: center;
  padding: 0 18px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--surface);
  color: var(--muted);
  font-size: 15px;
}

.direction-mark,
.between-panels {
  display: flex;
  align-items: center;
  justify-content: center;
}

.direction-mark {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: var(--surface);
  color: var(--text);
  font-size: 20px;
}

.panel {
  border-radius: 14px;
  border: 1px solid var(--border);
  padding: 24px;
  min-height: 190px;
}

.input-panel {
  background: var(--surface);
}

.output-panel {
  background: var(--surface-soft);
}

.panel-heading {
  min-height: 42px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
  color: var(--muted);
  font-size: 18px;
  font-weight: 700;
}

.source-input {
  width: 100%;
  min-height: 110px;
  resize: vertical;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--text);
  font-size: clamp(26px, 5vw, 42px);
  font-weight: 800;
  line-height: 1.25;
}

.source-input::placeholder {
  color: #b3b6ba;
}

.translate-button {
  min-width: 92px;
  height: 42px;
  border: 0;
  border-radius: 9px;
  background: var(--accent);
  color: #ffffff;
  cursor: pointer;
  font-size: 16px;
  font-weight: 800;
}

.translate-button:disabled {
  cursor: wait;
  opacity: 0.7;
}

.between-panels {
  height: 48px;
  color: var(--muted);
  font-size: 24px;
}

.status-text {
  color: var(--muted);
  font-size: 14px;
  font-weight: 600;
}

.translation-result {
  min-height: 92px;
  margin: 0;
  color: var(--text);
  font-size: clamp(28px, 5.2vw, 46px);
  font-weight: 850;
  line-height: 1.32;
}

.translation-result.is-muted {
  color: #a0a4aa;
  font-weight: 750;
}

.translation-result.is-error {
  color: var(--danger);
  font-size: clamp(22px, 4vw, 34px);
}

.translation-result.is-success {
  animation: result-in 180ms ease-out;
}

@keyframes result-in {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  .page-shell {
    align-items: stretch;
    padding: 16px;
  }

  .translator {
    padding: 16px;
    border-radius: 18px;
  }

  .language-row {
    grid-template-columns: 1fr;
  }

  .direction-mark {
    width: 100%;
    border-radius: 9px;
  }

  .panel {
    padding: 18px;
  }

  .panel-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .translate-button {
    width: 100%;
  }
}
```

- [ ] **Step 3: Create frontend interaction script**

Create `demo_frontend/app.js` with:

```javascript
const sourceText = document.querySelector("#sourceText");
const translateButton = document.querySelector("#translateButton");
const statusText = document.querySelector("#statusText");
const translationResult = document.querySelector("#translationResult");

function setResult(text, state) {
  translationResult.textContent = text;
  translationResult.classList.remove("is-muted", "is-error", "is-success");
  if (state) {
    translationResult.classList.add(state);
  }
}

function setLoading(isLoading) {
  translateButton.disabled = isLoading;
  translateButton.textContent = isLoading ? "翻译中" : "翻译";
  statusText.textContent = isLoading ? "模型解码中" : "等待输入";
}

async function translateCurrentText() {
  const text = sourceText.value.trim();
  if (!text) {
    statusText.textContent = "需要输入";
    setResult("请输入一句英文。", "is-error");
    return;
  }

  setLoading(true);
  setResult("正在翻译...", "is-muted");

  try {
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Translation failed.");
    }
    statusText.textContent = "完成";
    setResult(payload.translation, "is-success");
  } catch (error) {
    statusText.textContent = "出错";
    setResult(error.message || "翻译失败，请稍后再试。", "is-error");
  } finally {
    setLoading(false);
  }
}

translateButton.addEventListener("click", translateCurrentText);
sourceText.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    translateCurrentText();
  }
});
```

- [ ] **Step 4: Commit frontend files**

Run:

```powershell
git add demo_frontend\index.html demo_frontend\styles.css demo_frontend\app.js
git commit -m "Add translation demo frontend"
```

Expected: commit succeeds.

## Task 4: Integration Verification

**Files:**
- Existing: `code/demo_server.py`
- Existing: `demo_frontend/index.html`
- Existing: `demo_frontend/styles.css`
- Existing: `demo_frontend/app.js`
- Existing: `tests/test_demo_server.py`

- [ ] **Step 1: Run focused automated tests**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python -m pytest tests\test_demo_server.py tests\test_evaluate.py tests\test_checkpoint_loading.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Start the demo server**

Run:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python code\demo_server.py --device cuda
```

Expected console output:

```text
Loading translation model...
Translation demo running at http://127.0.0.1:8000
Press Ctrl+C to stop.
```

- [ ] **Step 3: Verify API from PowerShell while the server is running**

Run in a second PowerShell:

```powershell
$body = @{ text = "tom is a student ." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/translate" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
```

Expected: response contains a non-empty `translation` field with readable Chinese characters.

- [ ] **Step 4: Verify the browser UI**

Open:

```text
http://127.0.0.1:8000
```

Manual checks:

- the first screen is the translator, not a landing page
- input panel is above result panel
- typing `tom is a student .` and clicking `翻译` updates the lower panel
- Chinese text renders correctly
- repeated clicks while loading are disabled
- a blank input shows `请输入一句英文。`
- a longer input remains inside the panel without overlapping controls

- [ ] **Step 5: Commit any final polish from verification**

If verification requires changes, run the relevant tests again and commit only those changes:

```powershell
git add code\demo_server.py demo_frontend\index.html demo_frontend\styles.css demo_frontend\app.js tests\test_demo_server.py
git commit -m "Polish translation demo verification"
```

Expected: create this commit only if files changed during verification.

## Self-Review

Spec coverage:

- Local web demo for one sentence: covered by Tasks 2 and 3.
- Clean A visual direction with input above output: covered by Task 3.
- Three-checkpoint character ensemble defaults: covered by Task 2.
- `POST /api/translate` with UTF-8 JSON: covered by Tasks 1 and 2.
- Empty/loading/success/error states: covered by Task 3.
- Stubbed API tests avoiding checkpoint load: covered by Task 1.
- Manual conda-base verification: covered by Task 4.

Placeholder scan:

- No red-flag placeholder terms or vague implementation steps are present.
- The word `placeholder` appears only as a real HTML/CSS input attribute concept in the implementation content.

Type and interface consistency:

- `create_request_handler(frontend_dir, translator)` is defined in Task 2 and imported by Task 1.
- Stub translator and `DemoTranslator` both expose `translate(text) -> dict`.
- Frontend sends `{"text": "..."}` and backend reads `payload["text"]`.
- Backend returns `translation`, `source_tokens`, and `prediction_tokens`, and frontend uses `translation`.
