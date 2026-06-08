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

        def cleanup_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(cleanup_server)
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
