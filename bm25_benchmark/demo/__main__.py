"""Serve the local checkpoint demo with no web framework."""

import argparse
import json
import os
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cbr_bm25 import CORPUS_PATH, SearchIndex, load_corpus

from .answer import COARENA_FREE_MODELS, PROVIDER_NAME, answer_question, chat_completion


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "demo" / "web"
ASSETS = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
MAX_QUESTION = 1200
API_KEY_VARIABLE = "COARENA_API_KEY"


def read_api_key() -> str:
    key = os.environ.get(API_KEY_VARIABLE, "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == API_KEY_VARIABLE:
                return value.strip().strip("\"'")
    return ""


class DemoHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server, *, index: SearchIndex, generate, review=None):
        self.index = index
        self.generate = generate
        self.review = review
        super().__init__(request, client_address, server)

    def send_content(self, status: int, content: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{media_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status: int, payload: dict) -> None:
        self.send_content(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        asset = ASSETS.get(self.path)
        if not asset:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        name, media_type = asset
        self.send_content(HTTPStatus.OK, (WEB / name).read_bytes(), media_type)

    def do_POST(self) -> None:
        if self.path != "/api/answer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        port = self.server.server_port
        if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Демо доступно по локальному адресу."})
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers['Host']}":
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Запрос должен приходить со страницы демо."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Ожидается JSON."})
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(size))
            question = payload.get("question") if isinstance(payload, dict) else None
        except ValueError:
            question = None
        if not isinstance(question, str) or not 0 < len(question.strip()) <= MAX_QUESTION:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"Введите вопрос длиной от 1 до {MAX_QUESTION} символов."})
            return
        result = answer_question(self.index, question.strip(), self.generate, review=self.review)
        result["provider"] = PROVIDER_NAME
        self.send_json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", choices=COARENA_FREE_MODELS, default=COARENA_FREE_MODELS[0],
                        help="free Coarena model ID")
    args = parser.parse_args()

    try:
        index = SearchIndex(load_corpus(CORPUS_PATH), tokenizer="stem")
        key = read_api_key()
        generate = partial(chat_completion, api_key=key, model=args.model)
        review = partial(generate, fast=False) if args.model == COARENA_FREE_MODELS[0] else None
        handler = partial(DemoHandler, index=index, generate=generate, review=review)
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Демо: http://127.0.0.1:{server.server_port}", flush=True)
            if not key:
                print(f"Добавьте {API_KEY_VARIABLE} в .env и перезапустите сервер для генерации.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Ошибка запуска: {exc}\n")


if __name__ == "__main__":
    main()
