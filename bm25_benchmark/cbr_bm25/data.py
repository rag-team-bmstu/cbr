"""Read and validate a versioned corpus and relevance judgments."""

import gzip
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl.gz"


@dataclass(frozen=True)
class Passage:
    document_id: str
    passage_id: str
    title: str
    url: str
    kind: str
    section: str
    text: str
    references: tuple[str, ...]
    updated: str | None = None

    @property
    def index_text(self) -> str:
        return " ".join((self.title, self.section, self.text, *self.references))


@dataclass(frozen=True)
class Question:
    query_id: str
    split: str
    topic: str
    text: str
    judgments: dict[str, str]


def _records(path: Path):
    open_source = gzip.open if path.suffix == ".gz" else open
    with open_source(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location}: expected a JSON object")
            yield row, location


def _string(row: dict, field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a nonempty string")
    return value.strip()


def _valid_source(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme == "https" and parsed.hostname:
        return True
    # The public EBPI edition viewer is served over HTTP.
    return (parsed.scheme == "http" and parsed.netloc == "actual.pravo.gov.ru"
            and parsed.path == "/content/content.html")


def load_corpus(path: Path) -> list[Passage]:
    passages = []
    seen_ids = set()
    seen_urls = set()
    for row, location in _records(path):
        document_id = _string(row, "document_id", location)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", document_id) or document_id in seen_ids:
            raise ValueError(f"{location}: invalid or duplicate document_id")
        seen_ids.add(document_id)
        title = _string(row, "title", location)
        url = _string(row, "url", location)
        parsed = urlsplit(url)
        if not _valid_source(url):
            raise ValueError(f"{location}: invalid source URL")
        # Count a publication once even if a URL has a different section anchor.
        edition = parsed.fragment if parsed.hostname == "actual.pravo.gov.ru" else ""
        source_key = (parsed.hostname.removeprefix("www."), parsed.path.rstrip("/"), parsed.query, edition)
        if source_key in seen_urls:
            raise ValueError(f"{location}: duplicate source URL")
        seen_urls.add(source_key)
        kind = _string(row, "kind", location)
        references = row.get("references", [])
        if not isinstance(references, list) or not all(isinstance(ref, str) and ref.strip() for ref in references):
            raise ValueError(f"{location}: references must be a list of nonempty strings")
        entries = row.get("passages")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{location}: at least one passage is required")
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, dict):
                raise ValueError(f"{location}: passage {index} must be an object")
            passage_url = _string(entry, "url", location) if "url" in entry else url
            if not _valid_source(passage_url):
                raise ValueError(f"{location}: passage {index} has an invalid source URL")
            updated = None
            if "updated" in entry:
                raw_date = _string(entry, "updated", location)
                try:
                    updated = date.fromisoformat(raw_date).isoformat()
                except ValueError as exc:
                    raise ValueError(f"{location}: passage {index} has an invalid update date") from exc
            passages.append(Passage(
                document_id=document_id,
                passage_id=f"{document_id}#{index:02d}",
                title=title,
                url=passage_url,
                kind=kind,
                section=_string(entry, "section", location),
                text=_string(entry, "text", location),
                references=tuple(ref.strip() for ref in references),
                updated=updated,
            ))
    if not passages:
        raise ValueError(f"{path}: empty corpus")
    return passages


def load_questions(path: Path, document_ids: set[str]) -> list[Question]:
    questions = []
    seen_ids = set()
    seen_questions = set()
    for row, location in _records(path):
        query_id = _string(row, "query_id", location)
        text = _string(row, "question", location)
        canonical_text = " ".join(text.casefold().split())
        if query_id in seen_ids or canonical_text in seen_questions:
            raise ValueError(f"{location}: duplicate question or query_id")
        seen_ids.add(query_id)
        seen_questions.add(canonical_text)
        split = _string(row, "split", location)
        if split not in {"dev", "test"}:
            raise ValueError(f"{location}: split must be dev or test")
        topic = _string(row, "topic", location)
        judgments = row.get("relevant")
        if not isinstance(judgments, dict) or len(judgments) < 3:
            raise ValueError(f"{location}: at least three distinct relevant documents are required")
        unknown = judgments.keys() - document_ids
        if unknown:
            raise ValueError(f"{location}: unknown document IDs {', '.join(sorted(unknown))}")
        reasons = {document_id: _string(judgments, document_id, location) for document_id in judgments}
        questions.append(Question(query_id, split, topic, text, reasons))
    if sum(question.split == "test" for question in questions) < 15:
        raise ValueError(f"{path}: at least 15 test questions are required")
    if not any(question.split == "dev" for question in questions):
        raise ValueError(f"{path}: dev split is empty")
    return questions
