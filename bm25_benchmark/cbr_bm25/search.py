"""Inverted index, Russian text preprocessing and BM25 scoring."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from .data import Passage


_DASHES = str.maketrans({char: "-" for char in "‐‑‒–—−"})
_LAW_NUMBER = re.compile(r"(?<!\w)([0-9]+)\s*(?:-\s*|(?=фз))(фз|п|у|мр)(?!\w)")
_TOKENS = re.compile(r"[0-9]+(?:[.,][0-9]+)*(?:-(?:фз|п|у|мр)\b)?|[^\W\d_]+", re.UNICODE)
ACT = re.compile(r"(?<!\w)(\d+)\s*(?:[-‑–—]?\s*(фкз|фз)|[-‑–—]\s*([1iіпуи]|мр))(?!\w)", re.IGNORECASE)
ARTICLE = re.compile(r"(?:\bстать(?:[яеию]|[её]й)|\bст\.)\s*(\d+(?:[.-]\d+)*)", re.IGNORECASE)
POINT = re.compile(r"(?:\bпункт(?:ом|а|е|у)?|\bп\.)\s*(\d+(?:[.-]\d+)*)", re.IGNORECASE)
ARCHIVE_QUERY = re.compile(r"\b(?:архив\w*|первоначальн\w*|опубликован\w*)\b", re.IGNORECASE)
ARCHIVE_SUFFIXES = frozenset({"п", "у", "и", "мр"})
_RUSSIAN = re.compile(r"[а-я]+")
_STOP_WORDS = frozenset("""
    а бы был была были было быть в вам вас ведь во вот вы где да делать для
    его ее если есть еще же за и из или им их к как какая какие какой какую
    ко когда кто ли либо мне мой можно моя мы на над нам нас но нужно о об они оно от
    по под при про с сам свой со так такие такое также там те тем то того
    тоже той тот тут ты у уже чем что чтобы эта эти это этот я
""".split())


def make_tokenizer(mode: str) -> Callable[[str], list[str]]:
    if mode not in {"basic", "stem"}:
        raise ValueError(f"unknown tokenizer {mode!r}")
    stem_word = None
    if mode == "stem":
        try:
            import snowballstemmer
        except ImportError as exc:
            raise ValueError("stem mode requires python -m pip install -r requirements.txt") from exc
        @lru_cache(maxsize=65_536)
        def stem_word(word: str) -> str:
            return snowballstemmer.stemmer("russian").stemWord(word)

    def tokenize(text: str) -> list[str]:
        # NFKC would turn the number sign into the letters "No".
        text = text.replace("№", " ")
        text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").translate(_DASHES)
        text = _LAW_NUMBER.sub(r"\1-\2", text)
        words = [word.replace(",", ".") for word in _TOKENS.findall(text) if word not in _STOP_WORDS]
        if stem_word is not None:
            words = [stem_word(word) if _RUSSIAN.fullmatch(word) else word for word in words]
        return words

    return tokenize


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class BM25Index:
    """Rank passages, with one result per document by default."""

    def __init__(self, passages: Iterable[Passage], *, tokenizer: str = "basic", k1: float = 1.2, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self._passages = tuple(passages)
        if not self._passages:
            raise ValueError("cannot index an empty corpus")
        self._tokenize = make_tokenizer(tokenizer)
        self._k1 = k1
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for index, passage in enumerate(self._passages):
            words = self._tokenize(passage.index_text)
            lengths.append(len(words))
            for word, frequency in Counter(words).items():
                self._postings[word].append((index, frequency))
        size = len(lengths)
        average_length = sum(lengths) / size
        if average_length == 0:
            raise ValueError("corpus contains no searchable tokens")
        self._length_norm = [k1 * (1 - b + b * length / average_length) for length in lengths]
        self._idf = {
            word: math.log1p((size - len(postings) + 0.5) / (len(postings) + 0.5))
            for word, postings in self._postings.items()
        }

    def search(self, query: str, *, top_k: int = 3, passages_per_document: int = 1,
               accept: Callable[[Passage], bool] | None = None) -> list[Hit]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(passages_per_document) is not int or passages_per_document <= 0:
            raise ValueError("passages_per_document must be a positive integer")
        scores: dict[int, float] = defaultdict(float)
        for word in sorted(set(self._tokenize(query))):
            if word not in self._postings:
                continue
            weight = self._idf[word] * (self._k1 + 1)
            for index, frequency in self._postings[word]:
                scores[index] += weight * frequency / (frequency + self._length_norm[index])

        candidates = (i for i in scores if accept is None or accept(self._passages[i]))
        order = sorted(candidates, key=lambda i: (-scores[i], self._passages[i].document_id, self._passages[i].passage_id))
        hits = []
        document_counts: Counter[str] = Counter()
        for index in order:
            passage = self._passages[index]
            if document_counts[passage.document_id] >= passages_per_document:
                continue
            hits.append(Hit(passage, scores[index]))
            document_counts[passage.document_id] += 1
            if len(hits) == top_k:
                break
        return hits


def _act_number(match: re.Match) -> tuple[str, str]:
    suffix = (match[2] or match[3]).casefold()
    return match[1], "1" if suffix in {"i", "і"} else suffix


def _exact_legal_hits(index: BM25Index, query: str, *, passages_per_document: int) -> list[Hit]:
    numbers = {_act_number(match) for match in ACT.finditer(query)}
    article, point = ARTICLE.search(query), POINT.search(query)
    if len(numbers) != 1 or article is None or any(suffix in ARCHIVE_SUFFIXES for _, suffix in numbers):
        return []

    def matches_article(passage):
        number = ACT.search(passage.title)
        heading = ARTICLE.match(passage.section)
        return number and _act_number(number) in numbers and heading and heading[1] == article[1]

    if point is not None:
        begins_point = re.compile(rf"(?m)^\s*{re.escape(point[1])}[.)](?=\s)")
        hits = index.search(query, top_k=3, passages_per_document=min(3, passages_per_document),
                            accept=lambda passage: matches_article(passage) and begins_point.search(passage.text))
        if hits:
            return hits
    return index.search(query, top_k=3, passages_per_document=min(3, passages_per_document), accept=matches_article)


class SearchIndex:
    """Search reference material, current legislation and archived acts from one corpus."""

    def __init__(self, passages: Iterable[Passage], **options):
        references, laws, archive = [], [], []
        for passage in passages:
            if passage.kind == "effective_law":
                laws.append(passage)
            elif passage.kind == "archive_publication":
                archive.append(passage)
            else:
                references.append(passage)
        if not references and not laws and not archive:
            raise ValueError("cannot index an empty corpus")
        self._references = BM25Index(references, **options) if references else None
        self._laws = BM25Index(laws, **options) if laws else None
        self._archive = BM25Index(archive, **options) if archive else None

    def _named_archive(self, query: str, per_document: int) -> list[Hit]:
        if self._archive is None:
            return []
        numbers = {_act_number(match) for match in ACT.finditer(query)
                   if _act_number(match)[1] in ARCHIVE_SUFFIXES}
        if not numbers:
            return []

        def matches(passage: Passage) -> bool:
            number = ACT.search(passage.title)
            return number is not None and _act_number(number) in numbers

        return self._archive.search(query, top_k=4, passages_per_document=min(3, per_document), accept=matches)

    def search(self, query: str, *, top_k: int = 3, passages_per_document: int = 1) -> list[Hit]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(passages_per_document) is not int or passages_per_document <= 0:
            raise ValueError("passages_per_document must be a positive integer")
        options = {"top_k": top_k, "passages_per_document": passages_per_document}
        references = self._references.search(query, **options) if self._references else []
        exact = (_exact_legal_hits(self._laws, query, passages_per_document=passages_per_document)
                 if self._laws else [])
        if exact:
            return (exact + references)[:top_k]
        named_archive = self._named_archive(query, passages_per_document)
        archive_intent = ARCHIVE_QUERY.search(query) is not None
        if named_archive:
            if archive_intent:
                return (named_archive + references)[:top_k]
            return (references[:2] + named_archive + references[2:])[:top_k]
        if archive_intent and self._archive:
            return (self._archive.search(query, **options) + references)[:top_k]
        if len(references[:4]) >= top_k:
            return references
        laws = (self._laws.search(query, top_k=4 if references else top_k,
                                 passages_per_document=min(3, passages_per_document))
                if self._laws else [])
        archive = (self._archive.search(query, top_k=2 if references or laws else top_k,
                                       passages_per_document=min(2, passages_per_document))
                   if self._archive and len(references[:4]) + len(laws) < top_k else [])
        return (references[:4] + laws + archive + references[4:])[:top_k]
