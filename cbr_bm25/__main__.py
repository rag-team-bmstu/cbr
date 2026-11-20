"""Command line search and benchmark runner."""

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .benchmark import evaluate
from .data import CORPUS_PATH, load_corpus, load_questions
from .search import SearchIndex


DATA = Path(__file__).resolve().parents[1] / "data"


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="rank distinct documents for one query")
    search.add_argument("query")
    search.add_argument("--top-k", type=positive_integer, default=3)
    search.add_argument("--tokenizer", choices=("basic", "stem"), default="stem")

    benchmark = commands.add_parser("benchmark", help="run offline retrieval evaluation")
    benchmark.add_argument("--questions", type=Path, default=DATA / "questions.jsonl")
    benchmark.add_argument("--dataset-info", type=Path, default=DATA / "dataset.json")
    benchmark.add_argument("--split", choices=("dev", "test", "all"), default="test")
    benchmark.add_argument("--top-k", type=positive_integer, default=3)
    benchmark.add_argument("--tokenizer", choices=("basic", "stem", "both"), default="basic")
    benchmark.add_argument("--output", type=Path, help="write the full JSON report to this file")
    args = parser.parse_args()

    try:
        passages = load_corpus(args.corpus)
        options = {"k1": args.k1, "b": args.b}
        if args.command == "search":
            index = SearchIndex(passages, tokenizer=args.tokenizer, **options)
            hits = index.search(args.query, top_k=args.top_k)
            print(json.dumps([
                {**asdict(hit.passage), "rank": rank, "score": hit.score}
                for rank, hit in enumerate(hits, 1)
            ], ensure_ascii=False, indent=2))
            return

        document_ids = {passage.document_id for passage in passages}
        questions = load_questions(args.questions, document_ids)
        dataset = json.loads(args.dataset_info.read_text(encoding="utf-8"))
        if not isinstance(dataset, dict):
            raise ValueError("dataset info must be a JSON object")
        modes = ("basic", "stem") if args.tokenizer == "both" else (args.tokenizer,)
        splits = ("dev", "test") if args.split == "all" else (args.split,)
        report = {
            "dataset": dataset,
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
            "documents": len(document_ids),
            "passages": len(passages),
            "top_k": args.top_k,
            **options,
            "runs": [],
        }
        for mode in modes:
            index = SearchIndex(passages, tokenizer=mode, **options)
            for split in splits:
                selected = [question for question in questions if question.split == split]
                report["runs"].append({
                    "tokenizer": mode,
                    "snowballstemmer_version": version("snowballstemmer") if mode == "stem" else None,
                    "split": split,
                    **evaluate(index, selected, args.top_k),
                })

        output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
            for run in report["runs"]:
                print(f"{run['split']:4} {run['tokenizer']:5} Recall@{args.top_k}={run['recall_at_k']:.4f} "
                      f"MRR@{args.top_k}={run['mrr_at_k']:.4f} (n={run['queries']})")
        else:
            print(output, end="")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
