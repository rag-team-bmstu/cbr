"""Score fixed rankings with document-level Recall@k and reciprocal rank."""

from statistics import mean

from .data import Question
from .search import BM25Index, SearchIndex


def score_query(ranked_ids: list[str], relevant_ids: set[str], k: int) -> tuple[float, float]:
    if not relevant_ids or k <= 0:
        raise ValueError("positive k and nonempty relevance judgments are required")
    # A repeated ID never occupies a second position in the document ranking.
    top_ids = list(dict.fromkeys(ranked_ids))[:k]
    recall = len(relevant_ids.intersection(top_ids)) / len(relevant_ids)
    reciprocal_rank = next((1 / rank for rank, doc_id in enumerate(top_ids, 1) if doc_id in relevant_ids), 0.0)
    return recall, reciprocal_rank


def evaluate(index: SearchIndex | BM25Index, questions: list[Question], k: int) -> dict:
    rows = []
    for question in questions:
        hits = index.search(question.text, top_k=k)
        ranked = [hit.passage.document_id for hit in hits]
        relevant = set(question.judgments)
        recall, reciprocal_rank = score_query(ranked, relevant, k)
        rows.append({
            "query_id": question.query_id,
            "question": question.text,
            "topic": question.topic,
            "relevant_document_ids": sorted(relevant),
            "missed_document_ids": sorted(relevant.difference(ranked)),
            "recall": recall,
            "reciprocal_rank": reciprocal_rank,
            "results": [
                {
                    "rank": rank,
                    "document_id": hit.passage.document_id,
                    "passage_id": hit.passage.passage_id,
                    "url": hit.passage.url,
                    "score": hit.score,
                    "relevant": hit.passage.document_id in relevant,
                }
                for rank, hit in enumerate(hits, 1)
            ],
        })
    return {
        "queries": len(rows),
        "recall_at_k": mean(row["recall"] for row in rows),
        "mrr_at_k": mean(row["reciprocal_rank"] for row in rows),
        "recall_ceiling_at_k": mean(min(1, k / len(q.judgments)) for q in questions),
        "zero_hit_queries": sum(row["reciprocal_rank"] == 0 for row in rows),
        "per_query": rows,
    }
