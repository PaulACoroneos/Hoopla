import os

from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from utilities.inverted_index import InvertedIndex


def normalize(scores):
    minmax_scores = []
    min_score = min(scores)
    max_score = max(scores)

    if scores:
        if min_score == max_score:
            minmax_scores = [1.0] * len(scores)
        else:
            for score in scores:
                minmax_scores.append((score - min_score) / (max_score - min_score))
    return minmax_scores


def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_res = self._bm25_search(query, limit * 500)
        semantic_res = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = {doc_id: score for doc_id, score in bm25_res}
        semantic_scores = {
            self.documents[res["id"]]["id"]: res["score"] for res in semantic_res
        }

        bm25_normalized_scores = dict(zip(bm25_scores, normalize(bm25_scores.values())))
        semantic_normalized_scores = dict(
            zip(semantic_scores, normalize(semantic_scores.values()))
        )

        results = {}
        for document in self.documents:
            bm25 = bm25_normalized_scores.get(document["id"], 0.0)
            semantic = semantic_normalized_scores.get(document["id"], 0.0)

            results[document["id"]] = {
                "document": document,
                "bm_25": bm25,
                "semantic": semantic,
                "hybrid": hybrid_score(bm25, semantic, alpha),
            }

        return sorted(results.values(), key=lambda x: x["hybrid"], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
