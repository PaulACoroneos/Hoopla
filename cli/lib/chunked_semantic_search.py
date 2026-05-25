import json
import os

import numpy as np

from cli.lib.semantic_search import SemanticSearch, cosine_similarity, embed_query_text
from utilities.constants import CACHE_DIR
from utilities.search_utils import format_search_result
from utilities.text_utils import chunk_sentences


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        all_chunks = []
        chunk_metadata = []

        for idx, document in enumerate(documents):
            self.document_map[document["id"]] = document
            description = document["description"]
            if description:
                chunks = chunk_sentences(description, 4, 1)
                all_chunks.extend(chunks)
                for chunk_idx, _chunk in enumerate(chunks):
                    chunk_metadata.append(
                        {
                            "movie_idx": idx,
                            "chunk_idx": chunk_idx,
                            "total_chunks": len(chunks),
                        }
                    )

        self.chunk_embeddings = self.model.encode(
            inputs=all_chunks, show_progress_bar=True
        )
        self.chunk_metadata = chunk_metadata
        np.save(file=f"{CACHE_DIR}/chunk_embeddings.npy", arr=self.chunk_embeddings)
        with open(f"{CACHE_DIR}/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(f"{CACHE_DIR}/chunk_embeddings.npy") and os.path.exists(
            f"{CACHE_DIR}/chunk_metadata.json"
        ):
            self.chunk_embeddings = np.load(f"{CACHE_DIR}/chunk_embeddings.npy")
            with open(f"{CACHE_DIR}/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = self.generate_embedding(query)
        chunk_score = []
        for idx, chunk in enumerate(self.chunk_embeddings):
            cos_similarity = cosine_similarity(embedded_query, chunk)
            chunk_score.append(
                {
                    "chunk_idx": idx,
                    "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                    "score": cos_similarity,
                }
            )
        movie_idx_to_score = {}
        for idx, chunk in enumerate(chunk_score):
            if (
                chunk["movie_idx"] not in movie_idx_to_score
                or chunk["score"] > movie_idx_to_score[chunk["movie_idx"]]["score"]
            ):
                movie_idx_to_score[chunk["movie_idx"]] = chunk_score[idx]
        sorted_limited_movie_idx_to_score = sorted(
            movie_idx_to_score.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        results = []
        for score in sorted_limited_movie_idx_to_score:
            movie_idx = score["movie_idx"]
            results.append(
                format_search_result(
                    movie_idx,
                    self.documents[movie_idx]["title"],
                    self.documents[movie_idx]["description"],
                    score["score"],
                    {},
                )
            )

        return results
