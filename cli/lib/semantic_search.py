import os

import numpy as np
from sentence_transformers import SentenceTransformer

from utilities.text_utils import load_movies


def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(
        f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list | None = None
        self.document_map = {}

    def verify_model(self):
        semantic_search = SemanticSearch()

        print(f"Model loaded: {semantic_search.model}")
        print(f"Max sequence length: {semantic_search.model.max_seq_length}")

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        stringified_documents = []
        for document in documents:
            self.document_map[document["id"]] = document
            stringified_documents.append(
                f"{document['title']}: {document['description']}"
            )
        self.embeddings = self.model.encode(
            inputs=stringified_documents, show_progress_bar=True
        )
        np.save(file="cache/movie_embeddings.npy", arr=self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        stringified_documents = []
        for document in documents:
            self.document_map[document["id"]] = document
            stringified_documents.append(
                f"{document['title']}: {document['description']}"
            )
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        cos_similarities = []
        for embedding in self.embeddings:
            cos_similarities.append(cosine_similarity(query_embedding, embedding))
        scored_list = []
        for index, document in enumerate(self.documents):
            scored_list.append((cos_similarities[index], document))
        desc_scored_list = sorted(scored_list, key=lambda x: x[0], reverse=True)
        limited_scored_list = desc_scored_list[:limit]
        structured_top_results_list = []
        for scored_movie in limited_scored_list:
            structured_top_results_list.append(
                {
                    "score": scored_movie[0],
                    "title": scored_movie[1]["title"],
                    "description": scored_movie[1]["description"],
                    "doc_id": scored_movie[1]["id"],
                }
            )

        return structured_top_results_list
