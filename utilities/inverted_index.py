import math
import os
import pickle

from utilities.text_utils import generate_stop_words_list, load_movies, tokenize_text

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        # Tokenize the input text, then add each token to the index with the document ID.
        tokens = tokenize_text(text, generate_stop_words_list())
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = {}
            if token not in self.term_frequencies[doc_id]:
                self.term_frequencies[doc_id][token] = 1
            else:
                self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        # Return the set of document IDs for a given token, sorted in ascending order.
        tokens = tokenize_text(term, generate_stop_words_list())
        for token in tokens:
            return sorted(self.index.get(token, []))

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie

    def save(self):
        os.makedirs(f"{CACHE_DIR}", exist_ok=True)
        with open(f"{CACHE_DIR}/index.pkl", "wb") as handle:
            pickle.dump(self.index, handle)
        with open(f"{CACHE_DIR}/docmap.pkl", "wb") as handle:
            pickle.dump(self.docmap, handle)
        with open(f"{CACHE_DIR}/term_frequencies.pkl", "wb") as handle:
            pickle.dump(self.term_frequencies, handle)
        with open(f"{CACHE_DIR}/doc_lengths.pkl", "wb") as handle:
            pickle.dump(self.doc_lengths, handle)

    def load(self):
        try:
            with open(f"{CACHE_DIR}/index.pkl", "rb") as handle:
                self.index = pickle.load(handle)
        except Exception as e:
            raise Exception(f"Failed to load index.pkl {e}")
        try:
            with open(f"{CACHE_DIR}/docmap.pkl", "rb") as handle:
                self.docmap = pickle.load(handle)
        except Exception as e:
            raise Exception(f"Failed to load docmap.pkl {e}")
        try:
            with open(f"{CACHE_DIR}/term_frequencies.pkl", "rb") as handle:
                self.term_frequencies = pickle.load(handle)
        except Exception as e:
            raise Exception(f"Failed to load term_frequencies.pkl {e}")
        try:
            with open(f"{CACHE_DIR}/doc_lengths.pkl", "rb") as handle:
                self.doc_lengths = pickle.load(handle)
        except Exception as e:
            raise Exception(f"Failed to load doc_lengths.pkl {e}")

    def get_document(self, doc_id):
        return self.docmap[doc_id]

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term, generate_stop_words_list())
        if len(tokens) == 1:
            return self.term_frequencies.get(doc_id, {}).get(tokens[0], 0)
        raise Exception("Term must be a single token that is not a stopword")

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = tokenize_text(term, generate_stop_words_list())
        if len(tokenized_term) != 1:
            raise Exception("Term must be a single token that is not a stopword")

        N = len(self.docmap)
        df = len(self.get_documents(term) or [])
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def __get_avg_doc_length(self) -> float:
        if self.doc_lengths:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return 0.0

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query, generate_stop_words_list())
        scores = {}
        for token in tokens:
            for doc_id in self.index[token]:
                scores[doc_id] = scores.get(doc_id, 0) + self.bm25(doc_id, token)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:limit]
