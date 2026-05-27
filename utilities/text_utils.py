import json
import re
import string

from nltk.stem import PorterStemmer


def strip_punctuation(string_with_punctuation):
    replace_function = str.maketrans("", "", string.punctuation)

    return string_with_punctuation.translate(replace_function)


def generate_stop_words_list():
    with open("data/stopwords.txt", "r") as f:
        data = f.read()
        return data.splitlines()


def load_movies():
    with open("data/movies.json", "r") as f:
        return json.load(f)["movies"]


def remove_stopwords_from_phrase(phrase, stop_words):
    return " ".join(word for word in phrase.split() if word not in stop_words)


def stem_phrase(phrase):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(word) for word in phrase.split())


def tokenize_text(query, stop_words):
    lowered_query = query.lower()
    stipped_punction_and_lowered = strip_punctuation(lowered_query)
    stopwords_removed = remove_stopwords_from_phrase(
        stipped_punction_and_lowered, stop_words
    )
    return stem_phrase(stopwords_removed).split()


def sanitize_movie_titles(movies_data, stop_words):
    sanitized_movie_titles_with_original_titles = []
    for movie in movies_data["movies"]:
        clean_title = stem_phrase(
            remove_stopwords_from_phrase(
                strip_punctuation(movie["title"].lower()), stop_words
            )
        )
        sanitized_movie_titles_with_original_titles.append(
            (movie["title"], clean_title)
        )

    return sanitized_movie_titles_with_original_titles


def chunk_words(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def chunk_sentences(text, max_chunk_size, overlap):
    stripped_text = text.strip()
    if not stripped_text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", stripped_text)
    if len(sentences) == 1:
        return sentences
    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap):
        sub_chunk = [s.strip() for s in sentences[i : i + max_chunk_size] if s.strip()]
        # does the next sub chunk of sentences actually contain anything that isnt just previous overlap?
        if len(sub_chunk) > overlap:
            chunks.append(" ".join(sub_chunk))

    return chunks
