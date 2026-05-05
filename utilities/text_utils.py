from nltk.stem import PorterStemmer


def strip_punctuation(string_with_punctuation):
    replace_function = str.maketrans(
        {",": "", ".": "", "!": "", ":": "", "'": "", "-": ""}
    )
    return string_with_punctuation.translate(replace_function)


def generate_stop_words_list():
    with open("data/stopwords.txt", "r") as f:
        data = f.read()
        return data.splitlines()


def remove_stopwords_from_phrase(phrase, stop_words):
    return " ".join(word for word in phrase.split() if word not in stop_words)


def stem_phrase(phrase):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(word) for word in phrase.split())


def sanitize_query(query, stop_words):
    lowered_query = query.lower()
    stipped_punction_and_lowered = strip_punctuation(lowered_query)
    stopwords_removed = remove_stopwords_from_phrase(
        stipped_punction_and_lowered, stop_words
    )
    stemmed_query = stem_phrase(stopwords_removed)

    return stemmed_query


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
