import argparse
import json


def strip_punction(string_with_punctuation):
    replace_function = str.maketrans(
        {",": "", ".": "", "!": "", ":": "", "'": "", "-": ""}
    )
    return string_with_punctuation.translate(replace_function)


def generate_matched_movie_titles_list(query, movies_data):
    matched_titles = []
    query_no_punc_lower = strip_punction(query)
    for movie in movies_data["movies"]:
        title_no_punc_lower = strip_punction(movie["title"].lower())
        for query_word in query_no_punc_lower.split():
            for title_word in title_no_punc_lower.split():
                if query_word in title_word:
                    matched_titles.append(movie["title"])
    return matched_titles


def generate_stop_words_list():
    with open("data/stopwords.txt", "r") as f:
        data = f.read()
        return data.splitlines()


def remove_stopwords_from_phrase(phrase, stop_words):
    sanitized_text = " ".join(
        word for word in phrase.split() if word not in stop_words
    )
    return sanitized_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    stop_words = generate_stop_words_list()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                matched_titles = generate_matched_movie_titles_list(
                    args.query.lower(), data
                )

                count = 1
                for movie_title in list(dict.fromkeys(matched_titles))[:5]:
                    print(f"{count}. {movie_title}")
                    count = count + 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
