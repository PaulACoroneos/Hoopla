import argparse
import json

from utilities.text_utils import (
    generate_stop_words_list,
    sanitize_movie_titles,
    sanitize_query,
)


def generate_matched_movie_titles_list(query, movies_data, stop_words):
    matched_titles = []
    clean_query = sanitize_query(query, stop_words)
    santized_titles_with_originals = sanitize_movie_titles(movies_data, stop_words)

    for original_title, clean_title in santized_titles_with_originals:
        for query_word in clean_query.split():
            for title_word in clean_title.split():
                if query_word in title_word:
                    matched_titles.append(original_title)

    return matched_titles


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
                    args.query.lower(), data, stop_words
                )

                count = 1
                for movie_title in list(dict.fromkeys(matched_titles))[:5]:
                    print(f"{count}. {movie_title}")
                    count = count + 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
