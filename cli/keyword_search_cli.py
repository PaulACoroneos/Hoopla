import argparse

from utilities.inverted_index import InvertedIndex
from utilities.text_utils import (
    generate_stop_words_list,
    load_movies,
    sanitize_movie_titles,
    tokenize_text,
)


def generate_matched_movie_titles_list(query, movies_data, stop_words):
    matched_titles = []
    clean_query = tokenize_text(query, stop_words)
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
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser(
        "tf", help="Returns term frequency for a given document and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    stop_words = generate_stop_words_list()

    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Failed to load index: {e}")
                return
            tokenized_query = tokenize_text(args.query, stop_words)

            count = 0
            for token in tokenized_query:
                matched_documents = inverted_index.get_documents(token)
                if matched_documents:
                    for matched_id in matched_documents:
                        if count > 4:
                            break
                        count += 1
                        matched_doc = inverted_index.get_document(matched_id)
                        print(
                            matched_doc["title"],
                            matched_doc["id"],
                        )

        case "build":
            inverted_index.build()
            inverted_index.save()

        case "tf":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Failed to load index: {e}")
                return
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(tf if tf else "0")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
