#!/usr/bin/env python3

import argparse

from cli.lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
)
from utilities.text_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", help="Text to embed")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    embed_query_parser = subparsers.add_parser("embed_query", help="embed query")
    embed_query_parser.add_argument("query", help="query to embed")
    search_parser = subparsers.add_parser("search", help="perform search")
    search_parser.add_argument("query", help="query to embed")
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "embed_text":
            embed_text(args.text)
        case "verify":
            semantic_search.verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            movies = load_movies()
            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, args.limit)
            for index, result in enumerate(results):
                print(f"{index + 1}. {result['title']} (score: {result['score']})")
                print(result["description"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
