#!/usr/bin/env python3

import argparse

from cli.lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", help="Text to embed")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    embed_query_parser = subparsers.add_parser("embed_query", help="embed query")
    embed_query_parser.add_argument("query", help="query to embed")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
