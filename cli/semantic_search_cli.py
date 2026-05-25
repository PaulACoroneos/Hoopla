#!/usr/bin/env python3

import argparse

from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from cli.lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
)
from utilities.text_utils import chunk_sentences, chunk_words, load_movies


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
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", help="text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="Size of chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="Amount of overlap"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunk text"
    )
    semantic_chunk_parser.add_argument("text", help="text to semantic chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, nargs="?", default=4, help="Max size of chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="Amount of overlap"
    )

    subparsers.add_parser("embed_chunks", help="Embed chunks")

    search_chunked = subparsers.add_parser(
        "search_chunked", help="Search chunked embeddings"
    )
    search_chunked.add_argument("query", help="search query")
    search_chunked.add_argument(
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
            movies = load_movies()
            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, args.limit)
            for index, result in enumerate(results):
                print(f"{index + 1}. {result['title']} (score: {result['score']})")
                print(result["description"])

        case "chunk":
            chunked_text = chunk_words(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for idx, text in enumerate(chunked_text, 1):
                print(f"{idx}. {text}")

        case "semantic_chunk":
            chunks = chunk_sentences(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for idx, text in enumerate(chunks, 1):
                print(f"{idx}. {text}")

        case "embed_chunks":
            movies = load_movies()
            chunked_semantic_search = ChunkedSemanticSearch()
            embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)

            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            chunked_semantic_search = ChunkedSemanticSearch()
            chunked_semantic_search.load_or_create_chunk_embeddings(movies)
            results = chunked_semantic_search.search_chunks(args.query, args.limit)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
