#!/usr/bin/env python3

import argparse

from cli.lib.semantic_search import SemanticSearch, embed_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", help="Text to embed")

    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "embed_text":
            embed_text(args.text)
        case "verify":
            semantic_search.verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
