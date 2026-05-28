import argparse

from cli.lib.hybrid_search import HybridSearch, normalize
from utilities.text_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the input text"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="generate weighted search result"
    )
    weighted_search_parser.add_argument("query", help="search string")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="alpha to control weighting between bm25 and semantic search",
    )
    weighted_search_parser.add_argument(
        "--limit", default=5, type=int, help="limit results"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            minmax_scores = normalize(args.scores)

            for score in minmax_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            docs = load_movies()
            hybrid_search = HybridSearch(docs)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)

            for idx, result in enumerate(results, 1):
                print(f"{idx}. {result['document']['title']}")
                print(f"Hybrid Score: {result['hybrid']:.3f}")
                print(
                    f"BM25: {result['bm_25']:.3f}, Semantic: {result['semantic']:.3f}"
                )
                print(f"{result['document']['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
