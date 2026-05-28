import argparse
import os

from dotenv import load_dotenv
from google import genai

from cli.lib.hybrid_search import HybridSearch, normalize
from utilities.text_utils import load_movies

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


client = genai.Client(api_key=api_key)


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

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="generate weighted search result"
    )
    rrf_search_parser.add_argument("query", help="search string")
    rrf_search_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="k to control impact of ranking of score",
    )
    rrf_search_parser.add_argument("--limit", default=5, type=int, help="limit results")

    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        help="Query enhancement method",
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

        case "rrf-search":
            docs = load_movies()
            hybrid_search = HybridSearch(docs)
            query = args.query

            if args.enhance == "spell":
                res = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=f"""Fix any spelling errors in the user-provided movie search query below.
                    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                    Preserve punctuation and capitalization unless a change is required for a typo fix.
                    If there are no spelling errors, or if you're unsure, output the original query unchanged.
                    Output only the final query text, nothing else.
                    User query: "{query}"
                    """,
                )
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{res.text}'\n")
                query = res.text

            results = hybrid_search.rrf_search(query, args.k, args.limit)

            for idx, result in enumerate(results, 1):
                print(f"{idx}. {result['document']['title']}")
                print(f"RRF Score: {result['hybrid']:.3f}")
                print(
                    f"BM25 Rank: {result['bm_25']}, Semantic Rank: {result['semantic']}"
                )
                print(f"{result['document']['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
