import argparse
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

from cli.lib.hybrid_search import HybridSearch, normalize
from utilities.text_utils import load_movies

PROMPTS = {
    "spell": """Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.""",
    "enhance": """Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else."
    """,
    "expand": """Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    """,
}

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
        choices=["expand", "spell", "rewrite"],
        help="Query enhancement method",
    )

    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="method for reranking results",
        default=None,
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

            if args.enhance:
                res = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=f"""{PROMPTS[args.enhance]} User query: "{query}" """,
                )
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{res.text}'\n")
                query = res.text.strip() if res.text else query

            limit = args.limit * 5 if args.rerank_method else args.limit

            results = hybrid_search.rrf_search(query, args.k, limit)

            reranked_results_scores = []
            if args.rerank_method == "individual":
                for doc in results:
                    res = client.models.generate_content(
                        model="gemma-4-31b-it",
                        contents=f"""Rate how well this movie matches the search query.

                        Query: "{query}"
                        Movie: {doc["document"]["title"]} - {doc["document"]["description"]}

                        Consider:
                        - Direct relevance to query
                        - User intent (what they're looking for)
                        - Content appropriateness

                        Rate 0-10 (10 = perfect match).
                        Output ONLY the number in your response, no other text or explanation.

                        Score:""",
                    )
                    reranked_results_scores.append(
                        {"doc": doc, "score": float((res.text or "0").strip())}
                    )
                    time.sleep(3)

                sorted_reranked_results = sorted(
                    reranked_results_scores, key=lambda x: x["score"], reverse=True
                )[: args.limit]

                print(f"Re-ranking top {args.limit} results using individual method...")
                print(f'Reciprocal Rank Fusion Results for "{query}" (k={args.k}):')
                for idx, result in enumerate(sorted_reranked_results, 1):
                    print(f"{idx}. {result['doc']['document']['title']}")
                    print(f"Re-rank Score: {result['score']:.3f}/10")
                    print(f"RRF Score: {result['doc']['hybrid']:.3f}")
                    print(
                        f"BM25 Rank: {result['doc']['bm_25']}, Semantic Rank: {result['doc']['semantic']}"
                    )
                    print(f"{result['doc']['document']['description'][:100]}...")

            elif args.rerank_method == "batch":
                doc_list_str = json.dumps([doc["document"] for doc in results])

                res = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"

                    Movies:
                    {doc_list_str}

                    Return the movie IDs in order of relevance, best match first.

                    Your response must be a raw JSON array of integers.
                    Do not wrap the JSON in Markdown. Do not use a ```json code block.
                    Do not include any explanatory text.

                    For example:
                    [75, 12, 34, 2, 1]

                    Ranking:""",
                )
                scores = json.loads(res.text or "")
                doc_by_id = {doc["document"]["id"]: doc for doc in results}
                for idx, movie_id in enumerate(scores, 1):
                    reranked_results_scores.append(
                        {"doc": doc_by_id[movie_id], "score": idx}
                    )

                print(f"Re-ranking top {args.limit} results using batch method...")
                print(f'Reciprocal Rank Fusion Results for "{query}" (k={args.k}):')
                for idx, result in enumerate(reranked_results_scores[: args.limit], 1):
                    print(f"{idx}. {result['doc']['document']['title']}")
                    print(f"Re-rank Rank: {result['score']}")
                    print(f"RRF Score: {result['doc']['hybrid']:.3f}")
                    print(
                        f"BM25 Rank: {result['doc']['bm_25']}, Semantic Rank: {result['doc']['semantic']}"
                    )
                    print(f"{result['doc']['document']['description'][:100]}...")

            elif args.rerank_method == "cross_encoder":
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                pairs = []
                for result in results:
                    pairs.append(
                        [
                            query,
                            f"{result['document'].get('title', '')} - {result['document'].get('description', '')}",
                        ]
                    )
                scores = cross_encoder.predict(pairs)
                print(scores)
                sorted_results = sorted(
                    zip(scores, results), key=lambda x: x[0], reverse=True
                )

                for idx, (score, result) in enumerate(sorted_results[: args.limit], 1):
                    print(f"{idx}. {result['document']['title']}")
                    print(f"Cross Encoder Score: {score:.3f}")
                    print(f"RRF Score: {result['hybrid']:.3f}")
                    print(
                        f"BM25 Rank: {result['bm_25']}, Semantic Rank: {result['semantic']}"
                    )
                    print(f"{result['document']['description'][:100]}...")
            else:
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
