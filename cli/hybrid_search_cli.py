import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the input text"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores"
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            minmax_scores = []
            min_score = min(args.scores)
            max_score = max(args.scores)

            if args.scores:
                if min_score == max_score:
                    minmax_scores = [1.0] * len(args.scores)
                else:
                    for score in args.scores:
                        minmax_scores.append(
                            (score - min_score) / (max_score - min_score)
                        )

                for score in minmax_scores:
                    print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
