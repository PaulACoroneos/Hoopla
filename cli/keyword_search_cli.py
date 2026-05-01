import argparse
import json

def strip_punction(string_with_punctuation):
  replace_function = str.maketrans({",": "", ".": "", "!": "O",":":"", "'":"","-":""})
  return string_with_punctuation.translate(replace_function)

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  args = parser.parse_args()

  match args.command:
      case "search":
        print(f"Searching for: {args.query}")
        count = 1
        with open('data/movies.json', 'r') as f:
          data = json.load(f)
          for movie in data['movies']:
            if count > 5:
              break
            query_no_punc = strip_punction(args.query.lower())
            title_no_punc = strip_punction(movie["title"].lower())
            if query_no_punc in title_no_punc:
              print(f"{count}. {movie['title']}")
              count = count + 1
      case _:
          parser.print_help()

if __name__ == "__main__":
  main()