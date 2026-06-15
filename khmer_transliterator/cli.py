from __future__ import annotations

import argparse
import sys


def _version() -> str:
    try:
        from khmer_transliterator import __version__
        return __version__
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netra-transliterate",
        description="Transliterate romanized Khmer (English letters) into Khmer script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  netra-transliterate sokha
  netra-transliterate sokha srolanh
  netra-transliterate sokha -n 5
  netra-transliterate sokha --no-dict
  netra-transliterate --shell
  netra-transliterate --serve
  netra-transliterate --serve --port 8080
        """,
    )

    parser.add_argument(
        "words",
        nargs="*",
        metavar="WORD",
        help="One or more romanized Khmer words to transliterate.",
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=1,
        metavar="N",
        dest="top_n",
        help="Number of Khmer candidates to return per word (default: 1).",
    )
    parser.add_argument(
        "--no-dict",
        action="store_true",
        help="Skip dictionary validation and return raw model output.",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Start an interactive transliteration shell.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the Flask web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        metavar="PORT",
        help="Port for the web server (default: 5000, only used with --serve).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version()}",
    )
    return parser


def _run_batch(words: list[str], top_n: int, no_dict: bool) -> None:
    from khmer_transliterator import transliterate, transliterate_with_dict

    for word in words:
        if no_dict or top_n == 1:
            result = transliterate(word)
            print(f"{word}\t{result}")
        else:
            candidates = transliterate_with_dict(word, n=top_n)
            print(f"{word}:")
            for i, cand in enumerate(candidates, 1):
                print(f"  {i}. {cand}")


def _run_shell(top_n: int, no_dict: bool) -> None:
    from khmer_transliterator import transliterate, transliterate_with_dict

    print("=" * 52)
    print(" Khmer Transliterator — Interactive Shell ".center(52))
    print("=" * 52)
    print("Type a romanized Khmer word and press Enter.")
    print("Type 'exit' or press Ctrl-C to quit.\n")

    try:
        while True:
            try:
                word = input(">> ").strip()
            except EOFError:
                break
            if not word:
                continue
            if word.lower() in ("exit", "quit", "q"):
                break
            if no_dict or top_n == 1:
                result = transliterate(word)
                print(f"   \033[92m{result}\033[0m\n")
            else:
                candidates = transliterate_with_dict(word, n=top_n)
                for i, c in enumerate(candidates, 1):
                    print(f"   {i}. \033[92m{c}\033[0m")
                print()
    except KeyboardInterrupt:
        pass

    print("\nExiting.")


def _run_server(port: int) -> None:
    from khmer_transliterator.web import create_app

    app = create_app()
    print(f"Starting web server at http://localhost:{port}")
    print("Press Ctrl-C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    modes = [bool(args.words), args.shell, args.serve]
    if sum(modes) > 1:
        parser.error("Specify either WORD(s), --shell, or --serve — not multiple.")

    if args.serve:
        _run_server(args.port)
    elif args.shell:
        _run_shell(args.top_n, args.no_dict)
    elif args.words:
        _run_batch(args.words, args.top_n, args.no_dict)
    else:
        # No args: default to interactive shell
        _run_shell(args.top_n, args.no_dict)


if __name__ == "__main__":
    main()
