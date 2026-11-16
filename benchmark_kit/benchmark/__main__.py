"""Entry point for the copied-out scoring kit: ``python -m benchmark score …``."""

import sys

from .score import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "score":                 # accept the repository's spelling
        argv = argv[1:]
    sys.exit(main(argv))
