"""Run semantic residual-transmission and receiver-only experiments."""

from __future__ import annotations

import sys

from semantic_eeg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["semantic", *sys.argv[1:]]))
