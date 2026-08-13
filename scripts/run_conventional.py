"""Run conventional FBCSP--PCA transmission experiments."""

from __future__ import annotations

import sys

from semantic_eeg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["conventional", *sys.argv[1:]]))
