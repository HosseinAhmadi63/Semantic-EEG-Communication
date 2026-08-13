"""Download, preprocess, audit, and cache BNCI2014-001."""

from __future__ import annotations

import sys

from semantic_eeg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["cache", *sys.argv[1:]]))
