"""Reproduce publication tables, statistics, and figures from frozen CSV records."""

from __future__ import annotations

import sys

from semantic_eeg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["analysis", *sys.argv[1:]]))
