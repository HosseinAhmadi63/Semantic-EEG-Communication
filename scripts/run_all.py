"""Run the complete dataset and model-experiment sequence."""

from __future__ import annotations

import sys

from semantic_eeg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["all", *sys.argv[1:]]))
