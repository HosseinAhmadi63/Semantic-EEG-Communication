"""Checks for the immutable result-source bundle."""

from semantic_eeg.config import repository_root
from semantic_eeg.evaluation.publication import _verify_manifest


def test_publication_sources_match_manifest() -> None:
    source = repository_root() / "results" / "publication" / "source"
    verified = _verify_manifest(source)
    assert len(verified) == 15
    assert all(record["verified"] for record in verified)
