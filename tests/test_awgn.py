"""Checks for paired AWGN generation and seed construction."""

import numpy as np

from semantic_eeg.communication.awgn import add_awgn, channel_seed


def test_channel_seed_matches_publication_formula() -> None:
    assert channel_seed(500_000_000, 1, 0, 16, 0, 0) == 501_016_000
    assert channel_seed(500_000_000, 9, 1, 64, 7, 19) == 509_164_159


def test_awgn_is_deterministic_and_noise_free_is_identity() -> None:
    messages = np.ones((12, 16), dtype=np.float32)
    first = add_awgn(messages, -5.0, 2026)
    second = add_awgn(messages, -5.0, 2026)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, messages)
    assert np.array_equal(add_awgn(messages, None, 2026), messages)
