"""Checks for the frozen paper configuration."""

from semantic_eeg.config import load_config


def test_paper_configuration_is_frozen() -> None:
    config = load_config()
    assert config.subjects == tuple(range(1, 10))
    assert config.budgets == (16, 32, 64)
    assert config.seeds == (2026, 2027, 2028)
    assert config.training_snrs == (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    assert config.snr_specs[-1] == ("noise_free", None)
