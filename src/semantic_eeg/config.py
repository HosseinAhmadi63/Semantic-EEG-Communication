"""Validated loading of the frozen experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    """Return the repository root containing configuration and result directories."""

    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated view of the frozen manuscript experiment configuration."""

    raw: dict[str, Any]
    source_path: Path

    @property
    def subjects(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["dataset"]["subjects"])

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["communication"]["budgets"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["training"]["seeds"])

    @property
    def training_snrs(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["communication"]["training_snr_db"])

    @property
    def snr_specs(self) -> tuple[tuple[str, float | None], ...]:
        labels = self.raw["communication"]["evaluation_snr_labels"]
        values = self.raw["communication"]["evaluation_snr_db"]
        return tuple(
            (str(label), None if value is None else float(value))
            for label, value in zip(labels, values, strict=True)
        )

    @property
    def frequency_bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(tuple(float(item) for item in band) for band in self.raw["conventional"]["frequency_bands"])

    @property
    def output_root(self) -> Path:
        return repository_root() / self.raw["project"]["output_directory"]

    @property
    def data_root(self) -> Path:
        return repository_root() / self.raw["project"]["data_directory"]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load and validate the manuscript experiment configuration from YAML."""

    config_path = Path(path) if path is not None else repository_root() / "configs" / "paper.yaml"
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = ExperimentConfig(raw=raw, source_path=config_path)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Reject configurations that violate publication-defining protocol settings."""

    dataset = config.section("dataset")
    communication = config.section("communication")
    if dataset["name"] != "BNCI2014_001":
        raise ValueError("The publication configuration requires BNCI2014_001")
    if config.subjects != tuple(range(1, 10)):
        raise ValueError("The publication configuration requires Subjects 1 through 9")
    if config.budgets != (16, 32, 64):
        raise ValueError("The publication configuration requires K = 16, 32, and 64")
    if config.seeds != (2026, 2027, 2028):
        raise ValueError("The publication configuration requires seeds 2026, 2027, and 2028")
    if len(config.snr_specs) != 8 or config.snr_specs[-1] != ("noise_free", None):
        raise ValueError("The publication configuration requires seven noisy SNRs and one noise-free condition")
    if int(communication["channel_realizations"]) != 20:
        raise ValueError("The publication configuration requires 20 channel realizations")
    if float(dataset["sampling_frequency"]) != 160.0 or int(dataset["epoch_samples"]) != 560:
        raise ValueError("The publication configuration requires 160 Hz and 560 samples per trial")
    if float(dataset["epoch_start"]) != 0.5 or float(dataset["epoch_stop"]) != 4.0:
        raise ValueError("The publication configuration requires the 0.5-4.0 s epoch")
    if int(dataset["eeg_channels"]) != 22:
        raise ValueError("The publication configuration requires 22 EEG channels")
    if config.frequency_bands != (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 30.0),
    ):
        raise ValueError("The publication configuration requires the frozen five-band filter bank")
