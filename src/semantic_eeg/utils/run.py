"""Run directories, metadata, and structured event logging."""

from __future__ import annotations

import json
import platform
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import moabb
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from semantic_eeg.config import ExperimentConfig
from semantic_eeg.utils.io import atomic_csv, upsert_csv


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunContext:
    """Own one experiment run's directories, metadata, checkpoints, and logs."""

    method: str
    config: ExperimentConfig
    protocol_hash: str
    config_hash: str
    root: Path
    csv_dir: Path
    checkpoint_dir: Path
    data_dir: Path
    run_id: str
    started_at: str
    device: torch.device

    @classmethod
    def create(
        cls,
        method: str,
        config: ExperimentConfig,
        protocol_hash: str,
        config_hash: str,
        device: torch.device,
    ) -> "RunContext":
        slug = method.lower().replace(" ", "_").replace("–", "-").replace("-", "_")
        run_id = f"{slug}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        root = config.output_root / slug
        context = cls(
            method=method,
            config=config,
            protocol_hash=protocol_hash,
            config_hash=config_hash,
            root=root,
            csv_dir=root / "csv",
            checkpoint_dir=root / "checkpoints",
            data_dir=config.data_root,
            run_id=run_id,
            started_at=utc_now(),
            device=device,
        )
        for directory in (context.root, context.csv_dir, context.checkpoint_dir, context.data_dir):
            directory.mkdir(parents=True, exist_ok=True)
        mne.set_config("MNE_DATA", str(context.data_dir), set_env=True)
        moabb.set_log_level("warning")
        context.write_metadata("running")
        return context

    @classmethod
    def open_existing(
        cls,
        method: str,
        config: ExperimentConfig,
        protocol_hash: str,
        config_hash: str,
        device: torch.device,
    ) -> "RunContext":
        """Open a method directory without changing its recorded run metadata."""

        slug = method.lower().replace(" ", "_").replace("–", "-").replace("-", "_")
        root = config.output_root / slug
        return cls(
            method=method,
            config=config,
            protocol_hash=protocol_hash,
            config_hash=config_hash,
            root=root,
            csv_dir=root / "csv",
            checkpoint_dir=root / "checkpoints",
            data_dir=config.data_root,
            run_id="dependency",
            started_at=utc_now(),
            device=device,
        )

    def log(
        self,
        level: str,
        event: str,
        message: Any,
        subject: int | None = None,
        direction: str | None = None,
        seed: int | None = None,
        budget: int | None = None,
    ) -> None:
        row = {
            "timestamp_utc": utc_now(),
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "config_hash": self.config_hash,
            "method": self.method,
            "level": level,
            "event": event,
            "subject": subject,
            "direction": direction,
            "seed": seed,
            "budget_k": budget,
            "message": str(message),
        }
        upsert_csv(
            self.csv_dir / "job_log.csv",
            [row],
            ["timestamp_utc", "run_id", "event", "subject", "direction", "seed", "budget_k"],
        )
        print(f"[{level}] {event}: {message}")

    def record_failure(
        self,
        stage: str,
        exception: BaseException,
        subject: int | None = None,
        direction: str | None = None,
        seed: int | None = None,
        budget: int | None = None,
    ) -> None:
        row = {
            "timestamp_utc": utc_now(),
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "config_hash": self.config_hash,
            "method": self.method,
            "stage": stage,
            "subject": subject,
            "direction": direction,
            "seed": seed,
            "budget_k": budget,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
        }
        upsert_csv(
            self.csv_dir / "failed_jobs.csv",
            [row],
            ["run_id", "stage", "subject", "direction", "seed", "budget_k"],
        )
        self.log("ERROR", stage, f"{type(exception).__name__}: {exception}", subject, direction, seed, budget)

    def write_metadata(self, status: str) -> None:
        row = {
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(),
            "method": self.method,
            "protocol_hash": self.protocol_hash,
            "config_hash": self.config_hash,
            "python": platform.python_version(),
            "device": str(self.device),
            "torch": torch.__version__,
            "moabb": moabb.__version__,
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "configuration": json.dumps(self.config.raw, sort_keys=True),
            "root": str(self.root),
        }
        atomic_csv(pd.DataFrame([row]), self.csv_dir / "run_metadata.csv")
