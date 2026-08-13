"""Reproduce the final manuscript analysis from immutable publication CSV files."""

from __future__ import annotations

import json
import gzip
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_eeg.config import repository_root
from semantic_eeg.evaluation.statistics import comparison_family, group_summary
from semantic_eeg.plotting.figures import generate_publication_figures
from semantic_eeg.plotting.style import (
    METHOD_FBCSP,
    METHOD_RECEIVER_ONLY,
    METHOD_RECONSTRUCTION,
    METHOD_SEMANTIC,
)
from semantic_eeg.utils.io import atomic_csv, sha256_file

SNR_LABELS = ("-10_dB", "-5_dB", "0_dB", "5_dB", "10_dB", "15_dB", "20_dB", "noise_free")
BUDGETS = (16, 32, 64)
SUBJECTS = tuple(range(1, 10))


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _boolean_values(series: pd.Series) -> pd.Series:
    """Interpret serialized Boolean values without treating nonempty strings as true."""

    accepted = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.map(lambda value: str(value).strip().lower())
    unknown = sorted(set(normalized) - set(accepted))
    if unknown:
        raise ValueError(f"Unexpected Boolean values: {unknown}")
    return normalized.map(accepted).astype(bool)


def _subject_rows(path: Path, method: str) -> list[dict[str, Any]]:
    frame = _read(path)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "method": method,
                "subject": int(record["subject"]),
                "budget_k": int(record["budget_k"]),
                "snr_order": int(record["snr_order"]),
                "snr_label": str(record["snr_label"]),
                "snr_db": float(record["snr_db"]) if pd.notna(record["snr_db"]) else float("nan"),
                "balanced_accuracy": float(record["balanced_accuracy_mean"]),
                "n_observations": int(record["n_observations"]),
                "n_directions": int(record["n_directions"]),
                "n_model_seeds": int(record["n_model_seeds"]),
                "n_channel_realizations": int(record["n_channel_realizations"]),
            }
        )
    return rows


def _verify_manifest(source: Path) -> list[dict[str, Any]]:
    manifest_path = source / "source_manifest.csv"
    manifest = _read(manifest_path)
    verified: list[dict[str, Any]] = []
    for record in manifest.to_dict("records"):
        path = source / str(record["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != str(record["sha256"]):
            raise ValueError(f"Checksum mismatch for {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Size mismatch for {path}")
        if path.suffix == ".gz" and str(record.get("uncompressed_sha256", "")):
            uncompressed_digest = hashlib.sha256()
            with gzip.open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    uncompressed_digest.update(block)
            if uncompressed_digest.hexdigest() != str(record["uncompressed_sha256"]):
                raise ValueError(f"Uncompressed checksum mismatch for {path}")
        verified.append({**record, "verified": True})
    if not verified:
        raise ValueError("The publication source manifest is empty")
    return verified


def _detail_index(path: Path) -> dict[tuple[str, ...], tuple[str, str]]:
    frame = _read(path)
    fields = ("subject", "direction", "seed", "budget_k", "snr_order", "channel_realization")
    index: dict[tuple[str, ...], tuple[str, str]] = {}
    for record in frame.astype({field: str for field in fields}).to_dict("records"):
        key = tuple(record[field] for field in fields)
        if key in index:
            raise ValueError(f"Duplicate detailed result key in {path}: {key}")
        index[key] = (str(record["channel_seed"]), str(record["n_test_trials"]))
    return index


def _normalized_records(path: Path) -> list[tuple[tuple[str, str], ...]]:
    ignored = {"run_id", "protocol_hash", "config_hash", "method", "source_file"}
    records = []
    for record in _read(path).fillna("").astype(str).to_dict("records"):
        records.append(tuple(sorted((key, value) for key, value in record.items() if key not in ignored)))
    return sorted(records)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path in committed analysis metadata."""

    try:
        return str(path.relative_to(repository_root()))
    except ValueError:
        return str(path)


def reproduce_publication_analysis(
    source_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate frozen results and regenerate all data-driven manuscript outputs."""

    source = Path(source_directory).resolve()
    output = Path(output_directory).resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    provenance = _verify_manifest(source)

    paths = {
        "semantic_summary": source / "semantic" / "semantic_results_subject_summary.csv.gz",
        "fbcsp_summary": source / "semantic" / "baseline_results_subject_summary.csv.gz",
        "receiver_summary": source / "semantic" / "receiver_only_results_subject_summary.csv.gz",
        "reconstruction_summary": source / "reconstruction" / "results_subject_summary.csv.gz",
        "semantic_detail": source / "semantic" / "results_detailed.csv.gz",
        "fbcsp_detail": source / "semantic" / "baseline_results_detailed.csv.gz",
        "receiver_detail": source / "semantic" / "receiver_only_results_detailed.csv.gz",
        "reconstruction_detail": source / "reconstruction" / "results_detailed.csv.gz",
        "semantic_artifacts": source / "semantic" / "artifact_rejection_summary.csv.gz",
        "reconstruction_artifacts": source / "reconstruction" / "artifact_rejection_summary.csv.gz",
        "semantic_splits": source / "semantic" / "split_counts.csv.gz",
        "reconstruction_splits": source / "reconstruction" / "split_counts.csv.gz",
        "integrity": source / "semantic" / "full_experiment_integrity_audit.csv.gz",
        "status": source / "semantic" / "full_experiment_status.csv.gz",
        "job_log": source / "semantic" / "job_log.csv.gz",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    subject_rows = (
        _subject_rows(paths["semantic_summary"], METHOD_SEMANTIC)
        + _subject_rows(paths["fbcsp_summary"], METHOD_FBCSP)
        + _subject_rows(paths["receiver_summary"], METHOD_RECEIVER_ONLY)
        + _subject_rows(paths["reconstruction_summary"], METHOD_RECONSTRUCTION)
    )
    expected_counts = {
        METHOD_SEMANTIC: 216,
        METHOD_FBCSP: 216,
        METHOD_RECONSTRUCTION: 216,
        METHOD_RECEIVER_ONLY: 72,
    }
    actual_counts = Counter(row["method"] for row in subject_rows)
    if dict(actual_counts) != expected_counts:
        raise ValueError(f"Unexpected subject-summary counts: {dict(actual_counts)}")

    lookup = {
        (row["method"], row["subject"], row["budget_k"], row["snr_order"]): row[
            "balanced_accuracy"
        ]
        for row in subject_rows
    }
    group_rows = group_summary(subject_rows)
    primary = comparison_family(
        lookup,
        "semantic_vs_fbcsp",
        METHOD_SEMANTIC,
        METHOD_FBCSP,
        BUDGETS,
        SUBJECTS,
        SNR_LABELS,
    )
    semantic_vs_reconstruction = comparison_family(
        lookup,
        "semantic_vs_reconstruction",
        METHOD_SEMANTIC,
        METHOD_RECONSTRUCTION,
        BUDGETS,
        SUBJECTS,
        SNR_LABELS,
    )
    fbcsp_vs_reconstruction = comparison_family(
        lookup,
        "fbcsp_vs_reconstruction",
        METHOD_FBCSP,
        METHOD_RECONSTRUCTION,
        BUDGETS,
        SUBJECTS,
        SNR_LABELS,
    )
    semantic_vs_receiver = comparison_family(
        lookup,
        "semantic_vs_receiver_only",
        METHOD_SEMANTIC,
        METHOD_RECEIVER_ONLY,
        (32,),
        SUBJECTS,
        SNR_LABELS,
    )
    receiver_vs_fbcsp = comparison_family(
        lookup,
        "receiver_only_vs_fbcsp",
        METHOD_RECEIVER_ONLY,
        METHOD_FBCSP,
        (32,),
        SUBJECTS,
        SNR_LABELS,
    )

    semantic_detail = _detail_index(paths["semantic_detail"])
    fbcsp_detail = _detail_index(paths["fbcsp_detail"])
    reconstruction_detail = _detail_index(paths["reconstruction_detail"])
    receiver_detail = _detail_index(paths["receiver_detail"])
    main_keys_match = set(semantic_detail) == set(fbcsp_detail) == set(reconstruction_detail)
    main_pairing_matches = main_keys_match and all(
        semantic_detail[key] == fbcsp_detail[key] == reconstruction_detail[key]
        for key in semantic_detail
    )
    receiver_keys_match = set(receiver_detail) == {
        key for key in fbcsp_detail if int(key[3]) == 32
    }
    receiver_pairing_matches = receiver_keys_match and all(
        receiver_detail[key] == fbcsp_detail[key] for key in receiver_detail
    )

    status = _read(paths["status"])
    integrity = _read(paths["integrity"])
    log = _read(paths["job_log"])
    checks = [
        ("Frozen semantic run completed", str(status.iloc[0]["status"]) == "COMPLETE"),
        ("Frozen semantic integrity checks passed", _boolean_values(integrity["check_pass"]).all()),
        (
            "Frozen semantic log contains no warnings or errors",
            not log["level"].astype(str).str.upper().isin({"WARNING", "ERROR", "CRITICAL"}).any(),
        ),
        ("Subject-summary row counts are complete", dict(actual_counts) == expected_counts),
        (
            "Artifact rejection summaries match",
            _normalized_records(paths["semantic_artifacts"])
            == _normalized_records(paths["reconstruction_artifacts"]),
        ),
        (
            "Cross-session split counts match",
            _normalized_records(paths["semantic_splits"])
            == _normalized_records(paths["reconstruction_splits"]),
        ),
        ("Primary detailed keys match", main_keys_match),
        ("Primary channel seeds and test sizes are paired", main_pairing_matches),
        ("Receiver-only keys match the K=32 baseline", receiver_keys_match),
        ("Receiver-only channel seeds and test sizes are paired", receiver_pairing_matches),
        ("All group summaries contain nine subjects", all(row["n_subjects"] == 9 for row in group_rows)),
        ("Primary family contains 24 comparisons", len(primary) == 24),
    ]
    audit_rows = [
        {
            "check_name": name,
            "check_pass": bool(passed),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for name, passed in checks
    ]
    if not all(row["check_pass"] for row in audit_rows):
        raise RuntimeError(f"Publication analysis integrity failure: {audit_rows}")

    configuration = [
        {"parameter": "Dataset", "value": "BNCI2014-001 (BCI Competition IV Dataset 2a)"},
        {"parameter": "Subjects", "value": "9"},
        {"parameter": "Sessions", "value": "2 per subject; both source-target directions"},
        {"parameter": "Classes", "value": "4 motor imagery classes"},
        {"parameter": "Retained trials", "value": "4,696 of 5,184"},
        {"parameter": "EEG preparation", "value": "8-30 Hz; 160 Hz; 0.5-4.0 s; 22 x 560"},
        {"parameter": "Transmission budgets", "value": "K = 16, 32, and 64 real values"},
        {"parameter": "Test conditions", "value": "-10 to 20 dB in 5 dB steps and noise-free"},
        {"parameter": "Repetition seeds", "value": "2026, 2027, and 2028"},
        {"parameter": "Channel realizations", "value": "20 paired realizations per test condition"},
        {"parameter": "Classification outcome", "value": "Balanced accuracy"},
        {"parameter": "Independent statistical unit", "value": "Subject"},
    ]
    table_payloads = {
        "table_1_experimental_configuration.csv": configuration,
        "table_2_group_balanced_accuracy.csv": group_rows,
        "table_3_semantic_vs_fbcsp.csv": primary,
        "table_4_receiver_only_control.csv": semantic_vs_receiver + receiver_vs_fbcsp,
        "table_5_semantic_vs_reconstruction.csv": semantic_vs_reconstruction,
        "supplement_fbcsp_vs_reconstruction.csv": fbcsp_vs_reconstruction,
        "supplement_subject_level_results.csv": subject_rows,
        "supplement_all_pairwise_statistics.csv": (
            primary
            + semantic_vs_reconstruction
            + fbcsp_vs_reconstruction
            + semantic_vs_receiver
            + receiver_vs_fbcsp
        ),
        "analysis_integrity_audit.csv": audit_rows,
        "source_provenance.csv": provenance,
        "figure_data_group_summary.csv": group_rows,
        "figure_data_semantic_gain.csv": primary,
    }
    for filename, rows in table_payloads.items():
        atomic_csv(pd.DataFrame(rows), tables / filename)

    heatmap_rows = generate_publication_figures(group_rows, primary, lookup, figures)
    atomic_csv(pd.DataFrame(heatmap_rows), tables / "figure_data_subject_heatmap.csv")
    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": _portable_path(source),
        "output_directory": _portable_path(output),
        "source_files_verified": len(provenance),
        "integrity_checks_passed": len(audit_rows),
        "primary_comparisons": len(primary),
        "data_driven_figures": 4,
    }
    _write_json(output / "analysis_summary.json", summary)
    return summary
