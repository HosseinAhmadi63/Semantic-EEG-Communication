"""Command-line interface for experiments and publication reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from semantic_eeg.config import ExperimentConfig, load_config, repository_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-eeg",
        description="Run the frozen semantic EEG communication experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("cache", "Download and audit BNCI2014-001."),
        ("conventional", "Run conventional FBCSP-PCA transmission."),
        ("reconstruction", "Run reconstruction-oriented transmission."),
        ("semantic", "Run semantic residual transmission and the receiver-only control."),
        ("all", "Run all three experiment paths."),
    ):
        command = subparsers.add_parser(name, help=description)
        command.add_argument("--config", type=Path, default=repository_root() / "configs" / "paper.yaml")
        command.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        command.add_argument("--subjects", type=int, nargs="+")
        command.add_argument("--budgets", type=int, nargs="+")
        command.add_argument("--seeds", type=int, nargs="+")

    analysis = subparsers.add_parser(
        "analysis",
        help="Reproduce statistics and Figures 2-5 from immutable publication CSVs.",
    )
    analysis.add_argument(
        "--source",
        type=Path,
        default=repository_root() / "results" / "publication" / "source",
    )
    analysis.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results" / "publication" / "generated",
    )
    return parser


def _cache_dataset(config: ExperimentConfig, subjects: Sequence[int] | None, device: str) -> None:
    import torch

    from semantic_eeg.constants import METHOD_CONVENTIONAL
    from semantic_eeg.data.bnci2014_001 import load_filterbank_subject, load_wideband_subject
    from semantic_eeg.utils.run import RunContext

    publication = config.section("publication")
    context = RunContext.create(
        METHOD_CONVENTIONAL,
        config,
        str(publication["semantic_protocol_hash"]),
        str(publication["semantic_config_hash"]),
        torch.device(device),
    )
    selected = config.subjects if subjects is None else tuple(subjects)
    invalid = set(selected) - set(config.subjects)
    if invalid:
        raise ValueError(f"Unsupported subjects: {sorted(invalid)}")
    for subject in selected:
        filterbank, filterbank_labels, filterbank_metadata = load_filterbank_subject(
            subject, config, context
        )
        wideband, wideband_labels, wideband_metadata = load_wideband_subject(subject, config, context)
        if not (
            filterbank.shape[0]
            == wideband.shape[0]
            == len(filterbank_labels)
            == len(wideband_labels)
            == len(filterbank_metadata)
            == len(wideband_metadata)
        ):
            raise ValueError(f"The two preprocessing paths disagree for Subject {subject}")
        context.log("INFO", "subject_cached", f"Subject {subject}: {len(filterbank_labels)} trials")
    context.write_metadata("completed")


def _run_experiment_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    common = {
        "subjects": args.subjects,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "device": args.device,
    }
    if args.command == "cache":
        _cache_dataset(config, args.subjects, args.device)
        return

    from semantic_eeg.training.conventional import run_conventional_experiment
    from semantic_eeg.training.reconstruction import run_reconstruction_experiment
    from semantic_eeg.training.semantic import run_semantic_experiment

    if args.command in ("conventional", "all"):
        run_conventional_experiment(config, **common)
    if args.command in ("reconstruction", "all"):
        run_reconstruction_experiment(config, **common)
    if args.command in ("semantic", "all"):
        run_semantic_experiment(config, **common)


def main(arguments: Sequence[str] | None = None) -> int:
    """Dispatch a repository command and return a process status code."""

    parser = _parser()
    args = parser.parse_args(list(arguments) if arguments is not None else None)
    try:
        if args.command == "analysis":
            from semantic_eeg.evaluation.publication import reproduce_publication_analysis

            summary = reproduce_publication_analysis(args.source, args.output)
            print(json.dumps(summary, indent=2))
        else:
            _run_experiment_command(args)
        return 0
    except KeyboardInterrupt:
        print("Execution interrupted by the user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
