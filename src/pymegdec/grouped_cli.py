"""Grouped command-line dispatcher for PyMEGDec workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from pymegdec import alpha_cli
from pymegdec import cli as legacy_cli
from pymegdec import neureptrace_compat
from pymegdec import stimulus_cli, stimulus_cross_person, stimulus_hyperalignment, stimulus_mcca
from pymegdec.data_download import download_meg_data_files
from pymegdec.synthetic_data_cli import make_synthetic_data

CommandHandler = Callable[[Sequence[str] | None, str | None], int]


def _dispatch_group(group: str, description: str, handlers: dict[str, CommandHandler], argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog=f"pymegdec {group}", description=description)
        parser.add_argument("subcommand", nargs="?", choices=sorted(handlers), help="Subcommand to run.")
        parser.print_help()
        return 0

    subcommand, *remaining = argv
    if subcommand not in handlers:
        parser = argparse.ArgumentParser(prog=f"pymegdec {group}", description=description)
        parser.error(f"Unsupported {group} subcommand: {subcommand}")
    return handlers[subcommand](remaining, f"pymegdec {group} {subcommand}")


def _stimulus_handlers() -> dict[str, CommandHandler]:
    return {
        name: neureptrace_compat.deprecated_handler(handler, f"pymegdec stimulus {name}")
        for name, handler in {
            "cross-subject-cue-calibrated": stimulus_cli.stimulus_cross_subject_cue_calibrated,
            "cross-subject-hyperalignment": stimulus_hyperalignment.stimulus_cross_subject_hyperalignment,
            "cross-subject-mcca": stimulus_mcca.stimulus_cross_subject_mcca,
            "cross-subject-nested": stimulus_cli.stimulus_cross_subject_nested,
            "cross-person-decode": stimulus_cross_person.stimulus_cross_person_decode,
            "cross-subject-smoke": stimulus_cli.stimulus_cross_subject_smoke,
            "decoding": legacy_cli.stimulus_decoding,
            "predictions": stimulus_cli.stimulus_predictions,
            "robustness": stimulus_cli.stimulus_robustness,
            "temporal-generalization": stimulus_cli.stimulus_temporal_generalization,
            "onset-scan": stimulus_cli.stimulus_onset_scan,
        }.items()
    }


def _alpha_handlers() -> dict[str, CommandHandler]:
    return {
        name: neureptrace_compat.deprecated_handler(handler, f"pymegdec alpha {name}")
        for name, handler in {
            "metrics": alpha_cli.alpha_metrics,
            "movement": alpha_cli.alpha_movement,
            "movement-results": legacy_cli.alpha_movement_results,
            "reaction-time": alpha_cli.alpha_reaction_time,
            "rt": alpha_cli.alpha_reaction_time,
        }.items()
    }


def _data_handlers() -> dict[str, CommandHandler]:
    return {"download": download_meg_data_files}


def _top_level_handlers() -> dict[str, CommandHandler]:
    handlers: dict[str, CommandHandler] = {
        "migration": neureptrace_compat.migration_status,
        "migration-status": neureptrace_compat.migration_status,
        "neureptrace": neureptrace_compat.neureptrace_passthrough,
        "cross-validate": neureptrace_compat.deprecated_handler(legacy_cli.cross_validate, "pymegdec cross-validate"),
        "transfer": neureptrace_compat.deprecated_handler(legacy_cli.transfer, "pymegdec transfer"),
        "make-synthetic-data": make_synthetic_data,
        # Backward-compatible top-level aliases. Prefer grouped forms in new docs.
        "stimulus-decoding": neureptrace_compat.deprecated_handler(legacy_cli.stimulus_decoding, "pymegdec stimulus-decoding"),
        "stimulus-cross-subject-cue-calibrated": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_cross_subject_cue_calibrated, "pymegdec stimulus-cross-subject-cue-calibrated"),
        "stimulus-cross-subject-hyperalignment": neureptrace_compat.deprecated_handler(stimulus_hyperalignment.stimulus_cross_subject_hyperalignment, "pymegdec stimulus-cross-subject-hyperalignment"),
        "stimulus-cross-subject-mcca": neureptrace_compat.deprecated_handler(stimulus_mcca.stimulus_cross_subject_mcca, "pymegdec stimulus-cross-subject-mcca"),
        "stimulus-cross-subject-nested": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_cross_subject_nested, "pymegdec stimulus-cross-subject-nested"),
        "stimulus-cross-person-decode": neureptrace_compat.deprecated_handler(stimulus_cross_person.stimulus_cross_person_decode, "pymegdec stimulus-cross-person-decode"),
        "stimulus-cross-subject-smoke": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_cross_subject_smoke, "pymegdec stimulus-cross-subject-smoke"),
        "stimulus-predictions": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_predictions, "pymegdec stimulus-predictions"),
        "stimulus-robustness": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_robustness, "pymegdec stimulus-robustness"),
        "stimulus-temporal-generalization": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_temporal_generalization, "pymegdec stimulus-temporal-generalization"),
        "stimulus-onset-scan": neureptrace_compat.deprecated_handler(stimulus_cli.stimulus_onset_scan, "pymegdec stimulus-onset-scan"),
        "alpha-metrics": neureptrace_compat.deprecated_handler(alpha_cli.alpha_metrics, "pymegdec alpha-metrics"),
        "alpha-movement": neureptrace_compat.deprecated_handler(alpha_cli.alpha_movement, "pymegdec alpha-movement"),
        "alpha-movement-results": neureptrace_compat.deprecated_handler(legacy_cli.alpha_movement_results, "pymegdec alpha-movement-results"),
        "alpha-reaction-time": neureptrace_compat.deprecated_handler(alpha_cli.alpha_reaction_time, "pymegdec alpha-reaction-time"),
        "alpha-rt": neureptrace_compat.deprecated_handler(alpha_cli.alpha_reaction_time, "pymegdec alpha-rt"),
        "download-meg-data": download_meg_data_files,
    }
    handlers.update(neureptrace_compat.neureptrace_top_level_handlers())
    return handlers


def _print_main_help() -> None:
    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument("command", nargs="?", help="Command or command group to run.")
    parser.print_help()
    print(
        "\nCommand groups:\n"
        "  pymegdec stimulus <cross-subject-cue-calibrated|cross-subject-hyperalignment|cross-subject-mcca|cross-subject-nested|cross-subject-smoke|"
        "decoding|predictions|robustness|temporal-generalization|onset-scan>\n"
        "  pymegdec alpha <metrics|movement|movement-results|reaction-time|rt>\n"
        "  pymegdec data <download>\n"
        "\nCore commands:\n"
        "  pymegdec cross-validate ...\n"
        "  pymegdec transfer ...\n"
        "  pymegdec make-synthetic-data ...\n"
        "\nDeprecation:\n"
        "  PyMEGDec is now a compatibility CLI. Prefer NeuRepTrace dataset YAML workflows for new analyses.\n"
        "\nNeuRepTrace migration helpers:\n"
        "  pymegdec migration\n"
        "  pymegdec neureptrace <command> ...\n"
        "  pymegdec mne-time-decode ...\n"
        "  Compatibility aliases such as pymegdec stimulus-decoding and pymegdec alpha-metrics remain available, "
        "but now print migration guidance."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_main_help()
        return 0

    command, *remaining = argv
    if command == "stimulus":
        return _dispatch_group("stimulus", "Stimulus decoding and diagnostics.", _stimulus_handlers(), remaining)
    if command == "alpha":
        return _dispatch_group("alpha", "Alpha metric, movement, and reaction-time analyses.", _alpha_handlers(), remaining)
    if command == "data":
        return _dispatch_group("data", "Data download and data-management helpers.", _data_handlers(), remaining)

    handlers = _top_level_handlers()
    if command in handlers:
        return handlers[command](remaining, f"pymegdec {command}")

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
