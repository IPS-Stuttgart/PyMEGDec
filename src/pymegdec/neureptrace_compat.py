"""Compatibility shims for migrating PyMEGDec commands to NeuRepTrace."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module

CommandHandler = Callable[[Sequence[str] | None, str | None], int]


@dataclass(frozen=True)
class MigrationEntry:
    """Human-readable migration status for one PyMEGDec command."""

    replacement: str
    status: str
    notes: str = ""


SUPPRESS_ENV = "PYMEGDEC_SUPPRESS_MIGRATION_WARNINGS"
_EMITTED_COMMANDS: set[str] = set()

NEUREPTRACE_DIRECT_COMMANDS = (
    "benchmark",
    "continuous-stimulus-scan",
    "ensemble-observations",
    "event-detect",
    "event-detection",
    "metadata",
    "mne-time-decode",
    "observation-ensemble",
    "observation-schema",
    "onset-detect",
    "onset-detection",
    "plot-time-decode",
    "results",
    "stimulus-detect",
    "stimulus-detection",
    "temporal-model",
    "temporal-smoothing",
    "temporal-state-workflow",
    "validate-manifest",
    "validate-observations",
)

PYMEGDEC_COMMAND_MIGRATIONS: dict[str, MigrationEntry] = {
    "pymegdec cross-validate": MigrationEntry(
        replacement="neureptrace mne-time-decode --input-format fieldtrip-mat ...",
        status="partial",
        notes="Same-dataset decoding can move once FieldTrip MAT loading is enabled in NeuRepTrace.",
    ),
    "pymegdec transfer": MigrationEntry(
        replacement="neureptrace transfer-decode ...",
        status="pending",
        notes="Requires the two-recording train/test transfer runner proposed for NeuRepTrace.",
    ),
    "pymegdec stimulus decoding": MigrationEntry(
        replacement="neureptrace dataset run <dataset.yml> --analysis stimulus_main_to_cue",
        status="pending",
        notes="Requires dataset YAML plus transfer-decode support; keep this command until parity tests pass.",
    ),
    "pymegdec stimulus-decoding": MigrationEntry(
        replacement="neureptrace dataset run <dataset.yml> --analysis stimulus_main_to_cue",
        status="pending",
        notes="Top-level alias of pymegdec stimulus decoding.",
    ),
    "pymegdec stimulus predictions": MigrationEntry(
        replacement="neureptrace mne-time-decode --observations-out ...",
        status="partial",
        notes="NeuRepTrace probability observations should become the canonical prediction artifact.",
    ),
    "pymegdec stimulus-predictions": MigrationEntry(
        replacement="neureptrace mne-time-decode --observations-out ...",
        status="partial",
        notes="Top-level alias of pymegdec stimulus predictions.",
    ),
    "pymegdec stimulus temporal-generalization": MigrationEntry(
        replacement="neureptrace mne-time-decode --temporal-train-window START STOP ...",
        status="available",
        notes="Use NeuRepTrace's train-window ensemble or temporal-state workflow depending on the analysis.",
    ),
    "pymegdec stimulus-temporal-generalization": MigrationEntry(
        replacement="neureptrace mne-time-decode --temporal-train-window START STOP ...",
        status="available",
        notes="Top-level alias of pymegdec stimulus temporal-generalization.",
    ),
    "pymegdec stimulus onset-scan": MigrationEntry(
        replacement="neureptrace continuous-stimulus-scan / onset-detection / stimulus-detection ...",
        status="partial",
        notes="Use NeuRepTrace observation tables as input to onset/stimulus detection.",
    ),
    "pymegdec stimulus-onset-scan": MigrationEntry(
        replacement="neureptrace continuous-stimulus-scan / onset-detection / stimulus-detection ...",
        status="partial",
        notes="Top-level alias of pymegdec stimulus onset-scan.",
    ),
    "pymegdec stimulus robustness": MigrationEntry(
        replacement="neureptrace benchmark ...",
        status="partial",
        notes="Port robustness grids to NeuRepTrace manifests or dataset YAML analyses.",
    ),
    "pymegdec stimulus-robustness": MigrationEntry(
        replacement="neureptrace benchmark ...",
        status="partial",
        notes="Top-level alias of pymegdec stimulus robustness.",
    ),
    "pymegdec stimulus cross-subject-cue-calibrated": MigrationEntry(
        replacement="neureptrace mne-time-decode --group-column subject --emission-mode calibrated ...",
        status="partial",
        notes="Cross-subject calibration should be represented in NeuRepTrace manifests after FieldTrip staging.",
    ),
    "pymegdec stimulus-cross-subject-cue-calibrated": MigrationEntry(
        replacement="neureptrace mne-time-decode --group-column subject --emission-mode calibrated ...",
        status="partial",
        notes="Top-level alias of pymegdec stimulus cross-subject-cue-calibrated.",
    ),
    "pymegdec stimulus cross-subject-hyperalignment": MigrationEntry(
        replacement="neureptrace benchmark ...",
        status="pending",
        notes="Needs a NeuRepTrace alignment/preprocessing plugin before this can be retired.",
    ),
    "pymegdec stimulus-cross-subject-hyperalignment": MigrationEntry(
        replacement="neureptrace benchmark ...",
        status="pending",
        notes="Top-level alias of pymegdec stimulus cross-subject-hyperalignment.",
    ),
    "pymegdec stimulus cross-subject-mcca": MigrationEntry(
        replacement="neureptrace benchmark ...",
        status="pending",
        notes="Needs a NeuRepTrace alignment/preprocessing plugin before this can be retired.",
    ),
    "pymegdec stimulus-cross-subject-mcca": MigrationEntry(
        replacement="neureptrace benchmark ...",
        status="pending",
        notes="Top-level alias of pymegdec stimulus cross-subject-mcca.",
    ),
    "pymegdec stimulus cross-subject-nested": MigrationEntry(
        replacement="neureptrace benchmark --tune-hyperparameters ...",
        status="partial",
        notes="Represent the candidate grid in a NeuRepTrace benchmark manifest.",
    ),
    "pymegdec stimulus-cross-subject-nested": MigrationEntry(
        replacement="neureptrace benchmark --tune-hyperparameters ...",
        status="partial",
        notes="Top-level alias of pymegdec stimulus cross-subject-nested.",
    ),
    "pymegdec stimulus cross-subject-smoke": MigrationEntry(
        replacement="neureptrace validate-manifest && neureptrace benchmark ...",
        status="partial",
        notes="Use NeuRepTrace manifest validation plus a small benchmark subset.",
    ),
    "pymegdec stimulus-cross-subject-smoke": MigrationEntry(
        replacement="neureptrace validate-manifest && neureptrace benchmark ...",
        status="partial",
        notes="Top-level alias of pymegdec stimulus cross-subject-smoke.",
    ),
    "pymegdec alpha metrics": MigrationEntry(
        replacement="examples/bush_meg/alpha_metrics.py or future neureptrace meg alpha-metrics",
        status="project-specific",
        notes="Keep outside core NeuRepTrace unless the alpha workflow is generalized.",
    ),
    "pymegdec alpha-metrics": MigrationEntry(
        replacement="examples/bush_meg/alpha_metrics.py or future neureptrace meg alpha-metrics",
        status="project-specific",
        notes="Top-level alias of pymegdec alpha metrics.",
    ),
    "pymegdec alpha movement": MigrationEntry(
        replacement="examples/bush_meg/alpha_movement.py or future neureptrace meg alpha-movement",
        status="project-specific",
        notes="Depends on CTF/FieldTrip geometry and should remain Bush-specific until generalized.",
    ),
    "pymegdec alpha-movement": MigrationEntry(
        replacement="examples/bush_meg/alpha_movement.py or future neureptrace meg alpha-movement",
        status="project-specific",
        notes="Top-level alias of pymegdec alpha movement.",
    ),
    "pymegdec alpha movement-results": MigrationEntry(
        replacement="examples/bush_meg/alpha_movement_results.py",
        status="project-specific",
        notes="Paper-facing aggregation; keep as example/script unless generalized.",
    ),
    "pymegdec alpha-movement-results": MigrationEntry(
        replacement="examples/bush_meg/alpha_movement_results.py",
        status="project-specific",
        notes="Top-level alias of pymegdec alpha movement-results.",
    ),
    "pymegdec alpha reaction-time": MigrationEntry(
        replacement="examples/bush_meg/alpha_reaction_time.py",
        status="project-specific",
        notes="Reaction-time joins are dataset/paper-specific.",
    ),
    "pymegdec alpha-reaction-time": MigrationEntry(
        replacement="examples/bush_meg/alpha_reaction_time.py",
        status="project-specific",
        notes="Top-level alias of pymegdec alpha reaction-time.",
    ),
    "pymegdec alpha rt": MigrationEntry(
        replacement="examples/bush_meg/alpha_reaction_time.py",
        status="project-specific",
        notes="Alias of pymegdec alpha reaction-time.",
    ),
    "pymegdec alpha-rt": MigrationEntry(
        replacement="examples/bush_meg/alpha_reaction_time.py",
        status="project-specific",
        notes="Top-level alias of pymegdec alpha rt.",
    ),
}


def _migration_entry(command: str) -> MigrationEntry:
    return PYMEGDEC_COMMAND_MIGRATIONS.get(
        command,
        MigrationEntry(
            replacement="neureptrace <command> ... or examples/bush_meg/...",
            status="unknown",
            notes="No explicit migration mapping exists yet.",
        ),
    )


def migration_message(command: str) -> str:
    """Return one concise migration message for a legacy PyMEGDec command."""

    entry = _migration_entry(command)
    details = f"{command} is a PyMEGDec compatibility command [{entry.status}]. Prefer: {entry.replacement}."
    if entry.notes:
        details = f"{details} {entry.notes}"
    return details


def emit_migration_warning(command: str) -> None:
    """Emit a visible CLI warning and a Python deprecation warning."""

    if os.environ.get(SUPPRESS_ENV) or command in _EMITTED_COMMANDS:
        return
    _EMITTED_COMMANDS.add(command)
    message = migration_message(command)
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    print(f"PyMEGDec migration warning: {message}", file=sys.stderr)


def deprecated_handler(handler: CommandHandler, command: str) -> CommandHandler:
    """Wrap a grouped PyMEGDec command with migration guidance."""

    def _wrapped(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
        emit_migration_warning(command)
        return int(handler(argv, prog))

    return _wrapped


def run_neureptrace(argv: Sequence[str] | None = None) -> int:
    """Run the installed NeuRepTrace grouped CLI from a PyMEGDec shim."""

    if argv is None:
        argv = sys.argv[1:]
    neureptrace_cli = import_module("neureptrace.cli")
    result = neureptrace_cli.main(list(argv))
    return int(result) if isinstance(result, int) else 0


def neureptrace_passthrough(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Handle ``pymegdec neureptrace <command> ...``."""

    del prog
    return run_neureptrace(argv)


def neureptrace_command_handler(command: str) -> CommandHandler:
    """Create a PyMEGDec grouped alias for one NeuRepTrace command."""

    def _handler(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
        del prog
        return run_neureptrace([command, *(argv or ())])

    return _handler


def neureptrace_top_level_handlers() -> dict[str, CommandHandler]:
    """Return top-level PyMEGDec aliases that forward to NeuRepTrace."""

    return {command: neureptrace_command_handler(command) for command in NEUREPTRACE_DIRECT_COMMANDS}


def _migration_rows() -> list[tuple[str, str, str, str]]:
    return [
        (command, entry.status, entry.replacement, entry.notes)
        for command, entry in sorted(PYMEGDEC_COMMAND_MIGRATIONS.items())
    ]


def _print_text_table(rows: list[tuple[str, str, str, str]]) -> None:
    headers = ("PyMEGDec command", "status", "replacement", "notes")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def _print_markdown_table(rows: list[tuple[str, str, str, str]]) -> None:
    print("| PyMEGDec command | status | replacement | notes |")
    print("|---|---|---|---|")
    for command, status, replacement, notes in rows:
        print(f"| `{command}` | {status} | `{replacement}` | {notes} |")


def migration_status(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Print the PyMEGDec-to-NeuRepTrace command migration table."""

    parser = argparse.ArgumentParser(prog=prog, description="Show PyMEGDec command migration status.")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table.")
    parser.add_argument(
        "--status",
        choices=sorted({entry.status for entry in PYMEGDEC_COMMAND_MIGRATIONS.values()}),
        help="Show only commands with this migration status.",
    )
    args = parser.parse_args(argv)
    rows = _migration_rows()
    if args.status:
        rows = [row for row in rows if row[1] == args.status]
    if args.markdown:
        _print_markdown_table(rows)
    else:
        _print_text_table(rows)
    return 0


def migration_main() -> None:
    raise SystemExit(migration_status(sys.argv[1:], prog="pymegdec-migration"))


def neureptrace_main() -> None:
    raise SystemExit(run_neureptrace(sys.argv[1:]))


def _main_for_neureptrace_command(command: str) -> None:
    raise SystemExit(run_neureptrace([command, *sys.argv[1:]]))


def mne_time_decode_main() -> None:
    _main_for_neureptrace_command("mne-time-decode")


def plot_time_decode_main() -> None:
    _main_for_neureptrace_command("plot-time-decode")


def validate_manifest_main() -> None:
    _main_for_neureptrace_command("validate-manifest")


def validate_observations_main() -> None:
    _main_for_neureptrace_command("validate-observations")


def observation_ensemble_main() -> None:
    _main_for_neureptrace_command("observation-ensemble")


def results_main() -> None:
    _main_for_neureptrace_command("results")
