"""Backward-compatible wrapper for the grouped stimulus predictions command."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.stimulus_cli import stimulus_predictions  # noqa: E402


def main() -> int:
    return stimulus_predictions()


if __name__ == "__main__":
    raise SystemExit(main())
