#!/usr/bin/env python3
"""Backward-compatible wrapper for the grouped MEG data download command."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

_SPEC = spec_from_file_location("_pymegdec_data_download", _SRC / "pymegdec" / "data_download.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Could not load pymegdec.data_download")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
download_meg_data_files = _MODULE.download_meg_data_files


def main() -> int:
    return download_meg_data_files()


if __name__ == "__main__":
    raise SystemExit(main())
