"""Public compatibility facade for cross-subject stimulus decoding."""

from __future__ import annotations

import sys

from pymegdec import _stimulus_cross_subject_core as _core
from pymegdec import _stimulus_cross_subject_timefix as _timefix  # noqa: F401
from pymegdec import _stimulus_cross_subject_chance as _chance  # noqa: F401

sys.modules[__name__] = _core
globals().update(_core.__dict__)
