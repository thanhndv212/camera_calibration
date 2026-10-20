"""Calibration procedures: feedback while capturing, accept samples, solve, validate.

Each procedure follows the same steps so one GUI can drive any of them:
``feedback`` (what to show / tell the user), ``offer`` (auto-accept a steady,
new view, or take one by hand), ``solve``, and ``artifact`` (the calibration
file). Only intrinsics exist so far; camera-to-base and hand-eye follow.
"""

from .intrinsics import IntrinsicsProcedure, IntrinsicsResult, Sample, solve_from_source

__all__ = ["IntrinsicsProcedure", "IntrinsicsResult", "Sample", "solve_from_source"]
