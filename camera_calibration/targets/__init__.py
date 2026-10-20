"""Calibration targets: detect, draw, and render a printable image.

Every target answers ``detect(gray) -> Detection | None`` with matched image
and board points, so procedures never care which board is in use. The board
frame is the target's own: origin at a board corner, x along columns, y along
rows (down the printed image), z into the board, metres.

Targets are written as spec strings, like camera URIs:

====================================================================  ==========
``charuco?squares=7x5&square=0.03&marker=0.022&dict=DICT_4X4_50``     default
``chessboard?inner=9x6&square=0.025``                                 legacy data
====================================================================  ==========

Always pass the *measured* printed square size: printers scale.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Dict, Union

from .base import Detection, Target
from .charuco import CharucoTarget
from .chessboard import ChessboardTarget

DEFAULT_TARGET = "charuco?squares=7x5&square=0.03&marker=0.022&dict=DICT_4X4_50"

__all__ = ["Detection", "Target", "CharucoTarget", "ChessboardTarget", "DEFAULT_TARGET",
           "parse_target", "target_from_spec"]


def _wxh(text: str):
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def parse_target(text: str = DEFAULT_TARGET) -> Target:
    """Build a target from a spec string (see module docstring)."""
    kind, _, query = text.strip().partition("?")
    o = {k: v[-1] for k, v in urllib.parse.parse_qs(query).items()}
    kind = kind.lower()
    if kind == "charuco":
        return CharucoTarget(squares=_wxh(o.get("squares", "7x5")),
                             square_m=float(o.get("square", 0.03)),
                             marker_m=float(o.get("marker", 0.022)),
                             dictionary=o.get("dict", "DICT_4X4_50"))
    if kind == "chessboard":
        return ChessboardTarget(inner=_wxh(o.get("inner", "9x6")),
                                square_m=float(o.get("square", 0.025)))
    raise ValueError(f"unknown target {kind!r}: use charuco?... or chessboard?...")


def target_from_spec(spec: Union[str, Dict[str, Any]]) -> Target:
    """Rebuild a target from a spec string or a saved ``Target.spec()`` dict."""
    if isinstance(spec, str):
        return parse_target(spec)
    kind = spec.get("type")
    if kind == "charuco":
        return CharucoTarget(squares=tuple(spec["squares"]), square_m=spec["square_m"],
                             marker_m=spec["marker_m"], dictionary=spec["dictionary"])
    if kind == "chessboard":
        return ChessboardTarget(inner=tuple(spec["inner_corners"]), square_m=spec["square_m"])
    raise ValueError(f"unknown target spec {spec!r}")
