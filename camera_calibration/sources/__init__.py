"""Camera sources, chosen by one URI string so the CLI, GUI and config files agree.

=========================================  ===========================================
``uvc:0?size=1280x720&fps=30``             USB webcam by index
``uvc:name=USB Camera?size=1280x720``      USB webcam by name (indices move on re-plug)
``esp32:192.168.1.142``                    ESP32 CameraWebServer, with its controls
``mjpeg:http://host:81/stream``            any MJPEG-over-HTTP stream
``dir:runs/calib01/?fps=5&loop=1``         a directory of images
``video:clip.mp4?loop=1``                  a video file
``mock:?size=640x480&frames=100``          synthetic chessboard with known intrinsics
=========================================  ===========================================

A bare integer (``0``) means ``uvc:0``.
"""

from __future__ import annotations

import urllib.parse
from typing import Dict, Tuple

from .base import BaseSource, CameraSource, Control, Frame, SourceEnded, SourceInfo
from .files import DirSource, VideoSource
from .mjpeg import ESP32Source, MJPEGSource
from .mock import MockSource
from .uvc import UVCSource, list_cameras

__all__ = [
    "BaseSource", "CameraSource", "Control", "Frame", "SourceEnded", "SourceInfo",
    "DirSource", "VideoSource", "ESP32Source", "MJPEGSource", "MockSource", "UVCSource",
    "list_cameras", "open_source", "parse_uri",
]


def _wxh(text: str) -> Tuple[int, int]:
    w, sep, h = text.lower().partition("x")
    if not sep:
        raise ValueError(f"expected WxH, got {text!r}")
    return int(w), int(h)


def _flag(text: str) -> bool:
    return text.lower() in ("1", "true", "yes", "on")


def parse_uri(uri: str) -> Tuple[str, str, Dict[str, str]]:
    """Split ``scheme:target?k=v&...`` into its parts.

    ``mjpeg:`` keeps its query as part of the URL, since it belongs to the server.
    """
    uri = uri.strip()
    if uri.isdigit():
        return "uvc", uri, {}
    scheme, sep, rest = uri.partition(":")
    if not sep:
        raise ValueError(f"camera URI needs a scheme like uvc:, esp32:, mock: — got {uri!r}")
    scheme = scheme.lower()
    if scheme == "mjpeg":
        return scheme, rest, {}
    target, _, query = rest.partition("?")
    opts = {k: v[-1] for k, v in urllib.parse.parse_qs(query, keep_blank_values=True).items()}
    return scheme, urllib.parse.unquote(target), opts


def open_source(uri: str) -> BaseSource:
    """Build (not open) the source a URI names. Call ``.open()`` or use it in ``with``."""
    scheme, target, o = parse_uri(uri)
    if scheme == "uvc":
        kw = dict(size=_wxh(o["size"]) if "size" in o else (0, 0),
                  fps=float(o.get("fps", 0)), fourcc=o.get("fourcc", ""), uri=uri)
        if target.startswith("name="):
            return UVCSource(name=target[5:], **kw)
        return UVCSource(index=int(target or 0), **kw)
    if scheme == "esp32":
        return ESP32Source(target, timeout=float(o.get("timeout", 5)), uri=uri)
    if scheme == "mjpeg":
        return MJPEGSource(target, uri=uri)
    if scheme == "dir":
        return DirSource(target, loop=_flag(o.get("loop", "0")),
                         fps=float(o.get("fps", 0)), uri=uri)
    if scheme == "video":
        return VideoSource(target, loop=_flag(o.get("loop", "0")), uri=uri)
    if scheme == "mock":
        return MockSource(
            size=_wxh(o.get("size", "640x480")), fps=float(o.get("fps", 30)),
            frames=int(o.get("frames", 0)), board=_wxh(o.get("board", "9x6")),
            square=float(o.get("square", 0.025)), k1=float(o.get("k1", 0)),
            noise=float(o.get("noise", 2)), seed=int(o.get("seed", 0)), uri=uri)
    raise ValueError(f"unknown camera scheme {scheme!r} in {uri!r}")
