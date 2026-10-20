"""The camera-source protocol every workbench stage talks to.

A source is opened once, then read one frame at a time by a single owner (the
capture thread in :mod:`camera_calibration.capture`). Nothing here draws or
analyses; that keeps every source usable headless and in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


class SourceEnded(Exception):
    """Raised by :meth:`CameraSource.read` when a finite source has no more frames."""


@dataclass
class SourceInfo:
    """What a calibration file needs to know about the camera that made it.

    ``unique_id`` is whatever the transport offers that survives re-plugging
    (USB unique ID, MAC/IP, a directory path); empty when nothing stable exists.
    """

    name: str
    transport: str
    resolution: Tuple[int, int]
    fps: Optional[float] = None
    unique_id: str = ""
    uri: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unique_id": self.unique_id,
            "transport": self.transport,
            "uri": self.uri,
            "resolution": list(self.resolution),
            "fps": self.fps,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class Frame:
    """One image plus when it reached the host.

    ``t_host`` is ``time.monotonic()`` at receipt, ``seq`` counts frames the
    source delivered (gaps mean the source itself dropped), and ``jpeg`` holds
    the encoded bytes when the transport already had them, so a preview can be
    served without re-encoding.
    """

    img: np.ndarray
    t_host: float
    seq: int
    jpeg: Optional[bytes] = None
    path: Optional[str] = None


@dataclass
class Control:
    """One adjustable setting a source exposes (exposure, gain, JPEG quality...)."""

    name: str
    kind: str  # "int", "bool" or "choice"
    value: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    label: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


@runtime_checkable
class CameraSource(Protocol):
    uri: str

    def open(self) -> SourceInfo: ...

    def read(self) -> Frame: ...

    def close(self) -> None: ...

    def controls(self) -> List[Control]: ...

    def set_control(self, name: str, value: Any) -> None: ...


class BaseSource:
    """Defaults for the optional parts of the protocol, plus context-manager use."""

    uri: str = ""

    def controls(self) -> List[Control]:
        return []

    def set_control(self, name: str, value: Any) -> None:
        raise KeyError(f"{type(self).__name__} has no control {name!r}")

    def close(self) -> None:
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
