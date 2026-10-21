"""Serialisable render state: everything needed to reproduce one frame.

The interactive viewer and the (future) animation exporter both drive the
engine exclusively through these objects, so a frame is fully described by
(View, RenderParams, ColorParams, width, height).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, getcontext

BASE_HEIGHT = 3.0  # height of the view in the complex plane at zoom 1
# Perturbation keeps pixel offsets in float64, whose normal range ends near
# 1e-308; pixel spacing at this zoom is still well inside it.
MAX_LOG_ZOOM = 290.0


def set_decimal_precision(log_zoom: float) -> None:
    """Keep enough decimal digits for the view center at this zoom."""
    getcontext().prec = max(60, int(log_zoom) + 30)


set_decimal_precision(0.0)


def dec(x) -> Decimal:
    """Exact Decimal of a float (also numpy floats, whose repr is not plain)."""
    return Decimal(repr(float(x)))


@dataclass
class View:
    """Where we look. The center is kept as Decimal so deep zooms can later be
    computed with perturbation without changing this interface."""

    center_re: Decimal = Decimal("-0.5")
    center_im: Decimal = Decimal("0")
    log_zoom: float = 0.0  # log10 of the zoom factor; interpolates linearly in animations
    rotation: float = 0.0  # radians, reserved for later

    @property
    def zoom(self) -> float:
        return 10.0 ** self.log_zoom

    @property
    def height(self) -> float:
        """Height of the visible region in complex-plane units."""
        return BASE_HEIGHT / self.zoom

    def zoom_about(self, fx: float, fy: float, factor: float) -> None:
        """Zoom by `factor` keeping the point at offset (fx, fy) fixed.

        fx, fy are offsets from the screen center measured in view heights
        (x to the right, y downwards, as in screen coordinates).
        """
        new_log_zoom = min(self.log_zoom + math.log10(factor), MAX_LOG_ZOOM)
        factor = 10.0 ** (new_log_zoom - self.log_zoom)
        set_decimal_precision(new_log_zoom)
        h = dec(self.height)
        keep = Decimal(1) - dec(1.0 / factor)
        self.center_re += dec(fx) * h * keep
        self.center_im -= dec(fy) * h * keep
        self.log_zoom = new_log_zoom

    def pan(self, fx: float, fy: float) -> None:
        """Move the content by (fx, fy) view heights (screen directions)."""
        set_decimal_precision(self.log_zoom)
        h = dec(self.height)
        self.center_re -= dec(fx) * h
        self.center_im += dec(fy) * h

    def to_dict(self) -> dict:
        return {
            "center_re": str(self.center_re),
            "center_im": str(self.center_im),
            "log_zoom": self.log_zoom,
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "View":
        set_decimal_precision(float(d["log_zoom"]))
        return cls(Decimal(d["center_re"]), Decimal(d["center_im"]),
                   float(d["log_zoom"]), float(d.get("rotation", 0.0)))


@dataclass
class RenderParams:
    max_iter: int = 500        # base iteration limit (at zoom 1)
    auto_iter: bool = True     # raise the limit as we zoom in
    iter_growth: float = 50.0  # extra iterations per (log10 zoom)^2 when auto_iter is on
    precision: str = "auto"    # "auto", "f32" or "f64"
    supersample: int = 1       # s x s samples per pixel
    bailout: float = 256.0     # escape radius; large values make smooth colouring accurate

    def effective_max_iter(self, view: View) -> int:
        if not self.auto_iter:
            return self.max_iter
        lz = max(view.log_zoom, 0.0)
        return int(self.max_iter + self.iter_growth * lz * lz)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RenderParams":
        return cls(**d)


@dataclass
class ColorParams:
    palette: str = "default"
    offset: float = 0.0              # palette rotation phase, in palette lengths (wraps at 1)
    density: float = 1.0 / 64.0      # palette lengths per iteration
    mapping: str = "linear"          # "linear" or "log" (log suits deep zooms with huge counts)
    interior: tuple = field(default=(0, 0, 0))
    palette_filter: bool = True      # average the palette where it changes faster than a pixel

    def to_dict(self) -> dict:
        d = asdict(self)
        d["interior"] = list(self.interior)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ColorParams":
        d = dict(d)
        d["interior"] = tuple(d.get("interior", (0, 0, 0)))
        return cls(**d)


@dataclass
class FrameState:
    view: View = field(default_factory=View)
    render: RenderParams = field(default_factory=RenderParams)
    color: ColorParams = field(default_factory=ColorParams)

    def to_dict(self) -> dict:
        return {"view": self.view.to_dict(), "render": self.render.to_dict(),
                "color": self.color.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameState":
        return cls(View.from_dict(d["view"]), RenderParams.from_dict(d["render"]),
                   ColorParams.from_dict(d["color"]))
