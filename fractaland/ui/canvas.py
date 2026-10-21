"""The fractal canvas: input handling, the render loop and the HUD.

All rendering goes through Engine with a FrameState, exactly as an offline
animation tool would; this widget only decides *when* and at *what
resolution* to render.
"""

import json
import math
import pathlib
import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QWidget

from fractaland.engine.engine import Engine, log_zoom_digits
from fractaland.engine.temporal import FILTER_SCALE, TemporalAccumulator, jittered_view
from fractaland.state import FrameState

FRAME_MS = 16              # render loop period (~60 Hz)
TARGET_MS = 14.0           # compute budget per frame while moving
MOTION_HOLD = 0.15         # seconds after the last input that still count as "moving"
# Resolution fractions used while moving. Fixed steps with hysteresis keep the
# resolution from changing every frame, which itself caused shimmer.
SCALES = [1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125]
STEP_DOWN = 1.25           # step down when a frame takes this many budgets
STEP_UP = 0.8              # step up when the next step is predicted below this
WHEEL_STEP = 1.25          # zoom factor per mouse-wheel notch
PRECISIONS = ["auto", "f32", "f64", "perturb"]

HELP_TEXT = """\
Wheel            zoom at cursor
Hold left/right  continuous zoom in / out
Middle or Shift+left drag   pan
PgUp / PgDn      zoom in / out at centre
, / .            zoom speed down / up
[ / ]            rotate palette (Ctrl: larger step)
Space            auto palette rotation on/off
{ / }            rotation speed down / up
+ / -            max iterations up / down
I                set iteration depth
L                linear / log colour mapping
P                precision auto / f32 / f64 / perturbation
A                antialiasing (2x2) when idle
T                temporal smoothing while moving
G                palette filtering
F11 or F         fullscreen     Esc  leave fullscreen
Alt              menu bar in fullscreen
S                screenshot (PNG + JSON state)
R                reset view     H  HUD     F1  this help
Ctrl+Q           quit"""


def render_size(widget_w, widget_h, dpr, resolution, scale=1.0):
    """Pixel size of the computed image.

    resolution is ("fraction", f): f times the window's physical pixels, or
    ("lines", n): n rows, with the width following the window's aspect ratio.
    `scale` (the adaptive factor used while moving) multiplies either.
    """
    kind, value = resolution
    if kind == "lines":
        h = value
        w = value * widget_w / max(1, widget_h)
    else:
        w = widget_w * dpr * value
        h = widget_h * dpr * value
    return max(1, round(w * scale)), max(1, round(h * scale))


class FractalCanvas(QWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.state = FrameState()
        self.resolution = ("fraction", 1.0)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        # interaction
        self.zoom_speed = 2.0       # factor per second while a button is held
        self.rotate_speed = 0.1     # palette lengths per second
        self.auto_rotate = False
        self.idle_supersample = 2
        self.temporal = True        # accumulate jittered frames while moving
        self.show_hud = True
        self.show_help = False
        self._zoom_dir = 0
        self._pan_from = None
        self._cursor = None
        self._last_input = 0.0

        # rendering state
        self._view_dirty = True
        self._color_dirty = False
        self._needs_refine = False
        self._scale_idx = 2
        self._scale_used = 1.0
        self.accum = TemporalAccumulator(engine.backend)
        self._image = None
        self._buf = None
        self._img_size = (0, 0)
        self._compute_ms = 0.0
        self._fps = 0.0
        self._last_frame = time.perf_counter()
        self._last_tick = time.perf_counter()
        self._message = ("", 0.0)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(FRAME_MS)

    # ---------- geometry helpers ----------

    def _pixel_size(self, scale=1.0):
        return render_size(self.width(), self.height(), self.devicePixelRatioF(),
                           self.resolution, scale)

    def _offset(self, pos):
        """Cursor offset from the widget center in view heights."""
        h = max(1, self.height())
        return (pos.x() - self.width() / 2) / h, (pos.y() - self.height() / 2) / h

    def _touch(self):
        self._last_input = time.perf_counter()
        self._view_dirty = True

    # ---------- render loop ----------

    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now

        if self._zoom_dir and self._cursor is not None:
            fx, fy = self._offset(self._cursor)
            self.state.view.zoom_about(fx, fy, self.zoom_speed ** (self._zoom_dir * dt))
            self._touch()
        if self.auto_rotate:
            self.state.color.offset = (self.state.color.offset + self.rotate_speed * dt) % 1.0
            self._color_dirty = True

        moving = now - self._last_input < MOTION_HOLD
        computed = False
        if self._view_dirty and moving:
            self._compute(SCALES[self._scale_idx], 1, moving=True)
            self._needs_refine = True
            computed = True
        elif self._view_dirty or (self._needs_refine and not moving):
            self._compute(1.0, self.idle_supersample, moving=False)
            self._needs_refine = False
            computed = True
        self._view_dirty = False

        if computed or self._color_dirty:
            if not computed:  # palette change only: recolour the cached counts
                self._colorize()
            self._color_dirty = False
            dtf = now - self._last_frame
            self._last_frame = now
            if dtf > 0:
                self._fps = 0.9 * self._fps + 0.1 / dtf if self._fps else 1.0 / dtf
            self.update()
        elif self._message[0] and now > self._message[1]:
            self._message = ("", 0.0)
            self.update()

    def _compute(self, scale, ss, moving):
        """Compute and display one frame. While moving, frames are jittered
        and accumulated; the still frame (supersampled) restarts the history."""
        w, h = self._pixel_size(scale)
        W, H = self._pixel_size()
        view = self.state.view
        if moving and self.temporal:
            view = jittered_view(view, *self.accum.jitter(), h)
        self.state.render.supersample = ss
        t = time.perf_counter()
        self.engine.compute(view, self.state.render, w, h)
        self._img_size = (w, h)
        self._set_image(self._present(seed=not moving))  # also waits for the GPU
        self._compute_ms = (time.perf_counter() - t) * 1000
        self._scale_used = scale
        if moving:
            self._adapt_scale()

    def _adapt_scale(self):
        i, ms = self._scale_idx, self._compute_ms
        if ms > STEP_DOWN * TARGET_MS and i < len(SCALES) - 1:
            self._scale_idx = i + 1
        elif i > 0 and ms * (SCALES[i - 1] / SCALES[i]) ** 2 < STEP_UP * TARGET_MS:
            self._scale_idx = i - 1

    def _present(self, seed):
        """Colour the last computed counts and, with temporal smoothing,
        blend them into the history at display resolution."""
        if not self.temporal:
            return self.engine.colorize(self.state.color)
        W, H = self._pixel_size()
        cur = self.engine.colorize(self.state.color, to_host=False,
                                   filter_scale=1.0 if seed else FILTER_SCALE)
        return self.accum.blend(cur, self.state.view, W, H, seed=seed)

    def _colorize(self):
        if self._image is None or self._buf is None:
            return
        self._set_image(self._present(seed=True))

    def _set_image(self, rgba):
        self._buf = rgba  # keep the array alive while QImage points into it
        h, w = rgba.shape[:2]
        self._image = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBX8888)

    # ---------- painting ----------

    def paintEvent(self, event):
        p = QPainter(self)
        if self._image is None:
            p.fillRect(self.rect(), Qt.black)
        else:
            if self._image.width() < self.width() * self.devicePixelRatioF():
                p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(QRectF(self.rect()), self._image)
        if self.show_hud:
            self._draw_hud(p)
        if self.show_help:
            self._draw_box(p, HELP_TEXT.splitlines(), right=True)
        p.end()

    def _hud_lines(self):
        s = self.state
        v, r, c = s.view, s.render, s.color
        d = log_zoom_digits(v)
        prec = self.engine.last_precision or "?"
        w, h = self._pixel_size()
        lines = [
            f"re  {float(v.center_re): .{d}f}" if d <= 17 else f"re  {v.center_re:.{d}f}",
            f"im  {float(v.center_im): .{d}f}" if d <= 17 else f"im  {v.center_im:.{d}f}",
            f"zoom {v.zoom:.3e}   speed x{self.zoom_speed:.2f}/s",
            f"iter {self.engine.last_max_iter}   {prec} ({r.precision})   {self.engine.backend.name}",
            f"frame {w}x{h} @ {round(self._scale_used * 100)}%"
            f"{'  aa' if r.supersample > 1 else ''}"
            f"{'  temporal %.2f' % self.accum.last_alpha if self.temporal and r.supersample == 1 else ''}"
            f"   {self._compute_ms:.1f} ms   {self._fps:.0f} fps",
            f"palette {c.palette}  offset {c.offset:.3f}  {c.mapping}"
            f"{'  filtered' if c.palette_filter else ''}"
            f"{'  rotating %.2f/s' % self.rotate_speed if self.auto_rotate else ''}",
        ]
        ref = self.engine.last_reference
        if prec == "perturb" and ref is not None:
            lines.append(f"ref {ref.length:,} it{'' if ref.escaped else '+'}  {ref.bits} bits"
                         f"  {ref.compute_ms:.0f} ms"
                         f"{'   new reference: computing...' if self.engine.refs.pending else ''}")
        if self.engine.precision_exhausted(v, h):
            if prec == "perturb":
                lines.append("maximum zoom reached (float64 offset range)")
            else:
                lines.append(f"{prec} limit reached: use Auto or Perturbation precision")
        msg, until = self._message
        if msg and time.perf_counter() < until:
            lines.append(msg)
        lines.append("F1 help")
        return lines

    def _draw_hud(self, p):
        self._draw_box(p, self._hud_lines())

    def _draw_box(self, p, lines, right=False):
        font = QFont("Consolas", 10)
        p.setFont(font)
        fm = p.fontMetrics()
        pad = 8
        bw = max(fm.horizontalAdvance(l) for l in lines) + 2 * pad
        bh = fm.height() * len(lines) + 2 * pad
        x = self.width() - bw - 10 if right else 10
        y = 10
        p.fillRect(x, y, bw, bh, QColor(0, 0, 0, 150))
        p.setPen(QColor(235, 235, 235))
        for i, line in enumerate(lines):
            p.drawText(x + pad, y + pad + fm.ascent() + i * fm.height(), line)

    # ---------- input ----------

    def mousePressEvent(self, e):
        self._cursor = e.position()
        shift = e.modifiers() & Qt.ShiftModifier
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and shift):
            self._pan_from = e.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton:
            self._zoom_dir = 1
        elif e.button() == Qt.RightButton:
            self._zoom_dir = -1

    def mouseMoveEvent(self, e):
        self._cursor = e.position()
        if self._pan_from is not None:
            h = max(1, self.height())
            d = e.position() - self._pan_from
            self.state.view.pan(d.x() / h, d.y() / h)
            self._pan_from = e.position()
            self._touch()

    def mouseReleaseEvent(self, e):
        if self._pan_from is not None and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan_from = None
            self.setCursor(Qt.CrossCursor)
        if e.button() in (Qt.LeftButton, Qt.RightButton):
            self._zoom_dir = 0

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps:
            fx, fy = self._offset(e.position())
            self.state.view.zoom_about(fx, fy, WHEEL_STEP ** steps)
            self._touch()

    def resizeEvent(self, e):
        self._view_dirty = True
        super().resizeEvent(e)

    # ---------- commands (called by the menu actions) ----------

    def rotate_palette(self, step):
        self.state.color.offset = (self.state.color.offset + step) % 1.0
        self._color_dirty = True

    def reset_rotation(self):
        self.state.color.offset = 0.0
        self._color_dirty = True

    def set_auto_rotate(self, on):
        self.auto_rotate = on

    def scale_rotate_speed(self, f):
        self.rotate_speed *= f

    def zoom_step(self, factor):
        """Zoom at the window centre."""
        self.state.view.zoom_about(0.0, 0.0, factor)
        self._touch()

    def scale_zoom_speed(self, exponent):
        self.zoom_speed = max(1.05, self.zoom_speed ** exponent)

    def reset_view(self):
        self.state.view = FrameState().view
        self._view_dirty = True

    def scale_max_iter(self, f):
        r = self.state.render
        r.max_iter = max(16, int(r.max_iter * f))
        self._view_dirty = True

    def set_iterations(self, max_iter, auto_iter, iter_growth):
        r = self.state.render
        r.max_iter, r.auto_iter, r.iter_growth = max_iter, auto_iter, iter_growth
        self._view_dirty = True

    def set_precision(self, mode):
        self.state.render.precision = mode
        self._view_dirty = True

    def set_mapping(self, mode):
        self.state.color.mapping = mode
        self._color_dirty = True

    def set_antialias(self, on):
        self.idle_supersample = 2 if on else 1
        self._view_dirty = True

    def set_temporal(self, on):
        self.temporal = on
        self.accum.reset()
        self._view_dirty = True

    def set_palette_filter(self, on):
        self.state.color.palette_filter = on
        self._color_dirty = True

    def set_resolution(self, resolution):
        self.resolution = resolution
        self._view_dirty = True

    def set_show_hud(self, on):
        self.show_hud = on
        self.update()

    def set_show_help(self, on):
        self.show_help = on
        self.update()

    def screenshot(self):
        folder = pathlib.Path.cwd() / "screenshots"
        folder.mkdir(exist_ok=True)
        stem = folder / time.strftime("fractal_%Y%m%d_%H%M%S")
        w, h = self._pixel_size()
        render = FrameState.from_dict(self.state.to_dict()).render
        render.supersample = 2
        rgba = self.engine.render(self.state.view, render, self.state.color, w, h)
        QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBX8888).save(str(stem) + ".png")
        stem.with_suffix(".json").write_text(json.dumps(self.state.to_dict(), indent=2))
        self._message = (f"saved {stem.name}.png ({w}x{h})", time.perf_counter() + 3.0)
        # The engine buffer now holds the screenshot; recompute the display.
        self._view_dirty = True
