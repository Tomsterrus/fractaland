"""Temporal accumulation: supersampling spread over successive frames.

While the view moves, each frame is computed with a different sub-pixel
offset (jitter). The accumulated history is warped into the new view (the
view change is known exactly, so this is a pure scale and shift) and blended
with the new frame. Several frames of jittered samples then average out the
aliasing that otherwise makes fine detail sparkle during zoom.
"""

import math
from numba.extending import register_jitable

from fractaland.state import View, dec

ALPHA_MIN = 0.2      # weight of the new frame during slow, smooth motion
ALPHA_PER_LOG = 4.0  # extra weight per unit of |ln(scale change)| per frame
JITTER_CYCLE = 64    # length of the repeating jitter sequence
# Palette filter width while accumulating, relative to the full one-pixel
# width: the jitter already averages over the pixel area, and the full filter
# on top blurs. Measured with tools/flicker.py: 0.25 keeps flicker at
# 0.05 % and comes closer to a 4x4-supersampled image than 2x2 antialiasing.
FILTER_SCALE = 0.25


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def frame_transform(old: View, new: View, w: int, h: int):
    """Return (s, tx, ty) such that pixel (x, y) of a w x h frame of `new`
    shows the point found at (s*x + tx, s*y + ty) in the frame of `old`."""
    d_old = old.height / h
    s = new.height / old.height
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    shift_x = float(new.center_re - old.center_re) / d_old
    shift_y = -float(new.center_im - old.center_im) / d_old
    return s, shift_x + cx * (1 - s), shift_y + cy * (1 - s)


def blend_weight(s: float) -> float:
    """Weight of the new frame: low for slow motion (long averaging), up to 1
    for large jumps, where old samples would only smear."""
    return min(1.0, ALPHA_MIN + ALPHA_PER_LOG * abs(math.log(s)))


@register_jitable
def blend_pixel(cur, hist_in, hist_out, disp, y, x, s, tx, ty, alpha, valid):
    """Blend display pixel (y, x): the current frame `cur` (uint8 RGBA, any
    size, sampled bilinearly) with the history warped from the old view."""
    h = disp.shape[0]
    w = disp.shape[1]
    ch = cur.shape[0]
    cw = cur.shape[1]
    cx = (x + 0.5) * cw / w - 0.5
    cy = (y + 0.5) * ch / h - 0.5
    cx = min(max(cx, 0.0), cw - 1.0)
    cy = min(max(cy, 0.0), ch - 1.0)
    cx0 = int(cx)
    cy0 = int(cy)
    cx1 = min(cx0 + 1, cw - 1)
    cy1 = min(cy0 + 1, ch - 1)
    cfx = cx - cx0
    cfy = cy - cy0

    hx = s * x + tx
    hy = s * y + ty
    use_hist = valid != 0 and hx >= 0.0 and hy >= 0.0 and hx <= w - 1.0 and hy <= h - 1.0
    hx0 = 0
    hy0 = 0
    hx1 = 0
    hy1 = 0
    hfx = 0.0
    hfy = 0.0
    if use_hist:
        hx0 = int(hx)
        hy0 = int(hy)
        hx1 = min(hx0 + 1, w - 1)
        hy1 = min(hy0 + 1, h - 1)
        hfx = hx - hx0
        hfy = hy - hy0

    for k in range(3):
        # float() first: uint8 differences would wrap around.
        c00 = float(cur[cy0, cx0, k])
        c01 = float(cur[cy0, cx1, k])
        c10 = float(cur[cy1, cx0, k])
        c11 = float(cur[cy1, cx1, k])
        top = c00 + (c01 - c00) * cfx
        bot = c10 + (c11 - c10) * cfx
        v = top + (bot - top) * cfy
        if use_hist:
            ht = hist_in[hy0, hx0, k] + (hist_in[hy0, hx1, k] - hist_in[hy0, hx0, k]) * hfx
            hb = hist_in[hy1, hx0, k] + (hist_in[hy1, hx1, k] - hist_in[hy1, hx0, k]) * hfx
            v = alpha * v + (1.0 - alpha) * (ht + (hb - ht) * hfy)
        hist_out[y, x, k] = v
        disp[y, x, k] = min(255, int(v + 0.5))
    disp[y, x, 3] = 255


class TemporalAccumulator:
    """Owns the history buffers (on the backend's device) and the jitter."""

    def __init__(self, backend):
        self.backend = backend
        self.reset()

    def reset(self):
        self._hist = None      # (history in, history out)
        self._view = None
        self._frame = 0
        self.last_alpha = 1.0

    def jitter(self):
        """Sub-pixel offset in (-0.5, 0.5) for the next frame."""
        i = self._frame % JITTER_CYCLE + 1
        return halton(i, 2) - 0.5, halton(i, 3) - 0.5

    def blend(self, cur, view: View, w: int, h: int, seed=False):
        """Blend the colourised frame `cur` (backend array) for `view`
        (without jitter) into the history; return the host RGBA image.
        With seed, the history restarts from this frame alone."""
        if self._hist is None or self._hist[0].shape[:2] != (h, w):
            self._hist = (self.backend.empty((h, w, 3)), self.backend.empty((h, w, 3)))
            self._view = None
        valid = self._view is not None and not seed
        s, tx, ty = frame_transform(self._view, view, w, h) if valid else (1.0, 0.0, 0.0)
        alpha = blend_weight(s) if valid else 1.0
        hist_in, hist_out = self._hist
        out = self.backend.blend_history(cur, hist_in, hist_out, s, tx, ty, alpha, int(valid), h, w)
        self._hist = (hist_out, hist_in)
        self._view = View(view.center_re, view.center_im, view.log_zoom)
        self._frame += 1
        self.last_alpha = alpha
        return out


def jittered_view(view: View, jx: float, jy: float, h: int) -> View:
    """The view shifted by (jx, jy) pixels of a frame h pixels high."""
    d = dec(view.height / h)
    return View(view.center_re + dec(jx) * d, view.center_im - dec(jy) * d, view.log_zoom)
