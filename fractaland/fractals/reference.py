"""Reference orbit for perturbation rendering.

One point C (the view center) is iterated on the CPU in arbitrary precision;
its orbit Z_0 = 0, Z_1 = C, Z_2, ... is stored as float64. Pixels are then
iterated on the GPU only as small offsets from this orbit (see
mandelbrot_perturb_pixel).
"""

import math
import time
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from fractaland.state import dec

try:
    import gmpy2
except ImportError:  # pure-Python fallback, roughly 10x slower
    gmpy2 = None
    import mpmath

# Bits that must exceed the zoom depth (log2 of the zoom) for the orbit to be accurate.
MIN_MARGIN_BITS = 32
# Extra bits computed beyond the minimum, so continuous zooming rarely forces a recompute.
HEADROOM_BITS = 96
# The reference is replaced once the view center is this many view heights away.
MAX_OFFSET_HEIGHTS = 1000.0


def needed_bits(log_zoom: float) -> int:
    return int(max(log_zoom, 0.0) * math.log2(10.0)) + MIN_MARGIN_BITS


@dataclass
class Reference:
    re: Decimal
    im: Decimal
    bits: int
    zr: np.ndarray       # float64, Z_0 .. Z_L
    zi: np.ndarray
    escaped: bool        # True if the orbit left the bailout radius (length is final)
    max_iter: int        # iteration limit it was computed for
    compute_ms: float

    @property
    def length(self) -> int:
        return len(self.zr) - 1


def compute_reference(center_re: Decimal, center_im: Decimal, max_iter: int,
                      bits: int, bailout2: float) -> Reference:
    t = time.perf_counter()
    zr_out = np.empty(max_iter + 1)
    zi_out = np.empty(max_iter + 1)
    zr_out[0] = zi_out[0] = 0.0
    n = 0
    escaped = False
    if gmpy2 is not None:
        with gmpy2.context(gmpy2.get_context(), precision=bits):
            cr = gmpy2.mpfr(str(center_re))
            ci = gmpy2.mpfr(str(center_im))
            zr = gmpy2.mpfr(0)
            zi = gmpy2.mpfr(0)
            for n in range(1, max_iter + 1):
                zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                fr, fi = float(zr), float(zi)
                zr_out[n], zi_out[n] = fr, fi
                if fr * fr + fi * fi > bailout2:
                    escaped = True
                    break
    else:
        with mpmath.workprec(bits):
            c = mpmath.mpc(mpmath.mpf(str(center_re)), mpmath.mpf(str(center_im)))
            z = mpmath.mpc(0)
            for n in range(1, max_iter + 1):
                z = z * z + c
                fr, fi = float(z.real), float(z.imag)
                zr_out[n], zi_out[n] = fr, fi
                if fr * fr + fi * fi > bailout2:
                    escaped = True
                    break
    return Reference(center_re, center_im, bits, zr_out[:n + 1].copy(), zi_out[:n + 1].copy(),
                     escaped, max_iter, (time.perf_counter() - t) * 1000)


class ReferenceCache:
    """Keeps the last reference orbit and decides when it must be replaced.

    With background=True, a replacement is computed in a separate process
    while the old orbit keeps being used. That is safe: rebasing makes any
    reference orbit give correct pixels, a few bits too few degrade the image
    only marginally for a few frames, and float64 offsets tolerate a reference
    far outside the view. (A process rather than a thread: the gmpy2 loop
    holds Python's interpreter lock and would slow the UI.)
    """

    def __init__(self, background=False):
        self.background = background
        self.ref = None
        self.recomputed = False  # whether the last get() switched to a new orbit
        self._pool = None
        self._future = None

    @property
    def pending(self) -> bool:
        return self._future is not None

    def start(self):
        """Start the worker process now (Windows process start-up takes about
        a second), warming it with a tiny job."""
        if self.background and self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=1)
            self._pool.submit(compute_reference, Decimal(0), Decimal(0), 4, 64, 4.0).result()

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            self._future = None

    def get(self, view, max_iter: int, bailout2: float) -> Reference:
        self.recomputed = False
        if self._future is not None and self._future.done():
            try:
                self.ref = self._future.result()
                self.recomputed = True
            finally:
                self._future = None
        need = needed_bits(view.log_zoom)
        if self.ref is not None and self._usable(self.ref, view, max_iter, need):
            return self.ref
        args = (view.center_re, view.center_im, 2 * max_iter, need + HEADROOM_BITS, bailout2)
        if self.ref is not None and self.background:
            self.start()
            if self._future is None:
                self._future = self._pool.submit(compute_reference, *args)
            return self.ref
        self.ref = compute_reference(*args)
        self.recomputed = True
        return self.ref

    def wait(self):
        """Block until a pending background orbit is ready (for tests and tools)."""
        if self._future is not None:
            self._future.result()

    @staticmethod
    def _usable(ref, view, max_iter, need) -> bool:
        if need > ref.bits:
            return False
        if not ref.escaped and max_iter > ref.max_iter:
            return False
        h = dec(view.height)
        dist = max(abs(view.center_re - ref.re), abs(view.center_im - ref.im))
        return dist <= h * Decimal(MAX_OFFSET_HEIGHTS)
