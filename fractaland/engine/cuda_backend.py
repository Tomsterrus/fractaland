"""CUDA backend: one GPU thread per pixel. The smooth-count buffer stays on
the device so that palette changes only rerun the cheap colouring kernel."""

import math

import numpy as np
from numba import config, cuda, float32

# Small images (thumbnails, adaptive low-resolution frames) are expected.
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from fractaland.color.palettes import shade_sample
from fractaland.engine.temporal import blend_pixel
from fractaland.fractals.mandelbrot import mandelbrot_perturb_pixel, mandelbrot_pixel

_pixel = cuda.jit(device=True)(mandelbrot_pixel)
_perturb_pixel = cuda.jit(device=True)(mandelbrot_perturb_pixel)
_shade_sample = cuda.jit(device=True)(shade_sample)
_blend_pixel = cuda.jit(device=True)(blend_pixel)

BLOCK = (16, 16)
# Windows resets the driver when one kernel runs longer than ~2 s (WDDM
# watchdog), so large images are computed in bands of rows, one launch each.
BAND_ITER_BUDGET = 1.5e9  # pixel-iterations per launch, worst case


@cuda.jit(cache=True)
def _iterate_f64(out, row0, x0, y0, dx, max_iter, bailout2, eps2):
    j, i = cuda.grid(2)
    i += row0
    if i < out.shape[0] and j < out.shape[1]:
        out[i, j] = _pixel(x0 + j * dx, y0 - i * dx, max_iter, bailout2, eps2)


@cuda.jit(cache=True)
def _iterate_f32(out, row0, x0, y0, dx, max_iter, bailout2, eps2):
    j, i = cuda.grid(2)
    i += row0
    if i < out.shape[0] and j < out.shape[1]:
        # Coordinates are formed in float64, then rounded once to float32.
        cr = float32(x0 + j * dx)
        ci = float32(y0 - i * dx)
        out[i, j] = _pixel(cr, ci, max_iter, bailout2, eps2)


@cuda.jit(cache=True)
def _iterate_perturb(out, row0, dx0, dy0, dx, zr, zi, max_iter, bailout2):
    """dx0, dy0: offset of the top-left pixel from the reference point."""
    j, i = cuda.grid(2)
    i += row0
    if i < out.shape[0] and j < out.shape[1]:
        out[i, j] = _perturb_pixel(dx0 + j * dx, dy0 - i * dx, zr, zi, max_iter, bailout2)


@cuda.jit(cache=True)
def _colorize(nu, lut, cum, offset, density, mapping, filt, ir, ig, ib, ss, out):
    """Average ss x ss samples of `nu` into one RGBA pixel of `out`."""
    j, i = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        r = 0.0
        g = 0.0
        b = 0.0
        for a in range(ss):
            for c in range(ss):
                y = i * ss + a
                x = j * ss + c
                if nu[y, x] < 0.0:
                    r += ir
                    g += ig
                    b += ib
                else:
                    sr, sg, sb = _shade_sample(nu, y, x, lut, cum, offset, density, mapping, filt)
                    r += sr
                    g += sg
                    b += sb
        k = 1.0 / (ss * ss)
        out[i, j, 0] = min(255, int(r * k + 0.5))
        out[i, j, 1] = min(255, int(g * k + 0.5))
        out[i, j, 2] = min(255, int(b * k + 0.5))
        out[i, j, 3] = 255


@cuda.jit(cache=True)
def _blend_history(cur, hist_in, hist_out, disp, s, tx, ty, alpha, valid):
    x, y = cuda.grid(2)
    if y < disp.shape[0] and x < disp.shape[1]:
        _blend_pixel(cur, hist_in, hist_out, disp, y, x, s, tx, ty, alpha, valid)


def _grid(h, w):
    return (math.ceil(w / BLOCK[0]), math.ceil(h / BLOCK[1]))


class CudaBackend:
    name = "cuda"
    supports_f32 = True

    def __init__(self):
        self._nu = None
        self._rgba = None
        self._disp = None
        self._luts = {}
        self._ref = (None, None, None)  # (host reference, device zr, device zi)

    def _buffer(self, attr, shape, dtype):
        buf = getattr(self, attr)
        if buf is None or buf.shape != shape:
            buf = cuda.device_array(shape, dtype=dtype)
            setattr(self, attr, buf)
        return buf

    def compute(self, h, w, x0, y0, dx, max_iter, bailout2, precision):
        """Fill and return the device buffer of smooth counts (h, w)."""
        nu = self._buffer("_nu", (h, w), np.float32)
        if precision == "f32":
            kernel, args = _iterate_f32, (x0, y0, dx, max_iter,
                                          np.float32(bailout2), np.float32(1e-12))
            budget = BAND_ITER_BUDGET * 4  # float32 is several times faster
        else:
            kernel, args = _iterate_f64, (x0, y0, dx, max_iter, float(bailout2), 1e-24)
            budget = BAND_ITER_BUDGET
        self._launch_banded(kernel, nu, args, budget, max_iter)
        return nu

    def compute_perturb(self, h, w, dx0, dy0, dx, ref, max_iter, bailout2):
        """Perturbation render around the reference orbit `ref`."""
        nu = self._buffer("_nu", (h, w), np.float32)
        if self._ref[0] is not ref:
            self._ref = (ref, cuda.to_device(ref.zr), cuda.to_device(ref.zi))
        _, d_zr, d_zi = self._ref
        args = (dx0, dy0, dx, d_zr, d_zi, max_iter, float(bailout2))
        self._launch_banded(_iterate_perturb, nu, args, BAND_ITER_BUDGET, max_iter)
        return nu

    @staticmethod
    def _launch_banded(kernel, nu, args, budget, max_iter):
        h, w = nu.shape
        band = int(budget / (w * max_iter)) // BLOCK[1] * BLOCK[1]
        band = min(h, max(BLOCK[1], band))
        for row0 in range(0, h, band):
            rows = min(band, h - row0)
            kernel[_grid(rows, w), BLOCK](nu, row0, *args)
            if rows < h:
                cuda.synchronize()

    def colorize(self, nu, lut, cum, offset, density, mapping, filt, interior, ss, to_host=True):
        """RGBA uint8 image of shape (h/ss, w/ss, 4): a host array, or with
        to_host=False the device buffer (for blend_history)."""
        key = id(lut)
        if key not in self._luts:
            self._luts = {key: (lut, cuda.to_device(lut), cuda.to_device(cum))}
        _, d_lut, d_cum = self._luts[key]
        h, w = nu.shape[0] // ss, nu.shape[1] // ss
        out = self._buffer("_rgba", (h, w, 4), np.uint8)
        _colorize[_grid(h, w), BLOCK](nu, d_lut, d_cum, float(offset), float(density), mapping,
                                      float(filt), float(interior[0]), float(interior[1]),
                                      float(interior[2]), ss, out)
        return out.copy_to_host() if to_host else out

    def empty(self, shape):
        return cuda.device_array(shape, dtype=np.float32)

    def blend_history(self, cur, hist_in, hist_out, s, tx, ty, alpha, valid, h, w):
        """Blend the device frame `cur` into the history; return host RGBA (h, w, 4)."""
        disp = self._buffer("_disp", (h, w, 4), np.uint8)
        _blend_history[_grid(h, w), BLOCK](cur, hist_in, hist_out, disp, float(s), float(tx),
                                           float(ty), float(alpha), int(valid))
        return disp.copy_to_host()

    def to_host(self, nu):
        return nu.copy_to_host()
