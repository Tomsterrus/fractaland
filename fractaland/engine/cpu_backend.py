"""CPU fallback and correctness reference: the same per-pixel functions as the
GPU backend, compiled with numba.njit and parallelised over rows."""

import numpy as np
from numba import njit, prange

from fractaland.color.palettes import shade_sample
from fractaland.engine.temporal import blend_pixel
from fractaland.fractals.mandelbrot import mandelbrot_perturb_pixel, mandelbrot_pixel

_pixel = njit(cache=True)(mandelbrot_pixel)
_perturb_pixel = njit(cache=True)(mandelbrot_perturb_pixel)
_shade_sample = njit(cache=True)(shade_sample)
_blend_pixel = njit(cache=True)(blend_pixel)


@njit(parallel=True, cache=True)
def _iterate(out, x0, y0, dx, max_iter, bailout2, eps2):
    h, w = out.shape
    for i in prange(h):
        ci = y0 - i * dx
        for j in range(w):
            out[i, j] = _pixel(x0 + j * dx, ci, max_iter, bailout2, eps2)


@njit(parallel=True, cache=True)
def _iterate_perturb(out, dx0, dy0, dx, zr, zi, max_iter, bailout2):
    h, w = out.shape
    for i in prange(h):
        dci = dy0 - i * dx
        for j in range(w):
            out[i, j] = _perturb_pixel(dx0 + j * dx, dci, zr, zi, max_iter, bailout2)


@njit(parallel=True, cache=True)
def _colorize(nu, lut, cum, offset, density, mapping, filt, ir, ig, ib, ss, out):
    h, w = out.shape[0], out.shape[1]
    k = 1.0 / (ss * ss)
    for i in prange(h):
        for j in range(w):
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
                        sr, sg, sb = _shade_sample(nu, y, x, lut, cum, offset, density,
                                                   mapping, filt)
                        r += sr
                        g += sg
                        b += sb
            out[i, j, 0] = min(255, int(r * k + 0.5))
            out[i, j, 1] = min(255, int(g * k + 0.5))
            out[i, j, 2] = min(255, int(b * k + 0.5))
            out[i, j, 3] = 255


@njit(parallel=True, cache=True)
def _blend_history(cur, hist_in, hist_out, disp, s, tx, ty, alpha, valid):
    for y in prange(disp.shape[0]):
        for x in range(disp.shape[1]):
            _blend_pixel(cur, hist_in, hist_out, disp, y, x, s, tx, ty, alpha, valid)


class CpuBackend:
    name = "cpu"
    supports_f32 = False  # float32 brings no speed-up on the CPU

    def __init__(self):
        self._nu = None

    def compute(self, h, w, x0, y0, dx, max_iter, bailout2, precision):
        if self._nu is None or self._nu.shape != (h, w):
            self._nu = np.empty((h, w), dtype=np.float32)
        _iterate(self._nu, x0, y0, dx, max_iter, float(bailout2), 1e-24)
        return self._nu

    def compute_perturb(self, h, w, dx0, dy0, dx, ref, max_iter, bailout2):
        if self._nu is None or self._nu.shape != (h, w):
            self._nu = np.empty((h, w), dtype=np.float32)
        _iterate_perturb(self._nu, dx0, dy0, dx, ref.zr, ref.zi, max_iter, float(bailout2))
        return self._nu

    def colorize(self, nu, lut, cum, offset, density, mapping, filt, interior, ss, to_host=True):
        h, w = nu.shape[0] // ss, nu.shape[1] // ss
        out = np.empty((h, w, 4), dtype=np.uint8)
        _colorize(nu, lut, cum, float(offset), float(density), mapping, float(filt),
                  float(interior[0]), float(interior[1]), float(interior[2]), ss, out)
        return out

    def empty(self, shape):
        return np.empty(shape, dtype=np.float32)

    def blend_history(self, cur, hist_in, hist_out, s, tx, ty, alpha, valid, h, w):
        disp = np.empty((h, w, 4), dtype=np.uint8)
        _blend_history(cur, hist_in, hist_out, disp, float(s), float(tx), float(ty),
                       float(alpha), int(valid))
        return disp

    def to_host(self, nu):
        return nu
