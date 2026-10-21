"""Engine: the single entry point for rendering, used by the viewer and by
any offline tool. It turns (View, RenderParams, ColorParams, size) into
pixels and holds no UI state."""

import math

import numpy as np

from fractaland.color.palettes import MAPPING_CODES, build_cumulative_lut, get_lut
from fractaland.fractals.reference import ReferenceCache
from fractaland.state import MAX_LOG_ZOOM, ColorParams, RenderParams, View

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
# float32 is used while the pixel spacing exceeds this many float32 ulps of |c|;
# below it, auto mode uses perturbation. Direct float64 is not chosen
# automatically: compared with 200-bit results, it already gets 8 % of
# boundary pixels wrong at zoom 1e6 (rounding errors grow with iterations),
# while perturbation matches them.
F32_MARGIN = 32.0


def make_backend(prefer_gpu=True):
    if prefer_gpu:
        try:
            from numba import cuda
            if cuda.is_available():
                from fractaland.engine.cuda_backend import CudaBackend
                return CudaBackend()
        except Exception:
            pass
    from fractaland.engine.cpu_backend import CpuBackend
    return CpuBackend()


class Engine:
    def __init__(self, prefer_gpu=True, background_reference=False):
        self.backend = make_backend(prefer_gpu)
        self._luts = {}
        self._nu = None
        self._ss = 1
        self.refs = ReferenceCache(background_reference)
        self.last_precision = None
        self.last_max_iter = None
        self.last_reference = None  # Reference used by the last perturbation render

    def lut(self, name):
        """(LUT, cumulative LUT) of a palette, built once."""
        if name not in self._luts:
            lut = get_lut(name)
            self._luts[name] = (lut, build_cumulative_lut(lut))
        return self._luts[name]

    @staticmethod
    def geometry(view: View, w: int, h: int):
        """Return (x0, y0, dx): complex coordinates of the top-left pixel
        center and the spacing between pixel centers."""
        dx = view.height / h
        cr, ci = float(view.center_re), float(view.center_im)
        x0 = cr - (w / 2 - 0.5) * dx
        y0 = ci + (h / 2 - 0.5) * dx
        return x0, y0, dx

    def choose_precision(self, view: View, params: RenderParams, w: int, h: int) -> str:
        if params.precision != "auto":
            if params.precision == "f32" and not self.backend.supports_f32:
                return "f64"
            return params.precision
        dx = view.height / h
        mag = max(abs(float(view.center_re)), abs(float(view.center_im)), 0.5)
        if self.backend.supports_f32 and dx > F32_MARGIN * EPS32 * mag:
            return "f32"
        return "perturb"

    def precision_exhausted(self, view: View, h: int) -> bool:
        """True when the precision in use cannot resolve neighbouring pixels."""
        if self.last_precision == "perturb":
            return view.log_zoom >= MAX_LOG_ZOOM
        mag = max(abs(float(view.center_re)), abs(float(view.center_im)), 0.5)
        limit = 16.0 * (EPS32 if self.last_precision == "f32" else EPS64)
        return view.height / h < limit * mag

    def compute(self, view: View, params: RenderParams, w: int, h: int):
        """Compute smooth iteration counts for a w x h image (times the
        supersampling factor); the result stays inside the engine/backend."""
        ss = max(1, int(params.supersample))
        W, H = w * ss, h * ss
        x0, y0, dx = self.geometry(view, W, H)
        precision = self.choose_precision(view, params, W, H)
        max_iter = params.effective_max_iter(view)
        bailout2 = params.bailout ** 2
        if precision == "perturb":
            ref = self.refs.get(view, max_iter, bailout2)
            # Offsets from the reference point are small, so float64 holds them exactly enough.
            off_re = float(view.center_re - ref.re)
            off_im = float(view.center_im - ref.im)
            dx0 = off_re - (W / 2 - 0.5) * dx
            dy0 = off_im + (H / 2 - 0.5) * dx
            self._nu = self.backend.compute_perturb(H, W, dx0, dy0, dx, ref, max_iter, bailout2)
            self.last_reference = ref
        else:
            self._nu = self.backend.compute(H, W, x0, y0, dx, max_iter, bailout2, precision)
        self._ss = ss
        self.last_precision = precision
        self.last_max_iter = max_iter
        return self._nu

    def colorize(self, color: ColorParams, to_host=True, filter_scale=1.0):
        """Colour the last computed buffer; returns RGBA uint8 (h, w, 4) as a
        host array, or with to_host=False as a backend array (for blending).
        filter_scale narrows the palette filter (used with temporal smoothing,
        whose jitter already averages over the pixel area)."""
        if self._nu is None:
            raise RuntimeError("compute() must be called before colorize()")
        lut, cum = self.lut(color.palette)
        filt = filter_scale if color.palette_filter else 0.0
        return self.backend.colorize(self._nu, lut, cum, color.offset, color.density,
                                     MAPPING_CODES[color.mapping], filt,
                                     color.interior, self._ss, to_host)

    def render(self, view: View, params: RenderParams, color: ColorParams,
               w: int, h: int) -> np.ndarray:
        self.compute(view, params, w, h)
        return self.colorize(color)

    def smooth_counts(self) -> np.ndarray:
        """Host copy of the last smooth-count buffer (for tests and tools)."""
        return self.backend.to_host(self._nu)

    def warmup(self):
        """Trigger JIT compilation of all kernels on a tiny image."""
        view, color = View(), ColorParams()
        precisions = ["f64", "f32"] if self.backend.supports_f32 else ["f64"]
        for p in precisions + ["perturb"]:
            self.render(view, RenderParams(precision=p, max_iter=50), color, 32, 24)
        self.render(view, RenderParams(max_iter=50, supersample=2), color, 32, 24)
        from fractaland.engine.temporal import TemporalAccumulator
        acc = TemporalAccumulator(self.backend)
        for _ in range(2):
            acc.blend(self.colorize(color, to_host=False), view, 32, 24)
        self.refs = ReferenceCache(self.refs.background)
        self.refs.start()
        self.last_reference = None


def log_zoom_digits(view: View) -> int:
    """Decimal digits worth showing for the center at the current zoom."""
    return max(6, int(math.ceil(view.log_zoom)) + 6)
