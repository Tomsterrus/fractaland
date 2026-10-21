import pathlib
import sys
import warnings
from decimal import Decimal

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

from fractaland.color.palettes import build_lut, get_lut, PALETTES
from fractaland.engine.engine import Engine
from fractaland.state import ColorParams, FrameState, RenderParams, View

VIEWS = [
    View(),
    View(Decimal("-0.743643887037151"), Decimal("0.131825904205330"), 3.0),
    View(Decimal("-0.743643887037151"), Decimal("0.131825904205330"), 8.0),
]


@pytest.fixture(scope="module")
def cpu():
    return Engine(prefer_gpu=False)


@pytest.fixture(scope="module")
def gpu():
    e = Engine(prefer_gpu=True)
    if e.backend.name != "cuda":
        pytest.skip("no CUDA device")
    return e


def counts(engine, view, precision="f64", w=160, h=120):
    engine.compute(view, RenderParams(precision=precision), w, h)
    return engine.smooth_counts().copy()


@pytest.mark.parametrize("view", VIEWS)
def test_gpu_f64_matches_cpu(cpu, gpu, view):
    a, b = counts(cpu, view), counts(gpu, view)
    same_class = (a < 0) == (b < 0)
    assert same_class.mean() > 0.999
    both = (a >= 0) & (b >= 0)
    close = np.abs(a[both] - b[both]) < 1e-3
    assert close.mean() > 0.999


def exact_smooth(cr, ci, max_iter, prec=300, bailout2=65536.0):
    """Reference value computed directly at high precision with mpmath."""
    import math
    import mpmath
    with mpmath.workprec(prec):
        c = mpmath.mpc(mpmath.mpf(cr), mpmath.mpf(ci))
        z = mpmath.mpc(0)
        for n in range(1, max_iter + 1):
            z = z * z + c
            m = float(abs(z)) ** 2
            if m > bailout2:
                return n + 1 - math.log(0.5 * math.log(m) / math.log(2)) / math.log(2)
    return -1.0


def sample_against_exact(engine, view, n_samples=12, w=96, h=72, seed=0):
    """Max |engine - exact| over random escaping pixels."""
    import mpmath
    params = RenderParams(precision="perturb")
    engine.compute(view, params, w, h)
    nu = engine.smooth_counts()
    mi = params.effective_max_iter(view)
    rng = np.random.default_rng(seed)
    errs = []
    with mpmath.workprec(view.log_zoom * 3.33 + 80):
        dx = mpmath.mpf(view.height) / h
        for _ in range(n_samples):
            i, j = int(rng.integers(h)), int(rng.integers(w))
            cr = mpmath.mpf(str(view.center_re)) + (j - (w / 2 - 0.5)) * dx
            ci = mpmath.mpf(str(view.center_im)) - (i - (h / 2 - 0.5)) * dx
            t = exact_smooth(cr, ci, mi, prec=int(view.log_zoom * 3.33) + 80)
            if t >= 0 and nu[i, j] >= 0:
                errs.append(abs(nu[i, j] - t))
    assert errs, "no escaping samples"
    return max(errs)


# c = i is a Misiurewicz point: exact coordinates and structure at every depth.
MISIUREWICZ = (Decimal(0), Decimal(1))


@pytest.mark.parametrize("log_zoom", [9.0, 30.0])
def test_perturbation_matches_exact(cpu, log_zoom):
    view = View(MISIUREWICZ[0], MISIUREWICZ[1], log_zoom)
    assert sample_against_exact(cpu, view) < 1e-3


def test_perturbation_seahorse_matches_exact(cpu):
    assert sample_against_exact(cpu, VIEWS[2]) < 1e-3


def test_gpu_perturbation_matches_cpu(cpu, gpu):
    view = View(MISIUREWICZ[0], MISIUREWICZ[1], 20.0)
    a = counts(cpu, view, "perturb")
    b = counts(gpu, view, "perturb")
    assert ((a < 0) == (b < 0)).mean() > 0.999
    both = (a >= 0) & (b >= 0)
    assert (np.abs(a[both] - b[both]) < 1e-3).mean() > 0.999


def test_deep_view_has_detail(cpu):
    view = View(MISIUREWICZ[0], MISIUREWICZ[1], 50.0)
    nu = counts(cpu, view, "perturb")
    assert (nu >= 0).mean() > 0.5
    assert nu[nu >= 0].std() > 0.5


def test_reference_cache_reuse():
    from fractaland.fractals.reference import ReferenceCache
    cache = ReferenceCache()
    v = View(MISIUREWICZ[0], MISIUREWICZ[1], 12.0)
    r1 = cache.get(v, 1000, 65536.0)
    v.zoom_about(0.1, 0.1, 10.0)          # zoom toward a nearby point
    assert cache.get(v, 1000, 65536.0) is r1
    v.pan(5000.0, 0.0)                    # far beyond 1000 view heights
    assert cache.get(v, 1000, 65536.0) is not r1
    deep = View(MISIUREWICZ[0], MISIUREWICZ[1], 60.0)
    assert cache.get(deep, 1000, 65536.0).bits > r1.bits


def test_gpu_f32_close_at_shallow_zoom(cpu, gpu):
    a, b = counts(cpu, VIEWS[0]), counts(gpu, VIEWS[0], "f32")
    assert ((a < 0) == (b < 0)).mean() > 0.99


def test_auto_precision_switches(gpu):
    p = RenderParams()
    assert gpu.choose_precision(VIEWS[0], p, 1920, 1080) == "f32"
    assert gpu.choose_precision(VIEWS[2], p, 1920, 1080) == "perturb"


def test_supersampled_render_shape(cpu):
    rgba = cpu.render(View(), RenderParams(supersample=2), ColorParams(), 64, 48)
    assert rgba.shape == (48, 64, 4) and rgba.dtype == np.uint8


def test_lut_is_seamless():
    lut = get_lut("default")
    # The step across the wrap point is no larger than typical steps inside.
    steps = np.abs(np.diff(np.vstack([lut, lut[:1]]), axis=0)).max(axis=1)
    assert steps[-1] <= steps.max()
    first_stop = np.array(PALETTES["default"][0][1], dtype=np.float32)
    assert np.allclose(lut[0], first_stop)


def test_lut_passes_through_stops():
    stops = [(0.0, (0, 0, 0)), (0.5, (255, 255, 255))]
    lut = build_lut(stops, 1024)
    assert np.allclose(lut[512], 255) and np.allclose(lut[0], 0)


def test_palette_rotation_wraps(cpu):
    cpu.compute(View(), RenderParams(), 64, 48)
    a = cpu.colorize(ColorParams(offset=0.25))
    b = cpu.colorize(ColorParams(offset=1.25))
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_zoom_about_keeps_point_fixed():
    v = View()
    fx, fy = 0.3, -0.2
    before = (float(v.center_re) + fx * v.height, float(v.center_im) - fy * v.height)
    v.zoom_about(fx, fy, 7.0)
    after = (float(v.center_re) + fx * v.height, float(v.center_im) - fy * v.height)
    assert np.allclose(before, after, atol=1e-15)
    assert v.log_zoom == pytest.approx(np.log10(7.0))


def test_state_round_trip():
    s = FrameState()
    s.view.zoom_about(0.1, 0.1, 1e6)
    s.color.offset = 0.4
    s.render.iter_growth = 12.5
    assert FrameState.from_dict(s.to_dict()) == s


def test_old_state_without_iter_growth_loads():
    d = FrameState().to_dict()
    del d["render"]["iter_growth"]
    assert FrameState.from_dict(d).render.iter_growth == 50.0


def test_effective_max_iter():
    v = View(log_zoom=3.0)
    assert RenderParams(max_iter=100, iter_growth=10.0).effective_max_iter(v) == 190
    assert RenderParams(max_iter=100, auto_iter=False).effective_max_iter(v) == 100
    assert RenderParams(max_iter=100).effective_max_iter(View(log_zoom=-2.0)) == 100


def test_render_size():
    from fractaland.ui.canvas import render_size
    assert render_size(1280, 800, 1.25, ("fraction", 1.0)) == (1600, 1000)
    assert render_size(1280, 800, 1.25, ("fraction", 0.5)) == (800, 500)
    assert render_size(1280, 800, 1.25, ("lines", 1080)) == (1728, 1080)
    assert render_size(1280, 800, 1.0, ("lines", 1080), scale=0.5) == (864, 540)
    assert render_size(0, 0, 1.0, ("fraction", 1.0)) == (1, 1)


# ---------- palette filter ----------

def test_shade_filtered_limits():
    from fractaland.color.palettes import build_cumulative_lut, shade_filtered, shade_point
    lut = get_lut("default")
    cum = build_cumulative_lut(lut)
    for t in (0.0, 0.123, 0.5, 0.9999):
        assert np.allclose(shade_filtered(t, 0.0, lut, cum), shade_point(t, lut))
        # a one-entry box equals the point interpolation (continuity at the switch)
        assert np.allclose(shade_filtered(t, 1.0 / len(lut) + 1e-12, lut, cum),
                           shade_point(t, lut), atol=1e-3)
        assert np.allclose(shade_filtered(t, 1.0, lut, cum), lut.mean(axis=0), atol=1e-3)
        assert np.allclose(shade_filtered(t, 5.0, lut, cum), lut.mean(axis=0), atol=1e-3)


def test_filtered_colorize_cpu_matches_gpu(cpu, gpu):
    view = VIEWS[1]
    color = ColorParams(offset=0.3)
    for ss in (1, 2):
        params = RenderParams(precision="f64", supersample=ss)
        a = cpu.render(view, params, color, 160, 120).astype(int)
        b = gpu.render(view, params, color, 160, 120).astype(int)
        assert (np.abs(a - b) <= 1).mean() > 0.999


def test_filter_off_is_unchanged_in_smooth_areas(cpu):
    cpu.compute(View(), RenderParams(), 160, 120)
    a = cpu.colorize(ColorParams(palette_filter=True)).astype(int)
    b = cpu.colorize(ColorParams(palette_filter=False)).astype(int)
    # The full-set view is mostly smooth: filtering changes only a small part.
    assert (np.abs(a - b).max(axis=2) <= 1).mean() > 0.9


# ---------- temporal accumulation ----------

def test_frame_transform_keeps_zoom_point_fixed():
    from fractaland.engine.temporal import frame_transform
    w, h = 400, 300
    old = View(Decimal("-0.75"), Decimal("0.1"), 2.0)
    new = View(old.center_re, old.center_im, old.log_zoom)
    px, py = 300.0, 80.0                      # zoom toward this pixel
    fx, fy = (px - (w / 2 - 0.5)) / h, (py - (h / 2 - 0.5)) / h
    new.zoom_about(fx, fy, 1.5)
    s, tx, ty = frame_transform(old, new, w, h)
    assert s == pytest.approx(1 / 1.5)
    assert s * px + tx == pytest.approx(px, abs=1e-6)
    assert s * py + ty == pytest.approx(py, abs=1e-6)


def _blend_once(backend, cur, hist, s, tx, ty, alpha, valid):
    h, w = hist.shape[:2]
    if backend.name == "cuda":
        from numba import cuda
        d_cur, d_in, d_out = cuda.to_device(cur), cuda.to_device(hist), cuda.device_array_like(hist)
        disp = backend.blend_history(d_cur, d_in, d_out, s, tx, ty, alpha, valid, h, w)
        return disp, d_out.copy_to_host()
    out = np.empty_like(hist)
    disp = backend.blend_history(cur, hist, out, s, tx, ty, alpha, valid, h, w)
    return disp, out


def test_blend_history_cpu_matches_gpu(cpu, gpu):
    rng = np.random.default_rng(3)
    cur = rng.integers(0, 256, (30, 40, 4), dtype=np.uint8)
    hist = rng.uniform(0, 255, (60, 80, 3)).astype(np.float32)
    a = _blend_once(cpu.backend, cur, hist, 0.9, 4.0, 3.0, 0.3, 1)
    b = _blend_once(gpu.backend, cur, hist, 0.9, 4.0, 3.0, 0.3, 1)
    assert np.allclose(a[1], b[1], atol=1e-3)
    assert (np.abs(a[0].astype(int) - b[0].astype(int)) <= 1).all()


def test_blend_history_weights(cpu):
    cur = np.full((10, 10, 4), 200, dtype=np.uint8)
    hist = np.full((10, 10, 3), 100.0, dtype=np.float32)
    _, out = _blend_once(cpu.backend, cur, hist, 1.0, 0.0, 0.0, 0.25, 1)
    assert np.allclose(out, 0.25 * 200 + 0.75 * 100)
    _, out = _blend_once(cpu.backend, cur, hist, 1.0, 0.0, 0.0, 0.25, 0)   # no valid history
    assert np.allclose(out, 200)
    _, out = _blend_once(cpu.backend, cur, hist, 1.0, 50.0, 0.0, 0.25, 1)  # history out of frame
    assert np.allclose(out, 200)


def test_accumulator_converges_to_still_image(cpu):
    from fractaland.engine.temporal import TemporalAccumulator
    acc = TemporalAccumulator(cpu.backend)
    v = View()
    cpu.compute(v, RenderParams(), 64, 48)
    frame = cpu.colorize(ColorParams(), to_host=False)
    first = acc.blend(frame, v, 64, 48)
    for _ in range(5):
        out = acc.blend(frame, v, 64, 48)
    assert acc.last_alpha == pytest.approx(0.2)
    assert (np.abs(out.astype(int) - first.astype(int)) <= 1).all()


# ---------- background reference ----------

def test_background_reference_keeps_old_until_ready():
    from fractaland.fractals.reference import ReferenceCache
    cache = ReferenceCache(background=True)
    try:
        v = View(MISIUREWICZ[0], MISIUREWICZ[1], 12.0)
        r1 = cache.get(v, 1000, 65536.0)            # first orbit: synchronous
        assert not cache.pending
        deep = View(MISIUREWICZ[0], MISIUREWICZ[1], 80.0)
        assert cache.get(deep, 1000, 65536.0) is r1  # old orbit while the new one computes
        assert cache.pending
        cache.wait()
        r2 = cache.get(deep, 1000, 65536.0)
        assert r2 is not r1 and cache.recomputed and r2.bits > r1.bits
        assert not cache.pending
    finally:
        cache.shutdown()
