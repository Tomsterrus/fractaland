"""Palettes: colour stops -> cyclic lookup table (LUT), plus the per-pixel
shading functions shared by the CPU and GPU colouring kernels.

Helpers called from other compiled functions are marked register_jitable,
which makes them callable from both numba.njit and CUDA code."""

import math

import numpy as np
from numba.extending import register_jitable

LUT_SIZE = 1024

MAPPING_CODES = {"linear": 0, "log": 1}

# Stops: (position in [0, 1), (r, g, b)). The palette is cyclic: after the
# last stop it blends back into the first.
PALETTES = {
    # Classic deep blue -> white -> orange -> near-black cycle.
    "default": [
        (0.0, (0, 7, 100)),
        (0.16, (32, 107, 203)),
        (0.42, (237, 255, 255)),
        (0.6425, (255, 170, 0)),
        (0.8575, (0, 2, 0)),
    ],
}


def build_lut(stops, size=LUT_SIZE):
    """Return a (size, 3) float32 LUT with values 0..255.

    Between stops the blend uses smoothstep easing, which keeps the gradient
    free of visible kinks at the stops without overshooting the stop colours.
    """
    pos = np.array([p for p, _ in stops] + [stops[0][0] + 1.0])
    cols = np.array([c for _, c in stops] + [stops[0][1]], dtype=np.float64)
    t = np.arange(size) / size
    # Shift t so that it lies within [pos[0], pos[0] + 1).
    t = (t - pos[0]) % 1.0 + pos[0]
    k = np.searchsorted(pos, t, side="right") - 1
    k = np.clip(k, 0, len(stops) - 1)
    u = (t - pos[k]) / (pos[k + 1] - pos[k])
    u = u * u * (3.0 - 2.0 * u)
    lut = cols[k] + (cols[k + 1] - cols[k]) * u[:, None]
    return lut.astype(np.float32)


def get_lut(name):
    return build_lut(PALETTES[name])


def build_cumulative_lut(lut):
    """Running sums over three copies of the cyclic LUT, shape (3N + 1, 3),
    float64. The average colour over any palette interval then takes two
    lookups (see shade_filtered)."""
    tripled = np.vstack([lut, lut, lut]).astype(np.float64)
    return np.concatenate([np.zeros((1, 3)), np.cumsum(tripled, axis=0)])


@register_jitable
def palette_pos(nu, density, mapping):
    """Position in the palette (in palette lengths, not yet wrapped) of a
    smooth iteration count nu >= 0."""
    if mapping == 1:
        return math.log(nu + 1.0) * density * 16.0
    return nu * density


@register_jitable
def palette_gradient(nu, y, x, p, density, mapping):
    """Largest palette-position difference between sample (y, x), whose
    position is p, and its 4 neighbours; interior neighbours are ignored."""
    h = nu.shape[0]
    w = nu.shape[1]
    g = 0.0
    if x > 0 and nu[y, x - 1] >= 0.0:
        g = max(g, abs(palette_pos(nu[y, x - 1], density, mapping) - p))
    if x < w - 1 and nu[y, x + 1] >= 0.0:
        g = max(g, abs(palette_pos(nu[y, x + 1], density, mapping) - p))
    if y > 0 and nu[y - 1, x] >= 0.0:
        g = max(g, abs(palette_pos(nu[y - 1, x], density, mapping) - p))
    if y < h - 1 and nu[y + 1, x] >= 0.0:
        g = max(g, abs(palette_pos(nu[y + 1, x], density, mapping) - p))
    return g


@register_jitable
def shade_point(t, lut):
    """Colour at palette position t in [0, 1): linear interpolation in the
    cyclic LUT (entry k sits at position k / N)."""
    n = lut.shape[0]
    f = t * n
    i0 = int(f)
    if i0 >= n:
        i0 = n - 1
    i1 = i0 + 1
    if i1 == n:
        i1 = 0
    w = f - i0
    r = lut[i0, 0] + (lut[i1, 0] - lut[i0, 0]) * w
    g = lut[i0, 1] + (lut[i1, 1] - lut[i0, 1]) * w
    b = lut[i0, 2] + (lut[i1, 2] - lut[i0, 2]) * w
    return r, g, b


@register_jitable
def shade_filtered(t, width, lut, cum):
    """Average colour of the palette over [t - width/2, t + width/2].

    This is the colour a pixel would show if it were sampled infinitely
    finely while the palette position varies by `width` across it. For
    widths up to one LUT entry it equals shade_point exactly, so smooth areas
    are unchanged; for width >= 1 (the palette repeats within the pixel) it
    is the mean palette colour.
    """
    n = lut.shape[0]
    if width > 1.0:
        width = 1.0
    span = width * n
    if span <= 1.0:
        return shade_point(t, lut)
    # A box over the piecewise-constant entries is centred half an entry to
    # the right to line up with the point interpolation above.
    a = t * n + 0.5 - 0.5 * span
    a = a - math.floor(a / n) * n
    b = a + span
    ia = int(a)
    ib = int(b)
    fa = a - ia
    fb = b - ib
    r = ((cum[ib, 0] + (cum[ib + 1, 0] - cum[ib, 0]) * fb)
         - (cum[ia, 0] + (cum[ia + 1, 0] - cum[ia, 0]) * fa)) / span
    g = ((cum[ib, 1] + (cum[ib + 1, 1] - cum[ib, 1]) * fb)
         - (cum[ia, 1] + (cum[ia + 1, 1] - cum[ia, 1]) * fa)) / span
    bl = ((cum[ib, 2] + (cum[ib + 1, 2] - cum[ib, 2]) * fb)
          - (cum[ia, 2] + (cum[ia + 1, 2] - cum[ia, 2]) * fa)) / span
    return r, g, bl


def shade_sample(nu, y, x, lut, cum, offset, density, mapping, filt):
    """Colour of supersample (y, x) of the smooth-count buffer `nu`, which
    must be escaped (nu >= 0). With filt > 0, the palette is averaged over
    filt times the local rate of change (palette_gradient); 1 covers one
    pixel, 0 switches the filter off."""
    v = nu[y, x]
    p = palette_pos(v, density, mapping)
    t = p + offset
    t = t - math.floor(t)
    if filt > 0.0:
        g = palette_gradient(nu, y, x, p, density, mapping)
        return shade_filtered(t, filt * g, lut, cum)
    return shade_point(t, lut)
