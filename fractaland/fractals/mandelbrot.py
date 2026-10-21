"""Per-pixel Mandelbrot iteration, written once and compiled for both CPU
(numba.njit) and GPU (numba.cuda.jit device function).

Numba specialises the function on argument types, so calling it with float32
arguments yields a float32 inner loop. To keep it that way, the loop avoids
Python float literals (which would promote to float64): zero is made as
`cr - cr`, and doubling as `t + t`.
"""

import math


def mandelbrot_pixel(cr, ci, max_iter, bailout2, eps2):
    """Return the smooth iteration count of c = cr + i*ci, or -1.0 if c is
    judged to be inside the set.

    bailout2: squared escape radius; eps2: squared tolerance of the
    periodicity check (both of the same float type as cr, ci).
    """
    # Main cardioid and period-2 bulb: analytically inside, skip iterating.
    # (Done in float64 once per pixel; cheap.)
    x = cr - 0.25
    q = x * x + ci * ci
    if q * (q + x) <= 0.25 * ci * ci:
        return -1.0
    if (cr + 1.0) * (cr + 1.0) + ci * ci <= 0.0625:
        return -1.0

    zr = cr - cr
    zi = cr - cr
    zr2 = cr - cr
    zi2 = cr - cr
    # Periodicity check: compare z against a saved value whose save interval
    # doubles; if the orbit returns (within eps), it is an attracting cycle.
    sr = cr - cr
    si = cr - cr
    period_len = 8
    n_since = 0
    n = 0
    while n < max_iter:
        t = zr * zi
        zi = t + t + ci
        zr = zr2 - zi2 + cr
        zr2 = zr * zr
        zi2 = zi * zi
        n += 1
        if zr2 + zi2 > bailout2:
            # Smooth count: n + 1 - log2(log|z|), with log|z| = 0.5*log(|z|^2).
            log_zn = 0.5 * math.log(float(zr2 + zi2))
            return n + 1.0 - math.log(log_zn / math.log(2.0)) / math.log(2.0)
        dr = zr - sr
        di = zi - si
        if dr * dr + di * di < eps2:
            return -1.0
        n_since += 1
        if n_since == period_len:
            n_since = 0
            period_len += period_len
            sr = zr
            si = zi
    return -1.0


def mandelbrot_perturb_pixel(dcr, dci, Zr, Zi, max_iter, bailout2):
    """Smooth iteration count of the point C + d, where C is the reference
    point whose orbit Z is given (float64 arrays, Z[0] = 0, Z[1] = C).

    The pixel's orbit is written as z_n = Z_m + e and only the small
    difference e is iterated: e <- (2 Z_m + e) e + d. When z comes closer to 0
    than e (or the reference orbit runs out), the pixel restarts from the
    beginning of the reference orbit with e = z and m = 0 ("rebasing",
    Zhuoran 2021); since Z[1] = C, this continues the exact iteration.
    """
    last = Zr.shape[0] - 1
    er = 0.0
    ei = 0.0
    m = 0
    n = 0
    while n < max_iter:
        tr = 2.0 * Zr[m] + er
        ti = 2.0 * Zi[m] + ei
        ner = tr * er - ti * ei + dcr
        ei = tr * ei + ti * er + dci
        er = ner
        m += 1
        n += 1
        zr = Zr[m] + er
        zi = Zi[m] + ei
        mag = zr * zr + zi * zi
        if mag > bailout2:
            log_zn = 0.5 * math.log(mag)
            return n + 1.0 - math.log(log_zn / math.log(2.0)) / math.log(2.0)
        if mag < er * er + ei * ei or m == last:
            er = zr
            ei = zi
            m = 0
    return -1.0
