"""Measure flicker during a continuous zoom, using the viewer's pipeline.

    python tools/flicker.py [--cpu] [--frames 30] [--scale 1.0]

flicker: share of pixels whose colour differs by more than 48 of 255 levels
from the previous frame after compensating for the zoom.
error:   mean colour difference (0-255) of the last frame from a 4x4
supersampled rendering of the same view.
"""

import argparse
import pathlib
import sys
from decimal import Decimal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from fractaland.engine.engine import Engine
from fractaland.engine.temporal import FILTER_SCALE, TemporalAccumulator, jittered_view
from fractaland.state import ColorParams, RenderParams, View

W, H = 480, 300
STEP = 2 ** (1 / 60)  # default hold-to-zoom speed (2x per second) at 60 fps
THRESHOLD = 48

LOCATIONS = {
    "seahorse 1e2": ("-0.743643887037151", "0.131825904205330", 2.0),
    "seahorse 1e7": ("-0.743643887037151", "0.131825904205330", 7.0),
    "c = i 1e20": ("0", "1", 20.0),
}


def bilinear(img, xs, ys):
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = (xs - x0)[..., None], (ys - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def flicker(frames):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    px, py = cx + (xs - cx) / STEP, cy + (ys - cy) / STEP
    inside = (px > 2) & (px < W - 3) & (py > 2) & (py < H - 3)
    shares = [(np.abs(bilinear(a, px, py) - b).max(axis=2) > THRESHOLD)[inside].mean()
              for a, b in zip(frames, frames[1:])]
    return float(np.mean(shares))


def run(engine, views, max_iter, scale, palette_filter, temporal):
    """Frames as the viewer shows them while moving."""
    params = RenderParams(max_iter=max_iter, auto_iter=False)
    color = ColorParams(palette_filter=palette_filter)
    w, h = max(1, round(W * scale)), max(1, round(H * scale))
    acc = TemporalAccumulator(engine.backend)
    frames = []
    for v in views:
        view = jittered_view(v, *acc.jitter(), h) if temporal else v
        engine.compute(view, params, w, h)
        if temporal:
            rgba = acc.blend(engine.colorize(color, to_host=False, filter_scale=FILTER_SCALE),
                             v, W, H)
        else:
            rgba = engine.colorize(color)
            if (w, h) != (W, H):
                ys, xs = np.mgrid[0:H, 0:W]
                rgba = bilinear(rgba.astype(np.float32), (xs + 0.5) * w / W - 0.5,
                                (ys + 0.5) * h / H - 0.5)
        frames.append(np.asarray(rgba, dtype=np.float32)[:, :, :3])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0, help="render resolution while moving")
    args = ap.parse_args()
    engine = Engine(prefer_gpu=not args.cpu)
    engine.warmup()
    configs = [("before (1 sample)", False, False), ("palette filter", True, False),
               ("filter + temporal", True, True)]
    for name, (cre, cim, lz) in LOCATIONS.items():
        v = View(Decimal(cre), Decimal(cim), lz)
        views = []
        for _ in range(args.frames):
            views.append(View(v.center_re, v.center_im, v.log_zoom))
            v.zoom_about(0.0, 0.0, STEP)
        max_iter = RenderParams().effective_max_iter(views[0])
        truth = engine.render(views[-1], RenderParams(max_iter=max_iter, auto_iter=False,
                                                      supersample=4),
                              ColorParams(palette_filter=False), W, H)[:, :, :3].astype(np.float32)
        print(name)
        for label, filt, temporal in configs:
            frames = run(engine, views, max_iter, args.scale, filt, temporal)
            skip = 6 if temporal else 0  # let the history fill
            err = np.abs(frames[-1] - truth).mean()
            print(f"   {label:20} flicker {flicker(frames[skip:]):6.2%}   error {err:5.1f}")


if __name__ == "__main__":
    main()
