"""Time the iteration step at 1920x1080 for each backend and precision.

    python tools/benchmark.py [--cpu]
"""

import argparse
import pathlib
import sys
import time
from decimal import Decimal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from fractaland.engine.engine import Engine
from fractaland.state import RenderParams, View

VIEWS = {
    "full set": View(),
    "seahorse valley 1e3": View(Decimal("-0.743643887037151"), Decimal("0.131825904205330"), 3.0),
    "seahorse valley 1e9": View(Decimal("-0.743643887037151"), Decimal("0.131825904205330"), 9.0),
    # c = i lies on the boundary with structure at every depth (Misiurewicz point).
    "c = i, 1e20": View(Decimal(0), Decimal(1), 20.0),
    "c = i, 1e50": View(Decimal(0), Decimal(1), 50.0),
    "c = i, 1e200": View(Decimal(0), Decimal(1), 200.0),
}


def time_compute(engine, view, params, w, h, repeats=5):
    engine.compute(view, params, w, h)
    engine.smooth_counts()  # forces completion of the GPU work
    t = time.perf_counter()
    for _ in range(repeats):
        engine.compute(view, params, w, h)
    nu = engine.smooth_counts()
    ms = (time.perf_counter() - t) * 1000 / repeats
    escaped = nu[nu >= 0]
    mean_iter = float(escaped.mean()) if escaped.size else 0.0
    return ms, mean_iter, float((nu < 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="also time the CPU backend")
    args = ap.parse_args()
    w, h = 1920, 1080

    engines = [Engine(prefer_gpu=True)]
    if args.cpu and engines[0].backend.name != "cpu":
        engines.append(Engine(prefer_gpu=False))

    for engine in engines:
        engine.warmup()
        direct = ["f32", "f64"] if engine.backend.supports_f32 else ["f64"]
        for name, view in VIEWS.items():
            # Direct float precisions are meaningless beyond ~1e15.
            for p in (direct if view.log_zoom < 15 else []) + ["perturb"]:
                params = RenderParams(precision=p)
                ms, mean_iter, interior = time_compute(engine, view, params, w, h)
                ref = ""
                if p == "perturb":
                    r = engine.last_reference
                    ref = f"  (reference: {r.length} it, {r.bits} bits, {r.compute_ms:.0f} ms once)"
                print(f"{engine.backend.name:4} {p:7} {name:20} max_iter={params.effective_max_iter(view):6}"
                      f"  mean escape={mean_iter:7.1f}  interior={interior:5.1%}  {ms:7.1f} ms{ref}")
    if len(engines) == 1:
        print("(add --cpu to compare with the CPU backend)")
    np.seterr(all="ignore")


if __name__ == "__main__":
    main()
