"""Render one frame to PNG without the GUI.

    python tools/render_png.py out.png [--cpu] [--size 1600x1200] [--state view.json]
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from PIL import Image

from fractaland.engine.engine import Engine
from fractaland.state import FrameState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--size", default="1600x1200")
    ap.add_argument("--state", help="JSON file saved with a screenshot")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    args = ap.parse_args()

    w, h = map(int, args.size.lower().split("x"))
    state = FrameState()
    if args.state:
        state = FrameState.from_dict(json.loads(pathlib.Path(args.state).read_text()))
    state.render.supersample = args.ss

    engine = Engine(prefer_gpu=not args.cpu)
    engine.warmup()
    t = time.perf_counter()
    rgba = engine.render(state.view, state.render, state.color, w, h)
    print(f"{engine.backend.name} {engine.last_precision} {w}x{h} ss={args.ss}: "
          f"{(time.perf_counter() - t) * 1000:.1f} ms")
    Image.fromarray(rgba[:, :, :3]).save(args.out)


if __name__ == "__main__":
    main()
