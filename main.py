"""Fractaland: real-time Mandelbrot viewer.

    python main.py [--cpu] [--fullscreen] [--size 1280x800]
"""

import argparse
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from fractaland.engine.engine import Engine
from fractaland.ui.main_window import MainWindow


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", action="store_true", help="use the CPU backend even if a GPU is present")
    ap.add_argument("--fullscreen", action="store_true", help="start in fullscreen (F11 toggles)")
    ap.add_argument("--size", default="1280x800", help="initial window size, e.g. 1600x900")
    args = ap.parse_args()

    # Use the exact display scale (e.g. 1.25) so frames map 1:1 to physical pixels.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)

    engine = Engine(prefer_gpu=not args.cpu, background_reference=True)
    print(f"backend: {engine.backend.name}; compiling kernels (slow only on first run)...")
    t = time.perf_counter()
    engine.warmup()
    print(f"ready in {time.perf_counter() - t:.1f} s")

    win = MainWindow(engine)
    w, h = map(int, args.size.lower().split("x"))
    win.resize(w, h)
    if args.fullscreen:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
