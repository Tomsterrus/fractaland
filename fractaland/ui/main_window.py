from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox

from fractaland.engine.engine import Engine
from fractaland.ui.canvas import PRECISIONS, WHEEL_STEP, FractalCanvas
from fractaland.ui.dialogs import IterationDialog

VERSION = "0.2"

RESOLUTIONS = [
    ("Native (100%)", ("fraction", 1.0)),
    ("75%", ("fraction", 0.75)),
    ("50%", ("fraction", 0.5)),
    ("25%", ("fraction", 0.25)),
    None,
    ("720 lines", ("lines", 720)),
    ("1080 lines", ("lines", 1080)),
    ("1440 lines", ("lines", 1440)),
    ("2160 lines (4K)", ("lines", 2160)),
]
PRECISION_LABELS = {"auto": "Auto", "f32": "float32 (fast)", "f64": "float64 (direct)",
                    "perturb": "Perturbation (deep)"}
MAPPINGS = {"linear": "Linear", "log": "Logarithmic"}
SMALL_STEP, LARGE_STEP = 0.02, 0.1


class MainWindow(QMainWindow):
    def __init__(self, engine: Engine):
        super().__init__()
        self.setWindowTitle("Fractaland")
        self.canvas = FractalCanvas(engine)
        self.setCentralWidget(self.canvas)
        self._build_menus()
        self.canvas.setFocus()

    # ---------- menu construction ----------

    def _act(self, menu, text, slot, shortcuts=(), checkable=False, checked=False, group=None):
        a = QAction(text, self)
        if shortcuts:
            a.setShortcuts([QKeySequence(s) for s in shortcuts])
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda _=False: slot())
        if group is not None:
            group.addAction(a)
        if menu is not None:
            menu.addAction(a)
        # Also on the window, so shortcuts work while the menu bar is hidden.
        self.addAction(a)
        return a

    def _radio(self, menu, items, current, slot):
        """items: list of (label, value) or None for a separator."""
        group = QActionGroup(self)
        actions = {}
        for item in items:
            if item is None:
                menu.addSeparator()
                continue
            label, value = item
            a = self._act(menu, label, lambda on, v=value: on and slot(v),
                          checkable=True, checked=(value == current), group=group)
            actions[value] = a
        return actions

    def _build_menus(self):
        c = self.canvas
        mb = self.menuBar()

        m = mb.addMenu("&File")
        self._act(m, "Save &Screenshot", c.screenshot, ["S"])
        m.addSeparator()
        self._act(m, "&Quit", self.close, ["Ctrl+Q"])

        m = mb.addMenu("&View")
        self.fullscreen_action = self._act(m, "&Fullscreen", self._set_fullscreen,
                                           ["F11", "F"], checkable=True)
        self._act(m, "Show &HUD", c.set_show_hud, ["H"], checkable=True, checked=c.show_hud)
        self.help_action = self._act(m, "&Controls Overlay", c.set_show_help, ["F1"],
                                     checkable=True, checked=c.show_help)
        m.addSeparator()
        res = m.addMenu("Render &Resolution")
        self._radio(res, RESOLUTIONS, c.resolution, c.set_resolution)
        self._act(m, "&Antialiasing 2x2 When Idle", c.set_antialias, ["A"],
                  checkable=True, checked=c.idle_supersample > 1)
        self._act(m, "&Temporal Smoothing While Moving", c.set_temporal, ["T"],
                  checkable=True, checked=c.temporal)
        self._act(m, "Palette &Filtering", c.set_palette_filter, ["G"],
                  checkable=True, checked=c.state.color.palette_filter)
        m.addSeparator()
        self._act(m, "R&eset View", c.reset_view, ["R"])
        self._act(None, "Leave Fullscreen", lambda: self._set_fullscreen(False), ["Esc"])

        m = mb.addMenu("&Zoom")
        self._act(m, "Zoom &In", lambda: c.zoom_step(WHEEL_STEP ** 2), ["PgUp"])
        self._act(m, "Zoom &Out", lambda: c.zoom_step(WHEEL_STEP ** -2), ["PgDown"])
        m.addSeparator()
        self._act(m, "Zoom Speed &Faster", lambda: c.scale_zoom_speed(1.25), ["."])
        self._act(m, "Zoom Speed &Slower", lambda: c.scale_zoom_speed(0.8), [","])

        m = mb.addMenu("&Render")
        it = m.addMenu("&Iterations")
        self._act(it, "&Increase (x1.25)", lambda: c.scale_max_iter(1.25), ["+", "="])
        self._act(it, "&Decrease (x0.8)", lambda: c.scale_max_iter(0.8), ["-"])
        it.addSeparator()
        self._act(it, "&Set Iteration Depth...", self._iteration_dialog, ["I"])
        pr = m.addMenu("&Precision")
        self.precision_actions = self._radio(
            pr, [(PRECISION_LABELS[p], p) for p in PRECISIONS],
            c.state.render.precision, c.set_precision)
        pr.addSeparator()
        self._act(pr, "&Cycle Precision", self._cycle_precision, ["P"])

        m = mb.addMenu("&Palette")
        self._act(m, "Rotate &Forward", lambda: c.rotate_palette(SMALL_STEP), ["]"])
        self._act(m, "Rotate &Backward", lambda: c.rotate_palette(-SMALL_STEP), ["["])
        self._act(m, "Rotate Forward (Large)", lambda: c.rotate_palette(LARGE_STEP), ["Ctrl+]"])
        self._act(m, "Rotate Backward (Large)", lambda: c.rotate_palette(-LARGE_STEP), ["Ctrl+["])
        self._act(m, "Reset Rotation", c.reset_rotation)
        m.addSeparator()
        self._act(m, "&Auto Rotate", c.set_auto_rotate, ["Space"], checkable=True,
                  checked=c.auto_rotate)
        self._act(m, "Rotation Speed Faster", lambda: c.scale_rotate_speed(1.25), ["}"])
        self._act(m, "Rotation Speed Slower", lambda: c.scale_rotate_speed(0.8), ["{"])
        m.addSeparator()
        cm = m.addMenu("Colour &Mapping")
        self.mapping_actions = self._radio(cm, list((v, k) for k, v in MAPPINGS.items()),
                                           c.state.color.mapping, c.set_mapping)
        cm.addSeparator()
        self._act(cm, "&Toggle Mapping", self._toggle_mapping, ["L"])

        m = mb.addMenu("&Help")
        m.addAction(self.help_action)
        self._act(m, "&About Fractaland", self._about)

        # A menu bar shown with Alt in fullscreen hides again once its menu closes.
        for a in mb.actions():
            a.menu().aboutToHide.connect(lambda: QTimer.singleShot(0, self._rehide_menu_bar))

    # ---------- handlers ----------

    def _set_fullscreen(self, on):
        if on == self.isFullScreen():
            return
        if on:
            self.menuBar().hide()
            self.showFullScreen()
        else:
            self.showNormal()
            self.menuBar().show()
        self.fullscreen_action.setChecked(on)
        self.canvas.setFocus()

    def toggle_fullscreen(self):
        self._set_fullscreen(not self.isFullScreen())

    def _rehide_menu_bar(self):
        # Moving between menus closes one and opens the next: keep the bar then.
        if self.isFullScreen() and QApplication.activePopupWidget() is None:
            self.menuBar().hide()
            self.canvas.setFocus()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Alt and self.isFullScreen():
            self.menuBar().setVisible(not self.menuBar().isVisible())
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.canvas.engine.refs.shutdown()
        super().closeEvent(e)

    def _cycle_precision(self):
        i = PRECISIONS.index(self.canvas.state.render.precision)
        self.precision_actions[PRECISIONS[(i + 1) % len(PRECISIONS)]].setChecked(True)

    def _toggle_mapping(self):
        current = self.canvas.state.color.mapping
        self.mapping_actions["log" if current == "linear" else "linear"].setChecked(True)

    def _iteration_dialog(self):
        c = self.canvas
        dlg = IterationDialog(c.state.render, c.state.view, self)
        if dlg.exec() == QDialog.Accepted:
            v = dlg.values()
            c.set_iterations(v.max_iter, v.auto_iter, v.iter_growth)

    def _about(self):
        backend = self.canvas.engine.backend.name
        gpu = ""
        if backend == "cuda":
            from numba import cuda
            gpu = cuda.get_current_device().name
            gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
        QMessageBox.about(self, "About Fractaland",
                          f"<b>Fractaland</b> {VERSION}<br>Real-time Mandelbrot viewer<br><br>"
                          f"Backend: {backend} {gpu}")
