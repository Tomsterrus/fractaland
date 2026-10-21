from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QLabel, QSpinBox)

from fractaland.state import RenderParams, View


class IterationDialog(QDialog):
    """Set the iteration limit and how it grows with zoom."""

    def __init__(self, params: RenderParams, view: View, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Iteration Depth")
        self._view = view

        self.base = QSpinBox()
        self.base.setRange(16, 10_000_000)
        self.base.setSingleStep(100)
        self.base.setGroupSeparatorShown(True)
        self.base.setValue(params.max_iter)

        self.auto = QCheckBox("Increase with zoom")
        self.auto.setChecked(params.auto_iter)

        self.growth = QDoubleSpinBox()
        self.growth.setRange(0.0, 100_000.0)
        self.growth.setDecimals(1)
        self.growth.setSingleStep(10.0)
        self.growth.setValue(params.iter_growth)
        self.growth.setToolTip("limit = base + growth x (log10 zoom)^2")

        self.effective = QLabel()

        form = QFormLayout(self)
        form.addRow("Base iterations:", self.base)
        form.addRow("", self.auto)
        form.addRow("Growth coefficient:", self.growth)
        form.addRow("Effective at current zoom:", self.effective)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.base.valueChanged.connect(self._update)
        self.auto.toggled.connect(self._update)
        self.growth.valueChanged.connect(self._update)
        self._update()

    def values(self) -> RenderParams:
        """Only max_iter, auto_iter and iter_growth are meaningful."""
        return RenderParams(max_iter=self.base.value(), auto_iter=self.auto.isChecked(),
                            iter_growth=self.growth.value())

    def _update(self):
        self.growth.setEnabled(self.auto.isChecked())
        self.effective.setText(f"{self.values().effective_max_iter(self._view):,}")
