"""Realtime cross-correlation of the live acquisition against a reference image."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Collapse (H, W[, C]) to a float32 (H, W) image."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2) if a.shape[2] >= 3 else a[..., 0]
    return a.astype(np.float32, copy=False)


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample by an integer *factor*."""
    if factor <= 1:
        return img
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def load_reference(path: str) -> np.ndarray:
    """Load a reference image (.npy or any Qt-readable format) as float32 gray."""
    if path.lower().endswith(".npy"):
        return to_gray(np.load(path))
    qimg = QImage(path)
    if qimg.isNull():
        raise ValueError(f"Cannot read image: {path}")
    qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale16)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    ptr.setsize(qimg.sizeInBytes())
    stride = qimg.bytesPerLine() // 2
    buf = np.frombuffer(ptr, dtype=np.uint16).reshape(h, stride)[:, :w]
    return buf.astype(np.float32)


class ReferenceCorrelator:
    """FFT-based zero-mean normalised cross-correlation against a fixed reference.

    The reference is cropped/zero-padded to the frame size on first use. The score is
    the correlation peak normalised to [-1, 1]; (dx, dy) is the peak's shift in
    full-resolution pixels (circular correlation).
    """

    def __init__(self, reference: np.ndarray, factor: int = 1):
        self.factor = max(1, int(factor))
        ref = downsample(to_gray(reference), self.factor)
        self._ref_src = ref
        self._shape: Optional[tuple[int, int]] = None
        self._fref = None
        self._ref_norm = 1.0

    def _prepare(self, shape: tuple[int, int]) -> None:
        h, w = shape
        ref = self._ref_src[:h, :w]
        ref = ref - ref.mean()
        padded = np.zeros(shape, np.float32)
        padded[:ref.shape[0], :ref.shape[1]] = ref
        self._fref = np.fft.rfft2(padded)
        self._ref_norm = float(np.sqrt((ref ** 2).sum())) or 1.0
        self._shape = shape

    def correlate(self, frame: np.ndarray):
        """Return (score, dx, dy, zero_lag_score)."""
        img = downsample(to_gray(frame), self.factor)
        if self._shape != img.shape:
            self._prepare(img.shape)
        img = img - img.mean()
        norm = float(np.sqrt((img ** 2).sum())) or 1.0
        denom = norm * self._ref_norm
        cc = np.fft.irfft2(np.fft.rfft2(img) * np.conj(self._fref), s=img.shape) / denom
        iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape)
        h, w = cc.shape
        dy = iy - h if iy > h // 2 else iy
        dx = ix - w if ix > w // 2 else ix
        return float(cc[iy, ix]), dx * self.factor, dy * self.factor, float(cc[0, 0])


class CorrelationTab(QWidget):
    """Tab plotting the correlation between live frames and a reference image."""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._latest: Optional[np.ndarray] = None
        self._processed_id = -1
        self._frame_id = 0
        self._correlator: Optional[ReferenceCorrelator] = None
        self._reference: Optional[np.ndarray] = None
        self._t0 = time.monotonic()
        self._t = deque(maxlen=2000)
        self._score = deque(maxlen=2000)
        self._zero = deque(maxlen=2000)
        self._dx = deque(maxlen=2000)
        self._dy = deque(maxlen=2000)

        controls = QGroupBox("Reference")
        form = QHBoxLayout(controls)
        self.open_btn = QPushButton("Load reference...")
        self.use_btn = QPushButton("Use current frame")
        self.clear_btn = QPushButton("Clear history")
        self.running = QCheckBox("Running")
        self.running.setChecked(True)
        self.ds_spin = QSpinBox()
        self.ds_spin.setRange(1, 32)
        self.ds_spin.setValue(4)
        self.ds_spin.setToolTip("Block-mean downsampling before correlating (speed)")
        self.hist_spin = QSpinBox()
        self.hist_spin.setRange(10, 2000)
        self.hist_spin.setValue(300)
        self.hist_spin.setToolTip("Number of samples shown")
        for w in (self.open_btn, self.use_btn, self.clear_btn, self.running):
            form.addWidget(w)
        form.addWidget(QLabel("Downsample:"))
        form.addWidget(self.ds_spin)
        form.addWidget(QLabel("History:"))
        form.addWidget(self.hist_spin)
        form.addStretch(1)

        self.info = QLabel("No reference loaded.")
        self.readout = QLabel("-")

        glw = pg.GraphicsLayoutWidget()
        self.score_plot = glw.addPlot(row=0, col=0, title="Correlation with reference")
        self.score_plot.setLabel("left", "normalised cross-correlation")
        self.score_plot.setLabel("bottom", "time", units="s")
        self.score_plot.showGrid(x=True, y=True, alpha=0.3)
        self.score_plot.addLegend()
        self.peak_curve = self.score_plot.plot(pen=pg.mkPen("c", width=2), name="peak")
        self.zero_curve = self.score_plot.plot(pen=pg.mkPen("y"), name="zero lag")
        self.shift_plot = glw.addPlot(row=1, col=0, title="Shift of correlation peak")
        self.shift_plot.setLabel("left", "shift", units="px")
        self.shift_plot.setLabel("bottom", "time", units="s")
        self.shift_plot.showGrid(x=True, y=True, alpha=0.3)
        self.shift_plot.addLegend()
        self.shift_plot.setXLink(self.score_plot)
        self.dx_curve = self.shift_plot.plot(pen=pg.mkPen("r"), name="dx")
        self.dy_curve = self.shift_plot.plot(pen=pg.mkPen("g"), name="dy")

        lay = QVBoxLayout(self)
        lay.addWidget(controls)
        lay.addWidget(self.info)
        lay.addWidget(self.readout)
        lay.addWidget(glw, 1)

        self.open_btn.clicked.connect(self._open_reference)
        self.use_btn.clicked.connect(self.use_current_frame)
        self.clear_btn.clicked.connect(self.clear_history)
        self.ds_spin.valueChanged.connect(self._rebuild_correlator)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._process)
        self._timer.start()

    # -- inputs ---------------------------------------------------------------
    def set_frame(self, arr) -> None:
        """Receive the latest live frame; correlation runs on a timer, not per frame."""
        self._latest = arr
        self._frame_id += 1

    def clear_frame(self) -> None:
        self._latest = None

    def set_reference(self, ref: np.ndarray, description: str) -> None:
        self._reference = to_gray(ref)
        h, w = self._reference.shape
        self.info.setText(f"Reference: {description} ({w} x {h})")
        self._rebuild_correlator()
        self.clear_history()

    def _rebuild_correlator(self) -> None:
        if self._reference is None:
            return
        self._correlator = ReferenceCorrelator(self._reference, self.ds_spin.value())
        self.clear_history()

    def _open_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference image", "",
            "Images (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.npy)")
        if not path:
            return
        try:
            self.set_reference(load_reference(path), path)
        except Exception as exc:  # noqa: BLE001 - surface any loader error to the user
            self.status.emit(f"ERROR: {exc}")

    def use_current_frame(self) -> None:
        if self._latest is None:
            self.status.emit("No live frame available to use as reference.")
            return
        self.set_reference(np.array(self._latest, copy=True), "current frame")

    def clear_history(self) -> None:
        for d in (self._t, self._score, self._zero, self._dx, self._dy):
            d.clear()
        self._t0 = time.monotonic()
        self._processed_id = -1
        self._redraw()

    # -- processing -----------------------------------------------------------
    def _process(self) -> None:
        if (not self.running.isChecked() or self._correlator is None
                or self._latest is None or self._frame_id == self._processed_id):
            return
        self._processed_id = self._frame_id
        try:
            score, dx, dy, zero = self._correlator.correlate(self._latest)
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"Correlation error: {exc}")
            return
        self._t.append(time.monotonic() - self._t0)
        self._score.append(score)
        self._zero.append(zero)
        self._dx.append(dx)
        self._dy.append(dy)
        self.readout.setText(
            f"peak={score:.4f}  zero-lag={zero:.4f}  shift dx={dx} px  dy={dy} px")
        self._redraw()

    def _redraw(self) -> None:
        n = self.hist_spin.value()
        t = np.fromiter(self._t, float)[-n:]
        for curve, data in ((self.peak_curve, self._score), (self.zero_curve, self._zero),
                            (self.dx_curve, self._dx), (self.dy_curve, self._dy)):
            curve.setData(t, np.fromiter(data, float)[-n:])
