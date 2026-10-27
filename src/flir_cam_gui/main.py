"""FLIR Spinnaker camera GUI (PyQt6) — main window & entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                             QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
                             QWidget)

from .correlation import CorrelationTab
from .camera import DEMO_MODELS, PYSPIN_AVAILABLE, CameraLister, create_worker
from .live_view import ORIENTATIONS, PALETTES, LiveViewWidget
from .node_browser import NodeBrowser
from .savers import write_sidecar
from . import frame_reader

APP_TITLE = "FLIR Spinnaker Camera GUI"
_LEVEL_MAX = 65535


def format_mmss(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    return f"{total // 60:d}:{total % 60:02d}"


def stack_length_text(frames: int, fps: float) -> str:
    """Recorded length as minutes:seconds, or a placeholder when fps is unknown."""
    if fps <= 0:
        return f"{frames} frames (fps unknown)"
    return f"{format_mmss(frames / fps)}  ({frames} frames @ {fps:.1f} fps)"


class MainWindow(QMainWindow):
    def __init__(self, demo_default: bool = False, model_default: Optional[str] = None,
                 serial_default: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 950)

        self.worker = None
        self._last_tree: dict = {}
        self._frame_times: list[float] = []
        self._frames_since_meta = 0
        self._recording = False
        self._fps = 0.0
        self._rec_fps = 0.0
        self._lister: Optional[CameraLister] = None
        self._cameras: list = []          # last enumeration result
        self._wanted_serial = serial_default   # --serial hint, consumed once

        self._build_ui()
        self._connect_ui()

        # periodic value refresh (live values like temperature / counters)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(3000)
        self._refresh_timer.timeout.connect(self._poll_feature_values)

        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._update_fps)

        # The display polls the newest frame at its own pace; acquisition never waits for it.
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(33)
        self._display_timer.timeout.connect(self._poll_frame)
        # Start completely disconnected - user must manually connect
        if demo_default or not PYSPIN_AVAILABLE:
            key = model_default or next(iter(DEMO_MODELS))
            self.source_combo.setCurrentText(f"Simulate: {key}")
        else:
            self.source_combo.setCurrentText("Live (Spinnaker)")
        self._sync_camera_picker()
        if not self._current_is_demo():
            self._refresh_cameras()      # populate the picker before connecting

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        # --- source bar ---
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Camera source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Live (Spinnaker)")
        for key in DEMO_MODELS:
            self.source_combo.addItem(f"Simulate: {key}")
        src_row.addWidget(self.source_combo, 2)
        src_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Searching for cameras\u2026")
        self.camera_combo.setToolTip(
            "FLIR cameras detected by Spinnaker - pick which one to connect to")
        self.refresh_cameras_btn = QPushButton("Refresh")
        self.refresh_cameras_btn.setToolTip("Re-scan for connected cameras")
        self.refresh_cameras_btn.clicked.connect(self._refresh_cameras)
        src_row.addWidget(self.camera_combo, 2)
        src_row.addWidget(self.refresh_cameras_btn)
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        src_row.addWidget(self.connect_btn)
        src_row.addWidget(self.disconnect_btn)
        self.start_btn = QPushButton("▶ Start acquisition")
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("■ Stop acquisition")
        self.stop_btn.setEnabled(False)
        src_row.addWidget(self.start_btn)
        src_row.addWidget(self.stop_btn)
        src_row.addStretch(1)
        v.addLayout(src_row)

        # --- splitter: settings | centre tabs | right panel ---
        split = QSplitter()
        v.addWidget(split, 1)

        left = QTabWidget()
        self.browser = NodeBrowser()
        left.addTab(self.browser, "Camera features")
        split.addWidget(left)

        # Live acquisition and saved-stack playback each own a viewer, so
        # browsing a recorded stack never disturbs the running camera view.
        self.live = LiveViewWidget()
        self.reader_tab = frame_reader.FrameReaderTab()
        self.centre_tabs = QTabWidget()
        self.centre_tabs.addTab(self.live, "Live view")
        self.centre_tabs.addTab(self.reader_tab, "Saved images")
        self.corr_tab = CorrelationTab()
        self.centre_tabs.addTab(self.corr_tab, "Correlation")
        split.addWidget(self.centre_tabs)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        # frame info
        info_grp = QGroupBox("Image / camera info")
        info_form = QFormLayout(info_grp)
        self.lbl_frame = QLabel("-")
        self.lbl_dims = QLabel("-")
        self.lbl_format = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_cross = QLabel("-")
        self.lbl_state = QLabel("disconnected")
        for a, w in (("Frame", self.lbl_frame), ("Size", self.lbl_dims), ("Format", self.lbl_format),
                     ("FPS", self.lbl_fps), ("Crosshair", self.lbl_cross), ("State", self.lbl_state)):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info_form.addRow(a, w)
        right_lay.addWidget(info_grp)

        # display settings
        disp_grp = QGroupBox("Display")
        disp_form = QFormLayout(disp_grp)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES)
        self.palette_combo.setCurrentIndex(0)
        self.palette_combo.currentTextChanged.connect(self._palette_changed)
        # Viewer contrast only; independent of camera exposure and gain.
        self.auto_contrast_check = QCheckBox("Auto contrast")
        self.auto_contrast_check.setChecked(True)
        self.auto_contrast_check.toggled.connect(self._contrast_changed)
        self.auto_max_spin = QSpinBox()
        self.auto_max_spin.setRange(0, _LEVEL_MAX)
        self.auto_max_spin.setSpecialValueText("no limit")
        self.auto_max_spin.setToolTip("Upper limit for the auto contrast maximum (0 = no limit)")
        self.auto_max_spin.valueChanged.connect(self._contrast_changed)
        self.bits_combo = QComboBox()
        self.bits_combo.addItems(["8-bit", "10-bit", "12-bit", "16-bit"])
        self.bits_combo.setCurrentText("16-bit")
        self.bits_combo.setToolTip(
            "Pixel bit depth: limits the contrast sliders to 0 .. 2^bits - 1")
        self.bits_combo.currentTextChanged.connect(self._bits_changed)
        self.min_slider, self.min_spin, min_row = self._level_row(0)
        self.max_slider, self.max_spin, max_row = self._level_row(_LEVEL_MAX)
        self.follow_mouse = QCheckBox("Crosshair follows mouse")
        self.follow_mouse.setChecked(True)
        self.follow_mouse.toggled.connect(self._follow_mouse_changed)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(ORIENTATIONS)
        self.orientation_combo.currentTextChanged.connect(self._orientation_changed)
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip(
            "Fit the image to the viewport (does not move the crosshair)")
        self.reset_view_btn.clicked.connect(self._reset_view)
        disp_form.addRow("Palette:", self.palette_combo)
        disp_form.addRow("Bit depth:", self.bits_combo)
        disp_form.addRow("", self.auto_contrast_check)
        disp_form.addRow("Auto max:", self.auto_max_spin)
        disp_form.addRow("Manual min:", min_row)
        disp_form.addRow("Manual max:", max_row)
        self._sync_contrast_enabled()
        disp_form.addRow("Orientation:", self.orientation_combo)
        disp_form.addRow("", self.follow_mouse)
        self.hist_check = QCheckBox("Show histogram")
        self.hist_check.toggled.connect(self._histogram_toggled)
        disp_form.addRow("", self.hist_check)
        disp_form.addRow("", self.reset_view_btn)
        right_lay.addWidget(disp_grp)

        # output location shared by every save / record action
        out_grp = QGroupBox("Output location")
        out_form = QFormLayout(out_grp)
        self.dir_edit = QLineEdit(str(Path.home()))
        self.dir_edit.setToolTip("Working directory for saved files")
        self.dir_btn = QPushButton("Browse...")
        self.dir_btn.clicked.connect(self._browse_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(self.dir_btn)
        self.name_edit = QLineEdit("frame")
        self.name_edit.setToolTip(
            "File name; the extension for the chosen format is added if missing")
        out_form.addRow("Directory:", dir_row)
        out_form.addRow("File name:", self.name_edit)
        right_lay.addWidget(out_grp)

        # save last frame
        save_grp = QGroupBox("Save last frame")
        save_form = QFormLayout(save_grp)
        self.save_fmt_combo = QComboBox()
        self.save_fmt_combo.addItems(["raw (binary + sidecar)", "HDF5 (+ sidecar)",
                                      "Zarr v3 + Blosc (+ sidecar)",
                                      "NumPy .npy (+ sidecar)",
                                      "TIFF (+ sidecar)"])
        self.blosc_cname_combo = QComboBox()
        self.blosc_cname_combo.addItems(["lz4", "zstd"])
        self.blosc_clevel_combo = QComboBox()
        self.blosc_clevel_combo.addItems(["1", "2", "3"])
        self.save_btn = QPushButton("Save last frame...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(lambda _checked=False: self._save_last())
        save_form.addRow("Format:", self.save_fmt_combo)
        save_form.addRow("Blosc (Zarr):", self.blosc_cname_combo)
        save_form.addRow("Compression level:", self.blosc_clevel_combo)
        save_form.addRow("", self.save_btn)
        right_lay.addWidget(save_grp)

        # sequence record
        rec_grp = QGroupBox("Record sequence")
        rec_form = QFormLayout(rec_grp)
        self.rec_fmt_combo = QComboBox()
        self.rec_fmt_combo.addItems(["HDF5 (frames in one dataset)",
                                     "raw (folder of .raw + sidecar)",
                                     "Zarr (frames in one store)",
                                     "NumPy .npy (stack, N x H x W)",
                                     "TIFF (multi-page stack)"])
        self.rec_count = QSpinBox()
        self.rec_count.setRange(1, 100000)
        self.rec_count.setValue(100)
        self.rec_btn = QPushButton("Record N frames...")
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self._toggle_record)
        self.stream_check = QCheckBox("Stream to disk in chunks")
        self.stream_check.setToolTip(
            "Write frames to disk in a background thread while recording instead of "
            "holding the whole stack in RAM. Acquisition never waits for the disk.")
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 100000)
        self.chunk_spin.setValue(32)
        self.chunk_spin.setSuffix(" frames")
        self.chunk_spin.setEnabled(False)
        self.chunk_spin.setToolTip("Number of frames written to disk per chunk")
        self.stream_check.toggled.connect(self.chunk_spin.setEnabled)
        self.rec_length_lbl = QLabel("-")
        self.rec_last_lbl = QLabel("-")
        self.rec_last_lbl.setWordWrap(True)
        self.rec_count.valueChanged.connect(self._update_rec_length)
        rec_form.addRow("Format:", self.rec_fmt_combo)
        rec_form.addRow("Frames:", self.rec_count)
        rec_form.addRow("", self.stream_check)
        rec_form.addRow("Chunk size:", self.chunk_spin)
        rec_form.addRow("Length (m:ss):", self.rec_length_lbl)
        rec_form.addRow("Last recorded:", self.rec_last_lbl)
        rec_form.addRow("", self.rec_btn)
        self._update_rec_length()
        right_lay.addWidget(rec_grp)
        right_lay.addStretch(1)
        split.addWidget(right)

        split.setSizes([430, 800, 300])
        split.setStretchFactor(1, 1)

        # log dock
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        self._log_dock.setFixedHeight(140)

        # status bar
        self.statusBar().addPermanentWidget(QLabel(BUILD_INFO))

        # menus
        menu_file = self.menuBar().addMenu("&File")
        act_raw = menu_file.addAction("Save last frame as raw...", self._save_last_raw)
        act_raw.setShortcut("Ctrl+R")
        act_h5 = menu_file.addAction("Save last frame as HDF5...", self._save_last_hdf5)
        act_h5.setShortcut("Ctrl+H")
        menu_file.addSeparator()
        menu_file.addAction("Save settings snapshot...", self._save_snapshot)
        menu_file.addSeparator()
        menu_file.addAction("Quit", self.close)

        menu_help = self.menuBar().addMenu("&Help")
        menu_help.addAction("About", self._about)

    def _connect_ui(self):
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.connect_btn.clicked.connect(self._connect_camera)
        self.disconnect_btn.clicked.connect(self._disconnect_camera)
        self.start_btn.clicked.connect(self._start_acq)
        self.stop_btn.clicked.connect(self._stop_acq)
        self.browser.feature_write.connect(self._on_feature_write)
        self.browser.feature_execute.connect(self._on_feature_execute)
        self.browser.refresh_requested.connect(
            lambda: self.worker is not None and
            self.worker.submit({"action": "refresh_tree"}))
        self.live.crosshairMoved.connect(self._on_crosshair)
        self.reader_tab.status.connect(self.log_line)
        self.corr_tab.status.connect(self.log_line)

    # ------------------------------------------------------------ logging
    def log_line(self, text: str):
        self.log_view.appendPlainText(time.strftime("%H:%M:%S") + "  " + text.strip())

    # ------------------------------------------------------- camera connect
    @pyqtSlot()
    def _source_changed(self):
        # The camera picker only applies to real hardware; switching back to
        # "Live" re-scans so a camera plugged in meanwhile shows up.
        self._sync_camera_picker()
        if not self._current_is_demo() and self.worker is None:
            self._refresh_cameras()

    def _sync_camera_picker(self):
        """The picker is usable only for live sources and only while disconnected."""
        enabled = (not self._current_is_demo()) and self.worker is None
        self.camera_combo.setEnabled(enabled)
        self.refresh_cameras_btn.setEnabled(enabled)

    @pyqtSlot()
    def _refresh_cameras(self):
        """Scan for connected FLIR cameras without blocking the UI."""
        if self._current_is_demo() or self.worker is not None:
            return
        if self._lister is not None and self._lister.isRunning():
            return                       # one scan at a time
        if not self._cameras and self.camera_combo.count() <= 1:
            self.camera_combo.setItemText(0, "Searching for cameras\u2026")
        self.refresh_cameras_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self._lister = CameraLister(self)
        self._lister.camerasFound.connect(self._on_cameras_found)
        self._lister.failed.connect(self._on_cameras_failed)
        self._lister.finished.connect(self._sync_camera_picker)
        self._lister.start()

    def _on_cameras_found(self, cams):
        self._cameras = list(cams)
        # keep the current choice (or the --serial hint) selected across rescans
        wanted = self._wanted_serial or self.camera_combo.currentData()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if not self._cameras:
            self.camera_combo.addItem("No cameras found - click Refresh")
        else:
            for cam in self._cameras:
                model = str(cam.get("model") or "FLIR camera").strip()
                serial = str(cam.get("serial") or "").strip()
                transport = str(cam.get("transport") or "").strip()
                label = f"{model}   SN={serial}" if serial else model
                if transport:
                    label = f"{label}  [{transport}]"
                self.camera_combo.addItem(label, serial or None)
        index = 0
        if wanted:
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == wanted:
                    index = i
                    break
        self.camera_combo.setCurrentIndex(index)
        self.camera_combo.blockSignals(False)
        self._wanted_serial = None
        if self._cameras:
            self.log_line(f"Found {len(self._cameras)} camera(s).")
            self.statusBar().showMessage(
                f"{len(self._cameras)} camera(s) detected", 5000)
        else:
            self.log_line("No FLIR cameras detected.")

    def _on_cameras_failed(self, message: str):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("Camera scan failed - click Refresh")
        self.camera_combo.blockSignals(False)
        self.log_line("ERROR: " + str(message))
        self.statusBar().showMessage(str(message), 8000)

    def _current_is_demo(self):
        text = self.source_combo.currentText()
        return text.startswith("Simulate:")

    def _current_model_key(self):
        text = self.source_combo.currentText()
        if text.startswith("Simulate:"):
            return text[len("Simulate: "):].strip()
        return None

    def _connect_camera(self):
        if self.worker is not None:
            return
        demo = self._current_is_demo()
        serial = None if demo else self.camera_combo.currentData()
        if serial:
            self.log_line(f"Connecting to camera SN={serial} \u2026")
        self.worker = create_worker(demo=demo, serial=serial,
                                    model_key=self._current_model_key())
        self.worker.treeReady.connect(self._on_tree)
        self.worker.rowUpdated.connect(self.browser.row_updated)
        self.worker.saveDone.connect(self._on_save_done)
        self.worker.stateChanged.connect(self._on_state)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self.log_line)
        self.worker.set_display_settings(self.current_display_settings())
        self.worker.start()
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self._sync_camera_picker()   # lock the picker while connected
        self._refresh_timer.start()
        self._fps_timer.start()
        self._display_timer.start()

    def _disconnect_camera(self):
        self._refresh_timer.stop()
        self._fps_timer.stop()
        self._display_timer.stop()
        if self.worker is not None:
            worker = self.worker
            worker.stop()
            # Keep the reference until the thread has exited; a streamed
            # recording may still be flushing its last chunks, so wait for it.
            if not worker.wait(5000) and worker.has_pending_writes():
                self.log_line("Waiting for recording data to finish writing to disk ...")
                worker.wait()
            self.worker = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self._recording = False
        self.rec_btn.setText("Record N frames...")
        self.live.clear_frame()
        self.corr_tab.clear_frame()
        self.lbl_state.setText("disconnected")
        self._sync_camera_picker()   # the picker may be used again

    # ------------------------------------------------------------ acq control
    def _start_acq(self):
        if self.worker:
            self.worker.submit({"action": "start_acquisition"})

    def _stop_acq(self):
        if self.worker:
            self.worker.submit({"action": "stop_acquisition"})

    # -------------------------------------------------------------- signals
    @pyqtSlot(dict)
    def _on_tree(self, tree: dict):
        self._last_tree = tree
        self.browser.set_tree(tree)
        self.log_line("Settings tree refreshed.")

    @pyqtSlot(str)
    def _on_state(self, state: str):
        self.lbl_state.setText(state)
        acquiring = state in ("acquiring", "recording")
        connected = state != "disconnected"
        self.start_btn.setEnabled(connected and not acquiring)
        self.stop_btn.setEnabled(connected and acquiring)
        self.save_btn.setEnabled(connected)
        self.rec_btn.setEnabled(connected)
        if state == "disconnected" and self.worker is None:
            pass
        elif state == "recording":
            self.rec_btn.setText("■ Stop recording")
            self._recording = True
        elif acquiring:
            self.rec_btn.setText("Record N frames..." if not self._recording
                                 else "■ Stop recording")
        else:
            self._recording = False
            self.rec_btn.setText("Record N frames...")

    @pyqtSlot(str)
    def _on_error(self, text: str):
        self.log_line("ERROR: " + text)
        self.statusBar().showMessage(text.splitlines()[0], 8000)

    @pyqtSlot(object, object)
    def _on_frame(self, arr, meta):
        self.live.show_frame(arr, dict(meta) if meta else {})
        self.corr_tab.set_frame(arr)
        self._frames_since_meta += 1
        if self._frames_since_meta % 8 == 0:
            self._update_info(meta)

    def _poll_frame(self):
        worker = self.worker
        if worker is None:
            return
        item = worker.take_latest_frame()
        if item is not None:
            self._on_frame(*item)

    def _update_fps(self):
        # Rate of acquisition, not of what the display happened to draw.
        times = list(self.worker.acq_times) if self.worker is not None else []
        if len(times) < 2 or time.monotonic() - times[-1] > 2.0:
            self.lbl_fps.setText("0.0")
            return
        span = max(0.001, times[-1] - times[0])
        fps = (len(times) - 1) / span
        self._fps = fps
        self._update_rec_length()
        if self.live._arr is not None:
            self.live.set_hud_text(f"{fps:5.1f} fps")
        self.lbl_fps.setText(f"{fps:.1f}")

    def _follow_mouse_changed(self, follow: bool):
        self.live.set_follow_mouse(follow)
        self.reader_tab.view.set_follow_mouse(follow)

    def _histogram_toggled(self, show: bool):
        self.live.set_histogram_visible(show)
        self.reader_tab.view.set_histogram_visible(show)

    def _update_rec_length(self, *_):
        self.rec_length_lbl.setText(stack_length_text(int(self.rec_count.value()), self._fps))

    def _reset_view(self):
        """Fit whichever viewer the user is looking at."""
        widget = self.centre_tabs.currentWidget()
        if isinstance(widget, frame_reader.FrameReaderTab):
            widget.view.reset_view()
        else:
            self.live.reset_view()

    def _update_info(self, meta):
        if not meta:
            return
        self.lbl_frame.setText(str(meta.get("frame_id", "-")))
        self.lbl_dims.setText(f"{meta.get('width')} x {meta.get('height')}  "
                              f"{meta.get('acquired_dtype', '')}")
        self.lbl_format.setText(str(meta.get("pixel_format", "-")))

    @pyqtSlot(int, int, float)
    def _on_crosshair(self, x, y, val):
        self.lbl_cross.setText(f"x={x}  y={y}  val={val:.3f}")

    # --------------------------------------------------------- feature I/O
    def _on_feature_write(self, name: str, value):
        if self.worker:
            self.worker.submit({"action": "set_feature", "name": name, "value": value})

    def _on_feature_execute(self, name: str):
        if self.worker:
            self.worker.submit({"action": "execute_command", "name": name})

    def _poll_feature_values(self):
        if self.worker is not None:
            names = self.browser.collect_refresh_names()
            if names:
                self.worker.submit({"action": "refresh_rows", "names": names})

    # -------------------------------------------------------------- displays
    def current_display_settings(self):
        settings = {
            "palette": self.palette_combo.currentText(),
            "follow_mouse": self.follow_mouse.isChecked(),
            "auto_levels": self.auto_contrast_check.isChecked(),
            "bit_depth": self._bits(),
            "manual_min": self.min_spin.value(),
            "manual_max": self.max_spin.value(),
            "auto_max": self.auto_max_spin.value() or None,
            "orientation": self.orientation_combo.currentText(),
        }
        return settings

    def _palette_changed(self, name):
        self._apply_display_settings()

    def _bits_changed(self, *_):
        top = (1 << self._bits()) - 1
        was_top = self.max_spin.value() >= self.max_spin.maximum()
        widgets = (self.min_slider, self.min_spin, self.max_slider, self.max_spin,
                   self.auto_max_spin)
        for w in widgets:
            w.blockSignals(True)
            w.setMaximum(top)
        if was_top:
            self.max_spin.setValue(top)
            self.max_slider.setValue(top)
        for w in widgets:
            w.blockSignals(False)
        self._contrast_changed()

    def _bits(self) -> int:
        return int(self.bits_combo.currentText().split("-")[0])

    def _contrast_changed(self, *_):
        self._sync_contrast_enabled()
        self._apply_display_settings()

    def _sync_contrast_enabled(self):
        auto = self.auto_contrast_check.isChecked()
        self.auto_max_spin.setEnabled(auto)
        for w in (self.min_slider, self.min_spin, self.max_slider, self.max_spin):
            w.setEnabled(not auto)

    def _level_row(self, value: int):
        """Slider and spin box kept in sync; returns (slider, spin, layout widget)."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, _LEVEL_MAX)
        spin = QSpinBox()
        spin.setRange(0, _LEVEL_MAX)
        slider.setValue(value)
        spin.setValue(value)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        spin.valueChanged.connect(self._contrast_changed)
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider, 1)
        lay.addWidget(spin)
        return slider, spin, row

    def _orientation_changed(self, name):
        self._apply_display_settings()

    def _apply_display_settings(self):
        """Apply the display controls to every viewer and the worker sidecar
        record."""
        settings = self.current_display_settings()
        self.live.set_palette(settings["palette"])
        self.live.set_contrast(settings["auto_levels"], settings["manual_min"],
                               settings["manual_max"], settings["auto_max"])
        self.live.set_orientation(settings["orientation"])
        self.reader_tab.set_display_settings(settings)
        if self.worker:
            self.worker.set_display_settings(settings)

    # ---------------------------------------------------------------- saving
    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select working directory", self.dir_edit.text() or str(Path.home()))
        if path:
            self.dir_edit.setText(path)

    def _output_path(self, ext: str, what: str = "file") -> Optional[str]:
        """Build directory + name + extension from the path fields, or None to cancel."""
        directory = Path(self.dir_edit.text().strip() or Path.home()).expanduser()
        name = self.name_edit.text().strip() or "frame_" + time.strftime("%Y%m%d_%H%M%S")
        allowed = {"h5": {".h5", ".hdf5"}, "tif": {".tif", ".tiff"}}.get(ext, {"." + ext})
        if Path(name).suffix.lower() not in allowed:
            name += "." + ext
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Invalid directory", f"{directory}\n\n{exc}")
            return None
        path = directory / name
        if path.exists() and QMessageBox.question(
                self, "Overwrite?", f"{path} already exists. Overwrite this {what}?"
        ) != QMessageBox.StandardButton.Yes:
            return None
        return str(path)

    def _blosc_params(self):
        return {
            "blosc_cname": self.blosc_cname_combo.currentText(),
            "blosc_clevel": int(self.blosc_clevel_combo.currentText()),
            "blosc_shuffle": "shuffle",
        }

    def _save_last_hdf5(self):
        self._save_last(fmt="hdf5")

    def _save_last_raw(self):
        self._save_last(fmt="raw")

    def _save_last(self, fmt: Optional[str] = None):
        if self.worker is None:
            return
        if fmt is None:
            fmt = ("raw", "hdf5", "zarr", "npy", "tiff")[self.save_fmt_combo.currentIndex()]
        ext = {"raw": "raw", "hdf5": "h5", "zarr": "zarr", "npy": "npy", "tiff": "tif"}[fmt]
        path = self._output_path(ext)
        if not path:
            return
        self.worker.submit({"action": "save_last", "path": path, "fmt": fmt,
                            "display_settings": self.current_display_settings(),
                            "blosc": self._blosc_params()})

    def _toggle_record(self):
        if self.worker is None:
            return
        if self._recording:
            self.worker.submit({"action": "stop_record"})
            return
        fmt = ("hdf5_sequence", "raw_sequence", "zarr_sequence", "npy_sequence",
               "tiff_sequence")[self.rec_fmt_combo.currentIndex()]
        plain = {"hdf5_sequence": "HDF5", "raw_sequence": "raw",
                 "zarr_sequence": "Zarr", "npy_sequence": "npy",
                 "tiff_sequence": "TIFF"}[fmt]
        n = int(self.rec_count.value())
        path = self._output_path(
            {"hdf5_sequence": "h5", "raw_sequence": "raw_seq",
             "zarr_sequence": "zarr", "npy_sequence": "npy",
             "tiff_sequence": "tif"}[fmt], f"{plain} recording")
        if not path:
            return
        self.worker.submit({
            "action": "record", "path": path, "n_frames": n, "fmt": fmt,
            "display_settings": self.current_display_settings(),
            "blosc": self._blosc_params(),
            "stream": self.stream_check.isChecked(),
            "chunk_frames": int(self.chunk_spin.value())})
        self._rec_fps = self._fps
        self._recording = True
        self.rec_btn.setText(f"■ Stop recording (rec {n})")

    def _save_snapshot(self):
        if not self._last_tree:
            QMessageBox.information(self, "Nothing to save",
                                    "Connect to a camera first.")
            return
        path = self._output_path("json", "snapshot")
        if not path:
            return
        write_sidecar(Path(path), {
            "schema": 1, "application": "flir-cam-gui", "saved_at_utc":
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "camera_settings": self._last_tree})
        self.log_line(f"Settings snapshot written to {path}")

    @pyqtSlot(dict)
    def _on_save_done(self, result: dict):
        if result.get("ok"):
            self.log_line(result["message"])
            self.statusBar().showMessage(result["message"], 6000)
            if str(result.get("message", "")).startswith(("Recorded", "Stopped")):
                self._recording = False
                frames = result.get("frames")
                if frames is not None:
                    self.rec_last_lbl.setText(stack_length_text(int(frames), self._rec_fps))
        else:
            self._recording = False
            self.log_line("SAVE FAILED: " + result.get("message", ""))
            QMessageBox.warning(self, "Save failed", result.get("message", ""))

    def open_in_reader(self, path: str) -> bool:
        """Open *path* in the Saved images tab (also used by the CLI)."""
        self.centre_tabs.setCurrentWidget(self.reader_tab)
        self.reader_tab.set_display_settings(self.current_display_settings())
        return self.reader_tab.open_path(path)

    def _about(self):
        QMessageBox.information(
            self, "About", APP_TITLE + "\n"
            "PyQt6 + PySpin (FLIR Spinnaker) camera control.\n"
            "Simulation models are approximate, offline only.")

    # -------------------------------------------------------------- teardown
    def closeEvent(self, event):
        self._disconnect_camera()
        super().closeEvent(event)


BUILD_INFO = "flir-cam-gui 0.1.0"


def main():
    parser = argparse.ArgumentParser(description="FLIR Spinnaker camera GUI")
    parser.add_argument("--demo", action="store_true",
                        help="start in simulation mode even if PySpin is present")
    parser.add_argument("--live", action="store_true",
                        help="connect to real hardware even if PySpin appears missing")
    parser.add_argument("--model", type=str, default=None,
                        help="simulated camera model key (see README)")
    parser.add_argument("--serial", type=str, default=None, metavar="SN",
                        help="select this camera in the picker at startup")
    parser.add_argument("--open", type=str, default=None, metavar="PATH",
                        help="open a saved stack (raw/HDF5/Zarr) in the "
                             "Saved images tab at startup")
    args = parser.parse_args()
    import faulthandler
    faulthandler.enable()  # print a traceback on native crashes
    app = QApplication(sys.argv)
    app.setApplicationName("flir-cam-gui")
    win = MainWindow(demo_default=args.demo, model_default=args.model,
                      serial_default=args.serial)
    win.show()
    # Always start disconnected - user must manually click Connect
    if args.open:
        win.open_in_reader(args.open)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
