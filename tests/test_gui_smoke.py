"""GUI smoke tests (offscreen) and demo worker behavior checks."""

import json
import os
from pathlib import Path
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="PyQt6 required for GUI smoke tests")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from flir_cam_gui.camera import DEMO_MODELS, DemoWorker, create_worker  # noqa: E402
from flir_cam_gui.live_view import LiveViewWidget  # noqa: E402
from flir_cam_gui.main import MainWindow  # noqa: E402
from flir_cam_gui.node_browser import NodeBrowser  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Demo worker (simulated sensors)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(DEMO_MODELS.keys()))
def test_demo_frames_per_model(key):
    worker = DemoWorker(key)
    arr, meta = worker._make_frame()
    assert isinstance(arr, np.ndarray) and arr.ndim in (2, 3)
    if key.endswith("(simulated CMOS, color, RGB 8-bit)"):
        assert arr.shape[-1] == 3
        assert arr.dtype == np.uint8
    else:
        assert arr.dtype in (np.uint8, np.uint16)
    assert meta["width"] == arr.shape[1] and meta["height"] == arr.shape[0]
    assert meta["camera_model"] == key
    assert "sensor" in meta and "bit_depth" in meta


def test_demo_respects_exposure_and_no_negative():
    worker = DemoWorker()
    worker.FEATURES["ExposureTime"]["value"] = 500.0   # very low
    arr, _ = worker._make_frame()
    assert arr.min() >= 0
    worker.FEATURES["ExposureTime"]["value"] = 500000.0  # saturated
    arr, meta = worker._make_frame()
    assert arr.min() >= 0
    assert arr.max() <= np.iinfo(arr.dtype).max and meta["saturated_fraction"] >= 0


def test_demo_settings_snapshot_roundtrip():
    worker = DemoWorker()
    snap = worker.current_settings_snapshot()
    assert snap["camera"]["__type__"] == "category"
    features = snap["camera"]["features"]
    for expected in ("AcquisitionMode", "ExposureTime", "Gain", "Width", "Height"):
        assert expected in features
    assert features["DeviceTemperature"]["kind"] == "float"


def test_create_worker_defaults_to_demo_without_pyspin():
    from flir_cam_gui import camera as cam
    if getattr(cam, "PYSPIN_AVAILABLE", False):
        pytest.skip("PySpin present on this host")
    w = create_worker()
    assert isinstance(w, DemoWorker)


# ---------------------------------------------------------------------------
# Live view
# ---------------------------------------------------------------------------

def test_live_view_profiles_update(qapp):
    view = LiveViewWidget()
    arr = (np.arange(60 * 80, dtype=np.float32).reshape(60, 80) % 50).astype(np.uint16)
    view.show_frame(arr, {"frame_id": 1, "width": 80, "height": 60,
                          "pixel_format": "Mono16"})
    view._v_line.setPos(40)
    view._h_line.setPos(30)
    x_data, y_data = view._top_curve.xData, view._top_curve.yData
    assert x_data is not None and len(x_data) > 0
    assert y_data is not None and len(y_data) > 0
    rx = view._right_curve.xData
    assert rx is not None and len(rx) > 0
    # crosshair pixel value matches array
    value = int(arr[30, 40])
    emitted = {}
    view.crosshairMoved.connect(lambda x, y, v: emitted.update(x=x, y=y, v=v))
    view._update_profiles()
    assert emitted["x"] == 40 and emitted["y"] == 30 and emitted["v"] == value


def test_live_view_palette_levels(qapp):
    view = LiveViewWidget()
    arr = np.full((20, 20), 1000, dtype=np.uint16)
    view.show_frame(arr, {"frame_id": 2, "width": 20, "height": 20,
                          "pixel_format": "Mono16"})
    view.set_palette("viridis")
    view.set_bit_depth(12, auto_levels=False)
    lo, hi = view.img.levels
    assert hi == 4095


def test_demo_bfs_u3_200s6m_is_portrait(qapp):
    """BFS-U3-200S6M frames are 1536 px wide by 2048 px high, not the reverse."""
    key = next(k for k in DEMO_MODELS if k.startswith("BFS-U3-200S6M"))
    arr, meta = DemoWorker(key)._make_frame()
    assert (meta["width"], meta["height"]) == (1536, 2048)
    assert arr.shape == (2048, 1536)          # (height, width)

    view = LiveViewWidget()
    view.show_frame(arr, meta)
    assert view.displayed_size == (1536, 2048)
    assert "1536x2048" in view.hud_text.toPlainText()


def test_demo_models_do_not_share_feature_state(qapp):
    """Each simulated worker must own its feature values."""
    a = DemoWorker("BFS-U3-200S6M (simulated CMV4000 CMOS, mono 12-bit)")
    b = DemoWorker("BFS-U3-31S4M (simulated IMX265 CMOS, mono 12-bit)")
    a.FEATURES["ExposureTime"]["value"] = 12345.0
    assert a.FEATURES["ExposureTime"]["value"] != b.FEATURES["ExposureTime"]["value"]
    assert DemoWorker.FEATURES["ExposureTime"]["value"] != 12345.0


def test_live_view_orientation_is_display_only(qapp):
    view = LiveViewWidget()
    arr = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    meta = {"width": 8, "height": 6, "pixel_format": "Mono16"}
    view.show_frame(arr, meta)
    assert view.displayed_size == (8, 6)

    view.set_orientation("rotate 90° CW")
    assert view.displayed_size == (6, 8)
    assert np.array_equal(view._arr, np.rot90(arr, k=-1))
    assert view._src is arr                       # raw frame untouched

    view.set_orientation("as acquired")
    assert view.displayed_size == (8, 6)
    assert np.array_equal(view._arr, arr)


def test_crosshair_moves_while_frames_stream(qapp):
    """The crosshair must stay put while frames arrive and follow the user."""
    view = LiveViewWidget()
    arr = np.zeros((40, 60), dtype=np.uint16)
    meta = {"width": 60, "height": 40, "pixel_format": "Mono16"}
    view.show_frame(arr, meta)
    view._follow_mouse = False                    # simulate a drag, not a hover

    streamed = []
    view.crosshairMoved.connect(lambda x, y, v: streamed.append((x, y)))
    frames = np.arange(10, dtype=np.uint16)
    for i in frames:
        arr[:] = i                                # content changes every frame
        view.show_frame(arr, meta)
    assert len(streamed) == len(frames)           # profiles refreshed per frame

    view._v_line.setValue(51)
    view._h_line.setValue(33)
    assert (view._v_line.value(), view._h_line.value()) == (51, 33)
    assert streamed[-1] == (51, 33)
    assert view._top_curve.xData is not None and len(view._top_curve.xData) > 0

    # Incoming frames must not move the crosshair the user just placed.
    arr[:] = 7
    view.show_frame(arr, meta)
    assert (view._v_line.value(), view._h_line.value()) == (51, 33)
    assert streamed[-1] == (51, 33)


def _send_drag(view, kind: str, pos) -> None:
    """Send one realistic mouse event: while dragging, the button stays held."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}
    event_type = types[kind]
    button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    held = Qt.MouseButton.NoButton if kind == "release" else Qt.MouseButton.LeftButton
    QApplication.sendEvent(view.viewport(),
                           QMouseEvent(event_type, QPointF(pos), button, held,
                                       Qt.KeyboardModifier.NoModifier))


def test_crosshair_is_draggable_during_acquisition(qapp):
    """A left-drag on a line must track the cursor the whole way, frame after
    frame, and leave the other line (and the view) alone."""
    from PyQt6.QtCore import QPointF

    view = LiveViewWidget()
    view.resize(800, 700)
    view.show()
    qapp.processEvents()
    arr = np.zeros((2048, 1536), dtype=np.uint16)
    meta = {"width": 1536, "height": 2048, "pixel_format": "Mono16", "frame_id": 1}
    view.show_frame(arr, meta)
    qapp.processEvents()

    vb = view.image_plot.getViewBox()
    port = lambda x, y: view.mapFromScene(vb.mapViewToScene(QPointF(x, y)))  # noqa: E731
    view._v_line.setValue(300)
    view._h_line.setValue(500)
    qapp.processEvents()
    emitted = []
    view.crosshairMoved.connect(lambda x, y, v: emitted.append((x, y)))

    _send_drag(view, "press", port(300, 100))
    assert view._dragging is True
    assert view._drag_line is view._v_line

    # Every move during the drag must be followed, whatever the frame does.
    for x in (340, 400, 520, 600):
        _send_drag(view, "move", port(x, 100))
        arr[:] = x % 4096                      # a new frame arrives mid-drag
        view.show_frame(arr, meta)
        qapp.processEvents()
        assert abs(view._v_line.value() - x) < 3, (x, view._v_line.value())
        assert view._h_line.value() == 500     # only the dragged line moves
        assert emitted and abs(emitted[-1][0] - x) < 3

    # Dragging past the frame edges stays clamped to the frame.
    _send_drag(view, "move", port(5000, 100))
    assert view._v_line.value() == 1535
    _send_drag(view, "move", port(-500, 100))
    assert view._v_line.value() == 0

    _send_drag(view, "move", port(600, 100))
    _send_drag(view, "release", port(600, 100))
    assert view._dragging is False and view._drag_line is None
    assert abs(view._v_line.value() - 600) < 3
    assert view._h_line.value() == 500
    assert view._top_curve.xData is not None and len(view._top_curve.xData) > 0

    # 'Crosshair follows mouse' takes over again once the drag is finished.
    from PyQt6.QtTest import QTest

    view._hover_value(QPointF(700, 800))
    QTest.qWait(60)
    assert (view._v_line.value(), view._h_line.value()) == (700, 800)


def test_crosshair_drag_leaves_panning_alone(qapp):
    """Pressing away from both lines still pans the view (and moves nothing)."""
    from PyQt6.QtCore import QPointF

    view = LiveViewWidget()
    view.resize(800, 700)
    view.show()
    qapp.processEvents()
    view.show_frame(np.zeros((2048, 1536), dtype=np.uint16),
                    {"width": 1536, "height": 2048, "pixel_format": "Mono16"})
    qapp.processEvents()
    vb = view.image_plot.getViewBox()
    port = lambda x, y: view.mapFromScene(vb.mapViewToScene(QPointF(x, y)))  # noqa: E731
    view._v_line.setValue(300)
    view._h_line.setValue(1200)
    qapp.processEvents()
    before = list(vb.viewRange()[0])

    _send_drag(view, "press", port(900, 300))      # nowhere near either line
    assert view._drag_line is None
    _send_drag(view, "move", port(600, 300))
    _send_drag(view, "release", port(600, 300))
    assert list(vb.viewRange()[0]) != before     # the drag panned the view
    assert (view._v_line.value(), view._h_line.value()) == (300, 1200)


def test_reset_view_fits_without_auto_ranging(qapp):
    view = LiveViewWidget()
    view.resize(900, 700)
    arr = np.zeros((2048, 1536), dtype=np.uint16)
    view.show_frame(arr, {"width": 1536, "height": 2048, "pixel_format": "Mono16"})
    vb = view.image_plot.getViewBox()
    vb.setRange(xRange=(100, 200), yRange=(100, 200), padding=0)
    view.reset_view()
    # The whole frame is visible (the aspect lock pads the shorter axis, so the
    # range is never exactly the image extent -- see the aspect tests below).
    (x0, x1), (y0, y1) = vb.viewRange()
    assert x0 <= 0 and x1 >= 1536 and y0 <= 0 and y1 >= 2048
    # autoRange would rescan every pixel of every frame -> the stutter
    assert vb.autoRangeEnabled() == [False, False]


def _square_pixels(view) -> bool:
    """A pixel must have the same on-screen size along x and y."""
    vb = view.image_plot.getViewBox()
    px, py = vb.viewPixelSize()
    return abs(px - py) <= 1e-6 * max(px, py)


def test_image_aspect_ratio_is_never_stretched(qapp):
    """Sensor geometry decides the shape on screen: switching camera, rotating
    or transposing must never stretch the frame."""
    from flir_cam_gui.live_view import ORIENTATIONS

    view = LiveViewWidget()
    view.resize(1100, 620)          # deliberately not the sensor's aspect
    view.show()
    qapp.processEvents()

    # Rotating by 90 or transposing swaps the on-screen axes; flips do not.
    swap = {"rotate 90\u00b0 CW", "rotate 90\u00b0 CCW", "transpose"}

    def show(w, h, name):
        view.show_frame(np.zeros((h, w), dtype=np.uint16),
                        {"width": w, "height": h, "pixel_format": "Mono16"})
        qapp.processEvents()
        dw, dh = view.displayed_size
        assert (dw, dh) == ((h, w) if name in swap else (w, h)), (name, (w, h))
        (x0, x1), (y0, y1) = view.image_plot.getViewBox().viewRange()
        assert x0 <= 0 and x1 >= dw and y0 <= 0 and y1 >= dh, (name, (dw, dh))
        assert _square_pixels(view), (name, (dw, dh), (x0, x1), (y0, y1))

    for w, h in ((1536, 2048), (2048, 1536), (640, 480)):   # camera changes
        for name in ORIENTATIONS:
            view.set_orientation(name)
            show(w, h, name)
        view.set_orientation("as acquired")

    # Window resizing must not stretch the frame either.
    view.show_frame(np.zeros((2048, 1536), dtype=np.uint16),
                    {"width": 1536, "height": 2048, "pixel_format": "Mono16"})
    for size in ((420, 900), (1400, 300), (700, 700)):
        view.resize(*size)
        qapp.processEvents()
        assert _square_pixels(view), size
        (x0, x1), (y0, y1) = view.image_plot.getViewBox().viewRange()
        assert x0 <= 0 and x1 >= 1536 and y0 <= 0 and y1 >= 2048, size
    assert view.aspect_locked is True


def test_transpose_orientation_swaps_axes(qapp):
    view = LiveViewWidget()
    arr = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    view.show_frame(arr, {"width": 8, "height": 6, "pixel_format": "Mono16"})
    view.set_orientation("transpose")
    assert view.displayed_size == (6, 8)
    assert np.array_equal(view._arr, arr.T)
    assert _square_pixels(view)
    view.set_orientation("as acquired")
    assert view.displayed_size == (8, 6)


def _painted_frame_rect(view, qapp, step: int = 2):
    """Bounding box, in widget pixels, of the frame that was really painted.

    The crosshair, HUD and the dashed plot border are hidden/turned off so the
    only bright thing left inside the plot is the image itself.
    """
    for item in (view._v_line, view._h_line, view.hud_text):
        item.setVisible(False)
    vb = view.image_plot.getViewBox()
    vb.setBorder(None)                 # the default border is ~ value 128
    qapp.processEvents()
    r = vb.sceneBoundingRect()
    tl = view.mapFromScene(r.topLeft())
    br = view.mapFromScene(r.bottomRight())
    image = view.grab().toImage()
    xs, ys = [], []
    for x in range(tl.x(), min(br.x(), image.width()), step):
        for y in range(tl.y(), min(br.y(), image.height()), step):
            if image.pixelColor(x, y).value() > 160:      # only the frame is this bright
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def test_frame_is_painted_with_the_sensor_aspect(qapp):
    """Measured on the rendered pixels: a portrait sensor in a landscape window
    must come out portrait, whatever the orientation."""
    view = LiveViewWidget()
    view.resize(760, 400)
    view.show()
    qapp.processEvents()
    view.set_bit_depth(12, auto_levels=False)

    # (orientation, sensor w, sensor h, expected on-screen ratio)
    cases = (("as acquired", 1536, 2048, 1536 / 2048),
             ("flip horizontal", 1536, 2048, 1536 / 2048),   # a mirror keeps the shape
             ("transpose", 1536, 2048, 2048 / 1536),         # transposing swaps it
             ("rotate 90\u00b0 CW", 2048, 1536, 1536 / 2048))    # camera change + rotation
    for name, w, h, expected in cases:
        view.set_orientation(name)
        view.show_frame(np.full((h, w), 3000, dtype=np.uint16),
                        {"width": w, "height": h, "pixel_format": "Mono16"})
        qapp.processEvents()
        dw, dh = view.displayed_size
        assert abs(dw / dh - expected) < 1e-9, (name, (dw, dh))
        box = _painted_frame_rect(view, qapp)
        assert box is not None, f"{name}: nothing painted"
        x0, x1, y0, y1 = box
        ratio = (x1 - x0) / (y1 - y0)
        assert abs(ratio - expected) / expected < 0.03, (name, ratio, expected)


def test_frame_paints_when_levels_are_off(qapp):
    """Auto levels off with no frame yet used to leave the image unlevelled, and
    pyqtgraph paints nothing at all for a downsampled frame without levels."""
    view = LiveViewWidget()
    view.resize(760, 400)
    view.show()
    qapp.processEvents()
    view.set_bit_depth(12, auto_levels=False)      # before any frame arrives
    view.show_frame(np.full((2048, 1536), 3000, dtype=np.uint16),
                    {"width": 1536, "height": 2048, "pixel_format": "Mono16"})
    qapp.processEvents()
    assert [float(v) for v in view.img.levels] == [0.0, 4095.0]
    assert _painted_frame_rect(view, qapp) is not None


def test_clear_frame_keeps_the_widget_layout(qapp):
    """GraphicsLayoutWidget binds `clear` to a layout-wiping method; a cleared
    live view must keep its plots instead of destroying itself."""
    view = LiveViewWidget()
    assert view.clear.__func__ is LiveViewWidget.clear_frame
    meta = {"width": 10, "height": 8, "pixel_format": "Mono16"}
    view.show_frame(np.zeros((8, 10), dtype=np.uint16), meta)
    assert len(view.ci.items) == 3          # image plot + 2 profile plots
    view.clear_frame()
    assert view._arr is None
    assert len(view.ci.items) == 3
    view.show_frame(np.ones((8, 10), dtype=np.uint16), meta)
    assert view.displayed_size == (10, 8)


def test_profile_plots_stay_aligned_with_image(qapp):
    view = LiveViewWidget()
    view.resize(900, 700)
    arr = np.zeros((2048, 1536), dtype=np.uint16)
    view.show_frame(arr, {"width": 1536, "height": 2048, "pixel_format": "Mono16"})
    qapp.processEvents()
    assert view.top.getViewBox().viewRange()[0] == view.image_plot.getViewBox().viewRange()[0]
    assert view.right.getViewBox().viewRange()[1] == view.image_plot.getViewBox().viewRange()[1]


# ---------------------------------------------------------------------------
# Node browser
# ---------------------------------------------------------------------------

def test_node_browser_build_and_edit(qapp):
    browser = NodeBrowser()
    worker = DemoWorker()
    worker.refresh_tree()
    # use the source data directly via _emit path
    tree_payload = {}
    worker.treeReady.connect(lambda t: tree_payload.update(t))
    worker.refresh_tree()
    browser.set_tree(tree_payload)
    assert browser.tree.topLevelItemCount() >= 1
    # find the ExposureTime row and simulate a write
    log = []
    browser.feature_write.connect(lambda n, v: log.append((n, v)))
    ok = False
    from PyQt6.QtCore import Qt
    for key, item in browser._rows.items():
        node = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(node, dict) and node.get("name") == "ExposureTime":
            editor = browser._editors.get(key)
            assert editor is not None
            editor.setText("25000.5")
            editor.editingFinished.emit()
            ok = True
            break
    assert ok, "ExposureTime row not found"
    assert log and log[0][0] == "ExposureTime" and log[0][1] == 25000.5


def test_node_browser_row_update(qapp):
    browser = NodeBrowser()
    worker = DemoWorker()
    payload = {}
    worker.treeReady.connect(lambda t: payload.update(t))
    worker.refresh_tree()
    browser.set_tree(payload)
    row_info = {"name": "Gain", "kind": "float", "value": 1.5, "readable": True,
                "writable": True, "unit": "dB"}
    browser.row_updated(row_info)
    # editor text should reflect new value
    from PyQt6.QtCore import Qt
    found = False
    for key in list(browser._rows):
        node = browser._rows[key].data(0, Qt.ItemDataRole.UserRole)
        if isinstance(node, dict) and node.get("name") == "Gain":
            editor = browser._editors.get(key)
            assert editor is not None
            from PyQt6.QtWidgets import QLineEdit
            assert isinstance(editor, QLineEdit)
            assert editor.text() == "1.5"
            found = True
            break
    assert found, "Gain row not found"


# ---------------------------------------------------------------------------
# Main window wiring
# ---------------------------------------------------------------------------

def test_mainwindow_smoke(qapp):
    win = MainWindow(demo_default=True)
    arr = np.zeros((32, 48), dtype=np.uint16)
    meta = {"frame_id": 5, "width": 48, "height": 32,
            "pixel_format": "Mono16", "acquired_dtype": "uint16"}
    win._on_frame(arr, meta)
    assert win.live._arr is not None
    win._update_info(meta)
    assert "48 x 32" in win.lbl_dims.text()
    tree = DemoWorker().current_settings_snapshot()
    win._on_tree(tree)
    assert win._last_tree == tree
    result = {"ok": True, "path": "x", "message": "Saved x"}
    win._on_save_done(result)
    assert "Saved" in win.log_view.toPlainText()


def test_mainwindow_viewers_and_display_fanout(qapp, tmp_path):
    from flir_cam_gui import savers
    from flir_cam_gui.live_view import LiveViewWidget

    win = MainWindow(demo_default=True)
    assert win.centre_tabs.count() == 3
    assert isinstance(win.reader_tab.view, LiveViewWidget)
    assert win.centre_tabs.widget(0) is win.live

    # The display controls reach both viewers.
    win.orientation_combo.setCurrentText("rotate 90° CW")
    assert win.live.orientation == "rotate 90° CW"
    assert win.reader_tab.view.orientation == "rotate 90° CW"
    win.palette_combo.setCurrentText("viridis")
    win.auto_contrast_check.setChecked(False)
    win.max_spin.setValue(4095)
    assert win.live.auto_levels is False
    assert win.live._manual_levels == (0, 4095)
    assert win.reader_tab.view.auto_levels is False

    # A recorded stack can be opened straight from the main window.
    frames = [np.full((8, 10), i, dtype=np.uint16) for i in range(4)]
    out = savers.save_hdf5_sequence(tmp_path / "seq.hdf5", frames,
                                    [{"frame_id": i} for i in range(4)],
                                    pixel_format="Mono16", camera_settings={})
    assert win.open_in_reader(str(out)) is True
    assert win.centre_tabs.currentWidget() is win.reader_tab

    from time import monotonic, sleep
    deadline = monotonic() + 5.0
    while monotonic() < deadline and win.reader_tab.view._arr is None:
        qapp.processEvents()
        sleep(0.005)
    assert win.reader_tab.view._arr is not None
    assert win.reader_tab.frame_count() == 4

    # A demo frame shows up in the live viewer with the sensor's orientation.
    win.orientation_combo.setCurrentText("as acquired")
    arr, meta = DemoWorker(next(k for k in DEMO_MODELS
                                if k.startswith("BFS-U3-200S6M")))._make_frame()
    win._on_frame(arr, meta)
    win._update_info(meta)
    assert win.live.displayed_size == (1536, 2048)
    assert "1536 x 2048" in win.lbl_dims.text()


def test_sidecar_written_for_demo_save(tmp_path):
    worker = DemoWorker()
    arr, meta = worker._make_frame()
    path = worker._save_current(str(tmp_path / "frame"), "raw", arr, meta, True)
    assert path.exists()
    sidecar_path = Path(str(path) + ".json")
    sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sc["camera_settings"]["camera"]["__type__"] == "category"
    assert sc["frame_metadata"]["frame_id"] == meta["frame_id"]


# ---------------------------------------------------------------------------
# Camera discovery / selection (setups with several cameras)
# ---------------------------------------------------------------------------

class _FakeTlNode:
    def __init__(self, value):
        self._v = value

    def ToString(self):
        return self._v


class _FakeTlNodeMap:
    def __init__(self, info):
        self._info = info

    def GetNode(self, name):
        return _FakeTlNode(self._info.get(name, ""))


class _FakeDevice:
    """Stand-in for a Spinnaker CameraPtr before Init() (TL node map only)."""

    def __init__(self, model, serial, transport="U3V"):
        self._info = {"DeviceModelName": model,
                      "DeviceSerialNumber": serial,
                      "DeviceTransportType": transport}

    def GetTLDeviceNodeMap(self):
        return _FakeTlNodeMap(self._info)


class _FakeCamList:
    def __init__(self, devices):
        self.devices = list(devices)
        self.cleared = False

    def GetSize(self):
        return len(self.devices)

    def GetByIndex(self, i):
        return self.devices[i]

    def Clear(self):
        self.cleared = True


class _FakeSystem:
    devices: list = []
    last_list = None
    released = False

    @classmethod
    def GetInstance(cls):
        return cls()

    def GetCameras(self):
        type(self).last_list = _FakeCamList(type(self).devices)
        return type(self).last_list

    def ReleaseInstance(self):
        type(self).released = True


def _fake_pyspin(devices):
    """A PySpin stand-in exposing just what enumeration/picking uses."""
    import types

    _FakeSystem.devices = list(devices)
    _FakeSystem.last_list = None
    _FakeSystem.released = False
    return types.SimpleNamespace(System=_FakeSystem,
                                  CValuePtr=lambda node: node,
                                  IsReadable=lambda node: True)


def test_enumerate_cameras_lists_every_connected_unit(monkeypatch):
    from flir_cam_gui import camera as cam

    monkeypatch.setattr(cam, "PySpin", _fake_pyspin([
        _FakeDevice("BFS-U3-200S6M", "12345678"),
        _FakeDevice("BFS-U3-31S4M", "87654321", transport="GigE"),
    ]))
    monkeypatch.setattr(cam, "PYSPIN_AVAILABLE", True)

    found = cam.enumerate_cameras()
    assert [c["serial"] for c in found] == ["12345678", "87654321"]
    assert found[0]["model"] == "BFS-U3-200S6M"
    assert found[1]["transport"] == "GigE"
    # the scan must not leak the system / camera-list handles
    assert _FakeSystem.last_list.cleared and _FakeSystem.released

    # no SDK -> nothing to list, and never an exception
    monkeypatch.setattr(cam, "PYSPIN_AVAILABLE", False)
    assert cam.enumerate_cameras() == []


def test_camera_lister_reports_results_and_failures(qapp, monkeypatch):
    from flir_cam_gui import camera as cam
    from time import monotonic, sleep

    def pump(flag, timeout=2.0):
        end = monotonic() + timeout
        while monotonic() < end and not flag():
            qapp.processEvents()
            sleep(0.01)
        return flag()

    monkeypatch.setattr(cam, "enumerate_cameras", lambda: [
        {"model": "BFS-U3-200S6M", "serial": "111111", "transport": "U3V"}])
    found, failed = [], []
    lister = cam.CameraLister()
    lister.camerasFound.connect(lambda cams: found.append(cams))
    lister.failed.connect(failed.append)
    lister.start()
    assert lister.wait(5000)
    assert pump(lambda: bool(found)), "the lister never reported its result"
    assert not failed and found[0][0]["serial"] == "111111"

    def boom():
        raise RuntimeError("transport down")

    monkeypatch.setattr(cam, "enumerate_cameras", boom)
    lister2 = cam.CameraLister()
    lister2.camerasFound.connect(lambda cams: found.append(cams))
    lister2.failed.connect(failed.append)
    lister2.start()
    assert lister2.wait(5000)
    assert pump(lambda: bool(failed)), "the lister swallowed the error"
    assert "transport down" in failed[-1]
    assert len(found) == 1                      # nothing was invented


def test_pick_camera_by_serial_reads_the_device_node_map(qapp, monkeypatch):
    from flir_cam_gui import camera as cam

    monkeypatch.setattr(cam, "PySpin", _fake_pyspin([]))
    cam_list = _FakeCamList([_FakeDevice("BFS-U3-200S6M", "AAA111"),
                             _FakeDevice("BFS-U3-31S4M", "BBB222")])
    assert cam.CameraWorker(serial="BBB222")._pick_camera(cam_list) \
        is cam_list.devices[1]
    # no serial selected -> the first camera, as before
    assert cam.CameraWorker()._pick_camera(cam_list) is cam_list.devices[0]
    # an unknown serial must fail loudly and say what was actually detected
    with pytest.raises(RuntimeError, match="BBB999.*AAA111, BBB222"):
        cam.CameraWorker(serial="BBB999")._pick_camera(cam_list)


def test_camera_picker_populates_and_selects_the_camera(qapp, monkeypatch):
    """Connected cameras are listed, the chosen serial is what Connect uses."""
    from flir_cam_gui import camera as cam
    from flir_cam_gui import main as main_mod
    from time import monotonic, sleep

    monkeypatch.setattr(cam, "enumerate_cameras", lambda: [
        {"model": "BFS-U3-200S6M", "serial": "111111", "transport": "U3V"},
        {"model": "BFS-U3-31S4M", "serial": "222222", "transport": "U3V"},
    ])
    win = MainWindow(serial_default="222222")   # --serial hint
    win.source_combo.setCurrentText("Live (Spinnaker)")
    win._refresh_cameras()
    end = monotonic() + 5.0
    ready = lambda: (win.camera_combo.count() >= 2 and
                     win.camera_combo.isEnabled())  # noqa: E731
    while monotonic() < end and not ready():
        qapp.processEvents()
        sleep(0.01)
    assert ready(), win.camera_combo.itemText(0)
    assert "BFS-U3-200S6M" in win.camera_combo.itemText(0)
    assert win.camera_combo.itemData(0) == "111111"
    assert "SN=222222" in win.camera_combo.itemText(1)
    assert win.camera_combo.currentIndex() == 1    # the --serial hint won
    assert win.refresh_cameras_btn.isEnabled()

    # Connect must hand the selected serial to the worker factory
    seen = {}

    def fake_create(demo=False, serial=None, model_key=None):
        seen.update(demo=demo, serial=serial, model_key=model_key)
        return DemoWorker(model_key)

    monkeypatch.setattr(main_mod, "create_worker", fake_create)
    win._connect_camera()
    assert seen == {"demo": False, "serial": "222222", "model_key": None}
    assert not win.camera_combo.isEnabled()        # locked while connected
    assert not win.refresh_cameras_btn.isEnabled()

    win._disconnect_camera()
    assert win.camera_combo.isEnabled()            # usable again
    win.close()
