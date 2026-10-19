"""E2E verification: run the actual GUI headless, stream demo frames, save
raw/hdf5/zarr, record a sequence, and read everything back through the
"Saved images" tab the way a user would.
"""

import json
import os

import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

LOG = Path(__file__).parent / "e2e.log"
_lf = LOG.open("w", encoding="utf-8", buffering=1)


def p(*args):
    _lf.write(" ".join(str(a) for a in args) + "\n")


from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from flir_cam_gui.main import MainWindow

app = QApplication.instance() or QApplication([])
p("app ok")
win = MainWindow(demo_default=True)
win.show()
p("window ok")
out = Path(tempfile.mkdtemp())
state = {"frames": 0, "saves": [], "dims": set()}

orig_on_frame = win._on_frame


def on_frame(arr, meta):
    state["frames"] += 1
    state["dims"].add((meta.get("width"), meta.get("height")))
    orig_on_frame(arr, meta)


win._on_frame = on_frame

orig_done = win._on_save_done


def on_done(res):
    state["saves"].append((res["ok"], Path(res["path"]).name))
    orig_done(res)


win._on_save_done = on_done

win._connect_camera()
p("camera connected")
QTimer.singleShot(200, win._start_acq)


def save_and_record():
    win.worker.submit({"action": "save_last", "path": str(out / "frame_a"), "fmt": "raw"})
    win.worker.submit({"action": "save_last", "path": str(out / "frame_b.h5"), "fmt": "hdf5"})
    win.worker.submit({"action": "save_last", "path": str(out / "frame_c"), "fmt": "zarr",
                       "blosc": {"blosc_cname": "zstd", "blosc_clevel": 2,
                                 "blosc_shuffle": "shuffle"}})
    win.worker.submit({"action": "record", "path": str(out / "seq"), "n_frames": 6,
                       "fmt": "zarr_sequence", "display_settings": {},
                       "blosc": {"blosc_cname": "lz4", "blosc_clevel": 1,
                                 "blosc_shuffle": "shuffle"}})


QTimer.singleShot(1200, save_and_record)
p("timers armed")

reader_state = {}


def check_and_finish():
    if len(state["saves"]) >= 4 and all(ok for ok, _ in state["saves"]):
        stage_files()
    else:
        QTimer.singleShot(500, check_and_finish)


def stage_files():
    import zarr
    from zarr.storage import LocalStore

    g = zarr.open_group(store=LocalStore(root=out / "frame_c.zarr"), mode="r")
    p("zarr single shape:", g["image"].shape, g["image"].dtype)
    assert g["image"].shape == (2048, 1536)
    gz = zarr.open_group(store=LocalStore(root=out / "seq.zarr"), mode="r")
    p("zarr seq dataset:", gz["image"].shape)
    assert gz["image"].shape[0] == 6
    codecs = gz["image"].metadata.codecs
    gz_meta = [c for c in codecs if type(c).__name__ == "BloscCodec"][0]
    assert gz_meta.cname == "lz4" and gz_meta.clevel == 1
    p("saves:", state["saves"], "frames:", state["frames"])

    raw = out / "frame_a.raw"
    assert raw.stat().st_size == 6291456, raw.stat().st_size
    side = json.loads((out / "frame_a.raw.json").read_text())
    assert set(side) >= {"file", "frame_metadata", "camera_settings", "saved_at_utc"}
    # BFS-U3-200S6M is portrait: 1536 wide x 2048 high (not the other way round)
    assert side["file"]["width"] == 1536 and side["file"]["height"] == 2048
    cam_settings = json.dumps(side["camera_settings"])
    assert "AcquisitionMode" in cam_settings and "ExposureTime" in cam_settings
    zarr_sc = json.loads((out / "frame_c.zarr.json").read_text())
    assert zarr_sc["file"]["compressor"]["cname"] == "zstd"
    assert zarr_sc["file"]["compressor"]["clevel"] == 2
    seq_sc = json.loads((out / "seq.zarr.json").read_text())
    assert seq_sc["file"]["frames"] == 6

    # Live view orientation: the displayed frame is 1536 x 2048 (portrait).
    assert state["dims"] == {(1536, 2048)}, state["dims"]
    assert win.live.displayed_size == (1536, 2048), win.live.displayed_size
    assert "1536x2048" in win.live.hud_text.toPlainText()
    p("live view reported", win.live.displayed_size, "| hud:",
      win.live.hud_text.toPlainText())
    assert_not_blank(win.live, "live view")
    assert_sensor_aspect(win.live, "live view")
    drag_crosshair_while_streaming()
    stage_reader()


def drag_crosshair_while_streaming():
    """Drag a crosshair with a real mouse-event sequence while frames arrive."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    live = win.live
    vb = live.image_plot.getViewBox()
    port = lambda x, y: live.mapFromScene(vb.mapViewToScene(QPointF(x, y)))  # noqa: E731
    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}

    def send(kind, pos):
        button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
        held = Qt.MouseButton.NoButton if kind == "release" else Qt.MouseButton.LeftButton
        QApplication.sendEvent(live.viewport(),
                               QMouseEvent(types[kind], QPointF(pos), button, held,
                                           Qt.KeyboardModifier.NoModifier))

    live._v_line.setValue(400)
    live._h_line.setValue(1000)
    app.processEvents()
    send("press", port(400, 500))
    assert live._drag_line is live._v_line, "the press did not grab the crosshair"
    followed = []
    for x in (600, 900, 1200):
        send("move", port(x, 500))
        before = state["frames"]
        wait_for(lambda: state["frames"] > before, what="frames to arrive mid-drag")
        assert live._dragging is True, "the drag stopped while frames arrived"
        assert abs(live._v_line.value() - x) < 3, (x, live._v_line.value())
        assert live._h_line.value() == 1000, live._h_line.value()
        followed.append(round(live._v_line.value(), 1))
    send("release", port(1200, 500))
    assert live._drag_line is None and live._dragging is False
    p("crosshair dragged during acquisition:", followed,
      "| frame", state["frames"], "| horizontal line stayed at", live._h_line.value())


def assert_sensor_aspect(widget, what: str) -> None:
    """The image must be drawn with the sensor's aspect ratio, never stretched.

    Checks that one screen pixel is the same size along x and y, and that the
    frame's on-screen rectangle (in scene coordinates) has exactly the geometry
    the sensor reported.
    """
    from PyQt6.QtCore import QPointF

    app.processEvents()
    vb = widget.image_plot.getViewBox()
    w, h = widget.displayed_size
    assert vb.state["aspectLocked"] == 1.0, vb.state["aspectLocked"]
    px, py = vb.viewPixelSize()
    assert abs(px - py) <= 1e-6 * max(px, py), (what, px, py)
    (x0, x1), (y0, y1) = vb.viewRange()
    assert x0 <= 0 and x1 >= w and y0 <= 0 and y1 >= h, (what, vb.viewRange(), w, h)
    o = vb.mapViewToScene(QPointF(0, 0))
    e = vb.mapViewToScene(QPointF(w, h))
    drawn = (abs(e.x() - o.x()), abs(e.y() - o.y()))
    aspect = drawn[0] / drawn[1] if drawn[1] else 0.0
    assert abs(aspect - w / h) <= 1e-3 * (w / h), (what, drawn, (w, h))
    p(f"{what} aspect: {w}x{h} drawn as {drawn[0]:.1f}x{drawn[1]:.1f} px "
      f"(ratio {aspect:.4f} vs sensor {w / h:.4f}), pixel {px:.5f} == {py:.5f}")


def assert_not_blank(widget, what: str) -> None:
    """The viewer must actually paint the frame, not just hold an array."""
    app.processEvents()
    pix = widget.grab()
    assert pix.width() > 50 and pix.height() > 50, (what, pix.size())
    image = pix.toImage()
    values = [image.pixelColor(x, y).value()
              for x in range(0, image.width(), max(1, image.width() // 20))
              for y in range(0, image.height(), max(1, image.height() // 20))]
    spread = max(values) - min(values)
    assert spread > 20, f"{what} looks blank (value spread {spread})"
    p(f"{what} renders: {image.width()}x{image.height()}px, value spread {spread}")


def wait_for(predicate, timeout: float = 10.0, what: str = "") -> None:
    """Pump the event loop until *predicate* holds (frames stream meanwhile)."""
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what or predicate}")


def stage_reader():
    """Open the saved stacks through the Saved images tab, as a user would."""
    tab = win.reader_tab
    assert win.centre_tabs.count() == 2

    assert win.open_in_reader(str(out / "seq.zarr")) is True
    wait_for(lambda: tab.view._arr is not None, what="zarr sequence frame 0")
    assert tab.view._arr.shape == (2048, 1536), tab.view._arr.shape
    assert tab.frame_count() == 6 and tab.lbl_pos.text() == "0 / 6"
    p("reader zarr sequence frame 0 ok", tab.view._arr.shape, tab.lbl_pos.text())

    tab.seek(4)                       # jump straight to any frame
    wait_for(lambda: tab._index == 4, what="zarr sequence frame 4")
    assert tab.view._arr.shape == (2048, 1536) and tab.lbl_pos.text() == "4 / 6"
    p("reader seek ok", tab.lbl_pos.text())

    tab.play()
    wait_for(lambda: tab._index != 4, what="reader playback to advance")
    tab.stop()
    assert not tab.is_playing
    p("reader playback ok, stopped at", tab.lbl_pos.text())

    assert win.open_in_reader(str(out / "frame_a.raw")) is True
    wait_for(lambda: tab.view._arr is not None, what="raw single frame")
    assert tab.frame_count() == 1 and tab.view._arr.shape == (2048, 1536)
    p("reader raw single ok", tab.lbl_pos.text())
    assert_not_blank(tab.view, "saved image viewer")
    assert_sensor_aspect(tab.view, "saved image viewer")

    assert win.open_in_reader(str(out / "frame_b.h5")) is True
    wait_for(lambda: tab.view._arr is not None, what="hdf5 single frame")
    assert tab.frame_count() == 1, tab.frame_count()
    assert tab.view._arr.shape == (2048, 1536), tab.view._arr.shape
    p("reader hdf5 single ok", tab.view._arr.shape, tab.lbl_pos.text())

    finish()


def finish():
    win._disconnect_camera()
    assert win.live._arr is None
    p("ALL-GOOD")
    app.quit()


QTimer.singleShot(2000, check_and_finish)
QTimer.singleShot(45000, app.quit)
app.exec()
p("after exec; frames=", state["frames"], "saves=", state["saves"])
_lf.close()
