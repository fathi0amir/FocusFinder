"""Saved-stack reader: format detection, lazy loading and playback (offscreen)."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flir_cam_gui import savers  # noqa: E402
from flir_cam_gui.frame_reader import (FrameReaderTab, SequenceLoader,  # noqa: E402
                                       describe_source)

SETTINGS = {
    "camera": {"__type__": "category", "display": "Root", "writable": False,
               "features": {"ExposureTime": {"kind": "float", "value": 5000.0,
                                             "readable": True, "writable": True}}},
    "stream": {},
}


def _raw_sequence_dir(tmp_path: Path, n: int = 3, h: int = 4, w: int = 5) -> Path:
    d = tmp_path / "rawseq"
    d.mkdir()
    index = []
    for i in range(n):
        name = f"frame_{i:06d}.raw"
        arr = np.full((h, w), i, dtype=np.uint16)
        (d / name).write_bytes(np.ascontiguousarray(arr).tobytes())
        index.append({"file": name, "frame_id": i, "width": w, "height": h,
                      "acquired_dtype": "uint16", "pixel_format": "Mono16"})
    savers.write_sidecar(d / "sequence.json", savers.build_sidecar(
        image_info={"type": "raw_sequence", "frames": n, "width": w, "height": h,
                    "channels": 1, "dtype": "uint16", "byte_order": "little"},
        frame_meta={"frames": index}, camera_settings=SETTINGS))
    return d


# ---------------------------------------------------------------------------
# Format detection + lazy frame reading
# ---------------------------------------------------------------------------

def test_raw_sequence_is_described_from_sidecar(tmp_path):
    d = _raw_sequence_dir(tmp_path)
    info = describe_source(d)
    assert info.kind == "raw_sequence"
    assert (info.n_frames, info.width, info.height, info.dtype) == (3, 5, 4, "uint16")
    assert describe_source(d / "sequence.json").kind == "raw_sequence"

    loader = SequenceLoader(info)
    arr, meta = loader.load(2)
    assert arr.shape == (4, 5)
    assert np.all(arr == 2)
    assert meta["frame_id"] == 2
    loader.close()


def test_raw_single_via_sidecar_path(tmp_path):
    arr = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    p = savers.save_raw_with_sidecar(tmp_path / "f.raw", arr, pixel_format="Mono16",
                                     frame_meta={"frame_id": 4}, camera_settings={})
    info = describe_source(p)
    assert (info.kind, info.n_frames, info.width, info.height) == ("raw_single", 1, 8, 6)
    back, meta = SequenceLoader(info).load(0)
    assert np.array_equal(back, arr)
    assert meta["frame_id"] == 4
    assert describe_source(Path(str(p) + ".json")).kind == "raw_single"


def test_colour_raw_sidecar_keeps_frame_geometry(tmp_path):
    arr = np.zeros((6, 8, 3), dtype=np.uint8)
    p = savers.save_raw_with_sidecar(tmp_path / "rgb.raw", arr,
                                     pixel_format="RGB8Packed", frame_meta={},
                                     camera_settings={})
    sidecar = json.loads(Path(str(p) + ".json").read_text(encoding="utf-8"))
    assert (sidecar["file"]["width"], sidecar["file"]["height"],
            sidecar["file"]["channels"]) == (8, 6, 3)
    info = describe_source(p)
    assert (info.width, info.height, info.channels) == (8, 6, 3)
    back, _ = SequenceLoader(info).load(0)
    assert back.shape == (6, 8, 3)


def test_hdf5_sequence_is_read_lazily(tmp_path):
    frames = [np.full((4, 5), i, dtype=np.uint16) for i in range(5)]
    metas = [{"frame_id": i, "pixel_format": "Mono16", "acquired_dtype": "uint16"}
             for i in range(5)]
    out = savers.save_hdf5_sequence(tmp_path / "seq.hdf5", frames, metas,
                                    pixel_format="Mono16", camera_settings=SETTINGS)
    info = describe_source(out)
    assert info.kind == "hdf5_sequence"
    assert (info.n_frames, info.width, info.height, info.dtype) == (5, 5, 4, "uint16")

    loader = SequenceLoader(info)
    assert loader._h5 is None            # nothing is opened just to list a stack
    arr, meta = loader.load(3)
    assert np.all(arr == 3) and meta["frame_id"] == 3
    assert loader._h5 is not None
    loader.load(1)
    assert loader._cache[0] == 1          # only the newest frame is kept
    loader.close()
    assert loader._h5 is None


def test_hdf5_single_colour_is_one_frame(tmp_path):
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    out = savers.save_hdf5(tmp_path / "c.h5", arr, pixel_format="RGB8Packed",
                           frame_meta={}, camera_settings=SETTINGS)
    info = describe_source(out)
    assert info.n_frames == 1
    assert (info.width, info.height, info.channels) == (6, 4, 3)
    back, _ = SequenceLoader(info).load(0)
    assert back.shape == (4, 6, 3)


def test_zarr_sequence_and_frame_metadata(tmp_path):
    frames = [np.full((4, 5), i, dtype=np.uint8) for i in range(4)]
    metas = [{"frame_id": i, "pixel_format": "Mono8", "acquired_dtype": "uint8"}
             for i in range(4)]
    out = savers.save_zarr_sequence(tmp_path / "seq.zarr", frames, metas,
                                    pixel_format="Mono8", camera_settings=SETTINGS)
    info = describe_source(out)
    assert info.kind == "zarr_sequence"
    assert (info.n_frames, info.width, info.height, info.dtype) == (4, 5, 4, "uint8")
    arr, meta = SequenceLoader(info).load(2)
    assert np.all(arr == 2)
    assert meta["frame_id"] == 2


def test_describe_source_rejects_unknown_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe_source(tmp_path / "nope.raw")
    junk = tmp_path / "x.txt"
    junk.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        describe_source(junk)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        describe_source(empty)


# ---------------------------------------------------------------------------
# The reader tab
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _spin(app, predicate, timeout: float = 5.0) -> bool:
    """Run the event loop until *predicate* holds (the reader reads lazily)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _sequence(tmp_path: Path, n: int = 20) -> Path:
    frames = [np.full((8, 10), i * 10, dtype=np.uint16) for i in range(n)]
    metas = [{"frame_id": i, "pixel_format": "Mono16", "acquired_dtype": "uint16"}
             for i in range(n)]
    return savers.save_hdf5_sequence(tmp_path / "seq.hdf5", frames, metas,
                                     pixel_format="Mono16", camera_settings=SETTINGS)


def test_reader_tab_scrubs_and_plays(qapp, tmp_path):
    out = _sequence(tmp_path)
    tab = FrameReaderTab()
    assert tab.open_path(out) is True
    assert _spin(qapp, lambda: tab.view._arr is not None)
    assert tab.view._arr[0, 0] == 0
    assert tab.lbl_pos.text() == "0 / 20"

    tab.seek(12)
    assert _spin(qapp, lambda: tab.view._arr is not None and tab.view._arr[0, 0] == 120)
    assert tab.lbl_pos.text() == "12 / 20"

    # Dragging the scrubber quickly collapses to a single read of the last frame.
    tab.slider.setValue(15)
    tab.slider.setValue(7)
    assert _spin(qapp, lambda: tab.view._arr is not None and tab.view._arr[0, 0] == 70)
    assert tab.lbl_pos.text() == "7 / 20"

    start = tab._index
    tab.play()
    assert tab.is_playing
    assert _spin(qapp, lambda: tab._index != start)
    tab.stop()
    assert not tab.is_playing

    tab.set_display_settings({"palette": "viridis", "auto_levels": False,
                              "bit_depth": 12, "orientation": "rotate 90° CW"})
    assert tab.view.orientation == "rotate 90° CW"
    assert tab.view.displayed_size == (8, 10)   # 10 x 8 frame rotated


def test_reader_tab_reports_bad_path(qapp, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr("flir_cam_gui.frame_reader.QMessageBox.warning",
                        lambda *a, **k: warned.append(a[-1]))
    tab = FrameReaderTab()
    assert tab.open_path(tmp_path / "missing.raw") is False
    assert warned and "missing.raw" in str(warned[0])
