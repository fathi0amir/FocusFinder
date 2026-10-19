import json

import h5py
import numpy as np
import pytest

from flir_cam_gui.streaming import StreamingRecorder, resolve_stream_path
from flir_cam_gui.main import format_mmss, stack_length_text

SETTINGS = {"camera": {}, "stream": {}}
N = 10  # not a multiple of the chunk size, so a partial chunk must be flushed


def frames():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 4000, (6, 8), dtype=np.uint16) for _ in range(N)]


def record(tmp_path, fmt, name):
    data = frames()
    rec = StreamingRecorder(tmp_path / name, fmt, 4, camera_settings=SETTINGS)
    for i, a in enumerate(data):
        rec.push(a, {"frame_id": i, "pixel_format": "Mono16"})
    rec.finish()
    assert rec.join(30)
    return rec, np.stack(data)


def test_hdf5_all_frames_flushed(tmp_path):
    rec, data = record(tmp_path, "hdf5_sequence", "s.h5")
    assert rec.result["ok"] and rec.result["frames"] == N
    assert rec.chunks_written == 3 and rec.pending_frames == 0
    with h5py.File(rec.out_path) as f:
        np.testing.assert_array_equal(f["images"][...], data)
    side = json.loads(rec.out_path.with_suffix(".h5.json").read_text())
    assert len(side["frame_metadata"]["frames"]) == N


def test_npy_header_patched(tmp_path):
    rec, data = record(tmp_path, "npy_sequence", "s.npy")
    np.testing.assert_array_equal(np.load(rec.out_path), data)


def test_tiff_stack(tmp_path):
    import tifffile
    rec, data = record(tmp_path, "tiff_sequence", "s.tif")
    np.testing.assert_array_equal(tifffile.imread(rec.out_path), data)


def test_zarr_stack(tmp_path):
    zarr = pytest.importorskip("zarr")
    rec, data = record(tmp_path, "zarr_sequence", "s.zarr")
    np.testing.assert_array_equal(zarr.open_group(rec.out_path)["image"][...], data)


def test_raw_folder(tmp_path):
    rec, data = record(tmp_path, "raw_sequence", "s")
    files = sorted(rec.out_path.glob("frame_*.raw"))
    assert len(files) == N
    got = np.frombuffer(files[-1].read_bytes(), np.uint16).reshape(6, 8)
    np.testing.assert_array_equal(got, data[-1])
    assert (rec.out_path / "sequence.json").exists()


def test_no_frames_reports_failure(tmp_path):
    rec = StreamingRecorder(tmp_path / "e.h5", "hdf5_sequence", 4, camera_settings=SETTINGS)
    rec.finish()
    assert rec.join(10) and not rec.result["ok"]


def test_resolve_and_duration():
    assert resolve_stream_path("a/b", "npy_sequence").suffix == ".npy"
    assert format_mmss(125) == "2:05"
    assert stack_length_text(300, 30.0).startswith("0:10")
