"""Chunked, asynchronous recording to disk (no Qt, no PySpin).

The acquisition thread only calls ``StreamingRecorder.push`` (a queue put).  A
dedicated writer thread gathers frames into chunks of ``chunk_frames`` and
writes each chunk to disk, so slow storage never blocks the camera.  Call
``finish`` once acquisition has ended: the writer drains every queued frame,
writes the final partial chunk, finalises the file and sidecar, then reports.
"""

from __future__ import annotations

import queue
import struct
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import savers

STREAM_FORMATS = ("hdf5_sequence", "raw_sequence", "zarr_sequence",
                  "npy_sequence", "tiff_sequence")

_NPY_HEADER_LEN = 256  # fixed so the shape can be patched in place at the end


def resolve_stream_path(path, fmt: str) -> Path:
    """Final output path (file, Zarr store or folder) for a streamed recording."""
    p = Path(path)
    if fmt == "hdf5_sequence":
        return p if p.suffix in (".h5", ".hdf5") else p.with_suffix(".hdf5")
    if fmt == "zarr_sequence":
        return p if p.suffix == ".zarr" else p.with_suffix(".zarr")
    if fmt == "npy_sequence":
        return p if p.suffix.lower() == ".npy" else p.with_suffix(".npy")
    if fmt == "tiff_sequence":
        return p if p.suffix.lower() in (".tif", ".tiff") else p.with_suffix(".tif")
    if fmt == "raw_sequence":
        d = p if p.suffix == "" else p.with_suffix("")
        return d if str(d).endswith("_seq") else Path(str(d) + "_seq")
    raise ValueError(f"unsupported streaming format {fmt!r}")


def _npy_header(shape, dtype) -> bytes:
    desc = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False, "shape": tuple(int(s) for s in shape)}
    body = repr(desc).ljust(_NPY_HEADER_LEN - 11) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", _NPY_HEADER_LEN - 10) + body.encode("latin1")


# ---------------------------------------------------------------------------
# Per-format sinks: write(stack: N x H x W[ x C], first_index, metas) / close(ctx)
# ---------------------------------------------------------------------------

class _Hdf5Sink:
    def __init__(self, path, first, chunk_frames, blosc):
        import h5py
        self.f = h5py.File(path, "w")
        self.ds = self.f.create_dataset(
            "images", shape=(0, *first.shape), maxshape=(None, *first.shape),
            dtype=first.dtype, chunks=(1, *first.shape), compression="lzf")

    def write(self, stack, first_index, metas):
        n = self.ds.shape[0]
        self.ds.resize(n + len(stack), axis=0)
        self.ds[n:] = stack
        self.f.flush()

    def close(self, ctx):
        import json
        import time
        grp = self.f.require_group("metadata")
        grp.attrs["pixel_format"] = ctx["pixel_format"]
        grp.attrs["frame_count"] = ctx["frames"]
        grp.attrs["camera_settings_json"] = json.dumps(
            ctx["camera_settings"], default=savers._json_default)
        if ctx["display_settings"]:
            grp.attrs["display_settings_json"] = json.dumps(
                ctx["display_settings"], default=savers._json_default)
        self.f.attrs["application"] = savers.APP_NAME
        self.f.attrs["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for key, value in savers.flatten_settings(ctx["camera_settings"]).items():
            self.f.attrs[f"cam/{key}"] = value
        self.f.close()

    def abort(self):
        self.f.close()


class _ZarrSink:
    def __init__(self, path, first, chunk_frames, blosc):
        import zarr
        from zarr.storage import LocalStore
        blosc = dict(blosc or {})
        codec = savers.blosc_codec(blosc.get("blosc_cname", "lz4"),
                                   blosc.get("blosc_clevel", 1),
                                   blosc.get("blosc_shuffle", "shuffle"))
        self.blosc = {"name": "blosc", "cname": blosc.get("blosc_cname", "lz4"),
                      "clevel": int(blosc.get("blosc_clevel", 1)),
                      "shuffle": blosc.get("blosc_shuffle", "shuffle")}
        self.shape = first.shape
        self.group = zarr.open_group(store=LocalStore(root=path), mode="w")
        self.z = self.group.create_array(
            "image", shape=(0, *first.shape), dtype=first.dtype,
            chunks=(chunk_frames, *first.shape), compressors=[codec], overwrite=True)

    def write(self, stack, first_index, metas):
        n = self.z.shape[0]
        self.z.resize((n + len(stack), *self.shape))
        self.z[n:] = stack

    def close(self, ctx):
        savers._zarr_attrs(
            self.group, pixel_format=ctx["pixel_format"],
            camera_settings=ctx["camera_settings"],
            display_settings=ctx["display_settings"],
            frame_meta={"frames": ctx["metas"]}, array=self.z)
        ctx["info"]["compressor"] = self.blosc

    def abort(self):
        pass


class _NpySink:
    def __init__(self, path, first, chunk_frames, blosc):
        self.shape, self.dtype, self.n = first.shape, first.dtype, 0
        self.f = open(path, "wb")
        self.f.write(_npy_header((0, *self.shape), self.dtype))

    def write(self, stack, first_index, metas):
        self.f.write(np.ascontiguousarray(stack).tobytes(order="C"))
        self.n += len(stack)

    def close(self, ctx):
        self.f.flush()
        self.f.seek(0)
        self.f.write(_npy_header((self.n, *self.shape), self.dtype))
        self.f.close()

    def abort(self):
        self.f.close()


class _TiffSink:
    def __init__(self, path, first, chunk_frames, blosc):
        import tifffile
        self.tw = tifffile.TiffWriter(path, bigtiff=True)

    def write(self, stack, first_index, metas):
        for frame in stack:
            self.tw.write(frame, contiguous=True)

    def close(self, ctx):
        self.tw.close()

    def abort(self):
        self.tw.close()


class _RawSink:
    def __init__(self, path, first, chunk_frames, blosc):
        self.d = Path(path)
        self.d.mkdir(parents=True, exist_ok=True)

    def write(self, stack, first_index, metas):
        for k, arr in enumerate(stack):
            name = f"frame_{first_index + k:06d}.raw"
            (self.d / name).write_bytes(np.ascontiguousarray(arr).tobytes(order="C"))

    def close(self, ctx):
        index = [{"file": f"frame_{i:06d}.raw", **m} for i, m in enumerate(ctx["metas"])]
        ctx["info"]["type"] = "raw_sequence"
        ctx["info"]["byte_order"] = "little"
        ctx["frame_meta_frames"] = index

    def abort(self):
        pass


_SINKS = {"hdf5_sequence": _Hdf5Sink, "zarr_sequence": _ZarrSink,
          "npy_sequence": _NpySink, "tiff_sequence": _TiffSink,
          "raw_sequence": _RawSink}


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class StreamingRecorder:
    """Queue-fed chunked writer; see module docstring."""

    def __init__(self, path, fmt: str, chunk_frames: int, *,
                 camera_settings: Dict[str, Any],
                 display_settings: Optional[Dict[str, Any]] = None,
                 blosc: Optional[Dict[str, Any]] = None):
        if fmt not in _SINKS:
            raise ValueError(f"unsupported streaming format {fmt!r}")
        self.fmt = fmt
        self.chunk_frames = max(1, int(chunk_frames))
        self.out_path = resolve_stream_path(path, fmt)
        self.camera_settings = camera_settings
        self.display_settings = display_settings
        self.blosc = blosc
        self.error: Optional[str] = None
        self.frames_queued = 0      # producer thread only
        self.frames_written = 0     # writer thread only (also counts discarded)
        self.chunks_written = 0
        self.result: Optional[Dict[str, Any]] = None
        self._on_done: Optional[Callable[[Dict[str, Any]], None]] = None
        self._q: "queue.Queue" = queue.Queue()
        self._sink = None
        self._metas: List[Dict[str, Any]] = []
        self._first: Optional[np.ndarray] = None
        self._thread = threading.Thread(target=self._run, name="stream-writer")
        self._thread.start()

    # -- producer side (acquisition thread) ----------------------------------
    @property
    def pending_frames(self) -> int:
        return self.frames_queued - self.frames_written

    def push(self, arr: np.ndarray, meta: Optional[Dict[str, Any]]) -> None:
        """Hand a frame to the writer; never touches the disk."""
        self.frames_queued += 1
        self._q.put((arr, dict(meta or {})))

    def finish(self, on_done: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Request finalisation; the writer flushes everything queued so far."""
        self._on_done = on_done
        self._q.put(None)

    def join(self, timeout: Optional[float] = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    # -- writer thread -------------------------------------------------------
    def _run(self) -> None:
        arrs: List[np.ndarray] = []
        while True:
            item = self._q.get()
            if item is None:
                break
            if self.error is not None:
                self.frames_written += 1      # discard, keep the backlog bounded
                continue
            arrs.append(item[0])
            self._metas.append(item[1])
            if len(arrs) >= self.chunk_frames:
                self._write(arrs)
                arrs = []
        if arrs and self.error is None:
            self._write(arrs)
        self.result = self._close()
        if self._on_done is not None:
            try:
                self._on_done(self.result)
            except Exception:
                pass

    def _write(self, arrs: List[np.ndarray]) -> None:
        first_index = self.frames_written
        try:
            shapes = {a.shape for a in arrs}
            if self._first is not None:
                shapes.add(self._first.shape)
            if len(shapes) != 1:
                raise ValueError("mixed frame shapes; cannot stack sequence")
            if self._sink is None:
                self._first = arrs[0]
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                self._sink = _SINKS[self.fmt](self.out_path, arrs[0],
                                              self.chunk_frames, self.blosc)
            self._sink.write(np.stack([np.ascontiguousarray(a) for a in arrs]),
                             first_index, self._metas[first_index:first_index + len(arrs)])
            self.chunks_written += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self.frames_written += len(arrs)

    def _close(self) -> Dict[str, Any]:
        path = str(self.out_path)
        n = len(self._metas)
        if self.error is None and self._sink is None:
            self.error = "no frames captured"
        if self.error is not None:
            if self._sink is not None:
                try:
                    self._sink.abort()
                except Exception:
                    pass
            return {"ok": False, "path": path, "frames": self.frames_written,
                    "message": f"Record failed: {self.error}"}
        try:
            first = self._first
            h, w, c = savers.frame_geometry(first)
            pixel_format = str(self._metas[0].get("pixel_format") or "unknown")
            info = {"name": self.out_path.name, "type": self.fmt, "width": w, "height": h,
                    "channels": c, "dtype": str(first.dtype), "frames": n,
                    "shape": [n, *first.shape], "pixel_format": pixel_format,
                    "streamed": True, "chunk_frames": self.chunk_frames}
            ctx = {"pixel_format": pixel_format, "frames": n, "metas": self._metas,
                   "camera_settings": self.camera_settings,
                   "display_settings": self.display_settings, "info": info}
            self._sink.close(ctx)
            frame_meta = {"frames": ctx.get("frame_meta_frames", self._metas)}
            sidecar = savers.build_sidecar(
                image_info=info, frame_meta=frame_meta,
                camera_settings=self.camera_settings,
                display_settings=self.display_settings)
            target = (self.out_path / "sequence.json" if self.fmt == "raw_sequence"
                      else self.out_path.with_suffix(self.out_path.suffix + ".json"))
            savers.write_sidecar(target, sidecar)
        except Exception as exc:
            return {"ok": False, "path": path, "frames": n,
                    "message": f"Record failed while finalising: {exc}"}
        return {"ok": True, "path": path, "frames": n,
                "message": f"Recorded: {path}"}
