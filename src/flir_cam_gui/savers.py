"""Image persistence: raw binary + JSON sidecar, and HDF5 writers.

Pure-logic helpers (no Qt, no PySpin) so they can be unit-tested and reused.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

APP_NAME = "flir-cam-gui"


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def build_sidecar(
    *,
    image_info: Dict[str, Any],
    frame_meta: Optional[Dict[str, Any]],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON sidecar dictionary saved next to every image file.

    Parameters
    ----------
    image_info : details about the pixel payload (name, width, height, dtype,
        byte count, pixel format string, storage description).
    frame_meta : per-frame acquisition metadata (frame id, timestamps, chunk
        data such as exposure/gain when enabled).
    camera_settings : full GenICam node-map snapshot (as produced by the
        camera worker).
    display_settings : optional live-view display settings that were active.
    """
    sidecar: Dict[str, Any] = {
        "schema": 1,
        "application": APP_NAME,
        "app_version": _app_version(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": image_info,
        "frame_metadata": frame_meta or {},
        "camera_settings": camera_settings,
    }
    if display_settings:
        sidecar["display_settings"] = display_settings
    return sidecar


#: per-frame keys identical for every frame; kept once in the JSON, not in parquet
GLOBAL_FRAME_KEYS = ("camera_model", "sensor", "bit_depth", "width", "height",
                     "pixel_format", "acquired_shape", "acquired_dtype", "chunk")


def frames_parquet_path(sidecar_path: Path) -> Path:
    """Parquet file holding the per-frame metadata of *sidecar_path*."""
    s = str(sidecar_path)
    return Path((s[:-len(".json")] if s.lower().endswith(".json") else s) + ".frames.parquet")


def write_frames_parquet(path: Path, frames: Sequence[Dict[str, Any]]) -> None:
    """One row per frame, one column per metadata key."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys: List[str] = []
    for fr in frames:
        for k in fr:
            if k not in keys:
                keys.append(k)
    cols = {}
    for k in keys:
        values = [fr.get(k) for fr in frames]
        try:
            cols[k] = pa.array(values)
        except Exception:  # mixed / unsupported types
            cols[k] = pa.array([None if v is None else json.dumps(v, default=_json_default)
                                for v in values], type=pa.string())
    pq.write_table(pa.table(cols), path, compression="zstd")


def read_frames_parquet(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


def _add_fps(frames: List[Dict[str, Any]]) -> None:
    """Instantaneous fps from the gap to the previous frame (first frame uses the next gap)."""
    ts = [fr.get("device_timestamp_ns", fr.get("received_at_ns")) for fr in frames]
    for i, fr in enumerate(frames):
        if "fps" in fr or len(frames) < 2:
            continue
        a, b = (ts[i - 1], ts[i]) if i else (ts[0], ts[1])
        if a is not None and b is not None and b > a:
            fr["fps"] = 1e9 / (b - a)


def write_sidecar(path: Path, sidecar: Dict[str, Any]) -> None:
    """Write the JSON sidecar; a per-frame list goes to a parquet file instead."""
    path = Path(path)
    meta = sidecar.get("frame_metadata")
    if isinstance(meta, dict) and isinstance(meta.get("frames"), list):
        frames = [dict(m or {}) for m in meta["frames"]]
        constants = {k: frames[0][k] for k in GLOBAL_FRAME_KEYS if frames and k in frames[0]}
        frames = [{k: v for k, v in fr.items() if k not in GLOBAL_FRAME_KEYS} for fr in frames]
        _add_fps(frames)
        pq_path = frames_parquet_path(path)
        write_frames_parquet(pq_path, frames)
        sidecar = {**sidecar, "frame_metadata": {
            "frames_file": pq_path.name, "format": "parquet", "count": len(frames),
            "constant": constants}}
    path.write_text(json.dumps(sidecar, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return {"__bytes__": len(obj)}
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _app_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Raw binary
# ---------------------------------------------------------------------------

def write_raw(path: Path, arr: np.ndarray) -> None:
    """Write raw pixel data: row-major, C-contiguous, little-endian bytes."""
    Path(path).write_bytes(np.ascontiguousarray(arr).tobytes(order="C"))


def frame_geometry(arr: np.ndarray) -> Tuple[int, int, int]:
    """(height, width, channels) of one frame.

    A third axis is only a channel axis for colour frames -- using ``shape[-1]``
    blindly reports width=3 for an RGB image.
    """
    if arr.ndim == 3:
        return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1]), 1
    return 1, int(arr.shape[-1]), 1


def raw_image_info(path: Path, arr: np.ndarray, pixel_format: str) -> Dict[str, Any]:
    """Describe raw pixel payload. For colour frames the channel axis is kept
    separate so (height, width) never collapses into the channel count."""
    height, width, channels = frame_geometry(arr)
    return {
        "name": Path(path).name,
        "type": "raw_binary",
        "description": (
            "Raw pixel data, row-major (C order), no header. "
            "Reshape as (height, width[, channels]) in this dtype to recover "
            "the image."
        ),
        "byte_order": "little",
        "width": width,
        "height": height,
        "channels": channels,
        "shape": [int(s) for s in arr.shape],
        "dtype": str(arr.dtype),
        "bytes": int(arr.nbytes),
        "pixel_format": pixel_format,
    }


def save_raw_with_sidecar(
    path: Path,
    arr: np.ndarray,
    *,
    pixel_format: str,
    frame_meta: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save raw pixel binary + a sidecar holding every setting for that image."""
    path = Path(path)
    write_raw(path, arr)
    write_sidecar(
        path.with_suffix(path.suffix + ".json"),
        build_sidecar(
            image_info=raw_image_info(path, arr, pixel_format),
            frame_meta=frame_meta,
            camera_settings=camera_settings,
            display_settings=display_settings,
        ),
    )
    return path


# ---------------------------------------------------------------------------
# NumPy .npy / TIFF
# ---------------------------------------------------------------------------

def _save_array_file(path: Path, arr: np.ndarray, kind: str, writer, *, pixel_format,
                     frame_meta, camera_settings, display_settings) -> Path:
    path = Path(path)
    writer(path, np.ascontiguousarray(arr))
    info = raw_image_info(path, arr, pixel_format)
    info["type"] = kind
    info["description"] = f"Single frame stored as {kind}."
    write_sidecar(
        path.with_suffix(path.suffix + ".json"),
        build_sidecar(image_info=info, frame_meta=frame_meta,
                      camera_settings=camera_settings,
                      display_settings=display_settings))
    return path


def save_npy(path: Path, arr: np.ndarray, **kw) -> Path:
    return _save_array_file(path, arr, "npy", lambda p, a: np.save(p, a), **kw)


def save_tiff(path: Path, arr: np.ndarray, **kw) -> Path:
    import tifffile
    return _save_array_file(path, arr, "tiff",
                            lambda p, a: tifffile.imwrite(p, a, bigtiff=True), **kw)


def _save_stack_file(path: Path, frames: Sequence[np.ndarray], metas: Sequence[Dict[str, Any]],
                     kind: str, writer, *, pixel_format, camera_settings,
                     display_settings=None) -> Path:
    if not frames:
        raise ValueError("no frames captured")
    arrs = [np.ascontiguousarray(a) for a in frames]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"mixed frame shapes {shapes}; cannot stack sequence")
    path = Path(path)
    writer(path, np.stack(arrs))
    h, w, c = frame_geometry(arrs[0])
    write_sidecar(path.with_suffix(path.suffix + ".json"), build_sidecar(
        image_info={"name": path.name, "type": f"{kind}_sequence", "width": w, "height": h,
                    "channels": c, "dtype": str(arrs[0].dtype), "frames": len(arrs),
                    "shape": [len(arrs), *arrs[0].shape], "pixel_format": pixel_format},
        frame_meta={"frames": [dict(m or {}) for m in metas]},
        camera_settings=camera_settings, display_settings=display_settings))
    return path


def save_npy_sequence(path: Path, frames, metas, **kw) -> Path:
    """Stack of frames as one N x H x W[ x C] .npy array."""
    return _save_stack_file(path, frames, metas, "npy", lambda p, a: np.save(p, a), **kw)


def save_tiff_sequence(path: Path, frames, metas, **kw) -> Path:
    """Stack of frames as one multi-page TIFF (one page per frame)."""
    import tifffile
    return _save_stack_file(path, frames, metas, "tiff",
                            lambda p, a: tifffile.imwrite(p, a, bigtiff=True), **kw)


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def save_hdf5(
    path: Path,
    arr: np.ndarray,
    *,
    pixel_format: str,
    frame_meta: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    include_sidecar: bool = True,
) -> Path:
    """Save a single frame as HDF5 ('/image' dataset + metadata attributes)."""
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=arr)
        grp = f.require_group("metadata")
        grp.attrs["pixel_format"] = pixel_format
        grp.attrs["frame_metadata_json"] = json.dumps(frame_meta or {}, default=_json_default)
        grp.attrs["camera_settings_json"] = json.dumps(camera_settings, default=_json_default)
        if display_settings:
            grp.attrs["display_settings_json"] = json.dumps(display_settings, default=_json_default)
        f.attrs["application"] = APP_NAME
        f.attrs["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if camera_settings:
            for key, value in flatten_settings(camera_settings).items():
                f.attrs[f"cam/{key}"] = value
    if include_sidecar:
        write_sidecar(
            path.with_suffix(path.suffix + ".json"),
            build_sidecar(
                image_info={
                    "name": path.name, "type": "hdf5_single",
                    "width": frame_geometry(arr)[1],
                    "height": frame_geometry(arr)[0],
                    "channels": frame_geometry(arr)[2],
                    "dtype": str(arr.dtype), "pixel_format": pixel_format},
                frame_meta=frame_meta, camera_settings=camera_settings,
                display_settings=display_settings),
        )
    return path


def save_hdf5_sequence(
    path: Path,
    frames: Sequence[np.ndarray],
    metas: Sequence[Dict[str, Any]],
    *,
    pixel_format: str,
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a recorded burst as one HDF5 dataset ('/images', N x H x W[ x C])."""
    import h5py

    if not frames:
        raise ValueError("no frames captured")
    path = Path(path)
    frame_meta_list: List[Dict[str, Any]] = [dict(m or {}) for m in metas]
    with h5py.File(path, "w") as f:
        arrs = [np.ascontiguousarray(a) for a in frames]
        # Frames may differ in size (ROI changes mid-record). Pad stays: refuse
        # mismatched shapes to avoid silently corrupting the dataset.
        shapes = {a.shape for a in arrs}
        if len(shapes) != 1:
            raise ValueError(f"mixed frame shapes {shapes}; cannot stack sequence")
        f.create_dataset(
            "images",
            data=np.stack(arrs),
            chunks=(1,) + arrs[0].shape,
            compression="lzf",
        )
        grp = f.require_group("metadata")
        grp.attrs["pixel_format"] = pixel_format
        grp.attrs["frame_count"] = len(frame_meta_list)
        grp.attrs["camera_settings_json"] = json.dumps(camera_settings, default=_json_default)
        if display_settings:
            grp.attrs["display_settings_json"] = json.dumps(display_settings, default=_json_default)
        f.attrs["application"] = APP_NAME
        f.attrs["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if camera_settings:
            for key, value in flatten_settings(camera_settings).items():
                f.attrs[f"cam/{key}"] = value
    # per-frame metadata as sidecar json as well
    write_sidecar(path.with_suffix(path.suffix + ".json"), build_sidecar(
        image_info={
            "name": path.name,
            "type": "hdf5_sequence",
            "width": frame_geometry(frames[0])[1],
            "height": frame_geometry(frames[0])[0],
            "channels": frame_geometry(frames[0])[2],
            "dtype": str(frames[0].dtype),
            "frames": len(frames),
            "pixel_format": pixel_format,
        },
        frame_meta={"frames": frame_meta_list},
        camera_settings=camera_settings,
        display_settings=display_settings,
    ))
    return path


# ---------------------------------------------------------------------------
# Zarr + Blosc (lz4/zstd, level 1-3, shuffle)
# ---------------------------------------------------------------------------

BLOSC_CNAMES = ("lz4", "zstd")
BLOSC_CLEVELS = (1, 2, 3)
BLOSC_SHUFFLES = ("noshuffle", "shuffle", "bitshuffle")


def blosc_codec(
    cname: str = "lz4", clevel: int = 1, shuffle: str = "shuffle"
):
    """Build a Zarr v3 Blosc codec; specification allows level 1-3 + shuffle."""
    if cname not in BLOSC_CNAMES:
        raise ValueError(f"blosc cname must be one of {BLOSC_CNAMES}, got {cname!r}")
    clevel = int(clevel)
    if clevel not in BLOSC_CLEVELS:
        raise ValueError(f"blosc clevel must be one of {BLOSC_CLEVELS}, got {clevel}")
    if shuffle not in BLOSC_SHUFFLES:
        raise ValueError(f"blosc shuffle must be one of {BLOSC_SHUFFLES}, got {shuffle!r}")
    import zarr.codecs

    return zarr.codecs.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)


def zarr_store_meta(store_path: Path, array_name: str = "image") -> Dict[str, Any]:
    """Read a Zarr store's array metadata (codecs etc.) for verification."""
    meta_path = Path(store_path) / array_name / "zarr.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _zarr_attrs(group, *, pixel_format, camera_settings,
                display_settings, frame_meta, array=None):
    attrs = {
        "pixel_format": pixel_format,
        "application": APP_NAME,
        "camera_settings_json": json.dumps(camera_settings or {},
                                           default=_json_default),
    }
    if frame_meta is not None:
        attrs["frame_metadata_json"] = json.dumps(frame_meta, default=_json_default)
    if display_settings:
        attrs["display_settings_json"] = json.dumps(display_settings,
                                                    default=_json_default)
    group.attrs.update(attrs)
    if array is not None:
        array.attrs.update(attrs)


def save_zarr(
    path: Path,
    arr: np.ndarray,
    *,
    pixel_format: str,
    frame_meta: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    blosc_cname: str = "lz4",
    blosc_clevel: int = 1,
    blosc_shuffle: str = "shuffle",
    include_sidecar: bool = True,
) -> Path:
    """Save a single frame as a Zarr v3 store with Blosc-compressed chunks."""
    import zarr
    from zarr.storage import LocalStore

    path = Path(path)
    codec = blosc_codec(blosc_cname, blosc_clevel, blosc_shuffle)
    if path.suffix != ".zarr":
        path = path.with_suffix(".zarr")
    group = zarr.open_group(store=LocalStore(root=path), mode="w")
    z = group.create_array(
        "image", shape=arr.shape, dtype=arr.dtype,
        chunks=arr.shape, compressors=[codec], overwrite=True,
    )
    z[...] = arr
    _zarr_attrs(group, pixel_format=pixel_format, camera_settings=camera_settings,
                display_settings=display_settings, frame_meta=frame_meta, array=z)
    if include_sidecar:
        write_sidecar(
            path.with_suffix(path.suffix + ".json"),
            build_sidecar(
                image_info={
                    "name": path.name, "type": "zarr_single", "format": "zarr3",
                    "width": frame_geometry(arr)[1],
                    "height": frame_geometry(arr)[0],
                    "channels": frame_geometry(arr)[2],
                    "dtype": str(arr.dtype), "pixel_format": pixel_format,
                    "compressor": {"name": "blosc", "cname": blosc_cname,
                                   "clevel": int(blosc_clevel),
                                   "shuffle": blosc_shuffle},
                },
                frame_meta=frame_meta, camera_settings=camera_settings,
                display_settings=display_settings),
        )
    return path


def save_zarr_sequence(
    path: Path,
    frames: Sequence[np.ndarray],
    metas: Sequence[Dict[str, Any]],
    *,
    pixel_format: str,
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    blosc_cname: str = "lz4",
    blosc_clevel: int = 1,
    blosc_shuffle: str = "shuffle",
) -> Path:
    """Record a burst as one Zarr array ('/image', frames along axis 0)."""
    import zarr
    from zarr.storage import LocalStore

    if not frames:
        raise ValueError("no frames captured")
    path = Path(path)
    if path.suffix != ".zarr":
        path = path.with_suffix(".zarr")
    arrs = [np.ascontiguousarray(a) for a in frames]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"mixed frame shapes {shapes}; cannot stack sequence")
    shape = arrs[0].shape
    codec = blosc_codec(blosc_cname, blosc_clevel, blosc_shuffle)
    group = zarr.open_group(store=LocalStore(root=path), mode="w")
    z = group.create_array(
        "image", shape=(len(arrs), *shape), dtype=arrs[0].dtype,
        chunks=(1, *shape), compressors=[codec], overwrite=True,
    )
    z[...] = np.stack(arrs)
    _zarr_attrs(group, pixel_format=pixel_format, camera_settings=camera_settings,
                display_settings=display_settings,
                frame_meta={"frames": [dict(m or {}) for m in metas]}, array=z)
    write_sidecar(
        path.with_suffix(path.suffix + ".json"),
        build_sidecar(
            image_info={
                "name": path.name, "type": "zarr_sequence", "format": "zarr3",
                "width": frame_geometry(arrs[0])[1],
                "height": frame_geometry(arrs[0])[0],
                "channels": frame_geometry(arrs[0])[2],
                "dtype": str(arrs[0].dtype), "frames": len(arrs),
                "pixel_format": pixel_format,
                "compressor": {"name": "blosc", "cname": blosc_cname,
                               "clevel": int(blosc_clevel),
                               "shuffle": blosc_shuffle},
            },
            frame_meta={"frames": [dict(m or {}) for m in metas]},
            camera_settings=camera_settings, display_settings=display_settings),
    )
    return path


def flatten_settings(tree: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten the nested settings snapshot into 'Category/Feature: value'."""
    flat: Dict[str, str] = {}
    for name, node in tree.items():
        if not isinstance(node, dict):
            continue
        if node.get("__type__") == "category":
            flat.update(flatten_settings(node.get("features", {}), f"{prefix}{name}/"))
        elif node.get("readable"):
            flat[f"{prefix}{name}"] = str(node.get("value", ""))
    return flat

