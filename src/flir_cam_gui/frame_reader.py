"""Saved-stack reader: open a recorded stack (raw / HDF5 / Zarr) and play it
back lazily.

Only the frame currently on screen is read from disk, and the frame index is
coalesced so dragging the scrubber quickly across a 10 000 frame stack does not
read every intermediate frame.  Frame metadata (and the camera/display settings
recorded with the file) come from the JSON sidecar written next to the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSlider, QVBoxLayout,
                             QWidget)

from .live_view import LiveViewWidget

#: dataset names written by this application, in preference order
_DATASETS = ("images", "image")
#: pixel formats whose HDF5/Zarr frames are (H, W, 3) rather than a stack
_COLOUR_HINTS = ("rgb", "color", "bayer", "bgr")


class SequenceInfo:
    """What a saved stack is: geometry and dtype, without reading any pixels."""

    def __init__(self, *, kind: str, path: Path, n_frames: int, width: int,
                 height: int, dtype: str, pixel_format: str = "",
                 channels: int = 1, files: Sequence[Path] = (),
                 dataset: str = "", stacked: bool = False,
                 frame_meta: Optional[List[Dict[str, Any]]] = None,
                 camera_settings: Optional[Dict[str, Any]] = None):
        self.kind = kind
        self.path = Path(path)
        self.n_frames = int(n_frames)
        self.width = int(width)
        self.height = int(height)
        self.dtype = dtype
        self.pixel_format = pixel_format
        self.channels = int(channels)
        self.files = [Path(f) for f in files]
        self.dataset = dataset
        #: True when the first axis of the HDF5/Zarr dataset is the frame axis.
        self.stacked = bool(stacked)
        self.frame_meta = frame_meta or []
        self.camera_settings = camera_settings or {}

    def __repr__(self) -> str:
        return (f"<SequenceInfo {self.kind} n={self.n_frames} "
                f"{self.width}x{self.height} {self.dtype}>")

    @property
    def label(self) -> str:
        return f"{self.kind}  ({self.path.name})"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def describe_source(path: str | Path) -> SequenceInfo:
    """Inspect *path* and describe the stack it holds (no pixel data read).

    Accepts a data file (``.raw``/``.h5``/``.hdf5``), a Zarr store or raw
    sequence directory, or the JSON sidecar written next to any of those.
    """
    p = Path(path)
    if str(p).lower().endswith(".json"):
        # A sidecar describes the data file it sits next to.
        p = Path(str(p)[: -len(".json")])
        if not p.exists() and p.name == "sequence":
            p = p.parent          # a raw sequence folder's 'sequence.json'
    if p.is_dir():
        return _describe_dir(p)
    if not p.exists():
        raise FileNotFoundError(f"no such file or folder: {p}")
    return _describe_file(p)


def _read_sidecar(path: Path) -> Dict[str, Any]:
    sidecar = Path(str(path) + ".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _sidecar_frames(sidecar: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-frame metadata: a list for stacks, or the lone frame's own dict."""
    meta = sidecar.get("frame_metadata")
    if isinstance(meta, list):
        return [dict(m) for m in meta if isinstance(m, dict)]
    if isinstance(meta, dict):
        frames = meta.get("frames")
        if isinstance(frames, list):
            return [dict(m) for m in frames if isinstance(m, dict)]
        return [dict(meta)] if meta else []
    return []


def _dtype_from_sidecar(file_info: Dict[str, Any], frames: List[Dict[str, Any]]) -> str:
    dtype = file_info.get("dtype")
    if dtype:
        return str(dtype)
    for frame in frames:
        if frame.get("acquired_dtype"):
            return str(frame["acquired_dtype"])
    if file_info.get("channels") and int(file_info["channels"]) > 1:
        return "uint8"
    return "uint16"


def _describe_file(path: Path) -> SequenceInfo:
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return _describe_hdf5(path)
    if suffix == ".zarr":
        return _describe_zarr(path)
    if suffix == ".raw":
        return _describe_raw_single(path)
    if suffix == ".npy":
        return _describe_npy(path)
    if suffix in (".tif", ".tiff"):
        return _describe_tiff(path)
    raise ValueError(f"unsupported file type '{path.suffix}' ({path.name})")


def _describe_dir(d: Path) -> SequenceInfo:
    if (d / "zarr.json").exists() or (d / ".zgroup").exists() or (d / "image").is_dir():
        return _describe_zarr(d)
    raws = sorted(d.glob("*.raw"))
    if not raws:
        raise ValueError(f"no '.raw' frames or Zarr store in {d}")
    return _describe_raw_dir(d, raws)


# -- raw -------------------------------------------------------------------

def _describe_raw_single(path: Path) -> SequenceInfo:
    sidecar = _read_sidecar(path)
    file_info = sidecar.get("file") or {}
    frames = _sidecar_frames(sidecar)
    dtype = _dtype_from_sidecar(file_info, frames)
    width = int(file_info.get("width") or (frames[0].get("width") if frames else 0) or 0)
    height = int(file_info.get("height") or (frames[0].get("height") if frames else 0) or 0)
    if not width or not height:
        count = path.stat().st_size // np.dtype(dtype).itemsize
        width = int(round(count ** 0.5)) or 1
        height = max(1, count // width)
    return SequenceInfo(
        kind="raw_single", path=path, n_frames=1, width=width, height=height,
        dtype=dtype, pixel_format=str(file_info.get("pixel_format") or ""),
        channels=int(file_info.get("channels") or 1),
        frame_meta=frames, camera_settings=sidecar.get("camera_settings") or {})


def _describe_raw_dir(d: Path, raws: List[Path]) -> SequenceInfo:
    sidecar: Dict[str, Any] = {}
    if (d / "sequence.json").exists():
        try:
            sidecar = json.loads((d / "sequence.json").read_text(encoding="utf-8"))
        except Exception:
            sidecar = {}
    file_info = sidecar.get("file") or {}
    frames = _sidecar_frames(sidecar)
    dtype = _dtype_from_sidecar(file_info, frames)
    width = int(file_info.get("width") or (frames[0].get("width") if frames else 0) or 0)
    height = int(file_info.get("height") or (frames[0].get("height") if frames else 0) or 0)
    files: List[Path] = []
    for frame in frames:
        name = frame.get("file")
        if name:
            files.append(d / name)
    if not files:
        files = raws
    if not width or not height:
        count = files[0].stat().st_size // np.dtype(dtype).itemsize
        width = int(round(count ** 0.5)) or 1
        height = max(1, count // width)
    return SequenceInfo(
        kind="raw_sequence", path=d, n_frames=len(files), width=width,
        height=height, dtype=dtype,
        pixel_format=str(file_info.get("pixel_format") or ""),
        channels=int(file_info.get("channels") or 1), files=files,
        frame_meta=frames, camera_settings=sidecar.get("camera_settings") or {})


# -- npy / TIFF ------------------------------------------------------------

def _describe_npy(path: Path) -> SequenceInfo:
    sidecar = _read_sidecar(path)
    file_info = sidecar.get("file") or {}
    frames = _sidecar_frames(sidecar)
    arr = np.load(path, mmap_mode="r")
    shape = tuple(arr.shape)
    if len(shape) < 2:
        raise ValueError(f"{path.name}: expected an image array, got shape {shape}")
    ftype = str(file_info.get("type") or "")
    if ftype:
        stacked = ftype.endswith("_sequence") and len(shape) >= 3
    else:
        stacked = len(shape) == 4 or (len(shape) == 3 and shape[2] not in (3, 4))
    if stacked:
        n_frames, height, width = shape[0], shape[1], shape[2]
        channels = shape[3] if len(shape) == 4 else 1
    else:
        n_frames, height, width = 1, shape[0], shape[1]
        channels = shape[2] if len(shape) == 3 else 1
    return SequenceInfo(
        kind="npy", path=path, n_frames=n_frames, width=width, height=height,
        dtype=str(arr.dtype), pixel_format=str(file_info.get("pixel_format") or ""),
        channels=channels, stacked=stacked, frame_meta=frames,
        camera_settings=sidecar.get("camera_settings") or {})


def _describe_tiff(path: Path) -> SequenceInfo:
    import tifffile

    sidecar = _read_sidecar(path)
    file_info = sidecar.get("file") or {}
    frames = _sidecar_frames(sidecar)
    with tifffile.TiffFile(path) as tif:
        n_frames = len(tif.pages)
        page = tif.pages[0]
        shape = tuple(page.shape)
        dtype = str(page.dtype)
    height, width = shape[0], shape[1]
    channels = shape[2] if len(shape) == 3 else 1
    return SequenceInfo(
        kind="tiff", path=path, n_frames=n_frames, width=width, height=height,
        dtype=dtype, pixel_format=str(file_info.get("pixel_format") or ""),
        channels=channels, stacked=n_frames > 1, frame_meta=frames,
        camera_settings=sidecar.get("camera_settings") or {})


# -- HDF5 ------------------------------------------------------------------

def _describe_hdf5(path: Path) -> SequenceInfo:
    import h5py

    sidecar = _read_sidecar(path)
    file_info = sidecar.get("file") or {}
    frames = _sidecar_frames(sidecar)
    with h5py.File(path, "r") as f:
        name = next((n for n in _DATASETS if n in f and hasattr(f[n], "shape")), None)
        if name is None:
            raise ValueError(f"{path.name}: no 'images'/'image' dataset found")
        ds = f[name]
        shape = tuple(ds.shape)
        dtype = str(ds.dtype)
        pixel_format = ""
        if "metadata" in f:
            pixel_format = str(f["metadata"].attrs.get("pixel_format") or "")
        is_sequence = name == "images" or (
            len(shape) >= 3 and not _looks_like_colour(shape, file_info, pixel_format))
        if len(shape) == 2:
            is_sequence = False
        if is_sequence and len(shape) >= 3:
            n_frames, height, width = int(shape[0]), int(shape[1]), int(shape[2])
            channels = int(shape[3]) if len(shape) == 4 else 1
        else:
            n_frames = 1
            height, width = int(shape[0]), int(shape[1])
            channels = int(shape[2]) if len(shape) == 3 else 1
    if file_info.get("frames"):
        n_frames = int(file_info["frames"]) or n_frames
    return SequenceInfo(
        kind="hdf5_sequence" if n_frames > 1 else "hdf5_single", path=path,
        n_frames=n_frames, width=int(file_info.get("width") or width),
        height=int(file_info.get("height") or height),
        dtype=str(file_info.get("dtype") or dtype),
        pixel_format=str(file_info.get("pixel_format") or pixel_format),
        channels=int(file_info.get("channels") or channels), dataset=name,
        stacked=is_sequence, frame_meta=frames,
        camera_settings=sidecar.get("camera_settings") or {})


# -- Zarr ------------------------------------------------------------------

def _describe_zarr(path: Path) -> SequenceInfo:
    import zarr
    from zarr.storage import LocalStore

    sidecar = _read_sidecar(path)
    file_info = sidecar.get("file") or {}
    frames = _sidecar_frames(sidecar)
    root = zarr.open_group(store=LocalStore(root=path), mode="r")
    name = next((n for n in _DATASETS if n in root), None)
    if name is None:
        arrays = sorted(root.array_keys())
        if not arrays:
            raise ValueError(f"{path.name}: no array found in Zarr store")
        name = arrays[0]
    z = root[name]
    shape = tuple(z.shape)
    dtype = str(z.dtype)
    attrs = dict(getattr(z, "attrs", {}) or {})
    pixel_format = str(attrs.get("pixel_format") or "")
    is_sequence = (len(shape) >= 3
                   and not _looks_like_colour(shape, file_info, pixel_format))
    if len(shape) == 2:
        is_sequence = False
    if is_sequence:
        n_frames, height, width = int(shape[0]), int(shape[1]), int(shape[2])
        channels = int(shape[3]) if len(shape) == 4 else 1
    else:
        n_frames = 1
        height, width = int(shape[0]), int(shape[1])
        channels = int(shape[2]) if len(shape) == 3 else 1
    if not frames and attrs.get("frame_metadata_json"):
        try:
            frames = list(json.loads(attrs["frame_metadata_json"]).get("frames") or [])
        except Exception:
            frames = []
    if file_info.get("frames"):
        n_frames = int(file_info["frames"]) or n_frames
    return SequenceInfo(
        kind="zarr_sequence" if n_frames > 1 else "zarr_single", path=path,
        n_frames=n_frames, width=int(file_info.get("width") or width),
        height=int(file_info.get("height") or height),
        dtype=str(file_info.get("dtype") or dtype),
        pixel_format=str(file_info.get("pixel_format") or pixel_format),
        channels=int(file_info.get("channels") or channels), dataset=name,
        stacked=is_sequence, frame_meta=frames,
        camera_settings=sidecar.get("camera_settings") or {})


def _looks_like_colour(shape: Tuple[int, ...], file_info: Dict[str, Any],
                       pixel_format: str) -> bool:
    """True when a 3-D dataset is really one (H, W, 3) colour frame."""
    if len(shape) != 3 or shape[2] not in (3, 4):
        return False
    hint = f"{file_info.get('pixel_format', '')} {pixel_format}".lower()
    return any(h in hint for h in _COLOUR_HINTS)


# ---------------------------------------------------------------------------
# Lazy frame reading
# ---------------------------------------------------------------------------

class SequenceLoader:
    """Reads one frame at a time from a saved stack.

    The dataset handle is opened on first access (so listing a stack never
    reads pixels) and only the most recently read frame is kept in memory.
    """

    def __init__(self, info: SequenceInfo):
        self.info = info
        self._h5 = None
        self._h5_dataset = None
        self._zarr_array = None
        self._npy = None
        self._tiff = None
        self._cache: Optional[Tuple[int, np.ndarray, Dict[str, Any]]] = None

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
        self._h5 = None
        self._h5_dataset = None
        self._zarr_array = None
        if self._tiff is not None:
            try:
                self._tiff.close()
            except Exception:
                pass
        self._npy = None
        self._tiff = None
        self._cache = None

    @property
    def handle_open(self) -> bool:
        return (self._h5 is not None or self._zarr_array is not None
                or self._npy is not None or self._tiff is not None)

    def frame_count(self) -> int:
        return max(0, self.info.n_frames)

    # -- reading ------------------------------------------------------------
    def meta_for(self, idx: int) -> Dict[str, Any]:
        frames = self.info.frame_meta
        if 0 <= idx < len(frames):
            return dict(frames[idx] or {})
        return {}

    def load(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        idx = int(idx)
        if not 0 <= idx < self.frame_count():
            raise IndexError(f"frame {idx} out of range 0..{self.frame_count() - 1}")
        if self._cache is not None and self._cache[0] == idx:
            return self._cache[1], self._cache[2]
        arr = self._read(idx)
        meta = self.meta_for(idx)
        self._cache = (idx, arr, meta)
        return arr, meta

    def _read(self, idx: int) -> np.ndarray:
        kind = self.info.kind
        if kind == "raw_single":
            return self._read_raw(self.info.path)
        if kind == "raw_sequence":
            return self._read_raw(self.info.files[idx])
        if kind in ("hdf5_sequence", "hdf5_single"):
            return self._from_dataset(self._dataset(), idx)
        if kind in ("zarr_sequence", "zarr_single"):
            return self._from_dataset(self._array(), idx)
        if kind == "npy":
            return self._read_npy(idx)
        if kind == "tiff":
            return self._read_tiff(idx)
        raise NotImplementedError(f"no reader for '{kind}'")

    def _read_npy(self, idx: int) -> np.ndarray:
        if self._npy is None:
            self._npy = np.load(self.info.path, mmap_mode="r")
        return np.array(self._npy[idx] if self.info.stacked else self._npy)

    def _read_tiff(self, idx: int) -> np.ndarray:
        if self._tiff is None:
            import tifffile

            self._tiff = tifffile.TiffFile(self.info.path)
        return self._tiff.pages[idx].asarray()

    def _from_dataset(self, ds, idx: int) -> np.ndarray:
        # A single-frame file stores the frame itself, with no frame axis.
        return np.asarray(ds[idx] if self.info.stacked else ds[...])

    def _read_raw(self, path: Path) -> np.ndarray:
        info = self.info
        dtype = np.dtype(info.dtype)
        shape = ((info.height, info.width) if info.channels <= 1
                 else (info.height, info.width, info.channels))
        wanted = int(np.prod(shape))
        data = np.fromfile(path, dtype=dtype, count=wanted)
        if data.size < wanted:
            raise ValueError(f"{path.name}: {data.size} samples, expected {wanted}")
        return data.reshape(shape)

    def _dataset(self):
        if self._h5_dataset is None:
            import h5py

            self._h5 = h5py.File(self.info.path, "r")
            name = self.info.dataset or next(
                (n for n in _DATASETS if n in self._h5), "image")
            self._h5_dataset = self._h5[name]
        return self._h5_dataset

    def _array(self):
        if self._zarr_array is None:
            import zarr
            from zarr.storage import LocalStore

            root = zarr.open_group(store=LocalStore(root=self.info.path), mode="r")
            name = self.info.dataset or next(
                (n for n in _DATASETS if n in root), next(iter(root.array_keys()), "image"))
            self._zarr_array = root[name]
        return self._zarr_array


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------

class FrameReaderTab(QWidget):
    """Tab that opens a saved stack and plays it back frame by frame."""

    frame_changed = pyqtSignal(int, int)   # frame index, frame count
    status = pyqtSignal(str)               # human readable message

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._loader: Optional[SequenceLoader] = None
        self._info: Optional[SequenceInfo] = None
        self._index = 0
        self._play_index = 0
        self._playing = False
        self._rate_hz = 10.0
        self._loop = True
        self._display_settings: Dict[str, Any] = {}
        self.view = LiveViewWidget()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._pending_timer = QTimer(self)
        self._pending_timer.setSingleShot(True)
        self._pending_timer.setInterval(0)
        self._pending_timer.timeout.connect(self._load_pending)
        self._pending_index: Optional[int] = None

        self._build_ui()
        self._connect_ui()
        self._set_enabled(False)

    # -- UI -----------------------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        top = QHBoxLayout()
        btn_file = QPushButton("Open file...")
        btn_folder = QPushButton("Open folder...")
        self.btn_play = QPushButton("▶ Play")
        self.btn_stop = QPushButton("■ Stop")
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["1 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"])
        self.cmb_speed.setCurrentIndex(2)
        self.chk_loop = QPushButton("Loop: on")
        self.chk_loop.setCheckable(True)
        self.chk_loop.setChecked(True)
        for w in (btn_file, btn_folder, self.btn_play, self.btn_stop, self.cmb_speed,
                  self.chk_loop):
            top.addWidget(w)
        top.addStretch(1)
        lay.addLayout(top)
        self.btn_open_file = btn_file
        self.btn_open_folder = btn_folder

        self.lbl_title = QLabel("No stack loaded — open a .raw / .h5 / .zarr / .npy / .tif "
                               "(or a folder of raw frames) to browse it.")
        lay.addWidget(self.lbl_title)

        scrub = QHBoxLayout()
        scrub.addWidget(QLabel("Frame:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setTracking(True)
        scrub.addWidget(self.slider, 1)
        self.lbl_pos = QLabel("0 / 0")
        self.lbl_pos.setMinimumWidth(90)
        scrub.addWidget(self.lbl_pos)
        self.btn_first = QPushButton("|<")
        self.btn_prev = QPushButton("<")
        self.btn_next = QPushButton(">")
        self.btn_last = QPushButton(">|")
        for b in (self.btn_first, self.btn_prev, self.btn_next, self.btn_last):
            scrub.addWidget(b)
        lay.addLayout(scrub)

        lay.addWidget(self.view, 1)

        self.lbl_info = QLabel("frames -   size -   dtype -   format -")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)
        self.lbl_meta = QLabel("metadata -")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_meta)

    def _connect_ui(self):
        self.btn_open_file.clicked.connect(self._prompt_open_file)
        self.btn_open_folder.clicked.connect(self._prompt_open_folder)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(lambda *_: self.stop())
        self.cmb_speed.currentTextChanged.connect(self._update_speed)
        self.chk_loop.toggled.connect(self._update_loop)
        self.slider.valueChanged.connect(self._on_slider)
        self.btn_first.clicked.connect(lambda: self.seek(0))
        self.btn_prev.clicked.connect(lambda: self.seek(self._index - 1))
        self.btn_next.clicked.connect(lambda: self.seek(self._index + 1))
        self.btn_last.clicked.connect(lambda: self.seek(self.frame_count() - 1))

    def _set_enabled(self, on: bool):
        for w in (self.btn_play, self.btn_stop, self.slider, self.btn_first,
                  self.btn_prev, self.btn_next, self.btn_last):
            w.setEnabled(on)

    # -- opening ------------------------------------------------------------
    def _prompt_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open saved stack", str(Path.home()),
            "Stacks (*.h5 *.hdf5 *.raw *.npy *.tif *.tiff *.hdf5.json *.raw.json "
            "*.npy.json *.tif.json);;"
            "Zarr store (*.zarr);;All files (*)")
        if path:
            self.open_path(path)

    def _prompt_open_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open stack folder (Zarr store or raw frame folder)",
            str(Path.home()))
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        """Load the stack at *path*, returning True on success."""
        self.stop()
        try:
            info = describe_source(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open stack", f"{path}\n\n{exc}")
            self.status.emit(f"Could not open {path}: {exc}")
            return False
        if self._loader is not None:
            self._loader.close()
        self._info = info
        self._loader = SequenceLoader(info)
        self._index = 0
        self._play_index = 0
        self._populate()
        self._apply_display_settings()
        self.view.clear_frame()    # never show the previous stack's frame
        self.pending_show(0, force=True)
        self.status.emit(f"Opened {info.label}: {info.n_frames} frame(s), "
                         f"{info.width}x{info.height} {info.dtype}")
        return True

    # -- display ------------------------------------------------------------
    def set_display_settings(self, settings: Dict[str, Any]) -> None:
        self._display_settings = dict(settings or {})
        self._apply_display_settings()

    def _apply_display_settings(self):
        s = self._display_settings
        if not s:
            return
        if s.get("palette"):
            self.view.set_palette(str(s["palette"]))
        if s.get("orientation"):
            self.view.set_orientation(str(s["orientation"]))
        bits = s.get("bit_depth")
        auto = bool(s.get("auto_levels", True))
        if auto:
            bits = bits or self._bit_depth_hint()
        self.view.set_bit_depth(int(bits or 12), auto)
        if "manual_min" in s and "manual_max" in s:
            self.view.set_contrast(auto, s["manual_min"], s["manual_max"], s.get("auto_max"))

    def _bit_depth_hint(self) -> int:
        info = self._info
        if info is None:
            return 12
        try:
            return max(8, min(16, int(np.dtype(info.dtype).itemsize * 8)))
        except Exception:
            return 12

    # -- UI state -----------------------------------------------------------
    def frame_count(self) -> int:
        return self._loader.frame_count() if self._loader is not None else 0

    def _populate(self):
        info = self._info
        if info is None:
            self._set_enabled(False)
            self.lbl_title.setText("No stack loaded.")
            self.lbl_info.setText("frames -   size -   dtype -   format -")
            self.lbl_meta.setText("metadata -")
            self.slider.setMaximum(0)
            self.lbl_pos.setText("0 / 0")
            return
        n = self.frame_count()
        self._set_enabled(n > 0)
        self.lbl_title.setText(f"Opened: {info.label}")
        self.lbl_info.setText(
            f"frames {n}   size {info.width} x {info.height}   "
            f"dtype {info.dtype}   format {info.pixel_format or info.kind}")
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.lbl_pos.setText(f"0 / {n}")
        self.lbl_meta.setText("metadata -")

    # -- loading ------------------------------------------------------------
    def pending_show(self, idx: int, force: bool = False) -> None:
        """Request frame *idx*; rapid requests collapse into one read."""
        if self._loader is None:
            return
        idx = max(0, min(int(idx), self.frame_count() - 1))
        if not force and idx == self._index and self.view.displayed_size is not None:
            self._update_position(idx)
            return
        self._pending_index = idx
        if not self._pending_timer.isActive():
            self._pending_timer.start()

    def _load_pending(self):
        idx, self._pending_index = self._pending_index, None
        if idx is None or self._loader is None:
            return
        try:
            arr, meta = self._loader.load(idx)
        except Exception as exc:
            self.status.emit(f"Frame {idx} could not be read: {exc}")
            self.stop()
            return
        self._index = idx
        self.view.show_frame(arr, meta)
        self._update_position(idx)
        self.frame_changed.emit(idx, self.frame_count())

    def _update_position(self, idx: int):
        n = self.frame_count()
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_pos.setText(f"{idx} / {n}")
        self._update_meta_text(idx)

    def _update_meta_text(self, idx: int):
        info = self._info
        if info is None or self._loader is None:
            self.lbl_meta.setText("metadata -")
            return
        parts = [f"frame {idx} of {info.n_frames}", info.kind]
        meta = self._loader.meta_for(idx)
        for key in ("frame_id", "host_timestamp_s", "exposure_time_us", "gain_db",
                    "saturated_fraction"):
            value = meta.get(key)
            if value is None and isinstance(meta.get("chunk"), dict):
                value = meta["chunk"].get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        self.lbl_meta.setText("   ".join(str(p) for p in parts))

    # -- playback -----------------------------------------------------------
    def _toggle_play(self):
        if self._playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if self.frame_count() == 0:
            return
        self._playing = True
        self.btn_play.setText("⏸ Pause")
        interval = max(1, int(round(1000.0 / self._rate_hz)))
        self._timer.start(interval)

    def stop(self):
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("▶ Play")

    @property
    def is_playing(self) -> bool:
        return self._playing

    def _tick(self):
        n = self.frame_count()
        if n == 0:
            self.stop()
            return
        nxt = self._play_index + 1
        if nxt >= n:
            if not self._loop:
                self.stop()
                return
            nxt = 0
        self._play_index = nxt
        self.pending_show(nxt)

    def _update_speed(self, text: str):
        try:
            self._rate_hz = max(0.1, float(text.split()[0]))
        except Exception:
            return
        if self._playing:
            self._timer.setInterval(max(1, int(round(1000.0 / self._rate_hz))))

    def _update_loop(self, checked: bool):
        self._loop = bool(checked)
        self.chk_loop.setText(f"Loop: {'on' if self._loop else 'off'}")

    # -- seeking ------------------------------------------------------------
    def seek(self, idx: int):
        if self._loader is None or self.frame_count() == 0:
            return
        idx = max(0, min(int(idx), self.frame_count() - 1))
        self._play_index = idx
        self.pending_show(idx)

    def _on_slider(self, value: int):
        idx = int(value)
        self._play_index = idx
        self.pending_show(idx)

    def closeEvent(self, event):  # pragma: no cover - Qt teardown
        self.stop()
        if self._loader is not None:
            self._loader.close()
        super().closeEvent(event)
