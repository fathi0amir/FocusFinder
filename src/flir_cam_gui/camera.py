"""Camera backends: FLIR Spinnaker (PySpin) worker thread + offline demo camera.

All camera/node-map access happens inside the worker thread; the GUI talks to
the worker exclusively through message passing (commands in, signals out), so
GenICam nodes are never touched from the UI thread.
"""

from __future__ import annotations

import collections
import copy
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import PySpin

    PYSPIN_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host SDK
    PySpin = None
    PYSPIN_AVAILABLE = False

from PyQt6.QtCore import QThread, pyqtSignal

# --- feature kinds (plain strings, shared with the node browser) -----------
KIND_CATEGORY = "category"
KIND_INTEGER = "integer"
KIND_FLOAT = "float"
KIND_BOOLEAN = "boolean"
KIND_ENUMERATION = "enumeration"
KIND_STRING = "string"
KIND_COMMAND = "command"
KIND_VALUE = "value"

SIGNALS = dict(
    frameReady=pyqtSignal(object, object),   # ndarray, meta dict
    treeReady=pyqtSignal(object),            # full settings snapshot
    rowUpdated=pyqtSignal(object),           # single feature snapshot
    saveDone=pyqtSignal(object),             # result dict {ok, path, message}
    stateChanged=pyqtSignal(str),
    error=pyqtSignal(str),
    log=pyqtSignal(str),
)


class _WorkerMixin:
    """Command-queue plumbing shared by both backends."""

    def _init_common(self):
        self._cmd_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._display_settings: Dict[str, Any] = {}
        self._last_frame: Optional[tuple] = None  # (arr, meta)
        self._writers: list = []   # streaming recorders still flushing to disk
        # Latest-frame mailbox: the acquisition thread overwrites it, the GUI
        # polls it on its own timer, so display speed never throttles acquisition.
        self._mailbox_lock = threading.Lock()
        self._mailbox: Optional[tuple] = None
        self._mailbox_seq = 0
        self._taken_seq = 0
        self.acq_times: "collections.deque[float]" = collections.deque(maxlen=240)

    def _publish_frame(self, arr, meta) -> None:
        self._last_frame = (arr, meta)
        with self._mailbox_lock:
            self._mailbox = (arr, meta)
            self._mailbox_seq += 1
        self.acq_times.append(time.monotonic())

    def take_latest_frame(self):
        """Newest (arr, meta) not yet taken by the GUI, else None."""
        with self._mailbox_lock:
            if self._mailbox_seq == self._taken_seq:
                return None
            self._taken_seq = self._mailbox_seq
            return self._mailbox

    # -- streamed (chunked, asynchronous) recording --------------------------
    def _new_recording(self, path, fmt, n_frames, include_sidecar, blosc,
                       stream: bool, chunk_frames: int) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "path": Path(path), "fmt": fmt, "frames": [], "raws": [], "metas": [],
            "remaining": int(n_frames), "include_sidecar": include_sidecar,
            "blosc": dict(blosc or {}), "writer": None}
        if stream:
            from .streaming import StreamingRecorder
            # The node map may only be read from the worker thread, so the
            # settings are captured now rather than by the writer thread.
            rec["writer"] = StreamingRecorder(
                path, fmt, chunk_frames,
                camera_settings=self.current_settings_snapshot(),
                display_settings=dict(self._display_settings), blosc=blosc)
        return rec

    def _finish_stream(self, rec) -> None:
        """Hand the recording to its writer thread; saveDone fires when on disk."""
        writer = rec["writer"]
        self.log.emit(f"Acquisition ended - flushing {writer.pending_frames} queued "
                      f"frame(s) to disk ...")
        self._writers.append(writer)
        writer.finish(self.saveDone.emit)

    def has_pending_writes(self) -> bool:
        self._writers = [w for w in self._writers if not w.done]
        return bool(self._writers)

    def wait_for_writes(self) -> None:
        """Block until every streamed recording is fully written."""
        for w in list(self._writers):
            w.join()
        self._writers.clear()

    def submit(self, command: Dict[str, Any]) -> None:
        """Queue a command dict, e.g. {"action": "set_feature", ...}."""
        self._cmd_q.put(command)

    def drain_commands(self) -> List[Dict[str, Any]]:
        cmds: List[Dict[str, Any]] = []
        try:
            while True:
                cmds.append(self._cmd_q.get_nowait())
        except queue.Empty:
            pass
        return cmds

    def set_display_settings(self, settings: Dict[str, Any]) -> None:
        self._display_settings = dict(settings)

    @property
    def last_frame(self):
        return self._last_frame

    def _run_commands(self):
        for cmd in self.drain_commands():
            try:
                fn = getattr(self, f"_do_{cmd['action']}", None)
                if fn is None:
                    self.error.emit(f"Unknown command action: {cmd['action']}")
                else:
                    fn(**{k: v for k, v in cmd.items() if k != "action"})
            except Exception as exc:
                self.error.emit(f"Command failed: {exc}\n{traceback.format_exc(limit=3)}")

    def _save_current(self, path: str, fmt: str, arr, meta, include_sidecar=True,
                      blosc: Optional[Dict[str, Any]] = None):
        from . import savers
        settings = self.current_settings_snapshot()
        pixel_format = str((meta or {}).get("pixel_format") or arr.dtype)
        blosc_args = dict(blosc or {})
        blosc_args.setdefault("blosc_cname", "lz4")
        blosc_args.setdefault("blosc_clevel", 1)
        blosc_args.setdefault("blosc_shuffle", "shuffle")
        if fmt == "raw":
            out = Path(path)
            if out.suffix != ".raw":
                out = out.with_suffix(".raw")
            savers.save_raw_with_sidecar(
                out, arr, pixel_format=pixel_format, frame_meta=meta,
                camera_settings=settings,
                display_settings=self._display_settings)
            return out
        if fmt in ("npy", "tiff"):
            suffixes = (".npy",) if fmt == "npy" else (".tif", ".tiff")
            out = Path(path)
            if out.suffix.lower() not in suffixes:
                out = out.with_suffix(suffixes[0])
            (savers.save_npy if fmt == "npy" else savers.save_tiff)(
                out, arr, pixel_format=pixel_format, frame_meta=meta,
                camera_settings=settings,
                display_settings=self._display_settings)
            return out
        if fmt == "hdf5":
            out = path if str(path).endswith((".h5", ".hdf5")) else Path(str(path) + ".h5")
            savers.save_hdf5(
                out, arr, pixel_format=pixel_format, frame_meta=meta,
                camera_settings=settings,
                display_settings=self._display_settings,
                include_sidecar=include_sidecar)
            return out
        if fmt == "zarr":
            out = path if str(path).endswith(".zarr") else Path(str(path) + ".zarr")
            savers.save_zarr(
                out, arr, pixel_format=pixel_format, frame_meta=meta,
                camera_settings=settings,
                display_settings=self._display_settings,
                include_sidecar=include_sidecar, **blosc_args)
            return out
        raise ValueError(f"Unknown save format {fmt}")


# ===========================================================================
# Camera discovery (multi-camera selection)
# ===========================================================================

def _tl_value(cam, name: str, default: str = "") -> str:
    """Read a TL-device node (model / serial / transport) before Init()."""
    try:
        nodemap = cam.GetTLDeviceNodeMap()  # keep alive while node is used
        node = PySpin.CValuePtr(nodemap.GetNode(name))
        if PySpin.IsReadable(node):
            value = str(node.ToString())
            del node, nodemap
            return value
    except Exception:
        pass
    return default


def enumerate_cameras() -> List[Dict[str, str]]:
    """Detect every camera Spinnaker can see, without opening any of them.

    Returns a list of ``{"model", "serial", "transport"}`` dicts; ``serial``
    is empty when a device does not report one. The system instance acquired
    here is always released, so it can be called repeatedly (and from a worker
    thread) without leaking handles or disturbing a running connection.
    """
    if not PYSPIN_AVAILABLE:
        return []
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    try:
        out: List[Dict[str, str]] = []
        for i in range(cam_list.GetSize()):
            try:
                cam = cam_list.GetByIndex(i)
            except Exception:
                continue
            out.append({
                "model": _tl_value(cam, "DeviceModelName", "FLIR camera"),
                "serial": _tl_value(cam, "DeviceSerialNumber", ""),
                "transport": _tl_value(cam, "DeviceTransportType", ""),
            })
            del cam  # all camera refs must be dropped before ReleaseInstance
        return out
    finally:
        cam = None
        try:
            cam_list.Clear()
        except Exception:
            pass
        del cam_list
        try:
            system.ReleaseInstance()
        except Exception:
            pass


class CameraLister(QThread):
    """Enumerates cameras off the UI thread (Spinnaker discovery can block)."""

    camerasFound = pyqtSignal(object)   # list of {model, serial, transport}
    failed = pyqtSignal(str)

    def run(self):
        try:
            cams = enumerate_cameras()
        except Exception as exc:
            self.failed.emit(f"Camera scan failed: {exc}")
            return
        self.camerasFound.emit(cams)


# ===========================================================================
# Real Spinnaker camera
# ===========================================================================

class CameraWorker(QThread, _WorkerMixin):
    """Acquisition worker for one FLIR camera via PySpin."""

    frameReady = SIGNALS["frameReady"]
    treeReady = SIGNALS["treeReady"]
    rowUpdated = SIGNALS["rowUpdated"]
    saveDone = SIGNALS["saveDone"]
    stateChanged = SIGNALS["stateChanged"]
    error = SIGNALS["error"]
    log = SIGNALS["log"]

    def __init__(self, serial: Optional[str] = None, parent=None):
        QThread.__init__(self, parent)
        self._init_common()
        self.serial = serial
        self.cam = None
        self.nodemap = None
        self.stream_nodemap = None
        self.acquiring = False
        self._stop = False
        self._raw_needed = 0
        self._recording: Optional[Dict[str, Any]] = None
        self._pending_save: Optional[Dict[str, Any]] = None
        self.frame_timeout_ms = 5000

    # -- lifecycle ---------------------------------------------------------
    def stop(self) -> None:
        self._stop = True

    def run(self):
        if not PYSPIN_AVAILABLE:
            self.error.emit(
                "PySpin not available. Install the FLIR Spinnaker SDK (matching version) "
                "and add PySpin:  uv sync --extra spinnaker"
            )
            return
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        try:
            if cam_list.GetSize() == 0:
                self.error.emit("No FLIR cameras detected.")
                return
            self.cam = self._pick_camera(cam_list)
            self.cam.Init()
            self.nodemap = self.cam.GetNodeMap()
            self.stream_nodemap = self.cam.GetTLStreamNodeMap()
            self.log.emit(f"Connected: {self._camera_identity()}")
            self.stateChanged.emit("ready")
            self.refresh_tree()
            while not self._stop:
                self._run_commands()
                if self.acquiring:
                    self._grab_one()
                else:
                    time.sleep(0.02)
        except Exception as exc:
            self.error.emit(f"Camera thread error: {exc}\n{traceback.format_exc(limit=5)}")
        finally:
            self._shutdown(cam_list, system)

    def _pick_camera(self, cam_list):
        """Select the requested serial (or the first camera when none given).

        Serials live on the *TL device* node map, which is readable before
        Init(); a camera that cannot be matched raises an error naming what
        was actually detected instead of silently opening the wrong one.
        """
        if not self.serial:
            return cam_list.GetByIndex(0)
        wanted = self.serial.strip()
        seen: List[str] = []
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            serial = _tl_value(cam, "DeviceSerialNumber").strip()
            seen.append(serial or "?")
            if serial and serial == wanted:
                return cam
            del cam
        raise RuntimeError(
            f"No camera with serial '{self.serial}' (detected: "
            f"{', '.join(seen) or 'none'})")

    def _camera_identity(self) -> str:
        try:
            tl = self.cam.GetTLDeviceNodeMap()
            def val(name):
                n = PySpin.CValuePtr(tl.GetNode(name))
                return n.ToString() if PySpin.IsReadable(n) else "?"
            return f"{val('DeviceModelName')}   SN={val('DeviceSerialNumber')}"
        except Exception:
            return "FLIR camera"

    def _shutdown(self, cam_list, system):
        if self._recording is not None:
            self._finish_recording(early=True)
        self.wait_for_writes()
        try:
            if self.acquiring:
                self.cam.EndAcquisition()
        except Exception:
            pass
        self.acquiring = False
        try:
            if self.cam:
                self.cam.DeInit()
        except Exception:
            pass
        # Drop every camera/node-map reference first, or ReleaseInstance can crash.
        self.cam = None
        self.nodemap = None
        self.stream_nodemap = None
        try:
            cam_list.Clear()
        except Exception:
            pass
        try:
            system.ReleaseInstance()
        except Exception:
            pass
        self.stateChanged.emit("disconnected")

    # -- commands ------------------------------------------------------------
    def _do_start_acquisition(self):
        if self.acquiring:
            return
        node = PySpin.CEnumerationPtr(self.nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            entry = node.GetEntryByName("Continuous")
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                node.SetIntValue(entry.GetValue())
        self.cam.BeginAcquisition()
        self.acquiring = True
        self.stateChanged.emit("acquiring")

    def _do_stop_acquisition(self):
        if not self.acquiring:
            return
        self.acquiring = False
        if self._recording is not None:
            self._finish_recording(early=True)
        try:
            self.cam.EndAcquisition()
        except Exception:
            pass
        self.stateChanged.emit("ready")

    def _do_set_feature(self, name: str, value):
        node = self.nodemap.GetNode(name)
        if node is None or not PySpin.IsAvailable(node):
            self.error.emit(f"Feature '{name}' not found")
            return
        itype = node.GetPrincipalInterfaceType()
        try:
            if itype == PySpin.intfIEnumeration:
                enum = PySpin.CEnumerationPtr(node)
                entry = None
                if isinstance(value, str):
                    try:
                        e = enum.GetEntryByName(value)
                        if PySpin.IsAvailable(e) and PySpin.IsReadable(e):
                            entry = e
                    except Exception:
                        entry = None
                if entry is None:
                    try:
                        iv = int(value)
                        for e in enum.GetEntries():
                            ep = PySpin.CEnumEntryPtr(e)
                            if PySpin.IsAvailable(ep) and int(ep.GetValue()) == iv:
                                entry = ep
                                break
                    except Exception:
                        pass
                if entry is None:
                    raise ValueError(f"invalid entry {value!r}")
                enum.SetIntValue(entry.GetValue())
            elif itype == PySpin.intfIInteger:
                ptr = PySpin.CIntegerPtr(node)
                if not PySpin.IsWritable(node):
                    raise ValueError("feature is currently read-only (locked by another setting?)")
                target = int(round(float(value)))
                lo, hi, inc = int(ptr.GetMin()), int(ptr.GetMax()), max(1, int(ptr.GetInc()))
                target = min(hi, max(lo, target))
                target = lo + ((target - lo) // inc) * inc
                ptr.SetValue(target)
            elif itype == PySpin.intfIFloat:
                ptr = PySpin.CFloatPtr(node)
                if not PySpin.IsWritable(node):
                    raise ValueError("feature is currently read-only (locked by another setting?)")
                target = min(float(ptr.GetMax()), max(float(ptr.GetMin()), float(value)))
                ptr.SetValue(target)
            elif itype == PySpin.intfIBoolean:
                if isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                PySpin.CBooleanPtr(node).SetValue(bool(value))
            elif itype == PySpin.intfIString:
                PySpin.CStringPtr(node).SetValue(str(value))
            else:
                raise ValueError("feature is not writable")
            info = describe_node(node)
            if info:
                self.rowUpdated.emit(info)
        except Exception as exc:
            self.error.emit(f"Setting '{name}' -> '{value}' failed: {exc}")
            info = describe_node(node)  # revert the UI to the camera's real value
            if info:
                self.rowUpdated.emit(info)

    def _do_execute_command(self, name: str):
        node = self.nodemap.GetNode(name)
        if node is None or node.GetPrincipalInterfaceType() != PySpin.intfICommand:
            self.error.emit(f"Command '{name}' not found")
            return
        try:
            PySpin.CCommandPtr(node).Execute()
            self.log.emit(f"Executed: {name}")
        except Exception as exc:
            self.error.emit(f"Command '{name}' failed: {exc}")

    def _do_refresh_tree(self):
        self.refresh_tree()

    def _do_refresh_rows(self, names: List[str]):
        for name in names:
            if self.nodemap is None:
                return
            node = self.nodemap.GetNode(name)
            info = describe_node(node) if node is not None else None
            if info:
                self.rowUpdated.emit(info)

    def _do_save_last(self, path: str, fmt: str, include_sidecar: bool = True,
                      display_settings: Optional[Dict[str, Any]] = None,
                      blosc: Optional[Dict[str, Any]] = None):
        """Persist the most recent frame."""
        try:
            if display_settings:
                self._display_settings = dict(display_settings)
            if self._last_frame is None:
                raise RuntimeError("No frame captured yet (start acquisition first)")
            arr, meta = self._last_frame
            out = self._save_current(path, fmt, arr, meta, include_sidecar, blosc)
            self.saveDone.emit({"ok": True, "path": str(out), "message": f"Saved {out}"})
        except Exception as exc:
            self.saveDone.emit({"ok": False, "path": path, "message": str(exc)})

    def _do_record(self, path: str, n_frames: int, fmt: str,
                   include_sidecar: bool = True,
                   display_settings: Optional[Dict[str, Any]] = None,
                   blosc: Optional[Dict[str, Any]] = None,
                   stream: bool = False, chunk_frames: int = 64):
        if display_settings:
            self._display_settings = dict(display_settings)
        if not self.acquiring:
            self.saveDone.emit({"ok": False, "path": path,
                                "message": "Start acquisition before recording"})
            return
        if fmt not in SEQUENCE_FORMATS:
            self.saveDone.emit({"ok": False, "path": path, "message": f"bad fmt {fmt}"})
            return
        try:
            self._recording = self._new_recording(
                path, fmt, n_frames, include_sidecar, blosc, stream, chunk_frames)
        except Exception as exc:
            self.saveDone.emit({"ok": False, "path": path, "message": f"Record failed: {exc}"})
            return
        if fmt == "raw_sequence" and not stream:
            self._raw_needed += 1
        mode = f"streaming, chunks of {chunk_frames} frames" if stream else "in memory"
        self.log.emit(f"Recording {n_frames} frames to {path} ({mode}) ...")
        self.stateChanged.emit("recording")

    def _do_stop_record(self):
        if self._recording is not None:
            self._finish_recording(early=True)

    def refresh_tree(self):
        tree = {
            "camera": snapshot_nodemap(self.nodemap) if self.nodemap else {},
            "stream": snapshot_nodemap(self.stream_nodemap) if self.stream_nodemap else {},
        }
        self.treeReady.emit(tree)

    def current_settings_snapshot(self) -> Dict[str, Any]:
        """Nested snapshot (worker-thread context) used for sidecars."""
        return {
            "camera": snapshot_nodemap(self.nodemap) if self.nodemap else {},
            "stream": snapshot_nodemap(self.stream_nodemap) if self.stream_nodemap else {},
        }

    # -- frame grabbing ------------------------------------------------------
    def _grab_one(self):
        try:
            image = self.cam.GetNextImage(self.frame_timeout_ms)
        except Exception as exc:
            self.error.emit(f"GetNextImage failed: {exc}")
            self.acquiring = False
            self.stateChanged.emit("ready")
            return
        try:
            if image is None or image.IsIncomplete():
                try:
                    status = image.GetImageStatus() if image is not None else "timeout"
                except Exception:
                    status = "unknown"
                self.log.emit(f"Incomplete/timeout frame ({status})")
                return
            arr = self._to_ndarray(image)
            meta = self._collect_meta(image, arr)
            raw = None
            if self._raw_needed > 0:
                try:
                    count = image.GetBufferSize()
                    raw = bytes(bytearray(image.GetData()[:count]))
                except Exception:
                    raw = None
            self._publish_frame(arr, meta)
            if self._recording is not None:
                rec = self._recording
                if rec["writer"] is not None:
                    rec["writer"].push(arr, meta)
                    rec["remaining"] -= 1
                    if rec["remaining"] <= 0 or rec["writer"].error:
                        self._finish_recording(early=False)
                    return
                rec["metas"].append(meta)
                if rec["fmt"] == "raw_sequence" and raw is not None:
                    rec["raws"].append(raw)
                elif rec["fmt"] in ("hdf5_sequence", "zarr_sequence", "npy_sequence",
                                    "tiff_sequence"):
                    rec["frames"].append(arr.copy())
                rec["remaining"] -= 1
                if rec["remaining"] <= 0:
                    self._finish_recording(early=False)
        finally:
            try:
                image.Release()
            except Exception:
                pass

    def _to_ndarray(self, image) -> np.ndarray:
        try:
            return image.GetNDArray().copy()
        except Exception:
            processor = PySpin.ImageProcessor()
            processor.SetColorProcessing(
                PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
            fmt = (PySpin.PixelFormat_RGB8 if image.GetNumChannels() == 3
                   else PySpin.PixelFormat_Mono16)
            converted = processor.Convert(image, fmt)
            return converted.GetNDArray().copy()

    def _collect_meta(self, image, arr) -> Dict[str, Any]:
        def safe(fn, default=None):
            try:
                return fn()
            except Exception:
                return default
        meta: Dict[str, Any] = {
            "width": safe(image.GetWidth),
            "height": safe(image.GetHeight),
            "pixel_format": safe(image.GetPixelFormatName),
            "frame_id": safe(image.GetFrameID),
            "device_timestamp_ns": safe(image.GetTimeStamp),
            "received_at_ns": time.time_ns(),
            "host_timestamp_s": round(time.time(), 6),
            "acquired_shape": list(arr.shape),
            "acquired_dtype": str(arr.dtype),
        }
        try:
            if image.HasChunkData():
                chunks = image.GetChunkData()
                meta["chunk"] = {
                    "timestamp": safe(chunks.GetTimestamp),
                    "frame_id": safe(chunks.GetFrameID),
                    "exposure_time_us": safe(chunks.GetExposureTime),
                    "gain_db": safe(chunks.GetGain),
                    "frame_rate_hz": safe(chunks.GetFrameRate),
                    "offset_x": safe(chunks.GetOffsetX),
                    "offset_y": safe(chunks.GetOffsetY),
                    "width": safe(chunks.GetWidth),
                    "height": safe(chunks.GetHeight),
                }
        except Exception:
            pass
        return meta

    def _finish_recording(self, early: bool):
        rec, self._recording = self._recording, None
        if rec["writer"] is not None:
            self._finish_stream(rec)
            self.stateChanged.emit("acquiring" if self.acquiring else "ready")
            return
        if rec["fmt"] == "raw_sequence":
            self._raw_needed = max(0, self._raw_needed - 1)
        try:
            from . import savers
            settings = self.current_settings_snapshot()
            blosc = dict(rec.get("blosc") or {})
            blosc.setdefault("blosc_cname", "lz4")
            blosc.setdefault("blosc_clevel", 1)
            blosc.setdefault("blosc_shuffle", "shuffle")
            if rec["fmt"] in ("npy_sequence", "tiff_sequence"):
                out = _save_stack_recording(savers, rec, settings, self._display_settings)
            elif rec["fmt"] == "hdf5_sequence":
                out = rec["path"]
                if out.suffix not in (".h5", ".hdf5"):
                    out = out.with_suffix(".hdf5")
                savers.save_hdf5_sequence(
                    out, rec["frames"], rec["metas"],
                    pixel_format=str(rec["metas"][0].get("pixel_format") or "unknown"),
                    camera_settings=settings,
                    display_settings=self._display_settings)
            elif rec["fmt"] == "zarr_sequence":
                savers.save_zarr_sequence(
                    rec["path"], rec["frames"], rec["metas"],
                    pixel_format=str(rec["metas"][0].get("pixel_format") or "unknown"),
                    camera_settings=settings,
                    display_settings=self._display_settings, **blosc)
                out = rec["path"]
                if out.suffix != ".zarr":
                    out = Path(str(out) + ".zarr")
            else:  # raw_sequence -> folder of .raw files + sidecar
                d = rec["path"]
                d = d if d.suffix == "" else d.with_suffix("")
                if not str(d).endswith("_seq"):
                    d = Path(str(d) + "_seq")
                d.mkdir(parents=True, exist_ok=True)
                first_meta = rec["metas"][0] if rec["metas"] else {}
                width, height = int(first_meta.get("width", 0)), int(first_meta.get("height", 0))
                nchannels = 3 if (rec["raws"] and len(rec["raws"][0]) ==
                                  width * height * 3) else 1
                index = []
                for i, raw in enumerate(rec["raws"]):
                    fname = f"frame_{i:06d}.raw"
                    (d / fname).write_bytes(raw)
                    index.append({"file": fname, **rec["metas"][i]})
                savers.write_sidecar(
                    d / "sequence.json",
                    savers.build_sidecar(
                        image_info={"type": "raw_sequence", "frames": len(rec["raws"]),
                                    "width": width, "height": height,
                                    "channels": nchannels,
                                    "dtype": _raw_sequence_dtype(first_meta, nchannels),
                                    "byte_order": "little"},
                        frame_meta={"frames": index},
                        camera_settings=settings,
                        display_settings=self._display_settings))
                out = d
            msg = f"Recorded: {out}"
            self.saveDone.emit({"ok": True, "path": str(out), "message": msg,
                                "frames": len(rec["metas"])})
        except Exception as exc:
            self.saveDone.emit({"ok": False, "path": str(rec["path"]),
                                "message": f"Record failed: {exc}"})
        self.stateChanged.emit("acquiring" if self.acquiring else "ready")


# ===========================================================================
# PySpin node snapshot helpers
# ===========================================================================

SEQUENCE_FORMATS = ("hdf5_sequence", "raw_sequence", "zarr_sequence",
                    "npy_sequence", "tiff_sequence")


def _save_stack_recording(savers, rec, settings, display_settings) -> Path:
    """Write a finished recording as one .npy or multi-page TIFF file."""
    is_npy = rec["fmt"] == "npy_sequence"
    suffixes = (".npy",) if is_npy else (".tif", ".tiff")
    out = Path(rec["path"])
    if out.suffix.lower() not in suffixes:
        out = out.with_suffix(suffixes[0])
    (savers.save_npy_sequence if is_npy else savers.save_tiff_sequence)(
        out, rec["frames"], rec["metas"],
        pixel_format=str(rec["metas"][0].get("pixel_format") or "unknown"),
        camera_settings=settings, display_settings=display_settings)
    return out


def _raw_sequence_dtype(meta: Optional[Dict[str, Any]], channels: int) -> str:
    """Pixel dtype of a raw sequence, so the sidecar is self-describing."""
    dtype = (meta or {}).get("acquired_dtype")
    if dtype:
        return str(dtype)
    return "uint8" if int(channels) > 1 else "uint16"


def describe_node(node) -> Optional[Dict[str, Any]]:
    """Snapshot a GenICam node into a plain serializable dict (no PySpin refs)."""
    try:
        if not PySpin.IsAvailable(node):
            return None
        itype = node.GetPrincipalInterfaceType()
        info: Dict[str, Any] = {
            "name": node.GetName(),
            "display": node.GetDisplayName(),
            "tooltip": node.GetToolTip() or node.GetDescription() or "",
            "readable": bool(PySpin.IsReadable(node)),
            "writable": bool(PySpin.IsWritable(node)),
        }
        if itype == PySpin.intfICategory:
            info["kind"] = KIND_CATEGORY
            return info
        if itype == PySpin.intfIInteger:
            ptr = PySpin.CIntegerPtr(node)
            info["kind"] = KIND_INTEGER
            if info["readable"]:
                info["value"] = int(ptr.GetValue())
            for attr, getter in (("min", ptr.GetMin), ("max", ptr.GetMax)):
                try:
                    info[attr] = int(getter())
                except Exception:
                    pass
            try:
                info["inc"] = int(ptr.GetInc())
            except Exception:
                pass
            try:
                info["unit"] = ptr.GetUnit()
            except Exception:
                pass
        elif itype == PySpin.intfIFloat:
            ptr = PySpin.CFloatPtr(node)
            info["kind"] = KIND_FLOAT
            if info["readable"]:
                info["value"] = float(ptr.GetValue())
            for attr, getter in (("min", ptr.GetMin), ("max", ptr.GetMax)):
                try:
                    info[attr] = float(getter())
                except Exception:
                    pass
            try:
                info["unit"] = ptr.GetUnit()
            except Exception:
                pass
        elif itype == PySpin.intfIBoolean:
            ptr = PySpin.CBooleanPtr(node)
            info["kind"] = KIND_BOOLEAN
            if info["readable"]:
                info["value"] = bool(ptr.GetValue())
        elif itype == PySpin.intfIEnumeration:
            ptr = PySpin.CEnumerationPtr(node)
            info["kind"] = KIND_ENUMERATION
            current = None
            if info["readable"]:
                try:
                    current = ptr.ToString()
                except Exception:
                    pass
            entries: List[Dict[str, Any]] = []
            try:
                for entry in ptr.GetEntries():
                    ep = PySpin.CEnumEntryPtr(entry)
                    if not PySpin.IsAvailable(ep):
                        continue
                    entries.append({
                        "symbolic": ep.GetSymbolic(),
                        "value": int(ep.GetValue()),
                        "current": ep.GetSymbolic() == current,
                    })
            except Exception:
                pass
            info["entries"] = entries
            info["value"] = current
        elif itype == PySpin.intfIString:
            ptr = PySpin.CStringPtr(node)
            info["kind"] = KIND_STRING
            if info["readable"]:
                info["value"] = ptr.GetValue()
        elif itype == PySpin.intfICommand:
            info["kind"] = KIND_COMMAND
        else:
            info["kind"] = KIND_VALUE
            if info["readable"]:
                try:
                    info["value"] = node.ToString()
                except Exception:
                    info["value"] = ""
        return info
    except Exception:
        return None


def walk_category(category_node, depth: int = 0, max_depth: int = 8) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        cat = PySpin.CCategoryPtr(category_node)
        for feature in cat.GetFeatures():
            if feature.GetPrincipalInterfaceType() == PySpin.intfICategory:
                if depth < max_depth:
                    out[feature.GetName()] = {
                        "__type__": KIND_CATEGORY,
                        "display": feature.GetDisplayName(),
                        "writable": False,
                        "features": walk_category(feature, depth + 1, max_depth),
                    }
                continue
            info = describe_node(feature)
            if info:
                out[info["name"]] = info
    except Exception:
        pass
    return out


def snapshot_nodemap(nodemap) -> Dict[str, Any]:
    if nodemap is None:
        return {}
    root = nodemap.GetNode("Root")
    if root is None:
        return {}
    return {"__type__": KIND_CATEGORY, "display": "Root", "writable": False,
            "features": walk_category(root)}


# ===========================================================================
# Demo (simulated) camera — same interface, no hardware required
# ===========================================================================

# ---------------------------------------------------------------------------
# Simulated camera / sensor models (approximate public specs — offline only)
# ---------------------------------------------------------------------------

DEMO_MODELS: Dict[str, Dict[str, Any]] = {
    # NOTE: the sensor is mounted/delivered portrait on this module: 1536 px
    # wide by 2048 px high (width x height), so frames are (2048, 1536) arrays.
    "BFS-U3-200S6M (simulated CMV4000 CMOS, mono 12-bit)": {
        "sensor": "CMV4000 V3 CMOS, global shutter (simulated)",
        "width": 1536, "height": 2048,
        "pixel_format": "Mono16", "pixel_formats": ["Mono8", "Mono16"],
        "bit_depth": 12, "max_fps": 25.0,
        "adr_gain_adu_per_ke": 66.0,   # 4095 ADU / ~62 ke- full well
        "full_well_ke": 62.0,
        "read_noise_e": 11.0,
        "dark_current_e_per_s": 40.0,
        "baseline_adu": 120.0,
        "max_gain_db": 24.0,
    },
    "BFS-U3-31S4M (simulated IMX265 CMOS, mono 12-bit)": {
        "sensor": "Sony IMX265 CMOS, global shutter (simulated)",
        "width": 2048, "height": 1536,
        "pixel_format": "Mono16", "pixel_formats": ["Mono8", "Mono16"],
        "bit_depth": 12, "max_fps": 55.0,
        "adr_gain_adu_per_ke": 68.0,
        "full_well_ke": 60.0,
        "read_noise_e": 7.0,
        "dark_current_e_per_s": 12.0,
        "baseline_adu": 100.0,
        "max_gain_db": 20.0,
    },
    "GS3-U3-23S6M (simulated ICX672 CCD, mono 12-bit)": {
        "sensor": "Sony ICX672 CCD, interline transfer (simulated)",
        "width": 1920, "height": 1440,
        "pixel_format": "Mono16", "pixel_formats": ["Mono8", "Mono16"],
        "bit_depth": 12, "max_fps": 32.0,
        "adr_gain_adu_per_ke": 136.0,
        "full_well_ke": 30.0,
        "read_noise_e": 6.0,
        "dark_current_e_per_s": 4.0,
        "baseline_adu": 90.0,
        "max_gain_db": 16.0,
    },
    "BFS-U3-51S5C (simulated CMOS, color, RGB 8-bit)": {
        "sensor": "5.1 MP OSC CMOS, rolling shutter (simulated)",
        "width": 2448, "height": 2048,
        "pixel_format": "RGB8Packed", "pixel_formats": ["RGB8Packed", "Mono8"],
        "bit_depth": 8, "max_fps": 35.0,
        "adr_gain_adu_per_ke": 14.0,
        "full_well_ke": 18.5,
        "read_noise_e": 5.0,
        "dark_current_e_per_s": 15.0,
        "baseline_adu": 40.0,
        "max_gain_db": 20.0,
        "color": True,
    },
}

DEFAULT_DEMO_MODEL = next(iter(DEMO_MODELS))


class DemoWorker(QThread, _WorkerMixin):
    """Simulated camera emitting synthetic frames so the GUI runs everywhere."""

    frameReady = SIGNALS["frameReady"]
    treeReady = SIGNALS["treeReady"]
    rowUpdated = SIGNALS["rowUpdated"]
    saveDone = SIGNALS["saveDone"]
    stateChanged = SIGNALS["stateChanged"]
    error = SIGNALS["error"]
    log = SIGNALS["log"]

    FEATURES: Dict[str, Dict[str, Any]] = {
        "AcquisitionMode": {"kind": KIND_ENUMERATION, "value": "Continuous",
                            "entries": [{"symbolic": "Continuous", "value": 0, "current": True},
                                        {"symbolic": "SingleFrame", "value": 1, "current": False},
                                        {"symbolic": "MultiFrame", "value": 2, "current": False}]},
        "ExposureTime": {"kind": KIND_FLOAT, "value": 10000.0, "min": 10.0, "max": 500000.0, "unit": "us"},
        "Gain": {"kind": KIND_FLOAT, "value": 0.0, "min": 0.0, "max": 24.0, "unit": "dB"},
        "FrameRate": {"kind": KIND_FLOAT, "value": 30.0, "min": 1.0, "max": 120.0, "unit": "Hz"},
        "Width": {"kind": KIND_INTEGER, "value": 1280, "min": 8, "max": 3840, "inc": 8},
        "Height": {"kind": KIND_INTEGER, "value": 1024, "min": 8, "max": 2160, "inc": 8},
        "PixelFormat": {"kind": KIND_ENUMERATION, "value": "Mono16",
                        "entries": [{"symbolic": "Mono8", "value": 0, "current": False},
                                    {"symbolic": "Mono16", "value": 1, "current": True},
                                    {"symbolic": "RGB8Packed", "value": 2, "current": False}]},
        "TriggerMode": {"kind": KIND_ENUMERATION, "value": "Off",
                        "entries": [{"symbolic": "Off", "value": 0, "current": True},
                                    {"symbolic": "On", "value": 1, "current": False}]},
        "BalanceWhiteAuto": {"kind": KIND_ENUMERATION, "value": "Off",
                             "entries": [{"symbolic": "Off", "value": 0, "current": True}]},
        "SoftwareTriggerArm": {"kind": KIND_COMMAND},
        "DeviceTemperature": {"kind": KIND_FLOAT, "value": 33.5, "min": 0.0, "max": 125.0, "unit": "C"},
        "DeviceVendorName": {"kind": KIND_STRING, "value": "FLIR"},  # simulated
        "DeviceModelName": {"kind": KIND_STRING, "value": "DemoCam 1600"},
        "DeviceSerialNumber": {"kind": KIND_STRING, "value": "00000000"},
        "GevSCPHostPort": {"kind": KIND_INTEGER, "value": 0, "min": 0, "max": 65535},
        "ChunkModeActive": {"kind": KIND_BOOLEAN, "value": True},
        "StreamBinningH": {"kind": KIND_INTEGER, "value": 1, "min": 1, "max": 4, "inc": 1},
        "TLValueExample": {"kind": KIND_VALUE, "value": "read-only"},
    }

    def __init__(self, model_key: Optional[str] = None, parent=None):
        QThread.__init__(self, parent)
        self._init_common()
        if model_key not in DEMO_MODELS:
            model_key = DEFAULT_DEMO_MODEL
        self.model_key = model_key
        self.model: Dict[str, Any] = DEMO_MODELS[model_key]
        # FEATURES is a class attribute and _apply_model() edits it, so each
        # worker needs its own copy or one simulated model would leak into the
        # others (and into subsequent workers).
        self.FEATURES = copy.deepcopy(type(self).FEATURES)
        self._apply_model()
        self._stop = False
        self._acquiring = False
        self._counter = 0
        self._t0 = time.perf_counter()
        self._recording: Optional[Dict[str, Any]] = None

    def _apply_model(self):
        m = self.model
        self.FEATURES["Width"].update(value=m["width"], max=m["width"])
        self.FEATURES["Height"].update(value=m["height"], max=m["height"])
        self.FEATURES["PixelFormat"]["value"] = m["pixel_format"]
        self.FEATURES["PixelFormat"]["entries"] = [
            {"symbolic": pf, "value": i, "current": pf == m["pixel_format"]}
            for i, pf in enumerate(m["pixel_formats"])
        ]
        self.FEATURES["FrameRate"].update(value=min(30.0, m["max_fps"]), max=float(m["max_fps"]))
        self.FEATURES["Gain"].update(value=0.0, max=float(m["max_gain_db"]))
        self.FEATURES["DeviceModelName"]["value"] = m["sensor"].split(" (")[0]

    def stop(self):
        self._stop = True

    def current_settings_snapshot(self):
        return {
            "camera": {"__type__": KIND_CATEGORY, "display": "Root", "writable": False,
                       "features": self._flat_features()},
            "stream": {"__type__": KIND_CATEGORY, "display": "Root (TLStream)",
                       "writable": False,
                       "features": {"StreamBinningH": self._feature_info("StreamBinningH")}},
        }

    def _flat_features(self) -> Dict[str, Any]:
        return {name: self._feature_info(name) for name in self.FEATURES
                if name != "StreamBinningH"}

    def _feature_info(self, name: str) -> Dict[str, Any]:
        f = self.FEATURES[name]
        info = {**f, "name": name, "display": name, "tooltip": "",
                "readable": f["kind"] != KIND_COMMAND,
                "writable": f["kind"] not in (KIND_VALUE, KIND_COMMAND)}
        if f["kind"] == KIND_ENUMERATION:
            info["entries"] = [dict(e, current=e["symbolic"] == f["value"])
                               for e in f["entries"]]
        if f["kind"] == KIND_VALUE:
            info["writable"] = False
        return info

    def run(self):
        self.stateChanged.emit("ready")
        self.log.emit("Connected: DemoCam 1600 (simulated)")
        self.refresh_tree()
        while not self._stop:
            self._run_commands()
            if self._acquiring:
                arr, meta = self._make_frame()
                self._publish_frame(arr, meta)
                self._counter += 1
                if self._recording is not None:
                    rec = self._recording
                    if rec["writer"] is not None:
                        rec["writer"].push(arr, meta)
                    else:
                        rec["frames"].append(arr.copy())
                        rec["metas"].append(meta)
                    rec["remaining"] -= 1
                    if rec["remaining"] <= 0 or (rec["writer"] is not None
                                                 and rec["writer"].error):
                        self._finish_recording(early=False)
                rate = float(self.FEATURES["FrameRate"]["value"])
                time.sleep(max(0.004, 1.0 / rate))
            else:
                time.sleep(0.02)
        if self._recording is not None:
            self._finish_recording(early=True)
        self.wait_for_writes()
        self.stateChanged.emit("disconnected")

    def refresh_tree(self):
        tree = {
            "camera": {"__type__": KIND_CATEGORY, "display": "Root", "writable": False,
                       "features": {
                           "Acquisition Control": {
                               "__type__": KIND_CATEGORY, "display": "Acquisition Control",
                               "writable": False,
                               "features": {n: self._feature_info(n) for n in
                                            ("AcquisitionMode", "FrameRate", "TriggerMode",
                                             "SoftwareTriggerArm")}},
                           "Acquisition and Image Generation": {
                               "__type__": KIND_CATEGORY, "display": "Acquisition and Image Generation",
                               "writable": False,
                               "features": {n: self._feature_info(n) for n in
                                            ("ExposureTime", "Gain")}},
                           "Image Format Control": {
                               "__type__": KIND_CATEGORY, "display": "Image Format Control",
                               "writable": False,
                               "features": {n: self._feature_info(n) for n in
                                            ("Width", "Height", "PixelFormat",
                                             "ChunkModeActive")}},
                           "Device Control": {
                               "__type__": KIND_CATEGORY, "display": "Device Control",
                               "writable": False,
                               "features": {n: self._feature_info(n) for n in
                                            ("DeviceVendorName", "DeviceModelName",
                                             "DeviceSerialNumber", "DeviceTemperature")}},
                       }},
            "stream": {"__type__": KIND_CATEGORY, "display": "Root (TLStream)",
                       "writable": False,
                       "features": {"StreamBinningH": self._feature_info("StreamBinningH")}},
        }
        self.treeReady.emit(tree)

    # -- synthetic frame generator (sensor-shaped noise model) ------------
    def _make_frame(self):
        t = time.perf_counter() - self._t0
        m = self.model
        w = int(self.FEATURES["Width"]["value"])
        h = int(self.FEATURES["Height"]["value"])
        exposure_s = float(self.FEATURES["ExposureTime"]["value"]) * 1e-6
        gain_db = float(self.FEATURES["Gain"]["value"])
        # Out-of-range adjustments behave like a real sensor
        h = min(h, m["height"])
        w = min(w, m["width"])
        # Scene -> electrons (computed on a downscaled grid for speed) --------
        s = 4
        sw, sh = (w + s - 1) // s, (h + s - 1) // s
        x = np.linspace(0, 1, sw, dtype=np.float32)
        y = np.linspace(0, 1, sh, dtype=np.float32)
        scene = (x[None, :] * y[:, None]) * 3200.0
        scene += 300 * np.sin(2 * np.pi * x[None, :] * t) \
            * np.sin(2 * np.pi * y[:, None] * (t * 0.3))
        sx = 0.5 * sw + 0.3 * sw * np.sin(t)
        sy = 0.5 * sh + 0.3 * sh * np.cos(t)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        detail = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (90.0 / s) ** 2))) * 5000.0
        scene_small = np.clip(scene + detail, 0.0, None) * exposure_s
        scene = np.repeat(np.repeat(scene_small, s, axis=0), s, axis=1)[:h, :w]
        electrons = scene + np.float32(m["dark_current_e_per_s"] * exposure_s)
        # Shot noise (normal approx of Poisson) + read noise ------------------
        electrons += np.sqrt(np.maximum(electrons, 0.0)) * \
            np.random.normal(0.0, 1.0, electrons.shape).astype(np.float32)
        electrons += np.random.normal(0.0, m["read_noise_e"],
                                      electrons.shape).astype(np.float32)
        # ADC ------------------------------------------------------------------
        adu_gain = m["adr_gain_adu_per_ke"] * (10.0 ** (gain_db / 20.0))
        # Dark/electron noise can legitimately dip below the baseline floor
        adu = np.clip(m["baseline_adu"] + electrons * adu_gain, 0.0, None)
        bit_depth = int(m["bit_depth"])
        full_scale = (2 ** bit_depth) - 1
        out_of_range = float(np.count_nonzero(adu > full_scale)) / adu.size
        if self.FEATURES["PixelFormat"]["value"] == "Mono16":
            arr = np.clip(adu, 0, full_scale).astype(np.uint16)
        elif self.FEATURES["PixelFormat"]["value"] == "RGB8Packed":
            # Simulated color response: split channeted with weights
            rr = np.clip(adu * 1.05, 0, 255).astype(np.uint8)
            gg = np.clip(adu * 0.92, 0, 255).astype(np.uint8)
            bb = np.clip(adu * 0.80 + 12, 0, 255).astype(np.uint8)
            arr = np.stack([rr, gg, bb], axis=-1)
        else:  # Mono8
            arr = np.clip(adu, 0, full_scale).astype(np.uint8)
        corrupt = out_of_range > 0.02  # saturated frames flagged in metadata
        meta = {
            "camera_model": self.model_key,
            "sensor": m["sensor"],
            "bit_depth": bit_depth,
            "width": w, "height": h,
            "pixel_format": self.FEATURES["PixelFormat"]["value"],
            "frame_id": self._counter,
            "device_timestamp_ns": time.time_ns(),
            "received_at_ns": time.time_ns(),
            "host_timestamp_s": round(time.time(), 6),
            "acquired_shape": list(arr.shape),
            "acquired_dtype": str(arr.dtype),
            "saturated_fraction": round(out_of_range, 5),
            "chunk": {"exposure_time_us": float(self.FEATURES["ExposureTime"]["value"]),
                      "gain_db": gain_db,
                      "timestamp": int(time.time_ns())},
        }
        return arr, meta

    # -- commands ----------------------------------------------------------
    def _do_start_acquisition(self):
        self._acquiring = True
        self.stateChanged.emit("acquiring")

    def _do_stop_acquisition(self):
        self._acquiring = False
        if self._recording is not None:
            self._finish_recording(early=True)
        self.stateChanged.emit("ready")

    def _do_set_feature(self, name: str, value):
        node = self.FEATURES.get(name)
        if node is None:
            self.error.emit(f"Feature '{name}' not found")
            return
        v = value
        if isinstance(v, str):
            if node["kind"] in (KIND_INTEGER, KIND_FLOAT):
                try:
                    v = int(v) if node["kind"] == KIND_INTEGER else float(v)
                except ValueError:
                    pass
            elif node["kind"] == KIND_BOOLEAN:
                v = v.strip().lower() in ("1", "true", "yes", "on")
        if node["kind"] == KIND_INTEGER:
            v = int(v)
        elif node["kind"] == KIND_FLOAT:
            v = float(v)
        if isinstance(v, (int, float)) and "min" in node:
            v = min(max(v, float(node["min"])), float(node["max"]))
        node["value"] = v
        self.log.emit(f"Set {name} -> {value}")
        self._do_refresh_rows([name])

    def _do_execute_command(self, name: str):
        self.log.emit(f"Executed: {name}")

    def _do_refresh_tree(self):
        self.refresh_tree()

    def _do_refresh_rows(self, names: List[str]):
        for name in names:
            node = self.FEATURES.get(name)
            if node:
                self.rowUpdated.emit(self._feature_info(name))

    def _do_save_last(self, path: str, fmt: str, include_sidecar: bool = True,
                      display_settings: Optional[Dict[str, Any]] = None,
                      blosc: Optional[Dict[str, Any]] = None):
        try:
            if display_settings:
                self._display_settings = dict(display_settings)
            if self._last_frame is None:
                raise RuntimeError("No frame captured yet (start acquisition first)")
            arr, meta = self._last_frame
            out = self._save_current(path, fmt, arr, meta, include_sidecar, blosc)
            self.saveDone.emit({"ok": True, "path": str(out), "message": f"Saved {out}"})
        except Exception as exc:
            self.saveDone.emit({"ok": False, "path": path, "message": str(exc)})

    def _do_record(self, path: str, n_frames: int, fmt: str,
                   include_sidecar: bool = True,
                   display_settings: Optional[Dict[str, Any]] = None,
                   blosc: Optional[Dict[str, Any]] = None,
                   stream: bool = False, chunk_frames: int = 64):
        if display_settings:
            self._display_settings = dict(display_settings)
        if not self._acquiring:
            self.saveDone.emit({"ok": False, "path": path,
                                "message": "Start acquisition before recording"})
            return
        if fmt not in SEQUENCE_FORMATS:
            self.saveDone.emit({"ok": False, "path": path, "message": f"bad fmt {fmt}"})
            return
        try:
            self._recording = self._new_recording(
                path, fmt, n_frames, include_sidecar, blosc, stream, chunk_frames)
        except Exception as exc:
            self.saveDone.emit({"ok": False, "path": path, "message": f"Record failed: {exc}"})
            return
        mode = f"streaming, chunks of {chunk_frames} frames" if stream else "in memory"
        self.log.emit(f"Recording {n_frames} frames to {path} ({mode}) ...")
        self.stateChanged.emit("recording")

    def _do_stop_record(self):
        if self._recording is not None:
            self._finish_recording(early=True)

    def _finish_recording(self, early: bool):
        rec, self._recording = self._recording, None
        if rec["writer"] is not None:
            self._finish_stream(rec)
            self.stateChanged.emit("acquiring" if self._acquiring else "ready")
            return
        try:
            from . import savers
            settings = self.current_settings_snapshot()
            blosc = dict(rec.get("blosc") or {})
            blosc.setdefault("blosc_cname", "lz4")
            blosc.setdefault("blosc_clevel", 1)
            blosc.setdefault("blosc_shuffle", "shuffle")
            if rec["fmt"] in ("npy_sequence", "tiff_sequence"):
                out = _save_stack_recording(savers, rec, settings, self._display_settings)
            elif rec["fmt"] == "hdf5_sequence":
                out = rec["path"]
                if out.suffix not in (".h5", ".hdf5"):
                    out = out.with_suffix(".hdf5")
                savers.save_hdf5_sequence(
                    out, rec["frames"], rec["metas"],
                    pixel_format=str(rec["metas"][0].get("pixel_format") or "unknown"),
                    camera_settings=settings,
                    display_settings=self._display_settings)
            elif rec["fmt"] == "zarr_sequence":
                savers.save_zarr_sequence(
                    rec["path"], rec["frames"], rec["metas"],
                    pixel_format=str(rec["metas"][0].get("pixel_format") or "unknown"),
                    camera_settings=settings,
                    display_settings=self._display_settings, **blosc)
                out = rec["path"]
                if out.suffix != ".zarr":
                    out = Path(str(out) + ".zarr")
            else:
                d = rec["path"]
                d = d if d.suffix == "" else d.with_suffix("")
                if not str(d).endswith("_seq"):
                    d = Path(str(d) + "_seq")
                d.mkdir(parents=True, exist_ok=True)
                index = []
                for i, arr in enumerate(rec["frames"]):
                    fname = f"frame_{i:06d}.raw"
                    (d / fname).write_bytes(np.ascontiguousarray(arr).tobytes(order="C"))
                    index.append({"file": fname, **rec["metas"][i]})
                first = rec["metas"][0] if rec["metas"] else {}
                savers.write_sidecar(
                    d / "sequence.json",
                    savers.build_sidecar(
                        image_info={"type": "raw_sequence", "frames": len(rec["frames"]),
                                    "width": int(first.get("width", 0)),
                                    "height": int(first.get("height", 0)),
                                    "channels": 3 if (rec["frames"] and rec["frames"][0].ndim == 3) else 1,
                                    "dtype": _raw_sequence_dtype(
                                        first, 3 if (rec["frames"] and rec["frames"][0].ndim == 3) else 1),
                                    "byte_order": "little"},
                        frame_meta={"frames": index},
                        camera_settings=settings,
                        display_settings=self._display_settings))
                out = d
            self.saveDone.emit({"ok": True, "path": str(out), "frames": len(rec["frames"]),
                                "message": f"Recorded: {out}"})
        except Exception as exc:
            self.saveDone.emit({"ok": False, "path": str(rec["path"]),
                                "message": f"Record failed: {exc}"})
        self.stateChanged.emit("acquiring" if self._acquiring else "ready")


def create_worker(demo: bool = False, serial: Optional[str] = None,
                  model_key: Optional[str] = None):
    if demo or not PYSPIN_AVAILABLE:
        return DemoWorker(model_key=model_key)
    return CameraWorker(serial=serial)
