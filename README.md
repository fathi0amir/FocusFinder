# FLIR Spinnaker Camera GUI (PyQt6)

> **Requirement:** to run this GUI with a real camera you must install the FLIR
> **Spinnaker SDK** on the system and place the matching PySpin `.whl` in the
> `vendor/` directory (its contents are git-ignored). Without it, only the
> simulated cameras are available.

A comprehensive Python GUI for **FLIR Spinnaker cameras** (PySpin), built with
**Qt6 (PyQt6)**:

- **Multiple cameras**: every camera Spinnaker detects is listed in the
  *Camera* picker (model, serial, transport) with a **Refresh** button, so you
  pick which one to connect to; `--serial SN` preselects it from the CLI. The
  scan runs off the UI thread and the picker locks while connected.
- **Live view** at full frame rate with **crosshair line profiles** attached to
  the **top** and **right** edges of the image (profiles update as you move the
  crosshair or hover the mouse). The crosshair stays **draggable while
  acquisition is running** — incoming frames never move it — and the profile
  plots are pixel-aligned with the image.
- **Display orientation**: `as acquired` (the camera's own orientation),
  `rotate 90° CW/CCW`, `rotate 180°`, `flip horizontal/vertical`. Use it for a
  camera that is physically mounted rotated; the frame geometry readout always
  reports the orientation actually on screen. **Reset View** fits the image
  instantly (it sets the view range instead of auto-ranging over every pixel).
- **Saved images tab**: open a recorded stack and browse it — `.raw` frame or
  folder of raw frames, **HDF5** (`/image` or `/images`) or **Zarr** (`/image`)
  — with **play/pause**, playback speed, loop, a **scrubber** to jump to any
  frame, first/prev/next/last buttons, and the per-frame metadata
  (frame id, exposure, gain, saturation) from the JSON sidecar. Frames are read
  **lazily**: opening a stack reads no pixels, and only the frame on screen is
  read, with rapid scrubbing collapsed into a single read of the final
  position.
- **Full GenICam feature tree browser** — view and *modify every available
  camera node* (integer/float/boolean/enumeration/string editors, command
  execution buttons), including the transport-layer (stream) node map.
  Values refresh periodically live (temperature, counters, etc.).
- **Save the last frame several ways**:
  - `raw` binary plus a **JSON sidecar** with *every setting for that image*
    (full node-map snapshot, per-frame metadata incl. chunk data, display
    settings).
  - **HDF5** (`/image` dataset + `metadata` group attrs + flattened camera
    settings attrs `cam/...`) plus the same JSON sidecar.
  - **Zarr v3** store (`<name>.zarr`, `/image` array) compressed with
    **Blosc** — choose **lz4** or **zstd**, compression **level 1–3**,
    **shuffle** enabled — plus the same JSON sidecar.
- **Record sequences**: N frames into one **HDF5** file (`/images`, LZF
  compressed, chunked), one **Zarr** store (frames stacked on axis 0, one
  Blosc chunk per frame), or a folder of raw frames + `sequence.json` sidecar.
- **Menu → File → Save settings snapshot** for the raw node map dump.

The live camera view and the saved-image viewer are separate tabs, so browsing
recorded data never disturbs a running acquisition.

## Install (uv)

The environment is **pinned to Python 3.12** (`.python-version`, and the wheel
below is cp312) and fully locked by `uv.lock`:

```bash
uv sync                          # GUI deps (PyQt6, pyqtgraph, numpy, h5py, zarr)
uv sync --extra spinnaker        # + vendored FLIR PySpin 4.4.0.246 wheel
```

PySpin is not on PyPI; it is installed from the vendored wheel
(`vendor/spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl`, resolved via
`[tool.uv.sources]`). Real camera enumeration additionally requires the
[FLIR Spinnaker SDK 4.4](https://www.flir.com/products/spinnaker-sdk/)
(Windows installer) installed on the system.

Verify: `uv run python -c "import PySpin; print('pyspin ok')"`

## Run

```bash
uv run flir-cam            # real camera if PySpin is working; simulation otherwise
uv run flir-cam --live     # force hardware mode (errors if no PySpin)
uv run flir-cam --demo     # force simulation mode
uv run flir-cam --open seq.zarr   # open a saved stack in the Saved images tab
uv run flir-cam --serial 12345678 # preselect that camera in the picker
```

Simulated camera modules (approximate public sensor specs, offline only):

```bash
uv run flir-cam --model "BFS-U3-200S6M (simulated CMV4000 CMOS, mono 12-bit)"
```

Available simulated models (see `src/flir_cam_gui/camera.py::DEMO_MODELS`):

- `BFS-U3-200S6M` — CMV4000 CMOS, mono 12-bit, **1536 x 2048 (portrait)**
- `BFS-U3-31S4M` — Sony IMX265 CMOS, mono 12-bit, 2048 x 1536
- `GS3-U3-23S6M` — Sony ICX672 CCD, mono 12-bit, 1920 x 1440
- `BFS-U3-51S5C` — 5.1 MP OSC CMOS, color (RGB8Packed), 2448 x 2048

Frames are generated with a sensor-shaped noise model: photon **shot noise**
(normal approximation of Poisson), **read noise**, **dark current**,
**baseline offset**, ADC gain scaled by the configured dB gain, and clipping to
the sensor's full scale. Raising exposure/gain saturates frames just like real
hardware, and saturation fraction is reported in frame metadata.

## Usage notes

1. **Pick a source and a camera** — the source combo selects Live or a
   simulated model; with *Live* the *Camera* combo lists every connected FLIR
   camera (`model  SN=…  [transport]`), so choose the one to open and press
   **Connect**. *Refresh* re-scans after plugging a camera in.
2. **▶ Start acquisition** — live view and profiles begin.
3. **Crosshair** — drag either line with the left mouse button: it follows the
   cursor the whole way, *while acquisition is running*, and both profiles
   update live. The press must be within a few pixels of a line to grab it;
   dragging anywhere else pans the view as usual, and `Reset View` refits the
   frame.
4. Adjust any node in the *Camera features* tree; read-only values refresh
   every few seconds (and instantly after you write a value).
5. **Saved images** tab — *Open file...* for a `.raw`/`.h5`/`.hdf5` (or its
   `.json` sidecar) and *Open folder...* for a Zarr store or a folder of raw
   frames. Play/pause, drag the scrubber, or step with `|< < > >|`; the frame
   metadata line shows what the sidecar recorded for that frame. *Orientation*,
   *Levels* and *Palette* in the Display panel apply to this viewer too.
6. **Display panel** — *Orientation* renders the frame as acquired, rotated
   (90° CW/CCW, 180°), flipped (horizontal/vertical) or transposed, for a
   camera that is physically mounted at an angle. Whatever the window shape and
   whatever orientation is picked, the image keeps the sensor's own aspect ratio
   (square pixels, never stretched) and is centred with the whole frame visible;
   *Reset View* refits it instantly.
7. Save / record with the right-hand panels or the File menu. When it saves an
   image it **always writes a sidecar** (`<name>.json`) with:
   - `camera_settings` — full node maps (camera + stream),
   - `frame_metadata` — frame ID, device/host timestamps, chunk exposure/gain,
   - `file` — geometry, dtype, byte order, byte count,
   - `display_settings` — palette/bit-depth/levels used in the live view.

### Sidecar schema

**JSON sidecar** (`<name>.json`, or `sequence.json` inside a raw-sequence folder).
Global attributes only:

| Key | Content |
| --- | --- |
| `schema` | Sidecar schema version (`1`) |
| `application`, `app_version` | Writer name and version |
| `host`, `platform`, `python` | Machine that saved the file |
| `saved_at_utc` | ISO-8601 UTC save time |
| `file` | Pixel payload: `name`, `type`, `width`, `height`, `channels`, `shape`, `dtype`, `frames`, `pixel_format`, `byte_order`, plus format extras (e.g. `compressor`, `streamed`, `chunk_frames`) |
| `camera_settings` | Full node-map snapshot (camera + stream) |
| `display_settings` | Live-view settings active when saved (optional) |
| `frame_metadata` | Single frame: that frame's metadata dict. Sequence: `{"frames_file", "format": "parquet", "count", "constant"}` |

`frame_metadata.constant` holds the values identical for every frame (taken from the
first frame): `camera_model`, `sensor`, `bit_depth`, `width`, `height`,
`pixel_format`, `acquired_shape`, `acquired_dtype`, `chunk`.

**Parquet sidecar** (`<name>.frames.parquet`, `sequence.frames.parquet` for raw
sequences). One row per frame, zstd-compressed; null values are omitted when read back.

| Column | Type | Content |
| --- | --- | --- |
| `frame_id` | int | Camera frame counter |
| `device_timestamp_ns` | int | Camera timestamp, ns |
| `received_at_ns` | int | Host receive time, ns since epoch |
| `host_timestamp_s` | float | Host time, s since epoch |
| `fps` | float | `1e9 / (t_i - t_(i-1))` from `device_timestamp_ns`; first frame uses the next gap |
| `saturated_fraction` | float | Fraction of saturated pixels (simulated camera) |
| `file` | string | Frame file name (raw sequences only) |

Columns are the union of the per-frame metadata keys, so a camera may add more.

```python
import json, pyarrow.parquet as pq
side = json.load(open("seq.h5.json"))
frames = pq.read_table("seq.h5.frames.parquet").to_pylist()   # one dict per frame
```

Older recordings with inline `frame_metadata.frames` in the JSON still open.

### Reading a raw file back

The easiest way is the **Saved images** tab in the GUI. To do it in Python:

```python
import numpy as np, json, pathlib
meta = json.loads(pathlib.Path("frame.raw.json").read_text())["file"]
arr = np.fromfile("frame.raw", dtype=np.dtype(meta["dtype"])
                  ).reshape(meta["height"], meta["width"])
```

For a colour frame add the channel axis
(`meta["channels"]`, stored last: `(height, width, channels)`).

### Reading a HDF5 or Zarr store back

```python
import h5py, zarr
from zarr.storage import LocalStore

with h5py.File("seq.hdf5", "r") as f:
    stack = f["images"]            # (N, H, W): one chunk per frame
    frame = stack[3]
img = zarr.open_group(store=LocalStore(root="seq.zarr"), mode="r")["image"]  # (N, H, W)
```

## Development

```bash
uv sync              # installs dev deps (pytest) too
uv run pytest -q     # unit + offscreen GUI smoke tests
uv run flir-cam --demo   # manual GUI run
```

Architecture note: All Spinnaker/GenICam access happens inside a worker thread
(`src/flir_cam_gui/camera.py`); the UI communicates only via commands and Qt
signals, so camera nodes are never touched from the UI thread. Stack reading
(`src/flir_cam_gui/frame_reader.py`) keeps HDF5/Zarr handles open but reads a
single frame per request, so a 10 000 frame stack costs the same as a 10 frame
one.
