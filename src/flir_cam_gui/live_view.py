"""Live image view: the frame, a draggable crosshair and the two line profiles.

Layout::

      +-----------------------------------+
      |  top: horizontal profile (row)    |
+-----+-----------------------------------+
|     |                                   |
|     |            image + crosshair      |  right: vertical
|     |                                   |  profile (column)
+-----+-----------------------------------+-------+

The crosshair stays interactive at all times, including while the camera is
streaming: dragging a line refreshes the profiles immediately and incoming
frames never move the crosshair.  The frames are rendered in the camera's own
orientation unless the user picks a display orientation, in which case the
frame (and therefore the profiles) is rotated/flipped for display.

Pixels are always square: a frame is drawn with the sensor's own aspect ratio,
whatever the window shape, the sensor geometry or the display orientation, so
switching camera, rotating, flipping or transposing can never stretch it.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal

PALETTES = ("grayscale", "viridis", "inferno", "plasma", "turbo", "magma", "hot")

# Display orientations. "as acquired" shows the frame exactly as the camera
# delivered it (row 0 at the top); the others transform the displayed frame,
# e.g. for a camera that is physically mounted rotated on the bench.
ORIENTATIONS = (
    "as acquired",
    "rotate 90° CW",
    "rotate 90° CCW",
    "rotate 180°",
    "flip horizontal",
    "flip vertical",
    "transpose",
)
DEFAULT_ORIENTATION = ORIENTATIONS[0]

_ROT90 = {"rotate 90° CW": -1, "rotate 90° CCW": 1}

#: maximum number of points drawn per profile (frames can be 20+ MP)
_MAX_PROFILE_POINTS = 1200
#: how often auto levels may be recomputed while streaming
_LEVELS_INTERVAL_S = 0.4
#: histogram bins and refresh limit while streaming
_HIST_BINS = 256
_HIST_INTERVAL_S = 0.2
_HIST_PENS = ("#e6e6e6", "#ff5555", "#55dd55", "#5599ff")
#: coalescing interval for crosshair-follows-mouse updates
_HOVER_INTERVAL_MS = 15
#: how close (screen px) a press must be to a line to grab it
_GRAB_PIXELS = 8
#: fixed axis sizes (device independent px) that keep the plots aligned
_AXIS_W = 58
_AXIS_H = 26


def apply_orientation(arr: np.ndarray, name: str) -> np.ndarray:
    """Return *arr* as it should be displayed for the given orientation name."""
    if name in _ROT90:
        return np.rot90(arr, k=_ROT90[name])
    if name == "rotate 180°":
        return arr[::-1, ::-1]
    if name == "flip horizontal":
        return arr[:, ::-1]
    if name == "flip vertical":
        return arr[::-1, :]
    if name == "transpose":
        # Mirror about the main diagonal: swaps rows and columns (and keeps a
        # trailing colour axis where it is).
        return np.swapaxes(arr, 0, 1)
    return arr


class _HoverImage(pg.ImageItem):
    """ImageItem that reports hover positions for crosshair following."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def hoverEvent(self, ev):
        if ev.isExit():
            return
        try:
            pos = self.mapSceneToView(ev.scenePos())
        except Exception:
            pos = None
        if pos is not None:
            self._owner._hover_value(pos)


class LiveViewWidget(pg.GraphicsLayoutWidget):
    """Central live image with crosshair + line profiles."""

    crosshairMoved = pyqtSignal(int, int, float)  # x, y, value at intersection

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src: Optional[np.ndarray] = None          # frame as acquired
        self._arr: Optional[np.ndarray] = None          # frame as displayed
        self._last_meta: Optional[dict] = None
        self._shape: Optional[Tuple[int, int]] = None
        self._follow_mouse = True
        self._bit_depth = 16
        self._auto_levels = True
        self._manual_levels: Optional[Tuple[float, float]] = None
        self._auto_max: Optional[float] = None
        self._orientation = DEFAULT_ORIENTATION
        self._dragging = False
        self._drag_line: Optional[pg.InfiniteLine] = None
        self._levels_at = 0.0
        self._hover_pos: Optional[Tuple[float, float]] = None

        # --- image plot -----------------------------------------------------
        self.img = _HoverImage(self)
        # Rendering is cropped/strided to the viewport in _render_image(); pyqtgraph's
        # autoDownsample block-averages the whole frame on the GUI thread (>100 ms at 15 MP).
        self.img.setAutoDownsample(False)
        # Needed for "crosshair follows mouse"; ImageItem disables it by default.
        self.img.setAcceptHoverEvents(True)
        self.image_plot = self.addPlot(row=1, col=1)
        self.image_plot.addItem(self.img)
        self.image_plot.setMenuEnabled(False)
        self.image_plot.hideButtons()
        self.image_plot.setLabel("bottom", "Pixel X")
        self.image_plot.setLabel("left", "Pixel Y")
        # Fixed axis sizes keep the three plot areas exactly aligned, so the
        # linked profile plots line up with the image pixels.
        self.image_plot.getAxis("left").setWidth(_AXIS_W)
        self.image_plot.getAxis("bottom").setHeight(_AXIS_H)
        # Row 0 of the array is drawn at the top of the plot.
        self.image_plot.invertY(True)
        self.image_plot.getViewBox().disableAutoRange()
        # A pixel must stay square on screen.  Without this (pyqtgraph leaves
        # the axes independent by default) the frame is stretched to fill the
        # plot area, so any sensor whose aspect differs from the window -- or
        # the same sensor after a rotation/transpose -- comes out distorted.
        # ratio 1 == x and y keep the same scale, whatever the sensor gives us.
        self.image_plot.getViewBox().setAspectLocked(True, 1.0)
        self._v_line = self._add_crosshair_line(
            pos=0, angle=90, pen=pg.mkPen("#30d0ff", width=1))
        self._h_line = self._add_crosshair_line(
            pos=0, angle=0, pen=pg.mkPen("#ffd166", width=1))
        self.hud_text = pg.TextItem(anchor=(0, 0), color="#8ff")
        self.image_plot.addItem(self.hud_text, ignoreBounds=True)
        self.hud_text.setPos(0, 0)

        # --- horizontal profile (top) ---------------------------------------
        # x = pixel X (linked to the image), y = intensity (auto-ranged).
        self.top = self.addPlot(row=0, col=1)
        self.top.setMenuEnabled(False)
        self.top.hideButtons()
        self.top.setXLink(self.image_plot)
        self.top.setMouseEnabled(x=True, y=False)
        self.top.getViewBox().enableAutoRange(axis="y")
        self.top.getAxis("left").setWidth(_AXIS_W)
        self.top.hideAxis("bottom")  # the image plot already labels Pixel X
        self._top_curve = self.top.plot([], pen=pg.mkPen("#00e676", width=2))
        self.top.setLabel("left", "Intensity (row)")
        self.top.showGrid(x=True, y=True, alpha=0.3)

        # --- vertical profile (right) ---------------------------------------
        # x = intensity (auto-ranged), y = pixel Y (linked to the image, so it
        # shares the image's inverted axis: row 0 at the top).
        self.right = self.addPlot(row=1, col=2)
        self.right.setMenuEnabled(False)
        self.right.hideButtons()
        self.right.invertY(True)
        self.right.setYLink(self.image_plot)
        self.right.setMouseEnabled(x=False, y=True)
        self.right.getViewBox().enableAutoRange(axis="x")
        self.right.getAxis("bottom").setHeight(_AXIS_H)
        self.right.hideAxis("left")  # the image plot already labels Pixel Y
        self._right_curve = self.right.plot([], pen=pg.mkPen("#e879f9", width=2))
        self.right.setLabel("bottom", "Intensity (col)")
        self.right.showGrid(x=True, y=True, alpha=0.3)

        # --- histogram (bottom, added to the layout only while toggled on) ----
        self._hist_visible = False
        self._hist_at = 0.0
        self.hist = pg.PlotItem()
        self.hist.setMenuEnabled(False)
        self.hist.hideButtons()
        self.hist.getAxis("left").setWidth(_AXIS_W)
        self.hist.setLabel("bottom", "Intensity")
        self.hist.setLabel("left", "log10(1 + count)")
        self.hist.showGrid(x=True, y=True, alpha=0.3)
        self._hist_curves = [
            self.hist.plot([], [], pen=pg.mkPen(c, width=1), fillLevel=0,
                           brush=pg.mkBrush(c + "40"))
            for c in _HIST_PENS]
        dash = pg.mkPen("#ffd166", style=Qt.PenStyle.DashLine)
        self._hist_lo = pg.InfiniteLine(angle=90, pen=dash)
        self._hist_hi = pg.InfiniteLine(angle=90, pen=dash)
        self.hist.addItem(self._hist_lo)
        self.hist.addItem(self._hist_hi)

        self.ci.layout.setRowStretchFactor(0, 1)
        self.ci.layout.setRowStretchFactor(1, 6)
        self.ci.layout.setColumnStretchFactor(1, 6)
        self.ci.layout.setColumnStretchFactor(2, 1)

        # --- coalescing timers ----------------------------------------------
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(_HOVER_INTERVAL_MS)
        self._hover_timer.timeout.connect(self._apply_hover)

        # Zoom/pan/resize of a still frame needs a re-render at the new resolution.
        self._rerender_timer = QTimer(self)
        self._rerender_timer.setSingleShot(True)
        self._rerender_timer.setInterval(30)
        self._rerender_timer.timeout.connect(self._render_image)
        self.image_plot.getViewBox().sigRangeChanged.connect(
            lambda *_: self._rerender_timer.start())
        self._hud_base = ""
        self.clear = self.clear_frame

    def _add_crosshair_line(self, *, pos, angle, pen) -> pg.InfiniteLine:
        """Add a crosshair line the user can grab and drag at any time."""
        line = pg.InfiniteLine(pos=pos, angle=angle, movable=True, pen=pen)
        self.image_plot.addItem(line)
        line.sigPositionChanged.connect(self._on_crosshair_moved)
        return line

    # -- properties -----------------------------------------------------------
    @property
    def orientation(self) -> str:
        return self._orientation

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    @property
    def auto_levels(self) -> bool:
        return self._auto_levels

    @property
    def displayed_size(self) -> Optional[Tuple[int, int]]:
        """(width, height) of the image currently on screen, or None."""
        if self._arr is None:
            return None
        return int(self._arr.shape[1]), int(self._arr.shape[0])

    @property
    def aspect_locked(self) -> bool:
        """True while the view keeps pixels square (the image never stretches)."""
        return self.image_plot.getViewBox().state["aspectLocked"] is not False

    # -- incoming frames -----------------------------------------------------
    def show_frame(self, arr: np.ndarray, meta: Optional[dict] = None) -> None:
        """Display *arr* (height, width[, channels]) and refresh the profiles."""
        if arr is None:
            return
        arr = np.asarray(arr)
        self._src = arr
        self._last_meta = meta

        disp = apply_orientation(arr, self._orientation)
        if not disp.flags.c_contiguous:
            disp = np.ascontiguousarray(disp)
        self._arr = disp

        h, w = disp.shape[0], disp.shape[1]
        if self._shape != (h, w):
            # First frame, or the frame geometry changed (ROI/binning change):
            # fit the new extent once; user zoom/pan is preserved in between.
            self._shape = (h, w)
            self._fit(h, w)
        self._render_image()

        if self._auto_levels and (time.monotonic() - self._levels_at) >= _LEVELS_INTERVAL_S:
            self._apply_levels()
        if self.img.levels is None:
            # pyqtgraph raises inside paint() -- and therefore draws nothing --
            # when a downsampled image is painted with no levels at all.  A
            # frame must never be left unlevelled, whatever the Levels control
            # is set to: running with 'auto' off and no frame yet is exactly
            # that case.
            self._apply_levels()
        self._update_profiles()
        if self._hist_visible:
            self._update_histogram()

    def _render_image(self) -> None:
        """Draw the visible part of the frame at no more than screen resolution."""
        a = self._arr
        if a is None:
            return
        h, w = a.shape[0], a.shape[1]
        vb = self.image_plot.getViewBox()
        (vx0, vx1), (vy0, vy1) = vb.viewRange()
        x0, x1 = max(0, int(np.floor(min(vx0, vx1))) - 1), min(w, int(np.ceil(max(vx0, vx1))) + 1)
        y0, y1 = max(0, int(np.floor(min(vy0, vy1))) - 1), min(h, int(np.ceil(max(vy0, vy1))) + 1)
        if x1 <= x0 or y1 <= y0:
            x0, x1, y0, y1 = 0, w, 0, h
        px, py = vb.viewPixelSize()   # image pixels per screen pixel
        step = max(1, int(min(px, py))) if px > 0 and py > 0 else 1
        sub = np.ascontiguousarray(a[y0:y1:step, x0:x1:step])
        self.img.setImage(sub, autoLevels=False, axisOrder="row-major")
        self.img.setRect(QRectF(x0, y0, sub.shape[1] * step, sub.shape[0] * step))

    def _fit(self, h: int, w: int) -> None:
        """Fit the whole image extent -- an explicit range, never autoRange.

        With the aspect lock on, the requested extent is kept fully visible and
        the shorter axis of the plot area is padded instead, so the frame is
        centred with square pixels rather than stretched to fill the viewport.
        """
        vb = self.image_plot.getViewBox()
        vb.disableAutoRange()
        vb.setRange(xRange=(0, w), yRange=(0, h), padding=0.0)
        # Keep the crosshair on the frame: with square pixels the plot area can
        # be larger than the image, and a line parked in that empty margin would
        # sit on a profile that is not a real row/column.  Signals stay blocked
        # here -- show_frame refreshes the profiles right after this call.
        for line, upper in ((self._v_line, w), (self._h_line, h)):
            line.blockSignals(True)
            try:
                line.setBounds((0, upper))
                line.setValue(min(max(float(line.value()), 0.0), float(upper - 1)))
            finally:
                line.blockSignals(False)

    def _apply_levels(self) -> None:
        a = self._arr
        if a is None:
            return
        if self._auto_levels and a.size:
            step = max(1, a.size // 200_000)
            sub = a.ravel()[::step]
            try:
                lo, hi = (float(v) for v in np.percentile(sub, (0.5, 99.5)))
            except Exception:
                lo, hi = float(sub.min()), float(sub.max())
            if self._auto_max is not None:
                hi = min(hi, self._auto_max)
                lo = min(lo, hi - 1.0)
            if hi <= lo:
                hi = lo + 1.0
            self.img.setLevels([lo, hi])
        elif self._manual_levels is not None:
            lo, hi = self._manual_levels
            self.img.setLevels([lo, max(hi, lo + 1.0)])
        else:
            self.img.setLevels([0, float(2 ** self._bit_depth - 1)])
        self._levels_at = time.monotonic()

    # -- crosshair -----------------------------------------------------------
    def _on_crosshair_moved(self, _line=None) -> None:
        self._update_profiles(user_moved=True)

    def _hover_value(self, scene_point) -> None:
        if scene_point is None or self._arr is None:
            return
        if not self._follow_mouse or self._dragging:
            return
        h, w = self._arr.shape[0], self._arr.shape[1]
        x = min(max(float(scene_point.x()), 0.0), w - 1)
        y = min(max(float(scene_point.y()), 0.0), h - 1)
        self._hover_pos = (x, y)
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _apply_hover(self) -> None:
        if self._hover_pos is None or self._dragging or not self._follow_mouse:
            return
        x, y = self._hover_pos
        # Moving the lines triggers _on_crosshair_moved -> profile refresh.
        self._v_line.setValue(x)
        self._h_line.setValue(y)
        self._hover_pos = None

    # -- crosshair dragging ---------------------------------------------------
    #
    # The drag is handled here, at the view level, instead of by the lines
    # themselves: pyqtgraph only forwards a single start/finish drag event to a
    # movable item (the mouse moves in between never reach it), so a line
    # grabbed that way jumps once and then stops following the cursor -- exactly
    # the "cannot move the crosshair" behaviour.  Handling the press/move/release
    # ourselves keeps the crosshair glued to the cursor, mid-acquisition
    # included, and leaves pan/zoom alone when the press misses both lines.

    def _view_pos(self, widget_pos) -> Optional[Tuple[float, float]]:
        """Viewport coordinates -> image pixel coordinates (None if unmappable)."""
        vb = self.image_plot.getViewBox()
        # mapToScene() takes a QPoint (it rejects a QPointF outright).
        point = widget_pos.toPoint() if hasattr(widget_pos, "toPoint") else widget_pos
        try:
            view = vb.mapSceneToView(self.mapToScene(point))
        except Exception:
            return None
        return float(view.x()), float(view.y())

    def _line_near(self, widget_pos) -> Optional[pg.InfiniteLine]:
        """The crosshair line within _GRAB_PIXELS of *widget_pos* (or None)."""
        pos = self._view_pos(widget_pos)
        if pos is None:
            return None
        px, py = self.image_plot.getViewBox().viewPixelSize()
        candidates = ((self._v_line, abs(pos[0] - self._v_line.value()) / (px or 1.0)),
                      (self._h_line, abs(pos[1] - self._h_line.value()) / (py or 1.0)))
        line, distance = min(candidates, key=lambda c: c[1])
        return line if distance <= _GRAB_PIXELS else None

    def _drag_to(self, widget_pos) -> None:
        """Move the grabbed line to the cursor, staying on the frame."""
        line = self._drag_line
        pos = self._view_pos(widget_pos)
        if line is None or pos is None or self._arr is None:
            return
        h, w = self._arr.shape[0], self._arr.shape[1]
        # setValue() refreshes the profiles through _on_crosshair_moved().
        if line is self._v_line:
            line.setValue(min(max(pos[0], 0.0), float(w - 1)))
        else:
            line.setValue(min(max(pos[1], 0.0), float(h - 1)))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            line = self._line_near(ev.position())
            if line is not None:
                self._drag_line = line
                self._dragging = True
                self._hover_pos = None
                self.viewport().setCursor(Qt.CursorShape.SizeHorCursor
                                          if line is self._v_line
                                          else Qt.CursorShape.SizeVerCursor)
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_line is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            self._drag_to(ev.position())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_line is not None:
            self._drag_to(ev.position())
            self._drag_line = None
            self._dragging = False
            self.viewport().unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def _update_profiles(self, user_moved: bool = False) -> None:
        """Redraw both line profiles for the current crosshair position."""
        a = self._arr
        if a is None:
            return
        h, w = a.shape[0], a.shape[1]
        x = int(round(min(max(self._v_line.value(), 0), w - 1)))
        y = int(round(min(max(self._h_line.value(), 0), h - 1)))

        if a.ndim == 3:
            row = np.asarray(a[y, :, :], dtype=np.float64).mean(axis=1)
            col = np.asarray(a[:, x, :], dtype=np.float64).mean(axis=1)
            value = float(np.asarray(a[y, x, :], dtype=np.float64).mean())
        else:
            row = np.asarray(a[y, :], dtype=np.float64)
            col = np.asarray(a[:, x], dtype=np.float64)
            value = float(a[y, x])

        step = max(1, -(-row.size // _MAX_PROFILE_POINTS))
        self._top_curve.setData(np.arange(0, row.size, step), row[::step])
        step_v = max(1, -(-col.size // _MAX_PROFILE_POINTS))
        self._right_curve.setData(col[::step_v], np.arange(0, col.size, step_v))

        self._set_hud(self._hud_suffix)
        self.crosshairMoved.emit(x, y, value)

    # -- display -------------------------------------------------------------
    @property
    def _hud_suffix(self) -> str:
        """Frame information shown after the caller-supplied HUD text."""
        if self._arr is None:
            return ""
        w, h = self.displayed_size
        m = self._last_meta or {}
        parts = [f"{w}x{h}"]
        if self._orientation != DEFAULT_ORIENTATION:
            parts.append(self._orientation)
        if m.get("pixel_format"):
            parts.append(str(m["pixel_format"]))
        if m.get("frame_id") is not None:
            parts.append(f"#{m['frame_id']}")
        return "  ".join(parts)

    def set_hud_text(self, text: str) -> None:
        self._hud_base = text
        self._set_hud(self._hud_suffix)

    def _set_hud(self, suffix: str) -> None:
        self.hud_text.setText(self._hud_base + ("  |  " + suffix if suffix else ""))

    def set_orientation(self, name: str) -> None:
        """Rotate/flip the displayed image (and its profiles)."""
        if name not in ORIENTATIONS:
            name = DEFAULT_ORIENTATION
        if name == self._orientation:
            return
        self._orientation = name
        if self._src is not None:
            self._shape = None
            self.show_frame(self._src, self._last_meta)

    def set_palette(self, name: str) -> None:
        cmap = None
        lookup = "gray" if name in ("", "grayscale", "gray") else name
        for source in (None, "matplotlib"):
            try:
                cmap = pg.colormap.get(lookup, source=source) if source else pg.colormap.get(lookup)
            except Exception:
                cmap = None
            if cmap is not None:
                break
        if cmap is not None:
            self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        else:
            self.img.setLookupTable(None)
        self.img.updateImage()

    def set_bit_depth(self, bits: int, auto_levels: bool) -> None:
        self._bit_depth = int(bits)
        self._auto_levels = bool(auto_levels)
        self._apply_levels()

    def set_contrast(self, auto: bool, manual_min: float, manual_max: float,
                     auto_max: Optional[float] = None) -> None:
        """Auto: percentile stretch capped at *auto_max*; manual: fixed min/max."""
        self._auto_levels = bool(auto)
        self._manual_levels = (float(manual_min), float(manual_max))
        self._auto_max = None if auto_max is None else float(auto_max)
        self._apply_levels()
        if self._hist_visible:
            self._update_histogram(force=True)

    def set_follow_mouse(self, follow: bool) -> None:
        self._follow_mouse = bool(follow)

    # -- histogram -------------------------------------------------------------
    @property
    def histogram_visible(self) -> bool:
        return self._hist_visible

    def set_histogram_visible(self, show: bool) -> None:
        show = bool(show)
        if show == self._hist_visible:
            return
        self._hist_visible = show
        if show:
            self.ci.addItem(self.hist, row=2, col=1)
            self.ci.layout.setRowStretchFactor(2, 2)
            self._update_histogram(force=True)
        else:
            self.ci.removeItem(self.hist)
            self.ci.layout.setRowStretchFactor(2, 0)

    def _update_histogram(self, force: bool = False) -> None:
        a = self._arr
        now = time.monotonic()
        if a is None or (not force and now - self._hist_at < _HIST_INTERVAL_S):
            return
        self._hist_at = now
        step = max(1, a.shape[0] * a.shape[1] // 200_000)
        sub = a[::step]
        planes = [sub] if sub.ndim == 2 else [sub[..., c] for c in range(min(3, sub.shape[2]))]
        lo = float(min(p.min() for p in planes))
        hi = float(max(p.max() for p in planes))
        if hi <= lo:
            hi = lo + 1.0
        for i, curve in enumerate(self._hist_curves):
            if i >= len(planes):
                curve.setData([], [])
                continue
            counts, edges = np.histogram(planes[i], bins=_HIST_BINS, range=(lo, hi))
            color = _HIST_PENS[0] if len(planes) == 1 else _HIST_PENS[1 + i]
            curve.setPen(pg.mkPen(color, width=1))
            curve.setBrush(pg.mkBrush(color + "40"))
            curve.setData(0.5 * (edges[:-1] + edges[1:]), np.log10(1.0 + counts))
        levels = self.img.levels
        if levels is not None:
            self._hist_lo.setValue(float(levels[0]))
            self._hist_hi.setValue(float(levels[1]))

    # -- lifecycle -----------------------------------------------------------
    def clear_frame(self) -> None:
        """Forget the current frame (used when the camera disconnects)."""
        self._src = None
        self._arr = None
        self._last_meta = None
        self._shape = None
        self.img.clear()
        self._top_curve.setData([], [])
        self._right_curve.setData([], [])
        for curve in self._hist_curves:
            curve.setData([], [])
        self._set_hud("")

    # NOTE: GraphicsLayoutWidget.__init__ binds `self.clear` to
    # GraphicsLayout.clear, which *removes every plot from the layout*.  Rebind
    # it: callers that clear the view mean "drop the frame", not "destroy the
    # widget".  Use `self.ci.clear()` if the layout itself must be emptied.
    clear = clear_frame
    _clear_frame = clear_frame

    def reset_view(self) -> None:
        """Fit the image to the viewport (instant: sets an explicit range)."""
        if self._arr is None:
            self.image_plot.getViewBox().disableAutoRange()
            return
        h, w = self._arr.shape[0], self._arr.shape[1]
        self._fit(h, w)
