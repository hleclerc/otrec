"""Export of a 2D point cloud as a standalone HTML page, with a full-screen <canvas> and one
disk drawn per point -- an alternative to `matplotlib.pyplot.plot(..., '.')`, which is hard to read and
barely interactive at large scale (no zoom, markers of fixed size in screen pixels).

Also supports a SEQUENCE of clouds ("frames", e.g. the positions at each step of a
reconstruction): the page then adds a time bar + a play/pause button to replay the
movement of the points until convergence.

Each point is either a "point" or a "shape" (see `export_positions_html`), determined PER
POINT (not globally -- a single export can mix both, e.g. a cloud that interleaves
Dirac stages and disk stages):
- "point" (no `radii`/`vertex_offsets` at all, or a radius of 0 -- Dirac cloud) -- the
  drawn radius is a pure display setting, driven in WORLD units by the slider.
- "shape" (`radii` > 0, or `vertex_offsets` -- e.g. the disk reconstruction of `disks.py`,
  where the radius/geometry is part of the model) -- drawn at the EXACT exported size; the
  slider does not apply (a shape has no "scale", just the size it was given).

SINGLE, standalone file (the points are base64-encoded directly in the HTML, no side
file / no server -- opens directly from disk, `file://`).
"""
import base64
import json

import numpy as np

from loom import Tensor


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #ffffff; color: #111; }
  canvas { display: block; background: #ffffff; }
  #controls {
    position: fixed; top: 10px; left: 10px; z-index: 1;
    background: rgba(255,255,255,0.88); padding: 8px 12px; border-radius: 6px;
    font-family: sans-serif; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    width: 230px; box-sizing: border-box;
  }
  #controls *, #controls { box-sizing: border-box; }
  #controls label { display: block; }
  #controls input[type=range] { vertical-align: middle; width: 200px; }
  #controls .hint { color: #666; margin-top: 4px; }
  #timeControls { display: none; margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; }
  #timeControls .row { display: flex; align-items: center; gap: 6px; }
  #timeControls input[type=range] { flex: 1 1 auto; width: auto; min-width: 0; }
  #play {
    cursor: pointer; border: none; border-radius: 4px; background: #444; color: #fff;
    width: 26px; height: 26px; font-size: 12px; line-height: 1; flex: none;
  }
  #tval {
    min-width: 46px; flex: none; text-align: right; font-variant-numeric: tabular-nums;
  }
  #help {
    display: none; position: fixed; top: 10px; right: 10px; z-index: 2;
    background: rgba(255,255,255,0.95); padding: 10px 14px; border-radius: 6px;
    font-family: sans-serif; font-size: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  #help table { border-collapse: collapse; }
  #help td { padding: 1px 0; }
  #help td:first-child { padding-right: 10px; color: #333; white-space: nowrap; }
  #help td:last-child { color: #666; }
  body.dark { background: #1a1a1a; color: #ddd; }
  body.dark canvas { background: #1a1a1a; }
  body.dark #controls { background: rgba(30,30,30,0.92); color: #ddd; }
  body.dark #controls .hint { color: #aaa; }
  body.dark #controls input[type=range] { accent-color: #8ab4f8; }
  body.dark #timeControls { border-top-color: #555; }
  body.dark #play { background: #8ab4f8; color: #172033; }
  body.dark #help { background: rgba(30,30,30,0.95); color: #ccc; }
  body.dark #help td:first-child { color: #ddd; }
  body.dark #help td:last-child { color: #aaa; }
</style>
</head>
<body data-theme="light">
<div id="controls">
  <label><span id="rname">radius</span> : <span id="rval">__RADIUS__</span>
    <input id="r" type="range" min="0" max="1" step="0.001" value="__RT0__">
  </label>
  <div><span id="ptcount">__N__</span> points</div>
  <div id="timeControls">
    <div class="row">
      <button id="play">&#9654;</button>
      <input id="t" type="range" min="0" max="0" step="1" value="0">
      <span id="tval">1 / 1</span>
    </div>
  </div>
  <div class="hint">press <b>?</b> for help · <b>d</b>: <span id="modeLabel">light</span></div>
</div>
<div id="help">
  <b>Keyboard shortcuts</b>
  <table>
    <tr><td>&larr; / &rarr;</td><td>time -1 / +1</td></tr>
    <tr><td>Shift/Ctrl + &larr;/&rarr;</td><td>time, larger step</td></tr>
    <tr><td>Home / End</td><td>first / last frame</td></tr>
    <tr><td>Space</td><td>play / pause</td></tr>
    <tr><td>&uarr; / &darr;</td><td>radius + / - (Shift/Ctrl: faster)</td></tr>
    <tr><td>+ / -</td><td>zoom in/out (Shift/Ctrl: faster)</td></tr>
    <tr><td>0</td><td>reset the view</td></tr>
    <tr><td>drag</td><td>pan</td></tr>
    <tr><td>wheel / pinch</td><td>pan / zoom</td></tr>
    <tr><td>double-click</td><td>reset the view</td></tr>
    <tr><td>d</td><td>toggle light / dark</td></tr>
    <tr><td>?</td><td>show/hide this help</td></tr>
  </table>
</div>
<canvas id="c"></canvas>
<script>
function decodeF32(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(buf);
}
const pts = decodeF32("__B64__");            // all frames concatenated : [x0,y0, x1,y1, ...]
const COUNTS = __COUNTS__;                   // number of points per frame (may vary from one frame to the next)
const OFFSETS = (() => {                     // offset (in points) of the start of each frame
  let acc = 0;
  const o = [];
  for (const c of COUNTS) { o.push(acc); acc += c; }
  return o;
})();
const nFrames = COUNTS.length;
const FPS = __FPS__;
const bound = __BOUND__;                     // half-extent of the displayed world ([-bound,bound]^2)
const RMIN = __RMIN__, RMAX = __RMAX__;      // slider bounds, log scale

// Two categories of points, distinguished per point (not globally):
//  - "point" (no size data of its own: RADII/R0 <= 0, or SCALED = false) -- a simple
//    disk, always drawn at the SLIDER's size (absolute world units, `rPoint()`): it is
//    a point (Dirac), not a shape, it has its own display setting.
//  - "shape" (RADII[i] > 0 -- or R0 > 0 when RADII is uniform --, or VERT_OFFSETS) -- size
//    set by the MODEL (exported radius/vertices), drawn AS IS: the slider does not
//    apply (a shape has no "scale", just the size it was given).
// SCALED = false (no `radii`/`vertex_offsets` provided): historical behavior intact, all
// points are drawn with the requested marker (MARKER_VERTS or circle) at the slider's
// size -- no point/shape distinction possible without radius data.
const SCALED = __SCALED__;
const R0 = __R0__;                           // uniform world radius (SCALED, RADII == null) or 1
const RADII = __RADII__;                     // Float32Array per point, aligned on OFFSETS, or null
// unit vertices of the marker (e.g. a triangle -- see `export_positions_html`'s `marker`),
// or null for the historical disk (`draw()` then draws a circle via `ctx.arc`). Used
// only for "shape" points (see above); a "point" point is always a circle.
const MARKER_VERTS = __MARKER_VERTS__;
// EXPLICIT polygon per point (e.g. a deformed triangle, with its own orientation per point) --
// WORLD offsets from the center, NOT unit vertices, drawn AS IS (no
// slider): takes precedence over MARKER_VERTS/RADII when provided (see `export_positions_html`'s
// `vertex_offsets`). All points of a `vertex_offsets` export are "shapes".
const VERT_OFFSETS = __VOFF__;
const VOFF_K = __VOFF_K__;                   // vertices per polygon when VERT_OFFSETS is provided

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const rSlider = document.getElementById('r');
const rLabel = document.getElementById('rval');
const ptCountSpan = document.getElementById('ptcount');
const timeControls = document.getElementById('timeControls');
const tSlider = document.getElementById('t');
const tLabel = document.getElementById('tval');
const playBtn = document.getElementById('play');

// slider at t in [0,1] -> value = RMIN * (RMAX/RMIN)^t (log scale: equal relative steps everywhere,
// unlike a linear slider where most of the useful range gets squeezed at the bottom of the
// range -- which caused visual "jumps" at the bottom of the range)
function currentRadius() {
  return RMIN * Math.pow(RMAX / RMIN, parseFloat(rSlider.value));
}
function updateLabel() {
  // always an absolute WORLD radius -- the slider now only drives "point" points (the
  // "shapes" keep the size they were sent, see the consts above), so no notion
  // of scale/factor to display here.
  rLabel.textContent = currentRadius().toPrecision(3);
}
if (SCALED) document.getElementById('rname').textContent = 'radius (points)';

// interactive view: multiplicative zoom + pan in screen pixels, around (cx,cy) = canvas center
let zoom = 1, panX = 0, panY = 0;
const MIN_ZOOM = 0.02, MAX_ZOOM = 500;
let baseScale = 1, cx = 0, cy = 0;           // recomputed at each draw() (depend on w,h)

let frameIdx = 0;

function draw() {
  const w = canvas.width, h = canvas.height;
  const dark = document.body.classList.contains('dark');
  // The background is painted into the canvas bitmap (not only by CSS), which also guarantees
  // correct rendering when capturing or exporting the canvas.
  ctx.fillStyle = dark ? '#1a1a1a' : '#ffffff';
  ctx.fillRect(0, 0, w, h);
  baseScale = Math.min(w, h) / (2 * bound);
  cx = w / 2;
  cy = h / 2;
  const scale = baseScale * zoom;
  const ox = cx + panX, oy = cy + panY;
  const mult = currentRadius();
  const rUniform = Math.max(R0 * mult * scale, 0.4);   // screen size of points in !SCALED mode
  const rPoint = Math.max(mult * scale, 0.4);          // screen size of "point" points in SCALED mode
  ctx.fillStyle = dark ? '#ffffff' : '#000000';
  const path = new Path2D();
  const off = OFFSETS[frameIdx], cnt = COUNTS[frameIdx];
  // true if point `i` (GLOBAL index, not relative to the frame) has no size of its own --
  // a "point" (Dirac), to be drawn as a simple disk at the slider's size, NOT as
  // MARKER_VERTS (reserved for "shape" points) -- see the consts above.
  function isBarePoint(i) {
    if (!SCALED || VERT_OFFSETS) return false;
    const r = RADII ? RADII[i] : R0;
    return !(r > 0);
  }
  if (VERT_OFFSETS) {
    // EXPLICIT polygon per point: WORLD offsets (not unit) sent directly by
    // the caller -- e.g. deformed triangles whose shape/orientation varies point by point,
    // instead of a single MARKER_VERTS template rescaled by a common radius. Drawn AS
    // IS: the slider does not apply (an explicitly sent shape has no "scale").
    for (let i = 0; i < cnt; i++) {
      const x = ox + pts[2 * (off + i)] * scale;
      const y = oy - pts[2 * (off + i) + 1] * scale;
      const base = (off + i) * VOFF_K * 2;
      path.moveTo(x + VERT_OFFSETS[base] * scale, y - VERT_OFFSETS[base + 1] * scale);
      for (let k = 1; k < VOFF_K; k++) {
        const b = base + 2 * k;
        path.lineTo(x + VERT_OFFSETS[b] * scale, y - VERT_OFFSETS[b + 1] * scale);
      }
      path.closePath();
    }
  } else {
    for (let i = 0; i < cnt; i++) {
      const idx = off + i;
      const x = ox + pts[2 * idx] * scale;
      const y = oy - pts[2 * idx + 1] * scale;
      if (isBarePoint(idx)) {
        // "point": no size data -- always a disk, size driven by the slider,
        // never MARKER_VERTS (which only makes sense for a point that has a real size/shape).
        path.moveTo(x + rPoint, y);
        path.arc(x, y, rPoint, 0, 2 * Math.PI);
        continue;
      }
      // "shape": size set by the model (RADII[idx], or R0 if uniform), drawn as
      // is -- no slider; in !SCALED mode (no radii at all), we fall back to the
      // historical behavior (slider size, R0 = 1).
      const r = SCALED ? Math.max((RADII ? RADII[idx] : R0) * scale, 0.4) : rUniform;
      if (MARKER_VERTS) {
        path.moveTo(x + MARKER_VERTS[0][0] * r, y - MARKER_VERTS[0][1] * r);
        for (let k = 1; k < MARKER_VERTS.length; k++)
          path.lineTo(x + MARKER_VERTS[k][0] * r, y - MARKER_VERTS[k][1] * r);
        path.closePath();
      } else {
        path.moveTo(x + r, y);
        path.arc(x, y, r, 0, 2 * Math.PI);
      }
    }
  }
  ctx.fill(path);
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  draw();
}

// zooms by a factor `factor` keeping the screen point (px,py) fixed on screen
function zoomAt(px, py, factor) {
  const newZoom = Math.min(Math.max(zoom * factor, MIN_ZOOM), MAX_ZOOM);
  const applied = newZoom / zoom;
  panX = px - cx - (px - cx - panX) * applied;
  panY = py - cy - (py - cy - panY) * applied;
  zoom = newZoom;
  draw();
}

function resetView() {
  zoom = 1; panX = 0; panY = 0;
  draw();
}

rSlider.addEventListener('input', () => { updateLabel(); draw(); });
window.addEventListener('resize', resize);

// wheel/trackpad: the browser reports a trackpad pinch as a `wheel` with
// ctrlKey=true (and deltaY ~ the magnitude of the pinch) -- convention Ctrl+wheel = zoom on a
// regular mouse too. Without ctrlKey, a scroll (mouse or 2-finger trackpad) = pan.
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) {
    zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.01));
  } else {
    panX -= e.deltaX;
    panY -= e.deltaY;
    draw();
  }
}, { passive: false });

// mouse drag = pan
let dragging = false, lastX = 0, lastY = 0;
canvas.style.cursor = 'grab';
canvas.addEventListener('mousedown', (e) => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.style.cursor = 'grabbing';
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  panX += e.clientX - lastX;
  panY += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  draw();
});
window.addEventListener('mouseup', () => { dragging = false; canvas.style.cursor = 'grab'; });

canvas.addEventListener('dblclick', resetView);

// touch screens: 1 finger = pan, 2 fingers = pinch (zoom) + pan
let touch = null;
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  if (e.touches.length === 1) {
    touch = { mode: 'pan', x: e.touches[0].clientX, y: e.touches[0].clientY };
  } else if (e.touches.length === 2) {
    const [t1, t2] = e.touches;
    touch = {
      mode: 'pinch',
      dist: Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY),
    };
  }
}, { passive: false });
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (!touch) return;
  if (touch.mode === 'pan' && e.touches.length === 1) {
    const dx = e.touches[0].clientX - touch.x, dy = e.touches[0].clientY - touch.y;
    panX += dx; panY += dy;
    touch.x = e.touches[0].clientX; touch.y = e.touches[0].clientY;
    draw();
  } else if (touch.mode === 'pinch' && e.touches.length === 2) {
    const [t1, t2] = e.touches;
    const rect = canvas.getBoundingClientRect();
    const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    const mx = (t1.clientX + t2.clientX) / 2 - rect.left;
    const my = (t1.clientY + t2.clientY) / 2 - rect.top;
    zoomAt(mx, my, dist / touch.dist);
    touch.dist = dist;
  }
}, { passive: false });
canvas.addEventListener('touchend', (e) => {
  if (e.touches.length === 0) touch = null;
  else if (e.touches.length === 1) touch = { mode: 'pan', x: e.touches[0].clientX, y: e.touches[0].clientY };
});

// time bar + play/pause (only if several frames)
let playing = false;
let playTimer = null;

function stopPlaying() {
  playing = false;
  playBtn.innerHTML = '&#9654;';
  clearInterval(playTimer);
}

function setFrame(i) {
  frameIdx = Math.max(0, Math.min(nFrames - 1, i));
  tSlider.value = frameIdx;
  tLabel.textContent = `${frameIdx + 1} / ${nFrames}`;
  ptCountSpan.textContent = COUNTS[frameIdx];
  draw();
}

function togglePlay() {
  if (playing) { stopPlaying(); return; }
  playing = true;
  playBtn.innerHTML = '&#10074;&#10074;';
  if (frameIdx >= nFrames - 1) setFrame(0);
  playTimer = setInterval(() => {
    if (frameIdx >= nFrames - 1) { stopPlaying(); return; }
    setFrame(frameIdx + 1);
  }, 1000 / FPS);
}

if (nFrames > 1) {
  timeControls.style.display = 'block';
  tSlider.max = nFrames - 1;
  tSlider.addEventListener('input', () => { stopPlaying(); setFrame(parseInt(tSlider.value, 10)); });
  playBtn.addEventListener('click', togglePlay);
}

// keyboard shortcuts -- modifiers (Shift/Ctrl) = larger step, to navigate/adjust faster
function pick(e, normal, big, huge) {
  if (e.ctrlKey || e.metaKey) return huge;
  if (e.shiftKey) return big;
  return normal;
}
function timeStep(e) {
  return pick( e, 1, Math.max( 1, Math.round( nFrames / 20 ) ), Math.max( 1, Math.round( nFrames / 5 ) ) );
}
function adjustRadius(delta) {
  const t = Math.min(1, Math.max(0, parseFloat(rSlider.value) + delta));
  rSlider.value = t;
  updateLabel();
  draw();
}
function toggleHelp() {
  helpPanel.style.display = helpPanel.style.display === 'block' ? 'none' : 'block';
}

const helpPanel = document.getElementById('help');

window.addEventListener('keydown', (e) => {
  if (e.key === '?') { e.preventDefault(); toggleHelp(); return; }
  if (e.key === 'Escape' && helpPanel.style.display === 'block') {
    e.preventDefault(); helpPanel.style.display = 'none'; return;
  }
  switch (e.key) {
    case 'ArrowRight':
      if (nFrames > 1) { e.preventDefault(); stopPlaying(); setFrame(frameIdx + timeStep(e)); }
      break;
    case 'ArrowLeft':
      if (nFrames > 1) { e.preventDefault(); stopPlaying(); setFrame(frameIdx - timeStep(e)); }
      break;
    case 'Home':
      if (nFrames > 1) { e.preventDefault(); stopPlaying(); setFrame(0); }
      break;
    case 'End':
      if (nFrames > 1) { e.preventDefault(); stopPlaying(); setFrame(nFrames - 1); }
      break;
    case ' ':
      if (nFrames > 1) { e.preventDefault(); togglePlay(); }
      break;
    case 'ArrowUp':
      e.preventDefault(); adjustRadius(pick(e, 0.01, 0.05, 0.2));
      break;
    case 'ArrowDown':
      e.preventDefault(); adjustRadius(-pick(e, 0.01, 0.05, 0.2));
      break;
    case '+':
    case '=':
      e.preventDefault(); zoomAt(cx, cy, Math.pow(1.2, pick(e, 1, 2, 5)));
      break;
    case '-':
    case '_':
      e.preventDefault(); zoomAt(cx, cy, 1 / Math.pow(1.2, pick(e, 1, 2, 5)));
      break;
    case '0':
      e.preventDefault(); resetView();
      break;
  }
});

// Dark mode: system preference at load, then toggled with the [d] shortcut.
(function() {
  const darkML = window.matchMedia('(prefers-color-scheme: dark)');
  let curDark = darkML.matches;
  function setTheme(on) {
    curDark = on;
    document.body.classList.toggle('dark', on);
    document.body.dataset.theme = on ? 'dark' : 'light';
    const m = document.getElementById('modeLabel');
    if (m) m.textContent = on ? 'dark' : 'light';
    draw();
  }
  setTheme(curDark);
  darkML.addEventListener('change', e => setTheme(e.matches));
  document.addEventListener('keydown', (kd) => {
    if (kd.key === 'D' || kd.key === 'd') { kd.preventDefault(); setTheme(!curDark); }
  });
})();

// last frame by default (the final/converged state, what we want to see first)
updateLabel();
setFrame(nFrames - 1);
resize();
if (nFrames > 1) {
  // focus the time bar: the left/right arrows drive it immediately
  tSlider.focus();
}
</script>
</body>
</html>
"""


def _as_frames( positions ):
    """Normalizes `positions` into a list of `np.float32[n_t, 2]` frames (a single frame for
    historical usage, several for an animation):
    - Tensor/array [n,2]                       -> one frame.
    - Tensor/array [T,n,2] (fixed n)            -> T frames.
    - sequence (list/tuple) of Tensor/array [n_t,2], with n_t possibly varying from one frame to the next
      (e.g. the stages of a `Reconstruction.multiscale`) -> one frame per element.
    """
    def to_np( x ):
        arr = x.raw if isinstance( x, Tensor ) else np.asarray( x )
        return np.asarray( arr, dtype = np.float32 )

    if isinstance( positions, Tensor ):
        positions = positions.raw
    if isinstance( positions, np.ndarray ):
        if positions.ndim == 2:
            return [ to_np( positions ) ]
        if positions.ndim == 3:
            return [ to_np( positions[ t ] ) for t in range( positions.shape[ 0 ] ) ]
        raise ValueError( f"positions: unexpected ndim { positions.ndim } (expected 2 or 3)" )
    if isinstance( positions, ( list, tuple ) ):
        if len( positions ) == 0:
            raise ValueError( "positions: empty sequence" )
        first = to_np( positions[ 0 ] )
        if first.ndim == 1 and first.shape[ 0 ] == 2:
            # `positions` is directly a list of (x, y) pairs -- a single frame.
            return [ to_np( positions ) ]
        return [ to_np( f ) for f in positions ]
    raise TypeError( f"positions: unsupported type { type( positions ) }" )


def _as_radii( radii, frames ):
    """Normalizes `radii` into `( uniform, per_point )`:
    - `uniform`: the common world radius (float) when there is only one -- nothing is encoded.
    - `per_point`: a list of `np.float32[n_t]` aligned on `frames`, or `None`.

    Accepts a scalar (single radius, the case of a fixed-radius model), an `[n]` array reused
    for all frames, or a sequence of one array per frame.
    """
    def to_np( x ):
        arr = x.raw if isinstance( x, Tensor ) else np.asarray( x )
        return np.asarray( arr, dtype = np.float32 ).reshape( -1 )

    if np.isscalar( radii ):
        return float( radii ), None

    # a sequence of one array per frame -- distinguished from a simple array of per-point radii
    # by the fact that its elements are not scalars.
    if isinstance( radii, ( list, tuple ) ) and len( radii ) == len( frames ) \
            and not np.isscalar( radii[ 0 ] ):
        per_frame = [ to_np( r ) for r in radii ]
    else:
        shared = to_np( radii )
        if len( shared ) == 1:                        # single radius passed as an array
            return float( shared[ 0 ] ), None
        per_frame = [ shared for _ in frames ]

    for f, r in zip( frames, per_frame ):
        if len( r ) != len( f ):
            raise ValueError( f"radii: { len( r ) } radii for { len( f ) } points" )
    # a constant array reduces to the uniform case (no block of radii to encode)
    flat = np.concatenate( per_frame ) if per_frame else np.zeros( 0, dtype = np.float32 )
    if len( flat ) and np.all( flat == flat[ 0 ] ):
        return float( flat[ 0 ] ), None
    return None, per_frame


def _as_vertex_offsets( vertex_offsets, frames ):
    """Normalise `vertex_offsets` into a list of `np.float32[n_t, k, 2]` aligned on `frames`
    (the vertex count `k` must be the SAME for every frame/point), or `None`.

    Accepts a single `[n, k, 2]` array (reused for every frame) or a sequence of one
    `[n_t, k, 2]` array per frame -- see `export_positions_html`'s `vertex_offsets`.
    """
    if vertex_offsets is None:
        return None

    def to_np( x ):
        arr = x.raw if isinstance( x, Tensor ) else np.asarray( x )
        return np.asarray( arr, dtype = np.float32 )

    if isinstance( vertex_offsets, ( list, tuple ) ) and len( vertex_offsets ) == len( frames ) \
            and to_np( vertex_offsets[ 0 ] ).ndim == 3:
        per_frame = [ to_np( v ) for v in vertex_offsets ]
    else:
        shared = to_np( vertex_offsets )
        if shared.ndim != 3:
            raise ValueError( f"vertex_offsets: expected [n,k,2] (or a per-frame sequence "
                             f"thereof), got shape { shared.shape }" )
        per_frame = [ shared for _ in frames ]

    ks = { v.shape[ 1 ] for v in per_frame }
    if len( ks ) != 1:
        raise ValueError( f"vertex_offsets: vertex count k must be the same for every frame, "
                         f"got { sorted( ks ) }" )
    for f, v in zip( frames, per_frame ):
        if v.shape[ 0 ] != len( f ):
            raise ValueError( f"vertex_offsets: { v.shape[ 0 ] } polygons for { len( f ) } points" )
        if v.shape[ 2 ] != 2:
            raise ValueError( f"vertex_offsets: expected last axis of size 2, got { v.shape }" )
    return per_frame


#: markers built in -- unit vertices (x, y), circumradius 1, apex up. "circle" (default) draws
#: the historical `ctx.arc` disc instead of a polygon (see `MARKER_SHAPES` used as a lookup only
#: for the STRING convenience -- pass a raw vertex list to `marker` for any OTHER polygon, the
#: JS side (`MARKER_VERTS`) doesn't special-case the vertex count, "3 points" is not baked in).
MARKER_SHAPES = {
    "circle": None,
    "triangle": [ ( 0.0, 1.0 ), ( -0.8660254, -0.5 ), ( 0.8660254, -0.5 ) ],
}


def export_positions_html(
    positions, extent: float, out_path: str, point_radius: float | None = None,
    radius_range: tuple[float, float] | None = None, max_points: int = 500_000,
    title: str = "reconstruction", seed: int = 0, fps: float = 5.0,
    radii = None, marker: str | list = "circle", vertex_offsets = None,
):
    """Writes `out_path`, a standalone HTML page displaying `positions` as one disk per point
    on a full-screen <canvas>, with a slider (LOG scale) to adjust the radius of the disks
    (in WORLD units, not pixels -- stays consistent whatever the window size).
    `extent` sets the displayed window ([-extent/2, extent/2]^2, same conventions as `Sinogram`).
    `radius_range` must be strictly positive (bounds of the log slider) and bracket
    `point_radius`.

    `positions`: either ONE frame -- Tensor/array [n,2] (historical behavior) --, or
    SEVERAL frames to illustrate a movement over time (e.g. the convergence of a
    reconstruction): Tensor/array [T,n,2] (fixed n), or a sequence of Tensor/array [n_t,2] --
    `n_t` may vary from one frame to the next (e.g. the stages of `Reconstruction.multiscale`, which
    progressively refine the number of Diracs). With several frames, the page adds a
    time bar + a play/pause button (speed set by `fps`).

    `max_points`: subsamples beyond this count, PER FRAME -- an HTML file with 1e7
    points would encode ~80 MB of coordinates (base64 ~+33%), heavy to load for no
    visual gain (the disks overlap well before that density on screen anyway). If all
    frames have the same number of points, the subsampling uses the SAME indices
    for all frames (an animated point keeps its visual identity from one frame to the next);
    otherwise (variable number of points, e.g. multiscale) each frame is subsampled
    independently.

    `radii`: the WORLD radius of the points, when they have one that is part of the model and not
    just of the display (typically the disk reconstruction of `disks.py`, with fixed radius). A
    scalar (common radius), an `[n]` array (per point, reused for all frames), or
    a sequence of one array per frame. This radius is then EXPORTED and used for drawing, at the
    EXACT size provided -- the slider does NOT apply to it. `None` (default), or a radius of 0 for a given
    point: this point has no size of its own (a "point", e.g. a Dirac) -- it is always
    drawn as a simple disk (never `marker`), at the slider's size (the slider IS its
    radius, historical behavior). Both can coexist in the same export: per
    point, not a global mode -- useful for a cloud that interleaves Dirac stages (radius
    0) and disk stages (real radius), see `optim.recorder.Recorder.frame_radii`.

    `marker`: the SHAPE drawn by the points that HAVE a radius (`radii` > 0) -- `"circle"`
    (default, historical) or `"triangle"` (see `MARKER_SHAPES`), or directly a list of
    vertices `[(x0,y0), (x1,y1), ...]` (UNIT coordinates, circumscribed circle of radius 1,
    apex up) for an arbitrary polygon -- the JS rendering (`MARKER_VERTS`) does NOT assume 3
    vertices, a triangle is just a special case. Each vertex is scaled by the
    EXACT radius of the point (`RADII`/`R0`, no slider) and translated to its position; without
    `radii` at all (historical mode), it applies to ALL points, at the slider's size.
    Ignored if `vertex_offsets` is provided.

    `vertex_offsets`: EXPLICIT polygon per point, in WORLD units, when `marker` (a single unit
    template rescaled by a common radius) is no longer enough -- typically DEFORMED
    triangles (non-equilateral, oriented differently per point) computed by the
    model itself rather than deduced from a center + radius. `[n, k, 2]` (k vertices, WORLD
    offset from the point's center, NOT unit coordinates) reused for all
    frames, or a sequence of one `[n_t, k, 2]` array per frame (`k` must remain the same
    everywhere). Drawn at the EXACT size sent, the slider does not apply -- ALL
    points of a `vertex_offsets` export are "shapes" (no point/shape mixing possible
    here). Takes precedence over `marker`/`radii` when provided.
    """
    frames = _as_frames( positions )
    uniform_r, per_point_r = ( None, None ) if radii is None else _as_radii( radii, frames )
    per_point_vo = _as_vertex_offsets( vertex_offsets, frames )
    counts = [ len( f ) for f in frames ]

    # subsampling: radii/offsets follow EXACTLY the same indices as the positions.
    def take( seq, idx_per_frame ):
        return [ a if i is None else a[ i ] for a, i in zip( seq, idx_per_frame ) ]

    rng = np.random.default_rng( seed )
    if len( set( counts ) ) == 1 and counts[ 0 ] > max_points:
        idx = rng.choice( counts[ 0 ], max_points, replace = False )
        picks = [ idx ] * len( frames )
    elif any( c > max_points for c in counts ):
        picks = [ rng.choice( c, max_points, replace = False ) if c > max_points else None
                  for c in counts ]
    else:
        picks = [ None ] * len( frames )
    frames = take( frames, picks )
    if per_point_r is not None:
        per_point_r = take( per_point_r, picks )
    if per_point_vo is not None:
        per_point_vo = take( per_point_vo, picks )
    counts = [ len( f ) for f in frames ]

    all_points = np.concatenate( frames, axis = 0 ).astype( np.float32 )
    b64 = base64.b64encode( all_points.tobytes() ).decode( "ascii" )
    counts_js = "[" + ",".join( str( c ) for c in counts ) + "]"
    bound = extent / 2

    if per_point_r is None:
        radii_js = "null"
    else:
        flat_r = np.concatenate( per_point_r ).astype( np.float32 )
        radii_js = 'decodeF32("' + base64.b64encode( flat_r.tobytes() ).decode( "ascii" ) + '")'

    verts = MARKER_SHAPES[ marker ] if isinstance( marker, str ) else marker
    marker_js = "null" if verts is None else json.dumps( [ list( v ) for v in verts ] )

    if per_point_vo is None:
        voff_js, voff_k = "null", 0
    else:
        flat_vo = np.concatenate( per_point_vo, axis = 0 ).astype( np.float32 )   # [N, k, 2]
        voff_k = flat_vo.shape[ 1 ]
        voff_js = 'decodeF32("' + base64.b64encode( flat_vo.tobytes() ).decode( "ascii" ) + '")'

    # `point_radius` = initial slider position -- ALWAYS an absolute world radius (0.1 by
    # default), whether it drives all points (historical mode, no `radii`) or only the
    # "point" points (radius 0/absent) of a mixed export -- "shape" points ignore the slider.
    scaled = radii is not None or vertex_offsets is not None   # -> JS `SCALED` (see `_HTML`)
    if point_radius is None:
        point_radius = 0.1
    if radius_range is None:
        radius_range = ( point_radius / 20, point_radius * 20 )
    rmin, rmax = radius_range
    t0 = np.log( point_radius / rmin ) / np.log( rmax / rmin )

    html = ( _HTML
        .replace( "__TITLE__", title )
        .replace( "__RADIUS__", repr( point_radius ) )
        .replace( "__RMIN__", repr( rmin ) )
        .replace( "__RMAX__", repr( rmax ) )
        .replace( "__RT0__", repr( float( t0 ) ) )
        .replace( "__SCALED__", "true" if scaled else "false" )
        .replace( "__R0__", repr( float( uniform_r if uniform_r is not None else 1.0 ) ) )
        .replace( "__RADII__", radii_js )
        .replace( "__MARKER_VERTS__", marker_js )
        .replace( "__VOFF__", voff_js )
        .replace( "__VOFF_K__", str( voff_k ) )
        .replace( "__N__", str( counts[ 0 ] ) )
        .replace( "__COUNTS__", counts_js )
        .replace( "__FPS__", repr( float( fps ) ) )
        .replace( "__B64__", b64 )
        .replace( "__BOUND__", repr( float( bound ) ) )
    )
    with open( out_path, "w" ) as f:
        f.write( html )
    # total = sum( counts )
    print( f"OUTPUT: { out_path }" )
    # print( f"html saved: { out_path } ({ len( frames ) } frame(s), { total } points in total, "
    #        f"{ len( html ) / 1e6:.1f} MB)" )
