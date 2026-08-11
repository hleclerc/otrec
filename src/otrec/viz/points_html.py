"""Export d'un nuage de points 2D en page HTML autonome, avec un <canvas> plein écran et un
disque dessiné par point -- alternative à `matplotlib.pyplot.plot(..., '.')`, peu lisible et peu
interactif à grande échelle (pas de zoom, marqueurs de taille fixe en pixels écran).

Supporte aussi une SÉQUENCE de nuages ("frames", ex. les positions à chaque étape d'une
reconstruction) : la page ajoute alors une barre de temps + un bouton play/pause pour rejouer le
déplacement des points jusqu'à convergence.

Deux régimes de rayon (voir `export_positions_html`) :
- `radii = None` : les points n'ont pas de taille propre (nuage de diracs) -- le rayon des
  disques est un pur réglage d'affichage, piloté en unités MONDE par le slider.
- `radii` fourni : les points SONT des disques d'un rayon donné (ex. la reconstruction par
  disques de `disks.py`, où le rayon est fixé et fait partie du modèle) -- ce rayon est exporté
  avec les positions et sert au dessin ; le slider devient un simple FACTEUR d'échelle (1 par
  défaut), pour pouvoir aérer une figure trop dense sans mentir sur la géométrie.

Fichier UNIQUE et autonome (les points sont encodés en base64 directement dans le HTML, pas de
fichier annexe / pas de serveur -- s'ouvre directement depuis le disque, `file://`).
"""
import base64

import numpy as np

from sdot import Tensor


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
  <label><span id="rname">rayon</span> : <span id="rval">__RADIUS__</span>
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
  <div class="hint">appuyer sur <b>?</b> pour l’aide · <b>d</b> : <span id="modeLabel">clair</span></div>
</div>
<div id="help">
  <b>Raccourcis clavier</b>
  <table>
    <tr><td>&larr; / &rarr;</td><td>temps -1 / +1</td></tr>
    <tr><td>Maj/Ctrl + &larr;/&rarr;</td><td>temps, pas plus large</td></tr>
    <tr><td>Home / End</td><td>première / dernière frame</td></tr>
    <tr><td>Espace</td><td>lecture / pause</td></tr>
    <tr><td>&uarr; / &darr;</td><td>rayon + / - (Maj/Ctrl: plus vite)</td></tr>
    <tr><td>+ / -</td><td>zoom avant/arrière (Maj/Ctrl: plus vite)</td></tr>
    <tr><td>0</td><td>réinitialiser la vue</td></tr>
    <tr><td>glisser</td><td>pan</td></tr>
    <tr><td>molette / pincement</td><td>pan / zoom</td></tr>
    <tr><td>double-clic</td><td>réinitialiser la vue</td></tr>
    <tr><td>d</td><td>basculer clair / sombre</td></tr>
    <tr><td>?</td><td>afficher/masquer cette aide</td></tr>
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
const pts = decodeF32("__B64__");            // toutes les frames concaténées : [x0,y0, x1,y1, ...]
const COUNTS = __COUNTS__;                   // nb de points par frame (peut varier d'une frame à l'autre)
const OFFSETS = (() => {                     // offset (en points) du début de chaque frame
  let acc = 0;
  const o = [];
  for (const c of COUNTS) { o.push(acc); acc += c; }
  return o;
})();
const nFrames = COUNTS.length;
const FPS = __FPS__;
const bound = __BOUND__;                     // demi-étendue du monde affiché ([-bound,bound]^2)
const RMIN = __RMIN__, RMAX = __RMAX__;      // bornes du slider, échelle log

// Rayon dessiné = R_BASE(i) * slider.
//  - SCALED = false : les points n'ont pas de rayon propre -> R_BASE = 1 et le slider EST le
//    rayon, en unités monde (comportement historique).
//  - SCALED = true  : chaque point a un rayon monde exporté (RADII, ou R0 si uniforme) et le
//    slider est un facteur d'échelle sans dimension (1 = rayon réel).
const SCALED = __SCALED__;
const R0 = __R0__;                           // rayon monde uniforme (SCALED) ou 1
const RADII = __RADII__;                     // Float32Array par point, aligné sur OFFSETS, ou null

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const rSlider = document.getElementById('r');
const rLabel = document.getElementById('rval');
const ptCountSpan = document.getElementById('ptcount');
const timeControls = document.getElementById('timeControls');
const tSlider = document.getElementById('t');
const tLabel = document.getElementById('tval');
const playBtn = document.getElementById('play');

// slider en t in [0,1] -> valeur = RMIN * (RMAX/RMIN)^t (échelle log : pas relatifs égaux partout,
// contrairement à un slider linéaire où la majeure partie du réglage utile se tasse en bas de la
// plage -- c'est ce qui causait les "sauts" visuels en bas de plage)
function currentRadius() {
  return RMIN * Math.pow(RMAX / RMIN, parseFloat(rSlider.value));
}
function updateLabel() {
  const v = currentRadius();
  if (!SCALED) { rLabel.textContent = v.toPrecision(3); return; }
  // en mode échelle, on affiche le facteur ET, si le rayon est uniforme, le rayon monde obtenu
  rLabel.textContent = RADII ? ('x' + v.toPrecision(3))
                             : ('x' + v.toPrecision(3) + ' (' + (R0 * v).toPrecision(3) + ')');
}
if (SCALED) document.getElementById('rname').textContent = 'échelle rayon';

// vue interactive : zoom multiplicatif + pan en pixels écran, autour de (cx,cy) = centre canvas
let zoom = 1, panX = 0, panY = 0;
const MIN_ZOOM = 0.02, MAX_ZOOM = 500;
let baseScale = 1, cx = 0, cy = 0;           // recalculés à chaque draw() (dépendent de w,h)

let frameIdx = 0;

function draw() {
  const w = canvas.width, h = canvas.height;
  const dark = document.body.classList.contains('dark');
  // Le fond est peint dans le bitmap du canvas (et pas seulement par CSS), ce qui garantit
  // aussi le bon rendu lors d'une capture ou d'un export du canvas.
  ctx.fillStyle = dark ? '#1a1a1a' : '#ffffff';
  ctx.fillRect(0, 0, w, h);
  baseScale = Math.min(w, h) / (2 * bound);
  cx = w / 2;
  cy = h / 2;
  const scale = baseScale * zoom;
  const ox = cx + panX, oy = cy + panY;
  const mult = currentRadius();
  const rUniform = Math.max(R0 * mult * scale, 0.4);   // rayon écran commun quand RADII est absent
  ctx.fillStyle = dark ? '#ffffff' : '#000000';
  const path = new Path2D();
  const off = OFFSETS[frameIdx], cnt = COUNTS[frameIdx];
  for (let i = 0; i < cnt; i++) {
    const x = ox + pts[2 * (off + i)] * scale;
    const y = oy - pts[2 * (off + i) + 1] * scale;
    const r = RADII ? Math.max(RADII[off + i] * mult * scale, 0.4) : rUniform;
    path.moveTo(x + r, y);
    path.arc(x, y, r, 0, 2 * Math.PI);
  }
  ctx.fill(path);
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  draw();
}

// zoome d'un facteur `factor` en gardant le point écran (px,py) fixe à l'écran
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

// molette/trackpad : le navigateur reporte un pincement trackpad comme un `wheel` avec
// ctrlKey=true (et deltaY ~ l'ampleur du pincement) -- convention Ctrl+molette = zoom sur
// souris classique aussi. Sans ctrlKey, un scroll (souris ou 2 doigts trackpad) = pan.
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

// glisser à la souris = pan
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

// écrans tactiles : 1 doigt = pan, 2 doigts = pincement (zoom) + pan
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

// barre de temps + play/pause (uniquement si plusieurs frames)
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

// raccourcis clavier -- modifiers (Maj/Ctrl) = pas plus large, pour naviguer/régler plus vite
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

// Mode sombre : préférence système au chargement, puis bascule au raccourci [d].
(function() {
  const darkML = window.matchMedia('(prefers-color-scheme: dark)');
  let curDark = darkML.matches;
  function setTheme(on) {
    curDark = on;
    document.body.classList.toggle('dark', on);
    document.body.dataset.theme = on ? 'dark' : 'light';
    const m = document.getElementById('modeLabel');
    if (m) m.textContent = on ? 'sombre' : 'clair';
    draw();
  }
  setTheme(curDark);
  darkML.addEventListener('change', e => setTheme(e.matches));
  document.addEventListener('keydown', (kd) => {
    if (kd.key === 'D' || kd.key === 'd') { kd.preventDefault(); setTheme(!curDark); }
  });
})();

// dernière frame par défaut (l'état final/convergé, ce qu'on veut voir en premier)
updateLabel();
setFrame(nFrames - 1);
resize();
if (nFrames > 1) {
  // focus la barre de temps : les flèches gauche/droite la pilotent immédiatement
  tSlider.focus();
}
</script>
</body>
</html>
"""


def _as_frames( positions ):
    """Normalise `positions` en une liste de frames `np.float32[n_t, 2]` (une seule frame pour
    l'usage historique, plusieurs pour une animation) :
    - Tensor/array [n,2]                       -> une frame.
    - Tensor/array [T,n,2] (n fixe)             -> T frames.
    - séquence (list/tuple) de Tensor/array [n_t,2], n_t pouvant varier d'une frame à l'autre
      (ex. les étages d'un `Reconstruction.multiscale`) -> une frame par élément.
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
        raise ValueError( f"positions: ndim inattendu { positions.ndim } (attendu 2 ou 3)" )
    if isinstance( positions, ( list, tuple ) ):
        if len( positions ) == 0:
            raise ValueError( "positions: séquence vide" )
        first = to_np( positions[ 0 ] )
        if first.ndim == 1 and first.shape[ 0 ] == 2:
            # `positions` est directement une liste de paires (x, y) -- une seule frame.
            return [ to_np( positions ) ]
        return [ to_np( f ) for f in positions ]
    raise TypeError( f"positions: type non supporté { type( positions ) }" )


def _as_radii( radii, frames ):
    """Normalise `radii` en `( uniform, per_point )` :
    - `uniform` : le rayon monde commun (float) quand il n'y en a qu'un -- rien n'est encodé.
    - `per_point` : une liste de `np.float32[n_t]` alignée sur `frames`, ou `None`.

    Accepte un scalaire (rayon unique, cas d'un modèle à rayon fixe), un tableau `[n]` réutilisé
    pour toutes les frames, ou une séquence d'un tableau par frame.
    """
    def to_np( x ):
        arr = x.raw if isinstance( x, Tensor ) else np.asarray( x )
        return np.asarray( arr, dtype = np.float32 ).reshape( -1 )

    if np.isscalar( radii ):
        return float( radii ), None

    # une séquence d'un tableau par frame -- distinguée d'un simple tableau de rayons par point
    # par le fait que ses éléments ne sont pas des scalaires.
    if isinstance( radii, ( list, tuple ) ) and len( radii ) == len( frames ) \
            and not np.isscalar( radii[ 0 ] ):
        per_frame = [ to_np( r ) for r in radii ]
    else:
        shared = to_np( radii )
        if len( shared ) == 1:                        # rayon unique passé sous forme de tableau
            return float( shared[ 0 ] ), None
        per_frame = [ shared for _ in frames ]

    for f, r in zip( frames, per_frame ):
        if len( r ) != len( f ):
            raise ValueError( f"radii: { len( r ) } rayons pour { len( f ) } points" )
    # un tableau constant se réduit au cas uniforme (aucun bloc de rayons à encoder)
    flat = np.concatenate( per_frame ) if per_frame else np.zeros( 0, dtype = np.float32 )
    if len( flat ) and np.all( flat == flat[ 0 ] ):
        return float( flat[ 0 ] ), None
    return None, per_frame


def export_positions_html(
    positions, extent: float, out_path: str, point_radius: float | None = None,
    radius_range: tuple[float, float] | None = None, max_points: int = 500_000,
    title: str = "reconstruction", seed: int = 0, fps: float = 5.0,
    radii = None,
):
    """Écrit `out_path`, une page HTML autonome affichant `positions` comme un disque par point
    sur un <canvas> plein écran, avec un slider (échelle LOG) pour ajuster le rayon des disques
    (en unités MONDE, pas pixels -- reste cohérent quelle que soit la taille de la fenêtre).
    `extent` fixe la fenêtre affichée ([-extent/2, extent/2]^2, mêmes conventions que `Sinogram`).
    `radius_range` doit être strictement positif (bornes du slider log) et encadrer
    `point_radius`.

    `positions` : soit UNE frame -- Tensor/array [n,2] (comportement historique) --, soit
    PLUSIEURS frames pour illustrer un déplacement au fil du temps (ex. la convergence d'une
    reconstruction) : Tensor/array [T,n,2] (n fixe), ou séquence de Tensor/array [n_t,2] --
    `n_t` peut varier d'une frame à l'autre (ex. les étages de `Reconstruction.multiscale`, qui
    raffinent progressivement le nombre de diracs). Avec plusieurs frames, la page ajoute une
    barre de temps + un bouton play/pause (vitesse fixée par `fps`).

    `max_points` : sous-échantillonne au-delà de ce compte, PAR FRAME -- un fichier HTML à 1e7
    points encoderait ~80 Mo de coordonnées (base64 ~+33%), lourd à charger pour un gain visuel
    nul (les disques se recouvrent de toute façon bien avant cette densité à l'écran). Si toutes
    les frames ont le même nombre de points, le sous-échantillonnage utilise les MÊMES indices
    pour toutes les frames (un point animé garde son identité visuelle d'une frame à l'autre) ;
    sinon (nombre de points variable, ex. multiscale) chaque frame est sous-échantillonnée
    indépendamment.

    `radii` : le rayon MONDE des points, quand ils en ont un qui fait partie du modèle et non du
    seul affichage (typiquement la reconstruction par disques de `disks.py`, à rayon fixé). Un
    scalaire (rayon commun), un tableau `[n]` (par point, réutilisé pour toutes les frames), ou
    une séquence d'un tableau par frame. Le rayon est alors EXPORTÉ et sert au dessin, et le
    slider devient un facteur d'échelle (1 = rayon réel) -- `point_radius`/`radius_range` sont
    réinterprétés comme le facteur initial et ses bornes. `None` (défaut) : comportement
    historique, les points n'ont pas de rayon propre et le slider EST le rayon.
    """
    frames = _as_frames( positions )
    uniform_r, per_point_r = ( None, None ) if radii is None else _as_radii( radii, frames )
    counts = [ len( f ) for f in frames ]

    # sous-échantillonnage : les rayons suivent EXACTEMENT les mêmes indices que les positions.
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

    # `point_radius` = position initiale du slider : un RAYON monde en mode absolu (0.1 par défaut,
    # valeur historique), un FACTEUR d'échelle en mode `radii` (1 = rayon réel).
    scaled = radii is not None
    if point_radius is None:
        point_radius = 1.0 if scaled else 0.1
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
        .replace( "__N__", str( counts[ 0 ] ) )
        .replace( "__COUNTS__", counts_js )
        .replace( "__FPS__", repr( float( fps ) ) )
        .replace( "__B64__", b64 )
        .replace( "__BOUND__", repr( float( bound ) ) )
    )
    with open( out_path, "w" ) as f:
        f.write( html )
    total = sum( counts )
    print( f"html sauvé: { out_path } ({ len( frames ) } frame(s), { total } points au total, "
           f"{ len( html ) / 1e6:.1f} Mo)" )
