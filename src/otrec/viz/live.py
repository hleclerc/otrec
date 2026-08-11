"""Visualisation 2D en direct pour les simulations de reconstruction.

Le serveur est volontairement sans dépendance (``http.server`` + SSE).  Une simulation y publie
des contextes nommés, et chaque contexte ``canvas`` a sa propre page plein écran.  L'état est
conservé en mémoire : c'est un outil d'observation d'une exécution, pas un stockage de résultats.

Exemple minimal::

    from otrec.viz.live import LiveSimulation

    live = LiveSimulation("essai-lung")
    points = live.canvas("positions", extent=1.0)  # geometry="points" par défaut
    print(points.url)
    for x in optimisation():
        points.update(x)                 # x est de forme [n, 2]

Lancer le serveur dans un autre terminal::

    micromamba run -n sdot python -m applications.reconstruction.viz.live
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convertit Tensor / tableaux numpy en valeurs JSON, sans imposer un backend numérique."""
    if hasattr(value, "raw"):
        value = value.raw
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class _Context:
    name: str
    kind: str
    config: dict[str, Any]
    entries: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)


@dataclass
class _Simulation:
    id: str
    name: str
    created_at: str
    contexts: dict[str, _Context] = field(default_factory=dict)


class LiveStore:
    """État partagé par les requêtes HTTP du serveur."""

    def __init__(self):
        self.simulations: dict[str, _Simulation] = {}
        self.lock = threading.RLock()

    def create_simulation(self, name: str) -> _Simulation:
        sim = _Simulation(uuid.uuid4().hex[:12], name, dt.datetime.now(dt.timezone.utc).isoformat())
        with self.lock:
            self.simulations[sim.id] = sim
        return sim

    def latest(self) -> _Simulation:
        with self.lock:
            if not self.simulations:
                raise KeyError("aucune simulation")
            return next(reversed(self.simulations.values()))

    def create_context(self, sim_id: str, name: str, kind: str, config: dict[str, Any]) -> _Context:
        if kind != "canvas":
            raise ValueError(f"type de contexte non supporté: {kind!r} (seul 'canvas' est disponible)")
        with self.lock:
            sim = self.simulations[sim_id]
            context = sim.contexts.get(name)
            if context is None:
                context = _Context(name, kind, config)
                sim.contexts[name] = context
            elif context.kind != kind:
                raise ValueError(f"le contexte {name!r} existe déjà avec le type {context.kind!r}")
            return context

    def snapshot(self, sim_id: str, context_name: str) -> dict[str, Any]:
        with self.lock:
            context = self.simulations[sim_id].contexts[context_name]
            return {"type": "snapshot", "config": context.config, "entries": context.entries}

    def publish(self, sim_id: str, context_name: str, data: dict[str, Any]) -> None:
        with self.lock:
            context = self.simulations[sim_id].contexts[context_name]
            coordinates = data.get("coordinates", {})
            # Une même position dans l'espace des paramètres représente le même état : on le
            # remplace, sans encombrer l'historique lorsqu'une simulation réémet sa frame.
            for index, entry in enumerate(context.entries):
                if entry.get("coordinates", {}) == coordinates:
                    context.entries[index] = data
                    break
            else:
                context.entries.append(data)
            message = {"type": "update", "entry": data}
            for listener in list(context.subscribers):
                listener.put(message)

    def subscribe(self, sim_id: str, context_name: str) -> queue.Queue:
        listener: queue.Queue = queue.Queue()
        with self.lock:
            self.simulations[sim_id].contexts[context_name].subscribers.append(listener)
        return listener

    def unsubscribe(self, sim_id: str, context_name: str, listener: queue.Queue) -> None:
        with self.lock:
            context = self.simulations.get(sim_id, _Simulation("", "", "")).contexts.get(context_name)
            if context is not None and listener in context.subscribers:
                context.subscribers.remove(listener)


_PAGE = r"""<!doctype html><html lang="fr"><meta charset="utf-8"><title>Live canvas</title>
<style>html,body,canvas{margin:0;width:100%;height:100%;display:block;background:#fff}#info{position:fixed;top:10px;left:10px;padding:7px 10px;background:#fffd;border-radius:5px;font:13px sans-serif;box-shadow:0 1px 4px #0004}body.dark,body.dark canvas{background:#1a1a1a}body.dark #info{background:#222d;color:#ddd}</style>
<canvas id="canvas"></canvas><div id="info">connexion…</div><script>
const endpoint = __EVENTS__;
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d'), info = document.getElementById('info');
let config = {}, data = null, zoom = 1, panX = 0, panY = 0, dragging = null;
function dark() { return matchMedia('(prefers-color-scheme: dark)').matches; }
function draw() {
  const w=canvas.width=innerWidth, h=canvas.height=innerHeight, isDark=dark();
  ctx.fillStyle=isDark?'#1a1a1a':'#fff'; ctx.fillRect(0,0,w,h); if (!data) return;
  const pts=data.positions || [], extent=config.extent || 1, scale=Math.min(w,h)/extent*zoom, ox=w/2+panX, oy=h/2+panY;
  const radii=data.radii, radius=data.point_radius ?? config.point_radius ?? .003;
  ctx.fillStyle=isDark?'#fff':'#000'; const path=new Path2D();
  for(let i=0;i<pts.length;i++) { const p=pts[i], r=Math.max((radii ? radii[i] : radius)*scale,.4), x=ox+p[0]*scale, y=oy-p[1]*scale; path.moveTo(x+r,y);path.arc(x,y,r,0,2*Math.PI); }
  ctx.fill(path); info.textContent=(config.title || config.name || 'canvas')+' · '+pts.length+' points'+(data.step !== undefined ? ' · pas '+data.step : '');
}
function apply(msg) { if(msg.config) config=msg.config; if(msg.data) data=msg.data; draw(); }
const source=new EventSource(endpoint); source.onmessage=e=>apply(JSON.parse(e.data)); source.onerror=()=>{info.textContent='reconnexion…'};
addEventListener('resize',draw); canvas.addEventListener('wheel',e=>{e.preventDefault(); if(e.ctrlKey||e.metaKey) zoom*=Math.exp(-e.deltaY*.01); else {panX-=e.deltaX;panY-=e.deltaY} draw()},{passive:false});
canvas.addEventListener('mousedown',e=>dragging=[e.clientX,e.clientY]); addEventListener('mousemove',e=>{if(!dragging)return;panX+=e.clientX-dragging[0];panY+=e.clientY-dragging[1];dragging=[e.clientX,e.clientY];draw()});addEventListener('mouseup',()=>dragging=null);canvas.addEventListener('dblclick',()=>{zoom=1;panX=panY=0;draw()}); draw();
</script></html>"""

# Remplace la page minimale ci-dessus : cette version maintient une petite collection de frames
# indexées par des coordonnées de paramètres, manipulable sans aucune dépendance JavaScript.
_PAGE = r"""<!doctype html><html lang="fr"><meta charset="utf-8"><title>Live canvas</title>
<style>html,body,canvas{margin:0;width:100%;height:100%;display:block;background:#fff}#c{position:fixed;top:10px;left:10px;z-index:1;width:260px;padding:9px;background:#fffd;border-radius:6px;font:13px sans-serif;box-shadow:0 1px 4px #0004}.r{margin-top:6px}.r label{display:flex;gap:5px;align-items:center}.r input{flex:1;min-width:0}#hint{color:#666;margin-top:7px}#help{display:none;position:fixed;right:10px;top:10px;padding:9px;background:#fffd;border-radius:6px;font:12px sans-serif}body.dark,body.dark canvas{background:#1a1a1a}body.dark #c,body.dark #help{background:#222e;color:#ddd}body.dark #hint{color:#aaa}</style>
<canvas id="canvas"></canvas><div id="c"><b id="title">connexion…</b><div id="rows"></div><div id="hint">molette : pan · Ctrl/molette : zoom · glisser : pan · 0 : vue initiale · d : thème · ? : aide</div></div><div id="help"><b>Raccourcis</b><br>←/→ : premier curseur<br>Maj/Ctrl : pas plus grand<br>↑/↓ : taille des points<br>double-clic / 0 : réinitialiser la vue</div><script>
const endpoint=__EVENTS__,canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d'),rows=document.querySelector('#rows');let config={},entries=[],selected={},zoom=1,panX=0,panY=0,dragging=null,pointSize=.003;const key=v=>JSON.stringify(v),label=v=>typeof v==='string'?v:JSON.stringify(v);
function dims(){const a=[];for(const e of entries)for(const n of Object.keys(e.coordinates||{}))if(!a.includes(n))a.push(n);return a}function vals(n){const a=[];for(const e of entries){const v=(e.coordinates||{})[n];if(v!==undefined&&!a.some(x=>key(x)===key(v)))a.push(v)}return a}function cur(){return entries.filter(e=>dims().every(n=>selected[n]===undefined||key((e.coordinates||{})[n])===key(selected[n]))).at(-1)||entries.at(-1)}
function controls(){rows.replaceChildren();if((config.geometry||'points')==='points'){const r=document.createElement('div');r.className='r';r.innerHTML='<label>taille <input type="range" min="0" max="1" step=".001"><span></span></label>';const i=r.querySelector('input'),o=r.querySelector('span');i.value=Math.log(pointSize/(config.point_radius/20))/Math.log(400);const p=()=>{pointSize=(config.point_radius/20)*Math.pow(400,+i.value);o.textContent=pointSize.toPrecision(3);draw()};i.oninput=p;p();rows.append(r)}for(const n of dims()){const vs=vals(n);if(vs.length<2)continue;const r=document.createElement('div'),i=document.createElement('input'),o=document.createElement('span'),l=document.createElement('label');r.className='r';i.type='range';i.min=0;i.max=vs.length-1;i.step=1;let k=vs.findIndex(v=>key(v)===key(selected[n]));if(k<0)k=vs.length-1;i.value=k;const set=()=>{selected[n]=vs[+i.value];o.textContent=label(selected[n]);draw()};i.oninput=set;set();l.append(n+' ',i,o);r.append(l);rows.append(r)}}
function draw(){const w=canvas.width=innerWidth,h=canvas.height=innerHeight,dark=document.body.classList.contains('dark');ctx.fillStyle=dark?'#1a1a1a':'#fff';ctx.fillRect(0,0,w,h);const data=cur();if(!data)return;const ps=data.positions||[],s=Math.min(w,h)/(config.extent||1)*zoom,ox=w/2+panX,oy=h/2+panY,path=new Path2D(),disks=config.geometry==='disks';ctx.fillStyle=dark?'#fff':'#000';for(let j=0;j<ps.length;j++){const p=ps[j],rad=Math.max((disks?(data.radii?data.radii[j]:data.radius):pointSize)*s,.4),x=ox+p[0]*s,y=oy-p[1]*s;path.moveTo(x+rad,y);path.arc(x,y,rad,0,2*Math.PI)}ctx.fill(path);document.querySelector('#title').textContent=(config.title||config.name||'canvas')+' · '+ps.length+(disks?' disques':' points')}
function apply(m){if(m.config)config=m.config;if(m.entries){entries=m.entries;for(const n of dims())selected[n]=vals(n).at(-1)}if(m.entry){const co=m.entry.coordinates||{},at=entries.findIndex(e=>key(e.coordinates||{})===key(co));if(at<0)entries.push(m.entry);else entries[at]=m.entry;for(const n of dims()){const vs=vals(n);if(selected[n]===undefined||key(selected[n])===key(vs.at(-2)))selected[n]=vs.at(-1)}}controls();draw()}const source=new EventSource(endpoint);source.onmessage=e=>apply(JSON.parse(e.data));source.onerror=()=>document.querySelector('#title').textContent='reconnexion…';
addEventListener('resize',draw);canvas.addEventListener('wheel',e=>{e.preventDefault();if(e.ctrlKey||e.metaKey)zoom=Math.max(.02,Math.min(500,zoom*Math.exp(-e.deltaY*.01)));else{panX-=e.deltaX;panY-=e.deltaY}draw()},{passive:false});canvas.addEventListener('mousedown',e=>{dragging=[e.clientX,e.clientY];canvas.style.cursor='grabbing'});addEventListener('mousemove',e=>{if(!dragging)return;panX+=e.clientX-dragging[0];panY+=e.clientY-dragging[1];dragging=[e.clientX,e.clientY];draw()});addEventListener('mouseup',()=>{dragging=null;canvas.style.cursor='grab'});canvas.style.cursor='grab';canvas.addEventListener('dblclick',()=>{zoom=1;panX=panY=0;draw()});addEventListener('keydown',e=>{if(e.key==='0'){zoom=1;panX=panY=0;draw()}if(e.key==='d'){document.body.classList.toggle('dark');draw()}if(e.key==='?')document.querySelector('#help').style.display=document.querySelector('#help').style.display==='block'?'none':'block';if(['ArrowLeft','ArrowRight'].includes(e.key)){const n=dims().find(x=>vals(x).length>1);if(n){const vs=vals(n),i=vs.findIndex(v=>key(v)===key(selected[n])),jump=e.ctrlKey||e.metaKey?Math.ceil(vs.length/5):e.shiftKey?Math.ceil(vs.length/20):1;selected[n]=vs[Math.max(0,Math.min(vs.length-1,i+(e.key==='ArrowRight'?jump:-jump)))];controls();draw()}}if(['ArrowUp','ArrowDown'].includes(e.key)&&config.geometry==='points'){pointSize*=e.key==='ArrowUp'?1.2:1/1.2;controls();draw()}});draw();
</script></html>"""


def _make_handler(store: LiveStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # le serveur est souvent lancé à côté d'une simulation bruyante

        def _json(self, status: int, payload: Any) -> None:
            raw = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def _body(self) -> dict[str, Any]:
            size = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(size))

        def do_GET(self) -> None:
            parts = [unquote(x) for x in urlparse(self.path).path.strip("/").split("/")]
            try:
                if parts == ["api", "simulations"]:
                    with store.lock:
                        sims = [{"id": s.id, "name": s.name, "created_at": s.created_at,
                                 "contexts": list(s.contexts)} for s in store.simulations.values()]
                    return self._json(200, sims)
                if parts == ["latest"]:
                    sim = store.latest()
                    with store.lock:
                        contexts = list(sim.contexts)
                    if len(contexts) == 1:
                        return self._redirect(f"/s/{quote(sim.id)}/{quote(contexts[0])}")
                    links = "".join(f'<li><a href="/s/{quote(sim.id)}/{quote(name)}">{name}</a></li>' for name in contexts)
                    raw = f"<!doctype html><title>{sim.name}</title><h1>{sim.name}</h1><p>{sim.created_at}</p><ul>{links}</ul>".encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
                if len(parts) == 2 and parts[0] == "latest":
                    sim = store.latest()
                    with store.lock: sim.contexts[parts[1]]
                    return self._redirect(f"/s/{quote(sim.id)}/{quote(parts[1])}")
                if len(parts) == 6 and parts[:2] == ["api", "simulations"] and parts[3] == "contexts" and parts[5] == "events":
                    return self._events(parts[2], parts[4])
                if len(parts) == 3 and parts[0] == "s":
                    with store.lock: store.simulations[parts[1]].contexts[parts[2]]
                    raw = _PAGE.replace("__EVENTS__", json.dumps(f"/api/simulations/{quote(parts[1])}/contexts/{quote(parts[2])}/events")).encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
                self._json(404, {"error": "route inconnue"})
            except KeyError: self._json(404, {"error": "simulation ou contexte inconnu"})

        def _redirect(self, location: str) -> None:
            self.send_response(302); self.send_header("Location", location); self.end_headers()

        def _events(self, sim_id: str, context_name: str) -> None:
            listener = store.subscribe(sim_id, context_name)
            try:
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.end_headers()
                self.wfile.write(("data: " + json.dumps(store.snapshot(sim_id, context_name), allow_nan=False) + "\n\n").encode()); self.wfile.flush()
                while True:
                    try: message = listener.get(timeout=15)
                    except queue.Empty: self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); continue
                    self.wfile.write(("data: " + json.dumps(message, allow_nan=False) + "\n\n").encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            finally: store.unsubscribe(sim_id, context_name, listener)

        def do_POST(self) -> None:
            parts = [unquote(x) for x in urlparse(self.path).path.strip("/").split("/")]
            try:
                body = self._body()
                if parts == ["api", "simulations"]:
                    sim = store.create_simulation(str(body["name"]))
                    return self._json(201, {"id": sim.id, "name": sim.name, "created_at": sim.created_at})
                if len(parts) == 4 and parts[:2] == ["api", "simulations"] and parts[3] == "contexts":
                    context = store.create_context(parts[2], str(body["name"]), str(body.get("kind", "canvas")), dict(body.get("config", {})))
                    return self._json(201, {"name": context.name, "kind": context.kind})
                if len(parts) == 6 and parts[:2] == ["api", "simulations"] and parts[3] == "contexts" and parts[5] == "data":
                    store.publish(parts[2], parts[4], body); return self._json(200, {"ok": True})
                self._json(404, {"error": "route inconnue"})
            except (KeyError, ValueError, json.JSONDecodeError) as error: self._json(400, {"error": str(error)})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Construit le serveur. Appeler ``serve_forever`` ou utiliser :func:`run_server`."""
    return ThreadingHTTPServer((host, port), _make_handler(LiveStore()))


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = serve(host, port)
    print(f"Visualisation live : http://{host}:{server.server_port}/api/simulations")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


class LiveSimulation:
    """Client Python d'une simulation déclarée sur un serveur :class:`LiveStore`."""

    def __init__(self, name: str, server: str = "http://127.0.0.1:8765"):
        self.server = server.rstrip("/")
        reply = self._post("/api/simulations", {"name": name})
        self.id, self.created_at = reply["id"], reply["created_at"]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(payload, allow_nan=False, default=_jsonable).encode()
        request = Request(self.server + path, raw, {"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request) as response: return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"serveur live: {error.read().decode()}") from error

    def canvas(
        self, name: str, *, extent: float, geometry: str = "points", title: str | None = None,
        point_radius: float = .003,
    ) -> "LiveCanvas":
        """Déclare un canvas de ``points`` ou de ``disks``.

        Les points sont des positions sans rayon : ``point_radius`` est seulement le réglage
        initial du navigateur. Les disques doivent fournir leur rayon à chaque :meth:`update`.
        """
        if geometry not in {"points", "disks"}:
            raise ValueError("geometry doit valoir 'points' ou 'disks'")
        config = {"name": name, "title": title or name, "extent": float(extent),
                  "geometry": geometry, "point_radius": float(point_radius)}
        self._post(f"/api/simulations/{quote(self.id)}/contexts", {"name": name, "kind": "canvas", "config": config})
        return LiveCanvas(self, name, geometry)


class LiveCanvas:
    """Contexte canvas, à points sans rayon ou à disques avec rayon propre."""

    def __init__(self, simulation: LiveSimulation, name: str, geometry: str):
        self.simulation, self.name, self.geometry = simulation, name, geometry

    @property
    def url(self) -> str:
        return f"{self.simulation.server}/s/{quote(self.simulation.id)}/{quote(self.name)}"

    def update(
        self, positions: Any, *, radii: Any = None, step: Any = None,
        parameters: Mapping[str, Any] | None = None, **named_parameters: Any,
    ) -> None:
        """Publie une frame.

        ``step`` et chaque entrée de ``parameters`` (ou argument nommé supplémentaire) deviennent
        des dimensions de navigation. Leurs valeurs peuvent aussi être une liste ou un tuple
        (coordonnée multidimensionnelle).
        """
        points = np.asarray(positions.raw if hasattr(positions, "raw") else positions)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"positions doit être de forme [n, 2], reçu {points.shape}")
        coordinates = dict(parameters or {})
        overlap = coordinates.keys() & named_parameters.keys()
        if overlap:
            raise ValueError(f"paramètre(s) passé(s) deux fois: {', '.join(overlap)}")
        coordinates.update(named_parameters)
        payload: dict[str, Any] = {"positions": points, "coordinates": coordinates}
        if step is not None:
            payload["coordinates"]["step"] = step
        if self.geometry == "points":
            if radii is not None:
                raise ValueError("un canvas de points ne reçoit pas de rayons; utiliser geometry='disks'")
        elif radii is None:
            raise ValueError("un canvas de disques requiert radii (scalaire ou un rayon par point)")
        elif np.isscalar(radii):
            payload["radius"] = float(radii)
        else:
            radii = np.asarray(radii.raw if hasattr(radii, "raw") else radii).reshape(-1)
            if len(radii) != len(points): raise ValueError("radii doit contenir un rayon par point")
            payload["radii"] = radii
        self.simulation._post(f"/api/simulations/{quote(self.simulation.id)}/contexts/{quote(self.name)}/data", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="serveur de visualisation live")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(); run_server(args.host, args.port)


if __name__ == "__main__": main()
