"""Cinegraph editor UI: `cinegraph serve project.json` -> http://127.0.0.1:8765

Viewer + timeline you can edit: drag clips to move, drag edges to trim (slip-safe: left trim moves the in-point),
S = split at playhead, E = enable/disable, Delete = remove, snapping to playhead / markers / clip edges,
properties panel (edit any layer as JSON), Ctrl+Z / Ctrl+Shift+Z undo/redo, scopes, J/K/L + space transport.
Every edit is written straight back to the project file (atomic, .bak kept), so Claude and the UI can edit
the same project at the same time — external edits hot-reload."""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from . import project as PJ
from .compositor import ProjectRuntime
from .render import to_uint8

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Cinegraph</title>
<style>
:root{--bg:#0e1013;--panel:#16191e;--line:#262b33;--text:#d9dde3;--dim:#7d8590;--acc:#e0a63a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 Inter,system-ui,sans-serif;display:flex;flex-direction:column;height:100vh;user-select:none}
header{display:flex;gap:10px;align-items:center;padding:7px 12px;border-bottom:1px solid var(--line);background:var(--panel)}
header b{color:var(--acc);letter-spacing:.08em}.sp{flex:1}
button,select{background:#20252c;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}
#main{flex:1;display:flex;min-height:0}
#view{flex:1;display:flex;align-items:center;justify-content:center;min-width:0;background:#000;gap:8px}
#view img{max-width:100%;max-height:100%}
#props{width:340px;border-left:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column}
#props h4{margin:0;padding:8px 10px;border-bottom:1px solid var(--line);font-size:12px;color:var(--dim);font-weight:600}
#pj{flex:1;background:#0b0d10;color:#cfd6df;border:0;padding:8px;font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;resize:none;outline:none}
#pbar{display:flex;gap:6px;padding:6px;border-top:1px solid var(--line)}
#tl{height:36%;min-height:180px;border-top:1px solid var(--line);background:var(--panel);overflow:auto;position:relative}
#ruler{position:sticky;top:0;height:24px;background:#1b1f25;border-bottom:1px solid var(--line);z-index:3;cursor:pointer}
.row{display:flex;height:26px;border-bottom:1px solid #1d2128}
.name{width:190px;flex:none;padding:4px 8px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-right:1px solid var(--line);position:sticky;left:0;background:var(--panel);z-index:2}
.lane{position:relative;flex:1}
.bar{position:absolute;top:3px;height:20px;border-radius:4px;font-size:11px;padding:2px 8px;white-space:nowrap;overflow:hidden;color:#0b0d10;cursor:grab}
.bar.sel{outline:2px solid #fff}.bar.off{opacity:.35}.bar.ro{cursor:default}
.h{position:absolute;top:0;bottom:0;width:7px;cursor:ew-resize}.h.l{left:0}.h.r{right:0}
#head{position:absolute;top:0;bottom:0;width:2px;background:var(--acc);z-index:4;pointer-events:none}
.mk{position:absolute;top:0;height:24px;border-left:2px solid #6cc4ff;font-size:10px;padding-left:3px;color:#6cc4ff}
.tick{position:absolute;top:6px;font-size:10px;color:var(--dim)}
#tc{font-variant-numeric:tabular-nums;min-width:86px}#status{color:var(--dim);max-width:420px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
</style></head><body>
<header><b>CINEGRAPH</b><span id="name"></span><span class="sp"></span>
<button id="undo" title="Ctrl+Z">↶</button><button id="redo" title="Ctrl+Shift+Z">↷</button>
<button id="split" title="S">Split</button><button id="play">▶ Play</button><span id="tc">00:00.00</span>
<button id="sc">Scopes</button><select id="q"><option value="0.25">Quarter</option><option value="0.5" selected>Half</option><option value="1">Full</option></select>
<span id="status"></span></header>
<div id="main"><div id="view"><img id="img" alt=""><img id="scimg" alt="" style="display:none;max-height:100%"></div>
<div id="props"><h4 id="ptitle">Select a clip</h4><textarea id="pj" spellcheck="false"></textarea>
<div id="pbar"><button id="apply">Apply (Ctrl+Enter)</button><button id="toggle">Enable/Disable</button><button id="del">Delete</button></div></div></div>
<div id="tl"><div id="ruler"></div><div id="rows"></div><div id="head"></div></div>
<script>
const C={video:'#58a6ff',image:'#7ee787',text:'#e3b341',shape:'#f778ba',solid:'#a5a5a5',gradient:'#bc8cff',comp:'#ff7b72',adjustment:'#d2a8ff',null:'#555',particles:'#79c0ff',model:'#ffa657',light:'#fff2a8',audio:'#56d4bc'};
let P=null,t=0,playing=false,busy=false,ver=0,pending=null,sel=null,drag=null;
const $=id=>document.getElementById(id);const LX=190;
const PX=()=>Math.max(40,($('tl').clientWidth-LX-10)/Math.max(1,P.duration));
function fmt(s){const m=Math.floor(s/60);return String(m).padStart(2,'0')+':'+(s%60).toFixed(2).padStart(5,'0')}
function status(s){$('status').textContent=s}
async function load(){const r=await fetch('/project');if(!r.ok){status(await r.text());return}P=await r.json();ver=P.version;
 $('name').textContent=P.name+' · '+P.width+'×'+P.height+' @ '+P.fps+'fps · '+P.duration.toFixed(2)+'s';draw();show(t);if(sel)select(sel,true)}
function draw(){const px=PX(),R=$('ruler'),rows=$('rows');R.innerHTML='';rows.innerHTML='';
 const step=px>120?1:px>50?2:px>20?5:10;for(let s=0;s<=P.duration;s+=step){const d=document.createElement('div');d.className='tick';d.style.left=(LX+s*px)+'px';d.textContent=s+'s';R.appendChild(d)}
 for(const m of P.markers){const d=document.createElement('div');d.className='mk';d.style.left=(LX+m.t*px)+'px';d.textContent=m.name;R.appendChild(d)}
 for(const L of [...P.layers].reverse()){const r=document.createElement('div');r.className='row';r.innerHTML='<div class="name" title="'+L.id+'">'+L.id+'</div><div class="lane"></div>';
  const b=document.createElement('div');b.className='bar'+(L.id===sel?' sel':'')+(L.enabled===false?' off':'')+(L.editable?'':' ro');
  b.style.left=(L.start*px)+'px';b.style.width=Math.max(3,L.duration*px)+'px';b.style.background=C[L.type]||'#888';b.textContent=L.label;b.dataset.id=L.id;
  b.title=L.type+'  '+L.start.toFixed(2)+'s → '+(L.start+L.duration).toFixed(2)+'s'+(L.editable?'':'  (edit via properties)');
  if(L.editable){for(const side of ['l','r']){const h=document.createElement('div');h.className='h '+side;h.dataset.side=side;b.appendChild(h)}}
  b.onmousedown=e=>startDrag(e,L,b);r.children[1].appendChild(b);rows.appendChild(r)}
 head()}
function head(){$('head').style.left=(LX+t*PX())+'px';$('tc').textContent=fmt(t)}
function snapPts(except){const pts=[t,0];for(const m of P.markers)pts.push(m.t);for(const L of P.layers)if(L.id!==except){pts.push(L.start,L.start+L.duration)}return pts}
function snap(v,except){const tol=7/PX();let best=v,bd=tol;for(const p of snapPts(except)){const d=Math.abs(p-v);if(d<bd){bd=d;best=p}}return best}
function startDrag(e,L,b){e.stopPropagation();select(L.id);if(!L.editable)return;const side=e.target.dataset.side||'move';
 drag={id:L.id,side,x0:e.clientX,s0:L.start,d0:L.duration,b,L};
 const mv=ev=>{const dt=(ev.clientX-drag.x0)/PX();let s=drag.s0,d=drag.d0;
  if(side==='move'){s=Math.max(0,snap(drag.s0+dt,L.id));const e2=snap(s+d,L.id);if(Math.abs(e2-(s+d))<Math.abs(s-(drag.s0+dt))+1e-9&&e2!==s+d)s=e2-d}
  else if(side==='l'){const ns=Math.min(drag.s0+drag.d0-1/P.fps,Math.max(0,snap(drag.s0+dt,L.id)));d=drag.d0+(drag.s0-ns);s=ns}
  else{d=Math.max(1/P.fps,snap(drag.s0+drag.d0+dt,L.id)-drag.s0)}
  drag.s=s;drag.d=d;b.style.left=(s*PX())+'px';b.style.width=Math.max(3,d*PX())+'px';status(side+'  '+s.toFixed(2)+' → '+(s+d).toFixed(2))};
 const up=async()=>{window.removeEventListener('mousemove',mv);window.removeEventListener('mouseup',up);const D=drag;drag=null;
  if(D.s===undefined)return;const op=D.side==='move'?{op:'move',id:D.id,start:D.s}:D.side==='l'?{op:'trim_start',id:D.id,start:D.s}:{op:'trim_end',id:D.id,duration:D.d};
  await edit(op)};
 window.addEventListener('mousemove',mv);window.addEventListener('mouseup',up)}
async function edit(op){const r=await fetch('/api/edit',{method:'POST',body:JSON.stringify(op)});const j=await r.json();status(j.ok?('✓ '+j.msg):('✗ '+j.error));await load()}
async function select(id,keep){sel=id;const r=await fetch('/api/layer?id='+encodeURIComponent(id));if(!r.ok){$('ptitle').textContent=id+' (read-only)';$('pj').value='';return}
 const j=await r.json();$('ptitle').textContent=id;if(!keep||document.activeElement!==$('pj'))$('pj').value=JSON.stringify(j,null,2);
 document.querySelectorAll('.bar').forEach(b=>b.classList.toggle('sel',b.dataset.id===id))}
$('apply').onclick=async()=>{if(!sel)return;let v;try{v=JSON.parse($('pj').value)}catch(e){status('JSON error: '+e.message);return}await edit({op:'replace',id:sel,layer:v})};
$('toggle').onclick=()=>sel&&edit({op:'toggle',id:sel});$('del').onclick=()=>sel&&confirm('Delete '+sel+'?')&&edit({op:'delete',id:sel});
$('split').onclick=()=>sel&&edit({op:'split',id:sel,at:t});
$('undo').onclick=()=>edit({op:'undo'});$('redo').onclick=()=>edit({op:'redo'});
async function show(tt){t=Math.max(0,Math.min(P.duration-1/P.fps,tt));head();if(busy){pending=t;return}busy=true;const q=$('q').value;const t0=performance.now();
 const r=await fetch('/frame?t='+t.toFixed(4)+'&s='+q+'&v='+ver);if(r.ok){const b=await r.blob();const u=URL.createObjectURL(b);$('img').onload=()=>URL.revokeObjectURL(u);$('img').src=u;
 if(!drag)status((performance.now()-t0).toFixed(0)+' ms/frame'+(r.headers.get('X-Cache')==='hit'?' (cached)':''))}else status(await r.text());
 busy=false;if(!playing)loadScopes();if(pending!==null){const p=pending;pending=null;show(p)}}
async function loop(){if(!playing)return;const a=performance.now();await show(t+1/P.fps);if(t>=P.duration-1/P.fps){playing=false;setPlay()}
 const el=(performance.now()-a)/1000;t+=Math.max(0,el-1/P.fps);setTimeout(loop,0)}
function setPlay(){$('play').textContent=playing?'❚❚ Pause':'▶ Play'}
$('play').onclick=()=>{playing=!playing;setPlay();loop()};$('q').onchange=()=>show(t);
let scopesOn=false;$('sc').onclick=()=>{scopesOn=!scopesOn;$('scimg').style.display=scopesOn?'block':'none';loadScopes()};
function loadScopes(){if(scopesOn)$('scimg').src='/scopes?t='+t.toFixed(3)+'&v='+ver+'&r='+Math.random()}
function seek(e){const r=$('tl').getBoundingClientRect();const x=e.clientX-r.left+$('tl').scrollLeft-LX;if(x>=0)show(x/PX())}
$('ruler').onmousedown=e=>{seek(e);const mv=ev=>seek(ev);window.addEventListener('mousemove',mv);window.addEventListener('mouseup',()=>window.removeEventListener('mousemove',mv),{once:true})};
$('rows').onmousedown=e=>{if(!e.target.classList.contains('bar')&&!e.target.classList.contains('h'))seek(e)};
window.onkeydown=e=>{const inEd=document.activeElement===$('pj');
 if(inEd){if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();$('apply').click()}return}
 if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();edit({op:e.shiftKey?'redo':'undo'});return}
 if(e.key===' '){e.preventDefault();$('play').click()}else if(e.key==='ArrowRight')show(t+(e.shiftKey?1:1/P.fps));else if(e.key==='ArrowLeft')show(t-(e.shiftKey?1:1/P.fps));
 else if(e.key==='Home')show(0);else if(e.key==='l'){playing=true;setPlay();loop()}else if(e.key==='k'){playing=false;setPlay()}else if(e.key==='j')show(t-1);
 else if(e.key==='s'&&sel)edit({op:'split',id:sel,at:t});else if(e.key==='e'&&sel)edit({op:'toggle',id:sel});
 else if(e.key==='Delete'&&sel&&confirm('Delete '+sel+'?'))edit({op:'delete',id:sel})};
window.onresize=()=>P&&draw();
setInterval(async()=>{try{const v=await (await fetch('/version')).json();if(v!==ver&&!drag)await load()}catch(e){}},1000);
load();
</script></body></html>"""


class State:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.lock = threading.RLock()
        self.mtime = None
        self.version = 0
        self.rts = {}
        self.norm = None
        self.raw = None
        self.error = None
        self.undo, self.redo = [], []
        self.frames: OrderedDict = OrderedDict()
        self._ensure_ids()
        self.reload()

    # ---------------- file io
    def _read(self):
        with open(self.path) as f:
            return json.load(f)

    def _write(self, data):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def _ensure_ids(self):
        """Give every top-level layer an id so UI edits can address it (one-time .bak backup)."""
        data = self._read()
        changed = False
        seen = set()
        for i, L in enumerate(data.get("layers", [])):
            if not L.get("id") or L["id"] in seen:
                base = L.get("type") or ("seq" if L.get("clips") else "layer")
                k = i
                while f"{base}{k}" in seen:
                    k += 1
                L["id"] = f"{base}{k}"
                changed = True
            seen.add(L["id"])
        if changed:
            if not os.path.exists(self.path + ".bak"):
                shutil.copy2(self.path, self.path + ".bak")
            self._write(data)

    def reload(self):
        m = os.path.getmtime(self.path)
        if m == self.mtime:
            return
        self.mtime = m
        try:
            self.raw = self._read()
            self.norm = PJ.load(self.path)
            self.rts = {}
            self.frames.clear()
            self.error = None
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        self.version += 1

    def rt(self, scale):
        if scale not in self.rts:
            self.rts[scale] = ProjectRuntime(self.norm, scale, proxies=scale <= 0.5)
        return self.rts[scale]

    # ---------------- editing
    def _find(self, data, lid):
        for i, L in enumerate(data.get("layers", [])):
            if L.get("id") == lid:
                return data["layers"], i
        return None, None

    def apply(self, op):
        with self.lock:
            kind = op.get("op")
            if kind in ("undo", "redo"):
                src, dst = (self.undo, self.redo) if kind == "undo" else (self.redo, self.undo)
                if not src:
                    return {"ok": False, "error": f"nothing to {kind}"}
                dst.append(self._read())
                self._write(src.pop())
                self.reload()
                return {"ok": True, "msg": kind}
            data = self._read()
            before = copy.deepcopy(data)
            fps = float(data.get("comp", {}).get("fps", 30))
            q = lambda s: round(math.floor(float(s) * fps + 0.5 + 1e-9) / fps, 6)  # frame-accurate edits (half-up)
            lst, i = self._find(data, op.get("id"))
            if lst is None:
                return {"ok": False, "error": f"layer '{op.get('id')}' is not a top-level layer (edit its parent)"}
            L = lst[i]
            norm_L = next((x for x in self.norm["comps"]["main"]["layers"] if x.get("id") == L["id"]), {})
            cur_start = float(norm_L.get("start", PJ.parse_time(L.get("start", 0), fps)))
            cur_dur = float(norm_L.get("duration", self.norm["comps"]["main"]["duration"] - cur_start))
            spd = float(L.get("speed", 1.0))
            if kind == "move":
                L["start"] = max(0.0, q(op["start"]))
                msg = f"moved {L['id']} to {L['start']:.2f}s"
            elif kind == "trim_start":
                ns = max(0.0, q(op["start"]))
                delta = ns - cur_start
                L["start"] = ns
                L["duration"] = q(max(1 / fps, cur_dur - delta))
                L.pop("end", None)
                if L.get("type") == "video" or "in" in L:
                    L["in"] = round(max(0.0, float(L.get("in", 0)) + delta * spd), 6)
                msg = f"trimmed head of {L['id']}"
            elif kind == "trim_end":
                L["duration"] = q(max(1 / fps, float(op["duration"])))
                L.pop("end", None)
                msg = f"trimmed tail of {L['id']} to {L['duration']:.2f}s"
            elif kind == "split":
                at = q(op["at"])
                if not (cur_start < at < cur_start + cur_dur):
                    return {"ok": False, "error": "playhead is not inside the clip"}
                B = copy.deepcopy(L)
                L["duration"] = q(at - cur_start)
                L.pop("end", None)
                L.pop("transition_out", None)
                B["start"] = at
                B["duration"] = q(cur_start + cur_dur - at)
                B.pop("end", None)
                B.pop("transition_in", None)
                if B.get("type") == "video" or "in" in B:
                    B["in"] = round(float(B.get("in", 0)) + (at - cur_start) * spd, 6)
                ids = {x.get("id") for x in lst}
                k = 2
                while f"{L['id']}_{k}" in ids:
                    k += 1
                B["id"] = f"{L['id']}_{k}"
                lst.insert(i + 1, B)
                msg = f"split {L['id']} at {at:.2f}s"
            elif kind == "toggle":
                L["enabled"] = L.get("enabled", True) is False
                msg = f"{L['id']} {'enabled' if L['enabled'] else 'disabled'}"
            elif kind == "delete":
                lst.pop(i)
                msg = f"deleted {L['id']}"
            elif kind == "replace":
                new = op["layer"]
                if not isinstance(new, dict):
                    return {"ok": False, "error": "layer must be a JSON object"}
                new.setdefault("id", L["id"])
                lst[i] = new
                msg = f"updated {new['id']}"
            else:
                return {"ok": False, "error": f"unknown op {kind}"}
            # validate before committing
            try:
                PJ.load(data, os.path.dirname(self.path))
                ProjectRuntime(PJ.load(data, os.path.dirname(self.path)), 0.1)
            except Exception as e:
                return {"ok": False, "error": f"rejected: {type(e).__name__}: {e}"}
            self.undo.append(before)
            self.redo.clear()
            if len(self.undo) > 200:
                self.undo.pop(0)
            self._write(data)
            self.reload()
            return {"ok": True, "msg": msg}


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if urlparse(self.path).path == "/api/edit":
                try:
                    res = state.apply(body)
                except Exception as e:
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                return self._send(200, json.dumps(res).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_GET(self):
            u = urlparse(self.path)
            with state.lock:
                state.reload()
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/version":
                return self._send(200, json.dumps(state.version).encode(), "application/json")
            if state.error:
                return self._send(500, state.error.encode(), "text/plain")
            comp = state.norm["comps"]["main"]
            q = parse_qs(u.query)
            if u.path == "/api/layer":
                lid = q.get("id", [""])[0]
                for L in state.raw.get("layers", []):
                    if L.get("id") == lid:
                        return self._send(200, json.dumps(L).encode(), "application/json")
                return self._send(404, b"not top-level", "text/plain")
            if u.path == "/project":
                top = {L.get("id") for L in state.raw.get("layers", [])}
                layers = []
                for L in comp["layers"]:
                    dur = L.get("duration", comp["duration"] - L["start"])
                    label = L.get("text") or os.path.basename(str(L.get("src", ""))) or L.get("primitive") or L["type"]
                    layers.append({"id": L.get("id", "?"), "type": L["type"], "start": L["start"], "duration": dur,
                                   "label": str(label)[:40], "enabled": L.get("enabled", True), "editable": L.get("id") in top})
                for a in state.norm["audio"]:
                    layers.insert(0, {"id": "♪ " + str(a.get("id") or os.path.basename(a["src"])), "type": "audio",
                                      "start": a["start"], "duration": a.get("duration", comp["duration"] - a["start"]),
                                      "label": os.path.basename(a["src"]), "editable": False})
                body = {"name": os.path.basename(state.path), "width": comp["width"], "height": comp["height"],
                        "fps": comp["fps"], "duration": comp["duration"], "layers": layers,
                        "markers": [m for m in state.norm.get("markers", []) if not m.get("beat")], "version": state.version}
                return self._send(200, json.dumps(body).encode(), "application/json")
            if u.path == "/scopes":
                from .color import scopes
                with state.lock:
                    fr = state.rt(0.5).main.render(float(q.get("t", ["0"])[0]))
                ok, buf = cv2.imencode(".png", cv2.cvtColor(scopes(to_uint8(fr, False, dither=False), (384, 192)), cv2.COLOR_RGB2BGR))
                return self._send(200, buf.tobytes(), "image/png")
            if u.path == "/frame":
                t = float(q.get("t", ["0"])[0])
                s = float(q.get("s", ["0.5"])[0])
                fi = int(round(t * comp["fps"]))
                key = (state.version, fi, s)
                if key in state.frames:
                    state.frames.move_to_end(key)
                    return self._send(200, state.frames[key], "image/jpeg", {"X-Cache": "hit"})
                try:
                    with state.lock:
                        fr = state.rt(s).main.render(fi / comp["fps"])
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(to_uint8(fr, False, dither=False), cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 88])
                    data = buf.tobytes()
                    state.frames[key] = data
                    while len(state.frames) > 400:
                        state.frames.popitem(last=False)
                    return self._send(200, data, "image/jpeg")
                except Exception as e:
                    return self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")
            self._send(404, b"not found", "text/plain")
    return H


def serve(path: str, port: int = 8765, host: str = "127.0.0.1"):
    st = State(path)
    srv = ThreadingHTTPServer((host, port), make_handler(st))
    print(f"Cinegraph editor: http://{host}:{port}  (editing {st.path}; Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
