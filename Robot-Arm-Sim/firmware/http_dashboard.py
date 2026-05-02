"""
On-device HTTP dashboard for Robot Arm Pico firmware.

Kept separate from ``pico_control_script.py`` so UI/routes change without touching
the motion loop.  The main script calls ``configure()`` once, then ``bind_listen`` /
``serve_client`` / ``build_html`` from the LAN worker thread.

Motor APIs and trajectory logic stay in the main module; this file only:
  - serves HTML/CSS/JS,
  - reads angles/switches for JSON,
  - enqueues jog / LED commands into ``web_cmd_queue`` (drained on the main loop).
"""

import errno
import json

_CTX = {}


def configure(
    *,
    get_axes,
    web_cmd_queue,
    get_tcp_port,
    get_http_port,
    angles_deg_bundle,
    fk_positions_deg,
    axis_angle_deg,
):
    """Wire callbacks from ``pico_control_script`` (call from ``main()`` before LAN thread)."""
    _CTX.clear()
    _CTX.update(
        {
            "get_axes": get_axes,
            "web_cmd_queue": web_cmd_queue,
            "get_tcp_port": get_tcp_port,
            "get_http_port": get_http_port,
            "angles_deg_bundle": angles_deg_bundle,
            "fk_positions_deg": fk_positions_deg,
            "axis_angle_deg": axis_angle_deg,
        }
    )


def _q_append_motion(op):
    q = _CTX["web_cmd_queue"]
    q.append(op)
    while len(q) > 64:
        q.pop(0)


def _q_append_led():
    q = _CTX["web_cmd_queue"]
    q.append(("__LED__",))
    while len(q) > 64:
        q.pop(0)


def build_html():
    """Static page for ``GET /`` (ports interpolated from ``configure``)."""
    tcp = _CTX["get_tcp_port"]()
    http = _CTX["get_http_port"]()
    return (
        """<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Arm dashboard</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:12px;}
h1{font-size:1.1rem;margin:0 0 8px;}
.row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center;}
.dot{width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle;}
.dot.on{background:#0f0;box-shadow:0 0 6px #0f0;}.dot.off{background:#644;}.dot.na{background:#333;}
svg{background:#1a1a1a;border:1px solid #333;display:block;max-width:100%;}button{margin:2px;padding:6px 10px;cursor:pointer;touch-action:manipulation;-webkit-user-select:none;user-select:none;}
.ee{font-family:monospace;font-size:14px;}small{color:#888;}
.hdr{margin:10px 0 4px;color:#aaa;font-size:11px;}
#err{color:#f66;font-family:monospace;font-size:12px;min-height:1.2em;}
#tools pre{background:#1a1a1a;border:1px solid #444;padding:8px;max-height:180px;overflow:auto;font-size:11px;margin:6px 0;}
</style></head><body>
<h1>Pico arm dashboard</h1>
<p><small>TCP """
        + str(tcp)
        + """ · HTTP """
        + str(http)
        + """</small></p>
<p id="err"></p>
<div class="row" id="sw"></div>
<svg id="sv" width="320" height="240" viewBox="-110 -110 220 220"><path id="armPath" stroke="#6cf" stroke-width="3" fill="none" d=""/></svg>
<p class="ee" id="ee"></p>
<p class="hdr">Troubleshooting</p>
<div class="row">
<button type="button" id="btnLed">Toggle onboard LED</button>
<button type="button" id="btnHealth">GET /api/health</button>
<button type="button" id="btnInfo">GET /api/info</button>
</div>
<div id="tools"><pre id="toolsOut">—</pre></div>
<p class="hdr">End-effector jog — world XYZ (mm, hold)</p>
<div class="row">
<button type="button" id="exm">EE −X</button><button type="button" id="exp">EE +X</button>
<button type="button" id="eym">EE −Y</button><button type="button" id="eyp">EE +Y</button>
<button type="button" id="ezm">EE −Z</button><button type="button" id="ezp">EE +Z</button>
</div>
<p class="hdr">Joint jog — degrees (hold)</p>
<div class="row" id="jb"></div>
<script>
function armLine(points){if(!points||points.length<2)return '';
let mx=0,my=0,Mx=0,My=0;for(const p of points){mx=Math.min(mx,p[0]);Mx=Math.max(Mx,p[0]);my=Math.min(my,p[1]);My=Math.max(My,p[1]);}
const rw=Math.max(Mx-mx,1e-6),rh=Math.max(My-my,1e-6),sc=Math.min(180/rw,180/rh)*0.85,cx=(mx+Mx)/2,cy=(my+My)/2;
let d='';for(let i=0;i<points.length;i++){const x=(points[i][0]-cx)*sc,y=-(points[i][1]-cy)*sc;d+=(i?'L':'M')+x+','+y;}
return d;}
var jbBuiltSig='',swBuiltSig='',holdIv=null,holdUrl=null;
var stateBusy=false,statePending=false;
function setErr(msg){var e=document.getElementById('err');if(e)e.textContent=msg||'';}
function toolOut(t){var e=document.getElementById('toolsOut');if(e)e.textContent=t;}
function safeSwitches(j){return Array.isArray(j&&j.switches)?j.switches:[];}
function safeJoints(j){return Array.isArray(j&&j.joints)?j.joints:[];}
function rebuildSwitches(swArr){var el=document.getElementById('sw'),h='';for(var i=0;i<swArr.length;i++){h+='<span><span class="dot na"></span>J'+swArr[i].idx+' limit</span> ';}el.innerHTML=h.trimEnd();}
function updateSwitchDots(swArr){var el=document.getElementById('sw');for(var i=0;i<swArr.length;i++){var span=el.children[i];if(!span)return false;var dot=span.querySelector('.dot');if(!dot)return false;var s=swArr[i];dot.className='dot '+(s.pressed?'on':(s.wired?'off':'na'));}return true;}
function rebuildJointButtons(joints){var el=document.getElementById('jb');el.innerHTML='';for(var i=0;i<joints.length;i++){var jo=joints[i];var bm=document.createElement('button');bm.type='button';bm.textContent='J'+jo.idx+' −';bindHold(bm,'/api/jog_joint?j='+jo.idx+'&delta=-2');el.appendChild(bm);var bp=document.createElement('button');bp.type='button';bp.textContent='J'+jo.idx+' +';bindHold(bp,'/api/jog_joint?j='+jo.idx+'&delta=2');el.appendChild(bp);el.appendChild(document.createTextNode(' '));}}
async function refreshState(){
if(stateBusy){statePending=true;return;}
stateBusy=true;
try{
var r=await fetch('/api/state',{cache:'no-store'});
var txt=await r.text();
var j=null;
try{j=JSON.parse(txt);}catch(pe){j=null;}
if(!r.ok||!j||j.ok===false){
setErr('state: HTTP '+r.status+(j&&j.error?(' '+j.error):(' '+txt.slice(0,120))));
stateBusy=false;
if(statePending){statePending=false;refreshState();}
return;
}
setErr('');
var ee=j.ee||{x:0,y:0,z:0};
document.getElementById('ee').textContent='EE X='+ee.x+' Y='+ee.y+' Z='+ee.z+((j.fk_ok!==false)?'':' (FK off)');
var swArr=safeSwitches(j);
var swSig=swArr.map(function(s){return s.idx;}).join('|');
if(swSig!==swBuiltSig){rebuildSwitches(swArr);swBuiltSig=swSig;}else if(swArr.length&&!updateSwitchDots(swArr)){rebuildSwitches(swArr);}
var joints=safeJoints(j);
var jbSig=joints.map(function(x){return x.idx;}).join('|');
if(jbSig!==jbBuiltSig){rebuildJointButtons(joints);jbBuiltSig=jbSig;}
var pts=j.polyline_xy||[];
document.getElementById('armPath').setAttribute('d',pts.length?armLine(pts):'');
}catch(e){setErr(String(e));console.warn(e);}
finally{
stateBusy=false;
if(statePending){statePending=false;refreshState();}
}
}
async function pollLoop(){
while(true){
try{await refreshState();await new Promise(function(r){setTimeout(r,50);});}
catch(e){setErr(String(e));await new Promise(function(r){setTimeout(r,120);});}
}
}
pollLoop();
function stopHold(){if(holdIv){clearInterval(holdIv);holdIv=null;}holdUrl=null;}
function fireHold(){
if(!holdUrl)return;
fetch(holdUrl,{cache:'no-store'}).then(function(){refreshState();}).catch(function(){});
}
function startHold(url){stopHold();holdUrl=url;fireHold();holdIv=setInterval(fireHold,60);}
function bindHold(el,url){
el.addEventListener('pointerdown',function(e){if(e.pointerType==='mouse'&&e.button!==0)return;e.preventDefault();try{el.setPointerCapture(e.pointerId);}catch(x){}startHold(url);});
el.addEventListener('pointerup',stopHold);el.addEventListener('pointercancel',stopHold);el.addEventListener('lostpointercapture',stopHold);}
bindHold(document.getElementById('exm'),'/api/jog_ee?dx=-1.25');
bindHold(document.getElementById('exp'),'/api/jog_ee?dx=1.25');
bindHold(document.getElementById('eym'),'/api/jog_ee?dy=-1.25');
bindHold(document.getElementById('eyp'),'/api/jog_ee?dy=1.25');
bindHold(document.getElementById('ezm'),'/api/jog_ee?dz=-1.25');
bindHold(document.getElementById('ezp'),'/api/jog_ee?dz=1.25');
window.addEventListener('blur',stopHold);
window.addEventListener('pointerup',stopHold);
async function fetchJson(path,opts){
var r=await fetch(path,opts||{cache:'no-store'});
var t=await r.text();
try{return {ok:r.ok,status:r.status,json:JSON.parse(t),raw:t};}catch(x){return {ok:r.ok,status:r.status,json:null,raw:t};}
}
document.getElementById('btnLed').addEventListener('click',function(){
fetch('/api/led_toggle',{method:'POST',cache:'no-store'}).then(function(){toolOut('LED toggle queued');}).catch(function(e){toolOut(String(e));});
});
document.getElementById('btnHealth').addEventListener('click',async function(){
var x=await fetchJson('/api/health');
toolOut(JSON.stringify(x.json||x.raw,null,2));
});
document.getElementById('btnInfo').addEventListener('click',async function(){
var x=await fetchJson('/api/info');
toolOut(JSON.stringify(x.json||x.raw,null,2));
});
</script>
</body></html>"""
    )


def bind_listen(sta_ip):
    """Bind TCP listen socket for HTTP. ``sta_ip`` is STA/AP IPv4 from ``ifconfig()``."""
    import socket

    port = int(_CTX["get_http_port"]())
    candidates = []
    if sta_ip and sta_ip != "0.0.0.0":
        candidates.append((sta_ip, port))
    candidates.extend([("0.0.0.0", port), ("", port)])

    srv = None
    bound_addr = None
    last_be = None
    for addr in candidates:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(addr)
            srv = sock
            sock = None
            bound_addr = addr
            break
        except OSError as be:
            last_be = be
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    if srv is None:
        raise OSError("HTTP bind failed: {}".format(last_be))

    srv.listen(4)
    show_ip = bound_addr[0] if bound_addr and bound_addr[0] else "0.0.0.0"
    return srv, show_ip


def _parse_qs(qs):
    pr = {}
    for pair in (qs or "").split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            pr[k] = v
    return pr


def _json_state():
    axes = _CTX["get_axes"]()
    adg = _CTX["angles_deg_bundle"]
    fk = _CTX["fk_positions_deg"]
    adeg = _CTX["axis_angle_deg"]
    bd, pd = adg(axes) if axes else (0.0, [])
    pts3 = fk(bd, pd) if axes else []
    fk_ok = len(pts3) > 0
    poly_xy = [[round(p[0], 4), round(p[1], 4)] for p in pts3] if fk_ok else []
    joints_out = []
    sw_out = []
    for ax in sorted(axes or [], key=lambda a: a.idx):
        live = False
        if getattr(ax, "_home_pin", None) is not None:
            try:
                pv = ax._home_pin.value()
                live = (pv == 1) if ax.home_pin_polarity == "NC" else (pv == 0)
            except Exception:
                live = False
        joints_out.append({"idx": ax.idx, "deg": round(adeg(ax), 3)})
        sw_out.append(
            {"idx": ax.idx, "pressed": live, "wired": getattr(ax, "_home_pin", None) is not None}
        )
    ee = pts3[-1] if fk_ok else (0.0, 0.0, 0.0)
    return {
        "ok": True,
        "fk_ok": fk_ok,
        "joints": joints_out,
        "switches": sw_out,
        "ee": {"x": round(ee[0], 4), "y": round(ee[1], 4), "z": round(ee[2], 4)},
        "polyline_xy": poly_xy,
        "tcp_control_port": int(_CTX["get_tcp_port"]()),
        "http_port": int(_CTX["get_http_port"]()),
    }


def _api_info_doc():
    doc = {
        "ok": True,
        "tcp_control_port": int(_CTX["get_tcp_port"]()),
        "http_port": int(_CTX["get_http_port"]()),
    }
    try:
        import gc

        doc["mem_free"] = gc.mem_free()
        _ma = getattr(gc, "mem_alloc", None)
        if _ma is not None:
            doc["mem_alloc"] = _ma()
    except Exception:
        pass
    try:
        import sys

        doc["mp_version"] = getattr(sys, "version", "")
    except Exception:
        pass
    try:
        import network as n

        for mode, name in ((n.STA_IF, "sta"), (n.AP_IF, "ap")):
            try:
                w = n.WLAN(mode)
                doc[name] = {
                    "active": w.active(),
                    "connected": w.isconnected(),
                    "ifconfig": list(w.ifconfig()) if (w.active() and w.isconnected()) else None,
                }
            except Exception:
                doc[name] = {"error": "n/a"}
    except Exception as ex:
        doc["wlan"] = str(ex)
    return doc


def serve_client(conn, html):
    """Handle one HTTP connection (request → response, then close)."""
    try:
        conn.settimeout(3.0)
        req = b""
        while b"\r\n\r\n" not in req and len(req) < 4096:
            req += conn.recv(512)
        if not req:
            return
        head = req.split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace")
        lines = head.split("\r\n")
        req_line = lines[0] if lines else ""
        parts = req_line.split()
        method = parts[0].upper() if len(parts) > 0 else "GET"
        path_full = parts[1] if len(parts) > 1 else "/"
        path = path_full
        qs = ""
        if "?" in path_full:
            path, qs = path_full.split("?", 1)

        if method == "OPTIONS":
            hdr = (
                "HTTP/1.0 204 \r\nAllow: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Connection: close\r\nContent-Length: 0\r\n\r\n"
            )
            conn.send(hdr.encode("utf-8"))
            return

        status = "200 OK"
        body = ""
        ctype = "text/plain; charset=utf-8"

        if path in ("/", "/index.html"):
            body = html
            ctype = "text/html; charset=utf-8"
        elif path == "/api/health":
            body = json.dumps({"ok": True})
            ctype = "application/json"
        elif path == "/api/info":
            body = json.dumps(_api_info_doc())
            ctype = "application/json"
        elif path == "/api/state":
            try:
                body = json.dumps(_json_state())
            except Exception as ex:
                body = json.dumps({"ok": False, "error": str(ex)})
            ctype = "application/json"
        elif path == "/api/led_toggle" and method in ("GET", "POST"):
            try:
                _q_append_led()
                body = json.dumps({"ok": True})
            except Exception:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        elif path == "/api/jog_joint" and method == "GET":
            pr = _parse_qs(qs)
            try:
                jn = int(pr.get("j", "-1"))
                delta = float(pr.get("delta", "0"))
                _q_append_motion(("JJ", jn, delta))
                body = json.dumps({"ok": True})
            except Exception:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        elif path == "/api/jog_ee" and method == "GET":
            pr = _parse_qs(qs)
            try:
                dx = float(pr.get("dx", "0"))
                dy = float(pr.get("dy", "0"))
                dz = float(pr.get("dz", "0"))
                _q_append_motion(("JE", dx, dy, dz))
                body = json.dumps({"ok": True})
            except Exception:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        else:
            status = "404 Not Found"
            body = "Not found"

        raw_b = body.encode("utf-8")
        hdr = (
            "HTTP/1.0 "
            + status
            + "\r\nContent-Type: "
            + ctype
            + "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: "
            + str(len(raw_b))
            + "\r\n\r\n"
        )
        conn.send(hdr.encode("utf-8") + raw_b)
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


def accept_nonblocking(sock):
    """``accept()`` or ``None`` if no pending connection (EAGAIN / EWOULDBLOCK)."""
    try:
        return sock.accept()
    except OSError as e:
        _en = getattr(e, "errno", None)
        if _en is None and e.args:
            try:
                _en = int(e.args[0])
            except (TypeError, ValueError):
                _en = None
        if _en == errno.EAGAIN or _en == getattr(errno, "EWOULDBLOCK", -999) or _en in (11,):
            return None
        raise
