"""
Minimal HTTP dashboard — jog + status.
"""

import errno
import json

_CTX = {}


def configure(**kwargs):
    """Store callbacks from main script."""
    _CTX.clear()
    _CTX.update(kwargs)


def build_html():
    """Return complete dashboard HTML."""
    tcp = _CTX.get("get_tcp_port", lambda: 8888)()
    http = _CTX.get("get_http_port", lambda: 8080)()
    return """<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Arm</title><style>
body{font-family:system-ui;background:#111;color:#eee;margin:0;padding:12px;}
h1{font-size:1.1rem;margin:0 0 12px;}
.row{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center;}
button{padding:8px 12px;cursor:pointer;background:#007acc;color:white;border:none;border-radius:2px;}
button:hover{background:#005f99;}
button:active{background:#004578;}
#status{font-family:monospace;font-size:12px;color:#aaa;min-height:1.2em;margin:10px 0;}
#output{background:#1a1a1a;border:1px solid #444;padding:8px;margin:10px 0;max-height:200px;overflow:auto;font-family:monospace;font-size:11px;white-space:pre-wrap;word-break:break-all;}
.hdr{margin-top:16px;margin-bottom:6px;color:#aaa;font-size:11px;font-weight:bold;}
svg{border:1px solid #333;max-width:100%;background:#1a1a1a;}
</style></head><body>
<h1>Pico Arm Dashboard</h1>
<p>TCP:""" + str(tcp) + """ HTTP:""" + str(http) + """</p>
<div class="hdr">Test Connection</div>
<div class="row">
<button id="health">GET /api/health</button>
<button id="state">GET /api/state</button>
<button id="toggle">POST /api/led_toggle</button>
</div>
<div id="output">—</div>
<div class="hdr">Joint Jog (degrees)</div>
<div class="row" id="joints"></div>
<div id="status">Ready</div>
<div class="hdr">End-Effector Jog (mm)</div>
<div class="row">
<button data-jog="ee" data-axis="x" data-delta="-5">EE -X</button>
<button data-jog="ee" data-axis="x" data-delta="5">EE +X</button>
<button data-jog="ee" data-axis="y" data-delta="-5">EE -Y</button>
<button data-jog="ee" data-axis="y" data-delta="5">EE +Y</button>
<button data-jog="ee" data-axis="z" data-delta="-5">EE -Z</button>
<button data-jog="ee" data-axis="z" data-delta="5">EE +Z</button>
</div>
<svg id="arm" width="300" height="300" viewBox="-150 -150 300 300"><circle cx="0" cy="0" r="4" fill="#0f0"/><path id="armpath" stroke="#6cf" stroke-width="2" fill="none" d=""/></svg>
<script>
var out=document.getElementById('output'),status=document.getElementById('status'),lastJointsSig='';
function log(s){out.textContent=s;}
async function api(path,method){try{var r=await fetch(path,{method:method||'GET',cache:'no-store'});var t=await r.text();return {ok:r.ok,status:r.status,text:t,json:null};}catch(e){return {ok:false,status:0,text:String(e),json:null};}}
function tryJSON(t){try{return JSON.parse(t);}catch(e){return null;}}
async function doJog(path){var x=await api(path);status.textContent=(x.ok?'OK':'FAIL')+' '+x.text.slice(0,40);}
function attachHold(btn,urlFn){
var hold=null;
btn.onpointerdown=function(){doJog(urlFn());hold=setInterval(function(){doJog(urlFn());},100);};
btn.onpointerup=btn.onpointercancel=function(){if(hold){clearInterval(hold);hold=null;}};
}
document.getElementById('health').onclick=async function(){var x=await api('/api/health');log(JSON.stringify(tryJSON(x.text)||{error:x.text},null,2));};
document.getElementById('state').onclick=async function(){var x=await api('/api/state');log(x.text.slice(0,500));};
document.getElementById('toggle').onclick=async function(){var x=await api('/api/led_toggle','POST');log('LED: '+(x.ok?x.text:'FAIL '+x.status));};
document.querySelectorAll('button[data-jog="ee"]').forEach(function(b){
attachHold(b,function(){return '/api/jog_ee?d'+b.getAttribute('data-axis')+'='+b.getAttribute('data-delta');});
});
async function poll(){
try{
var x=await api('/api/state');var j=tryJSON(x.text);
if(j&&j.joints){
var sig=j.joints.map(function(a){return a.idx;}).join(',');
if(sig!==lastJointsSig){
var el=document.getElementById('joints');el.innerHTML='';
j.joints.forEach(function(a){
var idx=a.idx;
var wrap=document.createElement('div');wrap.style.display='inline-flex';wrap.style.alignItems='center';wrap.style.gap='4px';wrap.style.margin='2px';
var bm=document.createElement('button');bm.textContent='J'+idx+' -';
attachHold(bm,function(){return '/api/jog_joint?j='+idx+'&delta=-5';});
var lbl=document.createElement('span');lbl.id='jdeg'+idx;lbl.style.minWidth='55px';lbl.style.textAlign='center';lbl.style.fontFamily='monospace';lbl.style.fontSize='11px';lbl.textContent=a.deg.toFixed(1)+'°';
var bp=document.createElement('button');bp.textContent='J'+idx+' +';
attachHold(bp,function(){return '/api/jog_joint?j='+idx+'&delta=5';});
wrap.appendChild(bm);wrap.appendChild(lbl);wrap.appendChild(bp);el.appendChild(wrap);
});
lastJointsSig=sig;
}else{
j.joints.forEach(function(a){var l=document.getElementById('jdeg'+a.idx);if(l)l.textContent=a.deg.toFixed(1)+'°';});
}
}
}catch(e){}
setTimeout(poll,300);
}
poll();
</script>
</body></html>"""


def bind_listen(sta_ip):
    """Create and bind HTTP socket."""
    import socket
    port = int(_CTX.get("get_http_port", lambda: 8080)())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((sta_ip, port))
    except:
        sock.bind(("0.0.0.0", port))
    sock.listen(4)
    return sock, sta_ip or "0.0.0.0"


def serve_client(conn, html):
    """Handle one HTTP request."""
    try:
        conn.settimeout(2.0)
        req = b""
        while b"\r\n\r\n" not in req and len(req) < 4096:
            req += conn.recv(512)
        if not req:
            return

        head = req.split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace")
        parts = head.split("\r\n")[0].split()
        method = parts[0].upper() if parts else "GET"
        path = parts[1] if len(parts) > 1 else "/"
        qs = ""
        if "?" in path:
            path, qs = path.split("?", 1)

        status = "200 OK"
        body = ""
        ctype = "text/html; charset=utf-8"

        if path == "/" or path == "/index.html":
            body = html
        elif path == "/api/health":
            body = json.dumps({"ok": True})
            ctype = "application/json"
        elif path == "/api/state":
            try:
                axes = _CTX.get("get_axes", lambda: [])()
                angle_fn = _CTX.get("axis_angle_deg")
                joints = []
                for a in (axes or []):
                    try:
                        deg = float(angle_fn(a)) if angle_fn is not None else 0.0
                    except Exception:
                        deg = 0.0
                    joints.append({"idx": getattr(a, "idx", 0), "deg": round(deg, 2)})
                body = json.dumps({"ok": True, "joints": joints})
            except:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        elif path == "/api/led_toggle" and method in ("GET", "POST"):
            try:
                q = _CTX.get("web_cmd_queue", [])
                q.append(("__LED__",))
                body = json.dumps({"ok": True})
            except:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        elif path == "/api/jog_joint" and method == "GET":
            try:
                pr = {}
                for pair in (qs or "").split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        pr[k] = v
                j = int(pr.get("j", "-1"))
                d = float(pr.get("delta", "0"))
                q = _CTX.get("web_cmd_queue", [])
                q.append(("JJ", j, d))
                body = json.dumps({"ok": True})
            except:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        elif path == "/api/jog_ee" and method == "GET":
            try:
                pr = {}
                for pair in (qs or "").split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        pr[k] = v
                dx = float(pr.get("dx", "0"))
                dy = float(pr.get("dy", "0"))
                dz = float(pr.get("dz", "0"))
                q = _CTX.get("web_cmd_queue", [])
                q.append(("JE", dx, dy, dz))
                body = json.dumps({"ok": True})
            except:
                body = json.dumps({"ok": False})
            ctype = "application/json"
        else:
            status = "404 Not Found"
            body = "Not found"

        raw_b = body.encode("utf-8")
        hdr = f"HTTP/1.0 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(raw_b)}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        conn.send(hdr.encode("utf-8") + raw_b)
    except:
        pass
    finally:
        try:
            conn.close()
        except:
            pass
