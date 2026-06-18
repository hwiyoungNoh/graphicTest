#!/usr/bin/env python
"""
DaVinci Resolve - Remote control server (Free version compatible)

Runs INSIDE Resolve, where the global `resolve` object is auto-injected.
Deploy on the Mac under:
  ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/
Run via:
  Workspace -> Scripts -> Utility -> remote_player

Redeploy after editing this file:
  Copy it to the Utility folder, then RE-RUN it from the Scripts menu. Replacing the file
  alone does NOT reload a running server. The new run asks the old instance to /shutdown and
  rebinds the port automatically. Verify the live build via GET / -> {"version": ...}.

HTTP API (listens on 0.0.0.0:7777, JSON):
  GET  /                health/ping (also returns {"version": ...})
  GET  /info            project + current timeline summary
  GET  /timelines       list all timelines
  GET  /timeline        current timeline detail (tracks + items)
  GET  /media           media pool clips in current folder
  POST /timeline  {"name"} | {"index"}   switch current timeline
  POST /jump      {"index","track"?,"offset"?}  playhead -> INTO video item #index (default +1s)
  POST /add       {"path"}               import a Mac-local file -> append to timeline -> jump
  POST /upload?name=<f>&replace=<0|1>    body=raw bytes: save to PATTERN_DIR -> import -> append -> jump
                                         (replace=1 -> ReplaceClip the last uploaded clip in place)
  POST /goto      {"timecode"}           move playhead -> shows that FRAME in viewer
  POST /page      {"page"}               switch page (edit/color/cut/...)
  POST /signal    {"signal"}             legacy: switch to SIG_<signal> preset timeline
  POST /play                             transport: play forward  (osascript 'L')
  POST /pause                            transport: pause/stop    (osascript 'K')
  POST /reverse                          transport: play reverse  (osascript 'J')
  POST /toggle                           transport: play/pause    (osascript space)
  POST /key       {"key"}                raw key -> Resolve (space,j,k,l,home,end,left,right,up,down)
  POST /shutdown                         stop server, free the port (used by self-replace)

PLAYHEAD vs PLAYBACK:
  - /goto (SetCurrentTimecode) moves the playhead; the viewer/output shows that
    frame automatically (still). No key needed just to DISPLAY media.
  - It does NOT start playback. For motion, use transport. Tip: if Resolve is
    already playing, /goto and /timeline switches keep rolling from the new spot
    -> trigger /play once, then drive everything else via the API.

TRANSPORT NOTE: the Resolve scripting API has NO play/stop methods. Real transport
here is macOS GUI automation (osascript) and requires:
  - macOS + Accessibility permission for DaVinci Resolve
    (System Settings > Privacy & Security > Accessibility)
  - Resolve frontmost (server calls `activate` before each key); no text field focused
"""

import json
import os
import platform
import subprocess
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

# `resolve` is auto-injected by Resolve when run from the Scripts menu.
# Re-bind from globals so static linters see it defined, and fail clearly if run standalone.
resolve = globals().get("resolve")
if resolve is None:
    raise SystemExit(
        "Run this from inside DaVinci Resolve: Workspace > Scripts > Utility > remote_player"
    )
pm = resolve.GetProjectManager()

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 7777
TIMELINE_START_TC = "01:00:00:00"
PATTERN_DIR = os.environ.get(  # uploaded pattern images land here (override via env)
    "RESOLVE_PATTERN_DIR", os.path.expanduser("~/Pictures/pattern_new")
)
VERSION = "0.5-portable"  # bump on API change; shown in GET / to verify the live deploy
JUMP_OFFSET_SEC = 1.0  # land playhead this many seconds INTO a clip (avoid black start edge)

# Optional preset for legacy POST /signal. Use REAL macOS paths if you use it.
SIGNAL_MAP = {
    # "A": "/Users/me/videos/clipA.mov",
    # "B": "/Users/me/videos/clipB.mov",
}
_signal_timelines = {}  # signal -> Timeline, built lazily on first use
_server = None  # current HTTPServer instance (for self-replace / shutdown)
_last_pattern_item = None  # last uploaded pattern MediaPoolItem (for /upload?replace=1)


# ----------------------------------------------------------------------------
# Resolve API helpers (fetched live so project/timeline changes are picked up)
# ----------------------------------------------------------------------------
def _project():
    return pm.GetCurrentProject()


def _mediapool():
    p = _project()
    return p.GetMediaPool() if p else None


def _clip_props(clip):
    props = clip.GetClipProperty() or {}  # full property dict (no-arg form)
    return {
        "name": clip.GetName(),
        "path": props.get("File Path"),
        "duration": props.get("Duration"),
        "resolution": props.get("Resolution"),
        "fps": props.get("FPS"),
        "format": props.get("Format"),
    }


def q_info():
    p = _project()
    if not p:
        return {"project": None, "page": resolve.GetCurrentPage(), "msg": "no project open"}
    tl = p.GetCurrentTimeline()
    return {
        "project": p.GetName(),
        "page": resolve.GetCurrentPage(),
        "timelineCount": p.GetTimelineCount(),
        "currentTimeline": tl.GetName() if tl else None,
        "currentTimecode": tl.GetCurrentTimecode() if tl else None,
    }


def q_timelines():
    p = _project()
    if not p:
        return []
    out = []
    for i in range(1, p.GetTimelineCount() + 1):
        tl = p.GetTimelineByIndex(i)
        out.append({"index": i, "name": tl.GetName() if tl else None})
    return out


def q_timeline():
    p = _project()
    tl = p.GetCurrentTimeline() if p else None
    if not tl:
        return {"error": "no current timeline"}
    items = {}
    vcount = tl.GetTrackCount("video")
    for ti in range(1, vcount + 1):
        track = tl.GetItemListInTrack("video", ti) or []
        items[f"video{ti}"] = [
            {
                "name": it.GetName(),
                "start": it.GetStart(),
                "end": it.GetEnd(),
                "duration": it.GetDuration(),
            }
            for it in track
        ]
    return {
        "name": tl.GetName(),
        "timecode": tl.GetCurrentTimecode(),
        "videoTracks": vcount,
        "audioTracks": tl.GetTrackCount("audio"),
        "items": items,
    }


def q_media():
    mp = _mediapool()
    folder = mp.GetCurrentFolder() if mp else None
    clips = folder.GetClipList() if folder else []
    return {
        "folder": folder.GetName() if folder else None,
        "clips": [_clip_props(c) for c in clips],
    }


# ----------------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------------
def a_switch_timeline(body):
    p = _project()
    if not p:
        return False, "no project open", 400
    target = None
    if body.get("index"):
        target = p.GetTimelineByIndex(int(body["index"]))
    elif body.get("name"):
        for i in range(1, p.GetTimelineCount() + 1):
            tl = p.GetTimelineByIndex(i)
            if tl and tl.GetName() == body["name"]:
                target = tl
                break
    if not target:
        return False, "timeline not found", 404
    ok = bool(p.SetCurrentTimeline(target))
    return ok, f"switched to {target.GetName()}", 200 if ok else 400


def a_goto(body):
    p = _project()
    tl = p.GetCurrentTimeline() if p else None
    if not tl:
        return False, "no current timeline", 400
    tc = body.get("timecode", TIMELINE_START_TC)
    ok = bool(tl.SetCurrentTimecode(tc))
    return ok, f"goto {tc}", 200 if ok else 400


def a_page(body):
    ok = bool(resolve.OpenPage(body.get("page", "")))
    return ok, f"page {body.get('page')}", 200 if ok else 400


def a_signal(sig):
    if sig not in _signal_timelines and sig in SIGNAL_MAP:
        mp = _mediapool()
        clips = mp.ImportMedia([SIGNAL_MAP[sig]]) if mp else None
        if clips:
            _signal_timelines[sig] = mp.CreateTimelineFromClips(f"SIG_{sig}", clips)
    tl = _signal_timelines.get(sig)
    if not tl:
        return False, f"unknown/failed signal: {sig}", 404
    _project().SetCurrentTimeline(tl)
    tl.SetCurrentTimecode(TIMELINE_START_TC)
    return True, f"switched to {sig}", 200


def _timeline_fps(tl):
    try:
        return float(tl.GetSetting("timelineFrameRate") or 30)
    except Exception:  # noqa: BLE001
        return 30.0


def _frame_to_tc(frame, fps):
    """Absolute timeline frame -> 'HH:MM:SS:FF' (non-drop)."""
    f = int(round(fps)) or 30
    frame = int(frame)
    ff = frame % f
    s = frame // f
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{ff:02d}"


def _goto_item(tl, item, offset_sec=None):
    """Move playhead INTO a timeline item (offset_sec into it, clamped to the clip)."""
    fps = _timeline_fps(tl)
    if offset_sec is None:
        offset_sec = JUMP_OFFSET_SEC
    start, end = item.GetStart(), item.GetEnd()
    target = min(start + int(round(float(offset_sec) * fps)), max(start, end - 1))
    tc = _frame_to_tc(target, fps)
    ok = bool(tl.SetCurrentTimecode(tc))
    return ok, tc


def a_jump(body):
    """Move playhead into video-track item #index (1-based), JUMP_OFFSET_SEC into it."""
    p = _project()
    tl = p.GetCurrentTimeline() if p else None
    if not tl:
        return False, "no current timeline", 400
    items = tl.GetItemListInTrack("video", int(body.get("track", 1))) or []
    idx = int(body.get("index", 1)) - 1
    if not 0 <= idx < len(items):
        return False, f"index out of range (1..{len(items)})", 404
    item = items[idx]
    ok, tc = _goto_item(tl, item, body.get("offset"))
    return ok, f"jump #{idx + 1} {item.GetName()} @ {tc}", 200 if ok else 400


def a_add(body):
    """Import media from path -> media pool -> append to current timeline -> jump there."""
    path = body.get("path")
    if not path:
        return False, "missing 'path'", 400
    p = _project()
    mp = _mediapool()
    tl = p.GetCurrentTimeline() if p else None
    if not (p and mp and tl):
        return False, "no project/mediapool/timeline", 400
    clips = mp.ImportMedia([path])
    if not clips:
        return False, f"import failed (path exists on Mac?): {path}", 404
    items = mp.AppendToTimeline(clips)
    if not items:
        return False, "append to timeline failed", 500
    item = items[0]
    ok, tc = _goto_item(tl, item)
    return True, f"added {item.GetName()} @ {tc}", 200


def a_upload(filename, data, replace=False):
    """Write bytes to PATTERN_DIR/filename, then import+append+jump (or ReplaceClip in place)."""
    global _last_pattern_item
    if not filename:
        return False, "missing filename (?name=)", 400
    os.makedirs(PATTERN_DIR, exist_ok=True)
    dest = os.path.join(PATTERN_DIR, filename)
    with open(dest, "wb") as fh:
        fh.write(data)
    p = _project()
    mp = _mediapool()
    tl = p.GetCurrentTimeline() if p else None
    if not (p and mp and tl):
        return True, f"saved {dest} ({len(data)} bytes); no project/timeline to import into", 200
    if replace and _last_pattern_item is not None:
        ok = bool(_last_pattern_item.ReplaceClip(dest))
        return ok, f"replaced -> {dest} ({len(data)} bytes)", 200 if ok else 500
    clips = mp.ImportMedia([dest])
    if not clips:
        return False, f"saved but import failed: {dest}", 404
    items = mp.AppendToTimeline(clips)
    if not items:
        return False, "saved+imported but append failed", 500
    _last_pattern_item = clips[0]
    ok, tc = _goto_item(tl, items[0])
    return True, f"added {items[0].GetName()} @ {tc} ({len(data)} bytes)", 200


# ----------------------------------------------------------------------------
# Transport (macOS GUI automation via osascript)
# ----------------------------------------------------------------------------
IS_MAC = platform.system() == "Darwin"

# logical key -> AppleScript "System Events" snippet
_KEY_AS = {
    "space": 'keystroke " "',
    "j": 'keystroke "j"',
    "k": 'keystroke "k"',
    "l": 'keystroke "l"',
    "home": "key code 115",
    "end": "key code 119",
    "left": "key code 123",
    "right": "key code 124",
    "down": "key code 125",
    "up": "key code 126",
}


def a_key(key):
    if not IS_MAC:
        return False, "transport requires macOS (osascript)", 400
    snippet = _KEY_AS.get(key)
    if not snippet:
        return False, f"unknown key: {key}", 400
    script = (
        'tell application "DaVinci Resolve" to activate\n'
        "delay 0.12\n"
        'tell application "System Events" to ' + snippet
    )
    try:
        subprocess.run(["osascript", "-e", script], check=True, timeout=8)
        return True, f"key sent: {key}", 200
    except Exception as e:  # noqa: BLE001
        return False, f"osascript failed: {e}", 500


def _stop_server():
    """Shut down the running server and release the port (POST /shutdown)."""
    global _server
    srv, _server = _server, None
    if srv:
        srv.shutdown()
        srv.server_close()


# ----------------------------------------------------------------------------
# HTTP routing
# ----------------------------------------------------------------------------
def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            try:
                routes = {
                    "/": lambda: {"ok": True, "service": "remote_player", "version": VERSION, "mac": IS_MAC},
                    "/info": q_info,
                    "/timelines": lambda: {"timelines": q_timelines()},
                    "/timeline": q_timeline,
                    "/media": q_media,
                }
                fn = routes.get(self.path)
                if not fn:
                    return self._reply(404, {"ok": False, "msg": "not found"})
                return self._reply(200, fn())
            except Exception as e:  # noqa: BLE001
                return self._reply(500, {"ok": False, "msg": repr(e)})

        def do_POST(self):
            try:
                if self.path.split("?", 1)[0] == "/upload":
                    return self._handle_upload()
                body = self._json_body()
                ok, msg, code = self._dispatch(body)
                return self._reply(code, {"ok": ok, "msg": msg})
            except Exception as e:  # noqa: BLE001
                return self._reply(500, {"ok": False, "msg": repr(e)})

        def _handle_upload(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            filename = urllib.parse.unquote(q.get("name", [""])[0])
            replace = q.get("replace", ["0"])[0] in ("1", "true", "yes")
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n) if n else b""
            ok, msg, code = a_upload(filename, data, replace)
            return self._reply(code, {"ok": ok, "msg": msg})

        def _dispatch(self, body):
            p = self.path
            if p == "/play":
                return a_key("l")
            if p in ("/pause", "/stop"):
                return a_key("k")
            if p == "/reverse":
                return a_key("j")
            if p == "/toggle":
                return a_key("space")
            if p == "/key":
                return a_key(body.get("key", ""))
            if p == "/goto":
                return a_goto(body)
            if p == "/jump":
                return a_jump(body)
            if p == "/add":
                return a_add(body)
            if p == "/page":
                return a_page(body)
            if p == "/shutdown":
                Thread(target=_stop_server, daemon=True).start()
                return True, "shutting down", 200
            if p == "/timeline":
                return a_switch_timeline(body)
            if p in ("/signal", "/"):  # "/" kept for backward compat
                return a_signal(body.get("signal"))
            return False, f"unknown endpoint: {p}", 404

        def log_message(self, *a):
            pass

    return Handler


def _bind_server():
    """Bind on LISTEN_PORT; if a previous run of this script holds it, replace it."""
    try:
        return HTTPServer((LISTEN_HOST, LISTEN_PORT), make_handler())
    except OSError:
        try:  # ask the previous instance to step down, then retry
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{LISTEN_PORT}/shutdown", method="POST"),
                timeout=2,
            )
        except Exception:  # noqa: BLE001
            pass
        for _ in range(20):
            time.sleep(0.25)
            try:
                return HTTPServer((LISTEN_HOST, LISTEN_PORT), make_handler())
            except OSError:
                continue
        raise SystemExit(
            f"Port {LISTEN_PORT} still in use. Free it on the Mac:\n"
            f"  lsof -ti tcp:{LISTEN_PORT} | xargs kill -9"
        )


def main():
    global _server
    _server = _bind_server()
    Thread(target=_server.serve_forever, daemon=True).start()
    p = _project()
    print(f"remote_player {VERSION} listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"  project={p.GetName() if p else None} "
          f"page={resolve.GetCurrentPage()} mac={IS_MAC}")
    while _server is not None:
        time.sleep(1)
    print("remote_player stopped.")


if __name__ == "__main__":
    main()
