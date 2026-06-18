#!/usr/bin/env python3
r"""원격 DaVinci Resolve(remote_player.py) 제어 클라이언트.

구조:
    [이 PC] remote_control.py  --HTTP-->  [Mac] Resolve 내부 remote_player.py (:7777)

전제:
    - Mac에서 remote_player.py 가 Resolve 안에서 실행 중 (Workspace > Scripts > Utility)
    - 네트워크 도달 가능 + 대상 주소 일치

대상 서버 설정 (기본 192.168.0.3:7777, 다른 Mac/네트워크면 환경변수로 override):
    set RESOLVE_REMOTE_HOST=10.0.0.5
    set RESOLVE_REMOTE_PORT=7777

CLI 사용법:
    python remote_control.py check                # 연결 확인 (TCP + GET /, 서버 version 표시)
    python remote_control.py info                 # 프로젝트/타임라인 요약
    python remote_control.py timelines            # 타임라인 목록
    python remote_control.py timeline             # 현재 타임라인 상세
    python remote_control.py media                # 미디어풀 클립 목록
    python remote_control.py switch <name|index>  # 타임라인 전환
    python remote_control.py jump <index> [track] # N번째 클립 위치로 점프 (시작+1초)
    python remote_control.py add <mac_path>       # Mac 로컬 미디어 임포트→타임라인 끝→점프
    python remote_control.py upload <file> [--replace] [--as <name>]  # 파일 업로드→임포트→추가→점프
    python remote_control.py <file> [--replace]   # 단축: = upload <file> (경로만 줘도 됨)
    python remote_control.py goto <timecode>      # 플레이헤드 이동 (프레임 표시)
    python remote_control.py page <edit|color..>  # 페이지 전환
    python remote_control.py play|pause|reverse|toggle   # transport (osascript)
    python remote_control.py key <space|j|k|l|home|..>   # raw 키 전송
    python remote_control.py signal <A|B|..>      # 레거시 프리셋 신호

라이브러리 사용 (생성기에서 import):
    from remote_control import upload
    status, res = upload(r"C:\patterns\ramp.png")      # 추가 + 콘텐츠(+1초) 위치
    status, res = upload(path, replace=True)           # 실시간: 같은 클립 제자리 교체
    status, res = upload(path, as_name="ramp.png")     # Mac에 저장될 파일명 지정
    # res 예: {"ok": true, "msg": "added ramp.png @ 01:13:38:21 (N bytes)"}

실시간 갱신 루프 예:
    from remote_control import upload
    upload(path)                          # 최초 1회: 타임라인에 추가 + 점프
    while updating:
        path = generate_pattern(...)      # 생성기가 새 패턴 파일 생성
        upload(path, replace=True)        # 같은 클립 제자리 교체 -> 뷰어 즉시 갱신
"""
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

# Target Resolve server (override via env when reusing elsewhere).
HOST = os.environ.get("RESOLVE_REMOTE_HOST", "192.168.0.3")
PORT = int(os.environ.get("RESOLVE_REMOTE_PORT", "7777"))
TIMEOUT = 5
BASE = f"http://{HOST}:{PORT}"


def _req(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except Exception:  # noqa: BLE001
            body = {"msg": "(non-json)"}
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return None, {"error": str(e)}


def _show(status, body):
    print(f"[{status}]")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0 if (status and 200 <= status < 300) else 1


def check():
    try:
        with socket.create_connection((HOST, PORT), timeout=TIMEOUT):
            pass
    except OSError as e:
        print(f"[!] TCP {HOST}:{PORT} 연결 실패 - {e}")
        print("    -> Mac의 Resolve에서 remote_player.py 실행 중인지 확인")
        return 1
    print(f"[+] TCP {HOST}:{PORT} 연결 OK")
    return _show(*_req("GET", "/"))


def upload(local_path, replace=False, as_name=None):
    r"""Public API — copy a local file to the Mac pattern folder and ingest into DaVinci.

    Import from your generator and call directly:
        from remote_control import upload
        status, res = upload(r"C:\patterns\ramp_001.png")              # add clip + jump
        status, res = upload(r"C:\patterns\ramp.png", replace=True)    # update in place
        status, res = upload(path, as_name="ramp.png")                 # override remote name
    The Mac-side file copy is included (bytes are sent over HTTP; the server writes them
    to PATTERN_DIR). Returns (status_code, response_dict); status is None on local error.
    """
    if not os.path.isfile(local_path):
        return None, {"ok": False, "error": f"file not found: {local_path}"}
    with open(local_path, "rb") as f:
        data = f.read()
    name = as_name or os.path.basename(local_path)
    url = BASE + "/upload?name=" + urllib.parse.quote(name)
    if replace:
        url += "&replace=1"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:  # noqa: BLE001
            return e.code, {"ok": False, "msg": "(non-json)"}
    except Exception as e:  # noqa: BLE001
        return None, {"ok": False, "error": str(e)}


def _looks_like_path(s):
    return (
        "/" in s or "\\" in s or os.path.isfile(s)
        or s.lower().endswith(
            (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".exr", ".dpx", ".mov", ".mp4", ".mkv")
        )
    )


GET_CMDS = {"info", "timelines", "timeline", "media"}
TRANSPORT = {"play", "pause", "reverse", "toggle", "stop"}


def main():
    try:  # CLI only: make Korean output safe on cp949 consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = sys.argv[1:]
    cmd = args[0] if args else "check"

    if cmd in ("help", "-h", "--help"):
        print(__doc__)
        return 0
    if cmd == "check":
        return check()
    if cmd in GET_CMDS:
        return _show(*_req("GET", "/" + cmd))
    if cmd in TRANSPORT:
        return _show(*_req("POST", "/" + cmd))
    if cmd == "switch":
        if len(args) < 2:
            print("usage: switch <name|index>")
            return 2
        v = args[1]
        payload = {"index": int(v)} if v.isdigit() else {"name": v}
        return _show(*_req("POST", "/timeline", payload))
    if cmd == "goto":
        if len(args) < 2:
            print("usage: goto <timecode>  e.g. 01:00:05:00")
            return 2
        return _show(*_req("POST", "/goto", {"timecode": args[1]}))
    if cmd == "page":
        if len(args) < 2:
            print("usage: page <edit|color|cut|fairlight|deliver|media|fusion>")
            return 2
        return _show(*_req("POST", "/page", {"page": args[1]}))
    if cmd == "key":
        if len(args) < 2:
            print("usage: key <space|j|k|l|home|end|left|right|up|down>")
            return 2
        return _show(*_req("POST", "/key", {"key": args[1]}))
    if cmd == "signal":
        if len(args) < 2:
            print("usage: signal <id>")
            return 2
        return _show(*_req("POST", "/signal", {"signal": args[1]}))
    if cmd == "jump":
        if len(args) < 2:
            print("usage: jump <index> [track]")
            return 2
        payload = {"index": int(args[1])}
        if len(args) >= 3:
            payload["track"] = int(args[2])
        return _show(*_req("POST", "/jump", payload))
    if cmd == "add":
        if len(args) < 2:
            print("usage: add <media_path_on_mac>")
            return 2
        return _show(*_req("POST", "/add", {"path": " ".join(args[1:])}))
    if cmd == "upload":
        if len(args) < 2:
            print("usage: upload <local_file> [--replace] [--as <name>]")
            return 2
        rest = args[2:]
        as_name = None
        if "--as" in rest:
            i = rest.index("--as")
            as_name = rest[i + 1] if i + 1 < len(rest) else None
        return _show(*upload(args[1], "--replace" in rest, as_name))

    # shorthand: a bare path (optionally written --path) means `upload <path>`
    target = cmd[2:] if cmd.startswith("--") else cmd
    if _looks_like_path(target):
        return _show(*upload(target, "--replace" in args[1:]))

    print(f"unknown command: {cmd}\n")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
