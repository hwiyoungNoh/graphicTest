"""
core/tv_control.py — webOS TV 제어 연결 (device control plane).

패턴 표시(core.pattern_source)·측정(sensor_module)과 *별개 모듈*. webOS TV(기본
192.168.0.8)에 picture/OSD 설정·3D-LUT 명령을 보내는 TVControlAPI 연결을 만들고,
전송모드(transport)를 SSH(기본) / WS 중에서 선택할 수 있게 한다.

    conn = connect_tv_control("192.168.0.8", mode="ssh")     # 기본 SSH
    conn.api.set_picture_mode(...) ; conn.api.send_lut(lut)   # 제어/LUT 되돌리기
    conn.disconnect()

라우팅은 TVControlAPI 내부가 처리한다(_ssh_live 면 SSH, 아니면 WS). 호출부는
전송모드를 몰라도 된다.
"""

from __future__ import annotations

import os
import sys
import threading
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_EXPORT_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "export_pkg")
if os.path.isdir(_EXPORT_PKG) and _EXPORT_PKG not in sys.path:
    sys.path.insert(0, _EXPORT_PKG)

# 전송모드 식별자 (TVControlAPI.set_mode 가 받는 값과 동일)
SSH = "ssh"
WS = "websocket"
DEFAULT_TV_IP = "192.168.0.8"


class TVControlConnection:
    """연결된 TVControlAPI + 그 전송 백엔드의 수명주기 묶음."""

    def __init__(self, api: Any, mode: str, transport: Any = None,
                 ws_thread: Optional[threading.Thread] = None):
        self.api = api              # TVControlAPI
        self.mode = mode            # 'ssh' | 'websocket'
        self.transport = transport  # TVSSHClient | WebSocketClient
        self.ws_thread = ws_thread

    def is_connected(self) -> bool:
        if self.transport is None:
            return False
        if self.mode == SSH:
            # TVSSHClient.is_connected 는 property(bool)
            return bool(getattr(self.transport, "is_connected", False))
        # WS: run_forever 가 ws 객체를 만든 뒤 살아있는지(느슨한 판정)
        return getattr(self.transport, "ws", None) is not None

    def disconnect(self) -> None:
        t = self.transport
        if t is None:
            return
        try:
            if self.mode == SSH:
                t.disconnect()
            else:
                ws = getattr(t, "ws", None)
                if ws is not None:
                    ws.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("[tv_control] disconnect failed: %s", e)


def connect_tv_control(ip: str = DEFAULT_TV_IP, mode: str = SSH, *,
                       ssh_port: int = 22, ws_port: int = 3001,
                       username: str = "root", password: str = "",
                       message_callback: Optional[Callable[[str], None]] = None,
                       connect: bool = True) -> TVControlConnection:
    """전송모드(SSH/WS)를 선택해 연결된 TVControlAPI 를 만든다.

    mode='ssh'(기본): TVSSHClient.connect() (블로킹, bool 반환). paramiko 필요.
    mode='ws'       : WebSocketClient.connect() 는 run_forever 로 블로킹하므로
                      데몬 스레드에서 시작한다. 최초 연결 시 PIN 페어링이 필요할 수
                      있다(client-key 미저장 IP) — wss.pair_tv 참고.

    connect=False 면 객체만 구성/배선하고 실제 연결은 하지 않는다(테스트/지연 연결용).
    """
    from device.device_tv_control_panel import TVControlAPI
    api = TVControlAPI()
    mode = SSH if str(mode).lower() == "ssh" else WS

    if mode == SSH:
        from device.device_tv_ssh_client import TVSSHClient
        ssh = TVSSHClient(ip, port=ssh_port, username=username, password=password)
        if message_callback is not None and hasattr(ssh, "set_message_callback"):
            ssh.set_message_callback(message_callback)
        api.attach_ssh(ssh)
        api.set_mode(SSH)
        if connect:
            ok = bool(ssh.connect())
            logger.info("[tv_control] SSH %s:%d connect -> %s", ip, ssh_port, ok)
            if not ok:
                logger.warning("[tv_control] SSH connect to %s failed "
                               "(paramiko 설치/네트워크/dev SSH 확인)", ip)
        return TVControlConnection(api=api, mode=mode, transport=ssh)

    # ── WS ──
    from device.device_connect import WebSocketClient
    cb = message_callback or (lambda m: logger.info("[WS] %s", m))
    ws = WebSocketClient(ip, cb, port=ws_port)
    ws.add_response_handler(api.handle_response)   # 응답을 TVControlAPI 로 라우팅
    api.set_command_sender(ws.send_request)
    api.set_mode(WS)
    t = None
    if connect:
        # run_forever 가 블로킹하므로 데몬 스레드에서 연결 루프를 돌린다.
        t = threading.Thread(target=ws.connect, name="tv-ws-connect", daemon=True)
        t.start()
        logger.info("[tv_control] WS %s:%d connect thread started "
                    "(최초 연결 시 PIN 페어링 필요할 수 있음)", ip, ws_port)
    return TVControlConnection(api=api, mode=mode, transport=ws, ws_thread=t)
