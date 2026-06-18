"""대화형 PIN 페어링 (사용자가 직접 1회 실행).

TV 화면의 8자리 PIN을 터미널에 입력하면 페어링이 완료되고 client-key가
%APPDATA%\\PictureCalibration\\client_keys.json 에 저장된다. 이후 앱/검증
스크립트는 저장된 키로 .8에 자동 연결된다 (PIN 불필요).

실행:  python wss/pair_tv.py 192.168.0.8
"""
import sys
import os
import time
import json
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from device.device_connect import WebSocketClient  # noqa: E402

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.0.8"
state = {"registered": False, "pin_prompt": False, "key": None}


def cb(msg):
    print(msg)
    try:
        d = json.loads(msg)
        t = d.get("type")
        if t == "response" and d.get("payload", {}).get("pairingType") == "PIN":
            state["pin_prompt"] = True
        elif t == "registered":
            state["registered"] = True
            state["key"] = d.get("payload", {}).get("client-key")
    except Exception:
        pass


print("[*] %s:3001 연결 중 (SECURE_MANIFEST + pairingType PIN)..." % IP)
client = WebSocketClient(IP, cb, port=3001)
threading.Thread(target=client.connect, daemon=True).start()

# PIN 프롬프트(또는 저장키로 즉시 인증) 대기
for _ in range(15):
    if state["pin_prompt"] or state["registered"]:
        break
    time.sleep(1)

if state["registered"]:
    print("\n[OK] 저장된 client-key로 즉시 인증됨 — 페어링 불필요.")
    os._exit(0)

if not state["pin_prompt"]:
    print("\n[ERR] PIN 프롬프트 응답 없음. TV/네트워크 상태 확인.")
    os._exit(1)

pin = input("\n>>> TV 화면에 표시된 8자리 PIN 입력 후 Enter: ").strip()
client.set_pin(pin)

for _ in range(15):
    if state["registered"]:
        break
    time.sleep(1)

if state["registered"]:
    key = state["key"] or ""
    print("\n[OK] 페어링 성공! client-key 저장됨 (%s...). 이제 .8 자동 연결 가능." % key[:8])
    os._exit(0)
else:
    print("\n[FAIL] PIN 오류/만료. 스크립트를 다시 실행해 주세요.")
    os._exit(1)
