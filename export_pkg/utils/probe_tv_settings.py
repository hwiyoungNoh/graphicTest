"""Probe LG TV picture-setting descriptors + current values via SSH luna-send.

Pulls, for the A-direction target mapping (color_temperature / gamma / colorGamut
/ color), the VALID value vocabulary (getSystemSettingDesc) and the CURRENT value
(getSystemSettings) straight from the live TV — so the inference layer maps onto
the real WebOS API value names, not the stale enum guesses in the code.

    python utils/probe_tv_settings.py --ip 192.168.0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device.device_tv_ssh_client import TVSSHClient  # noqa: E402

# Keys driving the A-direction target (+ context). gammaLevel is the -3..+3
# sub-knob that applies when gamma == "mediumHavingLevel".
DESC_KEYS = [
    "pictureMode",
    "colorGamut",
    "gamma", "gammaLevel",
    "colorTemperature",
    "color",
    # context / adjacent
    "whiteBalanceColorTemperature", "dynamicToneMapping", "hdrDynamicToneMapping",
]


def _desc(client: TVSSHClient, keys: list[str], category: str = "picture") -> dict:
    return client.luna_raw(
        "luna://com.lge.settingsservice/getSystemSettingDesc",
        {"keys": keys, "category": category},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.0.8")
    ap.add_argument("--port", default=22, type=int)
    args = ap.parse_args()

    client = TVSSHClient(args.ip, port=args.port)
    client.set_message_callback(print)
    if not client.connect():
        print("[FAIL] SSH connect failed")
        return 1

    try:
        print("\n===== getSystemSettingDesc (valid value vocabulary) =====")
        # Per-key so one bad key doesn't drop the rest.
        for k in DESC_KEYS:
            try:
                r = _desc(client, [k])
                print(f"\n--- {k} ---")
                print(json.dumps(r, indent=2, ensure_ascii=False))
            except Exception as exc:
                print(f"\n--- {k} ---  [ERROR] {exc}")

        print("\n\n===== getSystemSettings (current values) =====")
        try:
            cur = client.get_system_settings(DESC_KEYS, "picture")
            print(json.dumps(cur, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"[ERROR] {exc}")
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
