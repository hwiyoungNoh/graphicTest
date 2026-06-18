"""Reset the TV's external 3D LUT to identity (bypass).

Sends the same proven sequence batch_cp_test uses:
    CAL_START -> (TV popup settle) -> send identity 33^3 LUT
With an identity LUT loaded, the TV's colorGamut/processing reverts to its
true bypass behavior (a non-identity LUT otherwise forces colorGamut="wide").

    python utils/reset_tv_lut.py --ip 192.168.0.8 --pic-mode cinema --profile 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests" / "lut_compare"))

from device.device_tv_ssh_client import TVSSHClient  # noqa: E402
from common import make_identity_lut  # noqa: E402


def reset_tv_lut(client: TVSSHClient, pic_mode: str = "cinema",
                 profile: int = 0, size: int = 33, color_temp: int = 0,
                 cal_settle: float = 6.0, lut_settle: float = 2.0) -> bool:
    """Standard measurement init — matches batch_cp_test setup.

    CAL_START -> restore colorTemperature (CAL_START perturbs it) -> identity LUT.
    Leaves the CAL session open so subsequent LUTs can be swapped in.
    """
    print(f"[reset] CAL_START (pic_mode={pic_mode}, profile={profile})")
    client.cal_start(pic_mode, profile)
    print(f"[reset] waiting {cal_settle}s for TV popup/transition...")
    time.sleep(cal_settle)
    print(f"[reset] restore colorTemperature = {color_temp}")
    client.set_color_temperature_value(color_temp)
    time.sleep(1.0)
    print(f"[reset] sending identity {size}^3 LUT")
    ok = client.send_3d_lut(make_identity_lut(size), pic_mode, profile)
    time.sleep(lut_settle)
    print(f"[reset] identity LUT send -> {'OK' if ok else 'FAILED'}")
    return ok


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.0.8")
    ap.add_argument("--port", default=22, type=int)
    ap.add_argument("--pic-mode", default="cinema")
    ap.add_argument("--profile", default=0, type=int)
    ap.add_argument("--size", default=33, type=int)
    ap.add_argument("--cal-end", action="store_true",
                    help="finalize with CAL_END (default: leave session open)")
    args = ap.parse_args()

    client = TVSSHClient(args.ip, port=args.port)
    client.set_message_callback(print)
    if not client.connect():
        print("[FAIL] SSH connect failed")
        return 1
    try:
        reset_tv_lut(client, args.pic_mode, args.profile, args.size)
        if args.cal_end:
            print("[reset] CAL_END")
            client.cal_end(args.pic_mode, args.profile)
        # Re-read the settings most affected by an active LUT.
        s = client.get_system_settings(
            ["colorGamut", "color", "colorTemperature", "gamma", "gammaLevel",
             "brightness", "contrast"], "picture")
        print("\n[verify] post-reset picture settings:")
        for k in ("colorGamut", "color", "colorTemperature", "gamma",
                  "gammaLevel", "brightness", "contrast"):
            print(f"  {k:18s}= {s.get(k)!r}")
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
