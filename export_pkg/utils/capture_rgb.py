"""Raw 8-bit RGB capture path for the TV (format="rgb").

The capture service writes headerless width*height*3 bytes for format "rgb"
(verified: UHD -> exactly 24,883,200 = 3840*2160*3). This module reshapes that
into an (H, W, 3) uint8 array — no container decode, no gAMA ambiguity, no JPEG
chroma loss — the cleanest 8-bit readback available on this TV.

    capture_rgb(client, w, h) -> np.ndarray (H, W, 3) uint8, channel order RGB

Run directly to sanity-check the path against a parallel PNG capture and pin
down the byte order:

    python utils/capture_rgb.py --ip 192.168.0.8 --width 3840 --height 2160
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device.device_tv_ssh_client import TVSSHClient  # noqa: E402


def capture_rgb(client: TVSSHClient, width: int, height: int,
                remote: str = "/tmp/_cap_rgb.bin",
                channel_order: str = "RGB") -> np.ndarray:
    """Capture the display as raw RGB and return an (H, W, 3) uint8 array.

    channel_order: byte order emitted by the service ("RGB" or "BGR").
    Returns the array already normalized to RGB order.
    """
    payload = {
        "method": "DISPLAY",
        "captureInputSize": {"width": width, "height": height},
        "width": width, "height": height,
        "format": "rgb",
        "path": remote,
    }
    cmd = (
        "luna-send -n 1 -f "
        "luna://com.webos.service.capture/executeOneShot "
        f"'{json.dumps(payload)}'"
    )
    raw = client._exec(cmd, timeout=15.0)
    resp = {}
    s, e = (raw or "").find("{"), (raw or "").rfind("}")
    if s != -1 and e != -1:
        try:
            resp = json.loads(raw[s:e + 1])
        except json.JSONDecodeError:
            pass
    if not resp.get("returnValue", False):
        raise RuntimeError(f"rgb capture failed: {raw[:200]!r}")

    w = int(resp.get("capturedWidth", width))
    h = int(resp.get("capturedHeight", height))
    expected = w * h * 3

    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "cap.bin")
        client.download_file(remote, local)
        buf = np.fromfile(local, dtype=np.uint8)
    if buf.size != expected:
        raise RuntimeError(
            f"rgb buffer size {buf.size} != expected {expected} ({w}x{h}x3)")
    img = buf.reshape(h, w, 3)
    if channel_order.upper() == "BGR":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _png_reference(client: TVSSHClient, width: int, height: int) -> np.ndarray:
    """Parallel PNG capture, returned as (H, W, 3) RGB uint8 (ground-truth order)."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "ref.png")
        client.capture_screen(local, width=width, height=height, fmt="PNG")
        return np.asarray(Image.open(local).convert("RGB"))


def main() -> int:
    try:  # avoid cp949 crashes on em-dash etc. in SSH client log messages
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.0.8")
    ap.add_argument("--port", default=22, type=int)
    ap.add_argument("--width", default=3840, type=int)
    ap.add_argument("--height", default=2160, type=int)
    ap.add_argument("--save", default="output/_cap_rgb_check.png",
                    help="save the raw-rgb capture (as RGB) here for eyeballing")
    args = ap.parse_args()

    client = TVSSHClient(args.ip, port=args.port)
    client.set_message_callback(print)
    if not client.connect():
        print("[FAIL] SSH connect failed")
        return 1

    try:
        # Assume RGB order first; verify against PNG reference below.
        rgb = capture_rgb(client, args.width, args.height, channel_order="RGB")
        print(f"\nraw rgb capture: shape={rgb.shape} dtype={rgb.dtype} "
              f"min={int(rgb.min())} max={int(rgb.max())}")

        ref = _png_reference(client, args.width, args.height)
        print(f"png reference  : shape={ref.shape}")

        if ref.shape == rgb.shape:
            # Correlate raw channel 0 against PNG R and B to pin the order.
            r_as_rgb = float(np.corrcoef(rgb[..., 0].ravel(),
                                         ref[..., 0].ravel())[0, 1])
            r_as_bgr = float(np.corrcoef(rgb[..., 0].ravel(),
                                         ref[..., 2].ravel())[0, 1])
            order = "RGB" if r_as_rgb >= r_as_bgr else "BGR"
            print(f"channel-0 corr  : vs PNG-R={r_as_rgb:.4f}  vs PNG-B={r_as_bgr:.4f}"
                  f"  -> raw byte order = {order}")
            md = float(np.abs(rgb.astype(int) - ref.astype(int)).mean())
            print(f"mean|rgb-png|   : {md:.3f} (assuming RGB order; "
                  f"two captures of same frame, small drift expected)")
        else:
            print("[warn] shape mismatch — cannot auto-determine order")

        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        from PIL import Image
        Image.fromarray(rgb).save(args.save)
        print(f"saved (as RGB)  : {args.save}")
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
