"""
TV SSH Client — Dev Mode
========================
LG TV (WebOS dev build) SSH client that executes luna-send commands
to read and write Picture settings.

Connection:
    SSH -> root@<IP>:22  (no password)

Key services:
    com.lge.settingsservice   : getSystemSettings / setSystemSettings
    externalpq                : setExternalPqData (CAL, 3D LUT)

Usage:
    from device.device_tv_ssh_client import TVSSHClient

    client = TVSSHClient("192.168.0.7")
    client.set_message_callback(print)
    client.connect()

    settings = client.get_picture_settings(["backlight", "brightness"])
    print(settings)  # {"backlight": "50", "brightness": "50"}

    client.set_system_settings({"backlight": "60"})
    client.disconnect()

Dependencies:
    pip install paramiko
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import paramiko
    from paramiko import AuthenticationException, SSHException
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    logger.warning("[SSH] paramiko not installed — run: pip install paramiko")


# ────────────────────────────────────────────────────────────────
# Known setting key lists
# ────────────────────────────────────────────────────────────────

ALL_PICTURE_KEYS: List[str] = [
    # ranges
    "backlight", "contrast", "brightness", "sharpness", "color", "tint",
    "colorTemperature",
    # arrays — existing
    "pictureMode", "gamma",
    "dynamicContrast", "dynamicColor", "peakBrightness",
    "superResolution", "noiseReduction", "smoothGradation",
    "blackLevel", "realCinema", "eyeComfortMode",
    "truMotionMode",
    # arrays — added from picture-quality spec
    "energySaving", "aiPicture", "colorGamut",
    "mpegNoiseReduction", "localDimming", "hdrDynamicToneMapping",
]

# CMS keys may not be available via settingsservice on some TVs;
# use externalpq struct instead
CMS_KEYS: List[str] = [
    f"cms{clr.capitalize()}{prm}"
    for clr in ("red", "green", "blue", "cyan", "magenta", "yellow")
    for prm in ("Hue", "Saturation", "Luminance")
]

WB_KEYS: List[str] = [
    f"whiteBalance{ch}{wt}"
    for ch in ("Red", "Green", "Blue")
    for wt in ("Gain", "Offset")
]


# ────────────────────────────────────────────────────────────────
# TVSSHClient
# ────────────────────────────────────────────────────────────────

class TVSSHClient:
    """LG TV dev-build SSH settings client."""

    def __init__(
        self,
        ip: str,
        port: int = 22,
        username: str = "root",
        password: str = "",
    ):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self._ssh: Optional[paramiko.SSHClient] = None  # type: ignore[name-defined]
        self._on_message: Optional[Callable[[str], None]] = None
        # Cached SFTP subsystem channel. Opened lazily by _get_sftp() and
        # reused across download_file/upload_file calls — opening SFTP per
        # transfer adds ~300-500ms of subsystem negotiation on dropbear.
        # Set to a sentinel (-1) once we've confirmed the subsystem is
        # unavailable so we don't retry on every call.
        self._sftp: Any = None
        self._sftp_unavailable: bool = False
        # Serialises every paramiko call (channel open / exec / sftp xfer) so
        # concurrent senders (LUT send + pair capture + settings) can't race
        # on the same transport — native OpenSSL stack overruns otherwise.
        # RLock allows nested calls (e.g. luna_raw -> _exec -> _exec_full).
        self._cmd_lock = threading.RLock()

    # ── Callback ──────────────────────────────────────────────────

    def set_message_callback(self, cb: Callable[[str], None]) -> None:
        self._on_message = cb

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self._on_message:
            self._on_message(msg)

    # ── Connection ─────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        if self._ssh is None:
            return False
        t = self._ssh.get_transport()
        return t is not None and t.is_active()

    def connect(self) -> bool:
        if not HAS_PARAMIKO:
            self._log("[SSH] paramiko not installed — run: pip install paramiko")
            return False
        try:
            self._log(f"[SSH] Connecting to {self.ip}:{self.port} as '{self.username}' ...")

            # ── Attempt 1: standard connect with empty-string password ──
            # dropbear requires password="" (empty string) rather than
            # password=None to attempt password authentication.
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(
                    self.ip,
                    port=self.port,
                    username=self.username,
                    password=self.password,   # "" OK — None skips password auth entirely
                    timeout=10,
                    allow_agent=False,
                    look_for_keys=False,
                )
                self._ssh = ssh
                self._log(f"[SSH] Connected to {self.ip} (password auth)")
                return True
            except (AuthenticationException, SSHException) as ex:
                self._log(f"[SSH] password auth failed ({ex}), trying 'none' auth...")

            # ── Attempt 2: none auth (some dropbear configurations) ──────
            transport = paramiko.Transport((self.ip, self.port))
            transport.start_client(timeout=10)
            try:
                transport.auth_none(self.username)
            except paramiko.BadAuthenticationType as bat:  # type: ignore[attr-defined]
                allowed = getattr(bat, "allowed_types", [])
                self._log(f"[SSH] none auth -> allowed methods: {allowed}")
                if "password" in allowed:
                    transport.auth_password(self.username, self.password)
                else:
                    raise
            ssh2 = paramiko.SSHClient()
            ssh2._transport = transport  # type: ignore[attr-defined]
            self._ssh = ssh2
            self._log(f"[SSH] Connected to {self.ip} (none/password fallback)")
            return True

        except Exception as exc:
            self._log(f"[SSH ERROR] {exc}")
            self._ssh = None
            return False

    def disconnect(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._ssh:
            self._ssh.close()
            self._ssh = None
            self._log(f"[SSH] Disconnected from {self.ip}")

    # ── SFTP (cached subsystem) ────────────────────────────────────

    def _get_sftp(self):
        """Return a cached SFTP client, opening the subsystem on first use.

        Returns None if SFTP is not available (caller should use fallback).
        Once we've confirmed unavailability, returns None without retrying.
        """
        with self._cmd_lock:
            if self._sftp_unavailable:
                return None
            if self._sftp is not None:
                return self._sftp
            if not self.is_connected:
                raise RuntimeError("SSH not connected")
            try:
                self._sftp = self._ssh.open_sftp()  # type: ignore[union-attr]
                return self._sftp
            except Exception as exc:
                self._log(f"[SFTP] Subsystem unavailable ({exc}) — using fallback for all transfers")
                self._sftp_unavailable = True
                self._sftp = None
                return None

    # ── Low-level exec ─────────────────────────────────────────────

    def _exec(self, cmd: str, timeout: float = 15.0) -> str:
        """Execute SSH command and return stdout."""
        out, _, _ = self._exec_full(cmd, timeout=timeout)
        return out

    def _exec_full(
        self, cmd: str, timeout: float = 15.0
    ) -> tuple:
        """Execute SSH command; return (stdout, stderr, exit_code).

        Both stdout and stderr are captured.
        On timeout or error, returns ("", error_str, -1) — does NOT retry
        reads (which would block again and cause a second timeout hang).
        """
        with self._cmd_lock:
            if not self.is_connected:
                raise RuntimeError("SSH not connected")
            _, out_ch, err_ch = self._ssh.exec_command(cmd)  # type: ignore[union-attr]
            chan = out_ch.channel
            chan.settimeout(timeout)
            try:
                out  = out_ch.read().decode("utf-8", errors="replace").strip()
                err  = err_ch.read().decode("utf-8", errors="replace").strip()
                code = chan.recv_exit_status()
            except Exception as exc:
                # Do NOT call out_ch.read() again here — it would block for
                # another full timeout cycle (double-hang bug).
                out  = ""
                err  = str(exc)
                code = -1
            return out, err, code

    # ── luna-send ─────────────────────────────────────────────────

    def luna_raw(
        self,
        service_uri: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run luna-send -n 1 -f <service_uri> '<payload_json>' and parse the result.

        Args:
            service_uri: e.g. "luna://com.lge.settingsservice/getSystemSettings"
            payload:     JSON-serializable dict
        Returns:
            TV response dict, e.g. {"returnValue": true, "settings": {...}}
        """
        payload_str = json.dumps(payload)
        cmd = f"luna-send -n 1 -f {service_uri} '{payload_str}'"
        raw = self._exec(cmd)
        if not raw:
            return {}
        return json.loads(raw)

    # ── System Settings ────────────────────────────────────────────

    def get_system_settings(
        self,
        keys: List[str],
        category: str = "picture",
    ) -> Dict[str, Any]:
        """
        Run luna://com.lge.settingsservice/getSystemSettings.

        Returns partial results when settings field is present even if returnValue=false.
        Unsupported keys are logged and ignored.

        Returns:
            {"backlight": 56, "brightness": 50, ...}
        """
        result = self.luna_raw(
            "luna://com.lge.settingsservice/getSystemSettings",
            {"keys": keys, "category": category},
        )
        settings = result.get("settings", {})
        if not result.get("returnValue"):
            error_keys = result.get("errorKey", [])
            if error_keys:
                self._log(f"[SSH] WARNING: Unsupported keys (ignored): {error_keys}")
            if not settings:
                raise RuntimeError(f"returnValue=false and no settings: {result}")
            # Partial success — use whatever settings were returned
        return settings

    def set_system_settings(
        self,
        settings: Dict[str, Any],
        category: str = "picture",
    ) -> bool:
        """
        Run luna://com.lge.settingsservice/setSystemSettings.

        Args:
            settings: {"backlight": 56, "pictureMode": "cinema", ...}
                      Pass numeric values as int/float, string values as str.
        Returns:
            returnValue bool
        """
        result = self.luna_raw(
            "luna://com.lge.settingsservice/setSystemSettings",
            {"category": category, "settings": settings},
        )
        ok = bool(result.get("returnValue"))
        if ok:
            self._log(f"[SSH] setSystemSettings OK - {settings}")
        else:
            self._log(f"[SSH] setSystemSettings FAILED: {result}")
        return ok

    def set_color_temperature_value(self, value: int = 0) -> bool:
        """Cal/Expert mode numeric Color Temperature.

        value: -50..+50 (0 = D65 neutral). After CAL_START the preset setter
        (warm2/natural/...) doesn't take effect; this numeric form does.
        Tries int first, falls back to str if firmware rejects.
        """
        value = max(-50, min(50, int(value)))
        ok = self.set_system_settings({"colorTemperature": value}, "picture")
        if not ok:
            ok = self.set_system_settings({"colorTemperature": str(value)}, "picture")
        return ok

    # ── Convenience getters ────────────────────────────────────────

    def get_picture_settings(
        self, keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Read picture settings. Pass keys=None to read all ALL_PICTURE_KEYS."""
        return self.get_system_settings(keys or ALL_PICTURE_KEYS, "picture")

    def get_white_balance(self) -> Dict[str, Any]:
        """Read white balance settings (R/G/B Gain and Offset)."""
        return self.get_system_settings(WB_KEYS, "picture")

    def get_cms(self) -> Dict[str, Any]:
        """Read Color Management System settings (6-axis)."""
        return self.get_system_settings(CMS_KEYS, "picture")

    def get_all_settings(self) -> Dict[str, Any]:
        """Read all Picture and WB settings at once (CMS excluded — not supported by settingsservice)."""
        return self.get_system_settings(
            ALL_PICTURE_KEYS + WB_KEYS, "picture"
        )

    def get_cms_settings(self) -> Dict[str, Any]:
        """
        CMS 6-axis settings. Returns empty dict if not supported by settingsservice.
        (Depends on TV firmware version.)
        """
        try:
            return self.get_system_settings(CMS_KEYS, "picture")
        except RuntimeError as exc:
            self._log(f"[SSH] CMS not available via settingsservice: {exc}")
            return {}

    # ── TV Spec / Info ─────────────────────────────────────────────

    TV_INFO_CONFIG_KEYS: List[str] = [
        # Panel / module
        "tv.model.panelGamutType",
        "tv.model.pwmFreqType",
        "tv.model.supportHDR",
        "tv.model.moduleBackLightType",
        "tv.model.moduleInchType",
        # SoC / HW
        "tv.hw.SoCOutputFrameRate",
        "tv.hw.panelOutputFrameRate",
        "tv.hw.displayType",
        "tv.hw.SoCChipType",
        "tv.hw.panelResolution",
        # Identification
        "tv.model.modelname",
        "tv.model.serialnumber",
        # Features (optional — may be missing on older FW)
        "tv.model.supportVRR",
        "tv.model.supportFreesync",
    ]

    def get_tv_info(self) -> Dict[str, Any]:
        """Fetch TV hardware/model spec via com.webos.service.config getConfigs
        plus SW version from /var/run/nyx/os_info.json.

        Returns a flat dict combining both sources. Missing keys are simply
        absent — never raises on partial data.
        """
        info: Dict[str, Any] = {}

        # 1) luna-send getConfigs
        try:
            result = self.luna_raw(
                "luna://com.webos.service.config/getConfigs",
                {"configNames": self.TV_INFO_CONFIG_KEYS},
            )
            configs = result.get("configs", {}) or {}
            info.update(configs)
        except Exception as exc:
            self._log(f"[SSH] get_tv_info getConfigs failed: {exc}")

        # 2) SW version from os_info.json
        try:
            raw = self._exec("cat /var/run/nyx/os_info.json", timeout=5.0)
            if raw:
                os_info = json.loads(raw)
                # Prefix to avoid colliding with tv.* keys
                for k, v in os_info.items():
                    info[f"os.{k}"] = v
        except Exception as exc:
            self._log(f"[SSH] get_tv_info os_info.json failed: {exc}")

        # 3) EDID from EIM service
        try:
            result = self.luna_raw(
                "luna://com.webos.service.eim/getEdidStatus", {}
            )
            for dev in result.get("devices", []):
                if dev.get("valid") and dev.get("edid"):
                    info["edid.hex"] = dev["edid"]
                    info["edid.id"] = dev.get("id", "")
                    break
        except Exception as exc:
            self._log(f"[SSH] get_tv_info EDID failed: {exc}")

        return info

    # ── Sync screen capture ────────────────────────────────────────

    def capture_screen(
        self,
        local_path: str,
        remote_path: str = "/tmp/_screen_capture.png",
        width: int = 1920,
        height: int = 1080,
        fmt: str = "PNG",
    ) -> str:
        """Synchronous luna-send screen capture. Returns local_path on success.

        Default format is PNG (lossless ~390 KB) — avoids JPEG 4:2:0 chroma
        subsampling so the captured image preserves per-channel precision.
        Pass fmt='JPEG' for compatibility with the main UI's capture path.

        Mirrors ScreenCaptureWorker._capture_once but without Qt — for use from
        analysis scripts that don't run a Qt event loop.
        """
        import json as _json
        if not self.is_connected:
            raise RuntimeError("SSH not connected")

        payload = _json.dumps({
            "path": remote_path,
            "method": "DISPLAY",
            "captureInputSize": {"width": width, "height": height},
            "width": width, "height": height,
            "format": fmt,
        })
        cmd = (
            "luna-send -n 1 -f "
            "luna://com.webos.service.capture/executeOneShot "
            f"'{payload}'"
        )
        raw = self._exec(cmd, timeout=10.0)

        written_bytes = 0
        if raw:
            try:
                resp = _json.loads(raw)
                if not resp.get("returnValue", True):
                    raise RuntimeError(
                        "luna-send failed: "
                        + resp.get("errorText", resp.get("errorCode", "unknown"))
                    )
                written_bytes = resp.get("writtenBytes", 0)
            except _json.JSONDecodeError:
                pass
        if written_bytes < 1000:
            raise RuntimeError(f"capture writtenBytes too small: {written_bytes}")

        self.download_file(remote_path, local_path)
        return local_path

    # ── File transfer ──────────────────────────────────────────────

    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Download a file from the TV.
        Tries SFTP first; falls back to SSH 'cat' pipe (dropbear compatibility).
        """
        with self._cmd_lock:
            if not self.is_connected:
                raise RuntimeError("SSH not connected")

            # --- Attempt 1: SFTP subsystem (cached) ---
            sftp = self._get_sftp()
            if sftp is not None:
                try:
                    sftp.get(remote_path, local_path)
                    self._log(f"[SFTP] Downloaded {remote_path} -> {local_path}")
                    return
                except Exception as exc:
                    # Per-transfer error — invalidate cached channel and fall
                    # back to cat. Subsystem itself might still work; next call
                    # will reopen.
                    self._log(f"[SFTP] get failed ({exc}), falling back to cat pipe...")
                    try:
                        sftp.close()
                    except Exception:
                        pass
                    self._sftp = None

            # --- Fallback: cat via exec_command ---
            _, stdout, stderr = self._ssh.exec_command(  # type: ignore[union-attr]
                f"cat {remote_path}", timeout=15.0
            )
            data = stdout.read()
            if not data:
                raise RuntimeError(
                    f"cat returned no data (file missing or permission denied): {remote_path}")
            with open(local_path, "wb") as f:
                f.write(data)
            self._log(f"[SSH] Downloaded via cat: {remote_path} -> {local_path} ({len(data)} bytes)")

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        Upload a file to the TV.
        Tries SFTP first; falls back to stdin dd pipe (dropbear compatibility).
        """
        with self._cmd_lock:
            if not self.is_connected:
                raise RuntimeError("SSH not connected")

            with open(local_path, "rb") as f:
                data = f.read()

            # --- Attempt 1: SFTP subsystem (cached) ---
            sftp = self._get_sftp()
            if sftp is not None:
                try:
                    sftp.put(local_path, remote_path)
                    self._log(f"[SFTP] Uploaded {local_path} -> {remote_path} ({len(data)} bytes)")
                    return
                except Exception as exc:
                    self._log(f"[SFTP] put failed ({exc}), falling back to dd pipe...")
                    try:
                        sftp.close()
                    except Exception:
                        pass
                    self._sftp = None

            # --- Fallback: pipe via dd over stdin ---
            stdin, stdout, stderr = self._ssh.exec_command(  # type: ignore[union-attr]
                f"dd of={remote_path}", timeout=60.0
            )
            stdin.write(data)
            stdin.channel.shutdown_write()
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                err_msg = stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"dd upload failed (exit {exit_code}): {err_msg}")
            self._log(
                f"[SSH] Uploaded via dd: {local_path} -> {remote_path} ({len(data)} bytes)"
            )

    # ── ExternalPQ ─────────────────────────────────────────────────

    def set_external_pq(self, payload: Dict[str, Any]) -> bool:
        """Run luna://com.webos.service.pqcontroller/setExternalPqData."""
        result = self.luna_raw("luna://com.webos.service.pqcontroller/setExternalPqData", payload)
        return bool(result.get("returnValue"))

    def get_external_pq(self) -> Dict[str, Any]:
        """Run luna://com.webos.service.pqcontroller/getExternalPqData."""
        return self.luna_raw("luna://com.webos.service.pqcontroller/getExternalPqData", {})

    def cal_start(self, pic_mode: str = "cinema", profile: int = 0) -> bool:
        return self.set_external_pq({
            "command": "CAL_START", "programID": 1,
            "picMode": pic_mode, "profileNo": profile, "dataOpt": 1,
            "dataType": "float", "dataCount": 9,
            "data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C2QO36MOb19P4U/",
        })

    def cal_end(self, pic_mode: str = "cinema", profile: int = 0) -> bool:
        return self.set_external_pq({
            "command": "CAL_END", "programID": 1,
            "picMode": pic_mode, "profileNo": profile, "dataOpt": 1,
            "dataType": "float", "dataCount": 9,
            "data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C2QO36MOb19P4U/",
        })

    def send_3d_lut(
        self,
        lut_data,           # np.ndarray shape (N, 3), float [0,1] or uint16 [0,4095]
        pic_mode: str = "cinema",
        profile: int = 0,
    ) -> bool:
        """Send a 3D LUT via SSH using Node.js palmbus call (no ARG_MAX limit).

        Why luna-send fails for large LUTs:
          luna-send passes the JSON as a command-line argument via exec().
          The OS ARG_MAX on this webOS build is ~128 KB, but the 3D LUT
          JSON is ~288 KB → "Argument list too long" regardless of how the
          argument is constructed (script literal, $(), etc.).

        Why Node.js works:
          Node.js reads the JSON from a /tmp file (small argv), then calls
          the webOS luna service via the palmbus IPC socket — no ARG_MAX.

        WSS-compatible parameters (dataOpt=1, palm:// URI) are used to
        match the working WebSocket path.

        dataType: "unsigned integer16", dataCount = numBytes / 2
        """
        import json as _json
        import numpy as np

        if not self.is_connected:
            raise RuntimeError("SSH not connected")

        # ── Convert to 12-bit UINT16 ─────────────────────────────
        arr = np.asarray(lut_data, dtype=np.float64)
        if arr.max() <= 1.0:            # float [0,1] -> 12-bit
            arr = arr * 4095.0
        arr = np.clip(np.round(arr), 0, 4095).astype(np.uint16)
        raw_bytes  = arr.flatten().tobytes()
        data_count = len(raw_bytes) // 2   # uint16 count (== file_size / 2)

        # ── Build JSON with "path" instead of inline "data" ────────
        # pqcontroller natively reads binary from a file path via fopen/fread,
        # so we upload raw bytes separately and pass the path reference.
        SERVICE   = "palm://com.webos.service.pqcontroller/setExternalPqData"
        BIN_DIR   = "/tmp/ssg/volatile/externalpq"
        BIN_TMP   = f"{BIN_DIR}/_lut3d.bin"
        JS_TMP    = "/tmp/_lut3d_call.js"

        json_str = _json.dumps({
            "command":   "BT709_3D_LUT_DATA",
            "programID": 1,
            "picMode":   pic_mode,
            "profileNo": profile,
            "dataOpt":   1,
            "dataType":  "unsigned integer16",
            "dataCount": data_count,
            "path":      BIN_TMP,
        }, separators=(",", ":"))

        self._log(
            f"[SSH][LUT] START: {data_count} uint16 values "
            f"({len(raw_bytes)} B raw, {len(json_str)} B JSON, "
            f"picMode={pic_mode}, profile={profile})"
        )

        # Build the luna payload once — used by both primary and fallback paths.
        # Because binary is uploaded separately and referenced via "path",
        # the JSON is only ~190 B — well under ARG_MAX, so luna-send works.
        luna_payload = {
            "command":   "BT709_3D_LUT_DATA",
            "programID": 1,
            "picMode":   pic_mode,
            "profileNo": profile,
            "dataOpt":   1,
            "dataType":  "unsigned integer16",
            "dataCount": data_count,
            "path":      BIN_TMP,
        }

        # ── Step 1: Upload raw binary data file ──────────────────
        # Ensure target directory exists, then upload raw uint16 bytes.
        self._exec_full(f"mkdir -p {BIN_DIR}", timeout=5.0)
        self._log(f"[SSH][LUT] Step 1: uploading binary ({len(raw_bytes)} B) -> {BIN_TMP}")
        if not self._upload_via_stdin(raw_bytes, BIN_TMP):
            return False
        self._log("[SSH][LUT] Step 1 OK")

        # ── Step 2: luna-send (primary path) ──────────────────────
        # CAL_START uses the same service (pqcontroller/setExternalPqData)
        # via luna-send and works reliably, so the LUT call should too now
        # that the JSON payload is ~190 B (path reference, not inline data).
        self._log("[SSH][LUT] Step 2: luna-send setExternalPqData (primary)")
        try:
            result = self.luna_raw(
                "luna://com.webos.service.pqcontroller/setExternalPqData",
                luna_payload,
            )
            if result.get("returnValue"):
                self._log(f"[SSH][LUT] DONE: OK (luna-send)  resp={result}")
                return True
            self._log(f"[SSH][LUT] luna-send returnValue=false, resp={result}")
        except Exception as exc:
            self._log(f"[SSH][LUT] luna-send threw: {exc}")
        self._log("[SSH][LUT] falling back to Node.js palmbus path...")

        # ── Step 3 (fallback): Upload tiny Node.js caller script ──
        # Enhanced diagnostics: capture each palmbus.Handle() exception so we
        # can see WHY all appIds fail (instead of the opaque ERR:no_handle).
        js_script = (
            "(function(){\n"
            "  var payload=JSON.stringify(" + json_str + ");\n"
            "  var svc='" + SERVICE + "';\n"
            "  var palmbus=null,modErr=[];\n"
            "  var mods=['palmbus','/usr/lib/nodejs/palmbus',\n"
            "            '/usr/palm/frameworks/location.ext/version/1.0/palmbus'];\n"
            "  for(var i=0;i<mods.length;i++){\n"
            "    try{palmbus=require(mods[i]);break;}catch(e){modErr.push(mods[i]+':'+e.message);}\n"
            "  }\n"
            "  if(!palmbus){process.stderr.write('ERR:no_palmbus ['+modErr.join(' | ')+']\\n');process.exit(1);}\n"
            "  var appIds=['',null,'com.webos.lunasend','com.palm.configurator',\n"
            "              'com.lge.chromalut','com.webos.lutsender.'+process.pid];\n"
            "  var h=null,handleErr=[];\n"
            "  for(var ai=0;ai<appIds.length;ai++){\n"
            "    try{h=new palmbus.Handle(appIds[ai],false);break;}\n"
            "    catch(e){handleErr.push(JSON.stringify(appIds[ai])+':'+e.message);}\n"
            "  }\n"
            "  if(!h){\n"
            "    process.stderr.write('ERR:no_handle ['+handleErr.join(' | ')+']\\n');\n"
            "    process.exit(1);\n"
            "  }\n"
            # Fire-and-forget: dispatch the call and exit after a short delay.
            # pqcontroller reads the binary directly via the 'path' field so the
            # LUT is applied as soon as the call is received; we do not need to
            # wait for the luna response.  h.run() / h.stop() on this webOS build
            # can block indefinitely when the response listener never fires, so we
            # avoid it entirely and just give the IPC socket 1 s to dispatch.
            "  var called=false,callErr='';\n"
            "  try{h.call(svc,payload,false);called=true;}catch(e1){\n"
            "    callErr+='3arg:'+e1.message+';';\n"
            "    try{h.call(svc,payload);called=true;}catch(e2){\n"
            "      callErr+='2arg:'+e2.message+';';\n"
            "    }\n"
            "  }\n"
            "  if(!called){\n"
            "    process.stderr.write('ERR:call_failed:'+callErr+'\\n');\n"
            "    process.exit(1);\n"
            "  }\n"
            "  process.stdout.write('{\"dispatched\":true}');\n"
            "  setTimeout(function(){process.exit(0);},1000);\n"
            "})();\n"
        )
        self._log(f"[SSH][LUT] Step 3a: uploading Node.js caller ({len(js_script)} B) -> {JS_TMP}")
        if not self._upload_via_stdin(js_script.encode("utf-8"), JS_TMP):
            return False
        self._log("[SSH][LUT] Step 3a OK")

        # ── Step 3b: Execute via node ──────────────────────────────
        self._log("[SSH][LUT] Step 3b: node " + JS_TMP)
        node_check, _, _ = self._exec_full("which node 2>/dev/null || which nodejs 2>/dev/null", timeout=5.0)
        node_bin = node_check.strip().split("\n")[0] if node_check.strip() else ""
        if not node_bin:
            self._log("[SSH][LUT] Step 3b FAILED: node/nodejs not found on TV")
            return False
        self._log(f"[SSH][LUT] using: {node_bin}")

        out, err, code = self._exec_full(f"{node_bin} {JS_TMP}", timeout=5.0)
        self._log(
            f"[SSH][LUT] Step 3b raw output (exit={code}):\n"
            f"  stdout: {out[:400]}\n  stderr: {err[:400]}"
        )

        # The JS script exits 0 after dispatching the call (fire-and-forget).
        # pqcontroller reads the binary via 'path', so the LUT is applied as
        # soon as pqcontroller processes the luna call — no response needed.
        # Exit code 1 means a hard failure (no palmbus module, no handle, or
        # h.call() threw).  Any other code (including -1 timeout) is treated
        # as success because Steps 1+2 already succeeded.
        if code == 1:
            self._log(f"[SSH][LUT] DONE: FAILED (node exit=1)  stderr: {err[:400]}")
            return False

        ok = (code == 0)
        if ok:
            self._log("[SSH][LUT] DONE: OK (dispatched via node)")
        else:
            self._log(f"[SSH][LUT] DONE: dispatched (exit={code}, treating as OK)")
        return True

    def _upload_via_stdin(self, data: bytes, remote_path: str) -> bool:
        """Upload bytes to remote_path using cat-stdin (dropbear-compatible)."""
        with self._cmd_lock:
            try:
                stdin, stdout, stderr = self._ssh.exec_command(  # type: ignore[union-attr]
                    f"cat > {remote_path}"
                )
                CHUNK = 32768
                offset = 0
                while offset < len(data):
                    stdin.write(data[offset:offset + CHUNK])
                    offset += CHUNK
                stdin.channel.shutdown_write()
                exit_code = stdout.channel.recv_exit_status()
                stderr.read()   # drain
                if exit_code != 0:
                    raise RuntimeError(f"cat exited with {exit_code}")
                return True
            except Exception as exc:
                self._log(f"[SSH] _upload_via_stdin -> {remote_path} FAILED: {exc}")
                return False


# ────────────────────────────────────────────────────────────────
# Qt Workers  (requires PySide6)
# ────────────────────────────────────────────────────────────────

try:
    from PySide6.QtCore import QThread, Signal  # type: ignore[attr-defined]

    class SSHConnectWorker(QThread):
        """Run SSH connect() in a background thread."""

        connected = Signal(bool)

        def __init__(self, client: TVSSHClient):
            super().__init__()
            self._client = client
            # NOTE: Set the message callback via client.set_message_callback()
            # before creating this worker so the callback outlives the worker.

        def run(self) -> None:
            ok = self._client.connect()
            self.connected.emit(ok)

        def stop(self) -> None:
            self._client.disconnect()

    class SSHCommandWorker(QThread):
        """Execute a single SSH luna-send command in a background thread."""

        result = Signal(str, object)   # (label, result_dict)
        error  = Signal(str, str)      # (label, error_message)

        def __init__(
            self,
            client: TVSSHClient,
            label: str,
            fn: Callable,
            *args: Any,
        ):
            super().__init__()
            self._client = client
            self._label  = label
            self._fn     = fn
            self._args   = args

        def run(self) -> None:
            try:
                r = self._fn(*self._args)
                self.result.emit(self._label, r)
            except Exception as exc:
                self.error.emit(self._label, str(exc))

    class SSHSetWorker(QThread):
        """
        Run setSystemSettings as a background fire-and-forget task.
        Results are reported via the client._on_message callback.
        """

        def __init__(
            self,
            client: TVSSHClient,
            settings: Dict[str, Any],
            category: str = "picture",
        ):
            super().__init__()
            self._client   = client
            self._settings = settings
            self._category = category

        def run(self) -> None:
            try:
                self._client.set_system_settings(self._settings, self._category)
            except Exception as exc:
                self._client._log(f"[SSH SET ERROR] {exc}")

    class SFTPWorker(QThread):
        """Background worker for SFTP/SSH file upload or download."""

        finished = Signal(bool, str)   # (success, message)

        def __init__(
            self,
            client: "TVSSHClient",
            direction: str,          # "upload" or "download"
            local_path: str,
            remote_path: str,
        ):
            super().__init__()
            self._client      = client
            self._direction   = direction
            self._local_path  = local_path
            self._remote_path = remote_path

        def run(self) -> None:
            try:
                if self._direction == "upload":
                    self._client.upload_file(self._local_path, self._remote_path)
                    self.finished.emit(True, f"Uploaded -> {self._remote_path}")
                else:
                    self._client.download_file(self._remote_path, self._local_path)
                    self.finished.emit(True, f"Downloaded -> {self._local_path}")
            except Exception as exc:
                self.finished.emit(False, str(exc))

    class ScreenCaptureWorker(QThread):
        """
        Persistent screen capture worker (continuous loop).

        Captures every interval_ms until stop() is called.
        A single QThread is reused for all frames to avoid per-frame overhead.
        Uses writtenBytes in the luna-send response to confirm the file is
        complete, so no sleep() delay or extra stat command is needed.
        """

        frame_ready   = Signal(str)   # local_path
        capture_error = Signal(str)   # error message (non-fatal; loop continues)

        def __init__(
            self,
            client: "TVSSHClient",
            remote_path: str,
            local_path: str,
            interval_ms: int = 2000,
        ):
            super().__init__()
            self._client      = client
            self._remote_path = remote_path
            self._local_path  = local_path
            self._interval    = interval_ms / 1000.0
            self._running     = False

        def stop(self) -> None:
            """Request the capture loop to stop. Safe to call from any thread."""
            self._running = False

        def run(self) -> None:
            self._running = True
            while self._running:
                t_start = time.monotonic()
                try:
                    self._capture_once()
                except Exception as exc:
                    self._client._log(f"[CAPTURE ERROR] {exc}")
                    self.capture_error.emit(str(exc))
                # Wait out the remaining interval in 0.05 s steps to react quickly to stop()
                elapsed = time.monotonic() - t_start
                remaining = self._interval - elapsed
                while remaining > 0 and self._running:
                    time.sleep(min(0.05, remaining))
                    remaining -= 0.05

        def _capture_once(self) -> None:
            # ── 1. Send luna-send capture command ────────────────────────
            payload = json.dumps({
                "path": self._remote_path,
                "method": "DISPLAY",
                "captureInputSize": {"width": 1920, "height": 1080},
                "width": 1920,
                "height": 1080,
                "format": "JPEG",
            })
            cmd = (
                "luna-send -n 1 -f "
                "luna://com.webos.service.capture/executeOneShot "
                f"'{payload}'"
            )
            t0 = time.monotonic()
            raw = self._client._exec(cmd, timeout=10.0)
            t_cmd = time.monotonic() - t0

            # ── 2. Parse response and verify via writtenBytes ─────────────
            # writtenBytes > 0 means the TV has finished writing the file,
            # so no sleep() delay or stat command is needed
            written_bytes = 0
            if raw:
                try:
                    resp = json.loads(raw)
                    if not resp.get("returnValue", True):
                        raise RuntimeError(
                            "luna-send failed: "
                            + resp.get("errorText", resp.get("errorCode", "unknown"))
                        )
                    written_bytes = resp.get("writtenBytes", 0)
                except json.JSONDecodeError:
                    pass  # ignore non-JSON response
            if written_bytes < 1000:
                raise RuntimeError(f"writtenBytes too small: {written_bytes}")

            # ── 3. Download file via cat (stat/sleep not needed) ──────────
            t1 = time.monotonic()
            self._client.download_file(self._remote_path, self._local_path)
            self.frame_ready.emit(self._local_path)

    HAS_QT = True

except ImportError:
    HAS_QT = False


# ────────────────────────────────────────────────────────────────
# Standalone test (python tv_ssh_client.py --ip <IP>)
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="LG TV SSH dev client smoke-test")
    parser.add_argument("--ip",   default="192.168.0.7", help="TV IP address")
    parser.add_argument("--port", default=22, type=int,   help="SSH port (default 22)")
    parser.add_argument("--user", default="root",         help="SSH username (default root)")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["backlight", "brightness", "pictureMode", "colorTemperature", "gamma"],
        help="Picture setting keys to read",
    )
    args = parser.parse_args()

    client = TVSSHClient(args.ip, port=args.port, username=args.user)
    client.set_message_callback(print)

    if not client.connect():
        print("[FAIL] SSH connection failed")
        raise SystemExit(1)

    print("\n── getSystemSettings ───────────────────────────────────")
    try:
        settings = client.get_system_settings(args.keys, "picture")
        print(json.dumps(settings, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[ERROR] {exc}")

    print("\n── backlight only ──────────────────────────────────────")
    try:
        s = client.get_system_settings(["backlight"], "picture")
        print(f"backlight = {s.get('backlight')!r}")
    except Exception as exc:
        print(f"[ERROR] {exc}")

    print("\n── brightness only ─────────────────────────────────────")
    try:
        s = client.get_system_settings(["brightness"], "picture")
        print(f"brightness = {s.get('brightness')!r}")
    except Exception as exc:
        print(f"[ERROR] {exc}")

    client.disconnect()
    print("\n── Done ─────────────────────────────────────────────────")
