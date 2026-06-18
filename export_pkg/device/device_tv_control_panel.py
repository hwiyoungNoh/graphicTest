"""
TV Control Panel Module
=======================
LG TV Picture 설정을 제어하기 위한 GUI 패널 및 API 인터페이스.

실제 통신은 별도 모듈(connect 등)에서 처리하며,
이 모듈은 UI + API 인터페이스만 제공합니다.

사용법:
    # 1) 독립 실행
    python -m device.device_tv_control_panel

    # 2) 외부 모듈에서 패널 열기
    from device.device_tv_control_panel import TVControlPanel, TVControlAPI
    api = TVControlAPI()
    api.set_command_sender(my_websocket_send_function)
    panel = TVControlPanel(api=api)
    panel.show()

    # 3) 통신 모듈 연결
    api.set_command_sender(ws_client.send_request)
    api.set_response_handler(my_handler)

Author: LUT Project
Date: 2026-03-01
"""

import json
import logging
import base64
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
#  상수 정의 (LG TV Picture Settings)
# ============================================================

class PictureMode(Enum):
    """LG TV 화질 모드."""
    VIVID = "vivid"
    STANDARD = "standard"
    ECO = "eco"
    CINEMA = "cinema"
    SPORTS = "sports"
    GAME = "game"
    FILMMAKER = "filmMaker"
    EXPERT_BRIGHT = "expert1"       # Expert (밝은 방)
    EXPERT_DARK = "expert2"         # Expert (어두운 방)
    DOLBY_CINEMA_BRIGHT = "dolbyHdrCinemaBright"
    DOLBY_CINEMA_DARK = "dolbyHdrCinemaDark"


class ColorTemperature(Enum):
    """색온도 프리셋."""
    WARM3 = "warm3"
    WARM2 = "warm2"
    WARM1 = "warm1"
    NATURAL = "natural"
    COOL1 = "cool1"
    COOL2 = "cool2"


class DynamicLevel(Enum):
    """Dynamic Contrast / Color 레벨."""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GammaPreset(Enum):
    """감마 프리셋 (webOS gamma 키 값)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH1 = "high1"
    HIGH2 = "high2"
    MEDIUM_HAVING_LEVEL = "mediumHavingLevel"


class HDRMode(Enum):
    """HDR 모드."""
    SDR = "sdr"
    HDR10 = "hdr10"
    DOLBY_VISION = "dolbyVision"
    HLG = "hlg"


# White Balance 채널
WB_CHANNELS = ["red", "green", "blue"]
WB_TYPES = ["gain", "offset"]

# Color Management (6축)
CMS_COLORS = ["red", "green", "blue", "cyan", "magenta", "yellow"]
CMS_PARAMS = ["hue", "saturation", "luminance"]


# ============================================================
#  Setting Parameter 정의 (이름 기반 접근용 레지스트리)
# ============================================================

@dataclass
class SettingParam:
    """개별 설정 파라미터 메타데이터.

    Attributes:
        name:       내부 이름 (PictureSettings 필드명과 동일)
        api_key:    WebOS API 키 (ssap payload에서 사용)
        label:      UI 표시용 한글/영문 라벨
        value_type: "int" / "str" / "bool"
        min_val:    숫자 최소값 (int일 때)
        max_val:    숫자 최대값 (int일 때)
        choices:    문자열 선택지 (str일 때)
        category:   그룹 분류 ("basic" / "advanced" / "wb" / "cms" / "calibration")
    """
    name: str
    api_key: str
    label: str
    value_type: str = "int"       # "int", "str", "bool"
    min_val: int = 0
    max_val: int = 100
    choices: List[str] = field(default_factory=list)
    category: str = "basic"


def _build_setting_params() -> Dict[str, SettingParam]:
    """전체 설정 파라미터 레지스트리 생성.

    Returns:
        {name: SettingParam} 딕셔너리.  name = PictureSettings 필드명.
    """
    params: List[SettingParam] = []

    # --- 기본 설정 ---
    params += [
        SettingParam("picture_mode",   "pictureMode",      "Picture Mode",
                     "str", choices=[m.value for m in PictureMode], category="basic"),
        SettingParam("backlight",      "backlight",        "OLED Light",
                     "int", 0, 100, category="basic"),
        SettingParam("contrast",       "contrast",         "Contrast",
                     "int", 0, 100, category="basic"),
        SettingParam("brightness",     "brightness",       "Brightness",
                     "int", 0, 100, category="basic"),
        SettingParam("sharpness",      "sharpness",        "Sharpness",
                     "int", 0, 50,  category="basic"),
        SettingParam("color",          "color",            "Color",
                     "int", 0, 100, category="basic"),
        SettingParam("tint",           "tint",             "Tint (R↔G)",
                     "int", -50, 50, category="basic"),
    ]

    # --- 고급 설정 ---
    dyn_levels = [d.value for d in DynamicLevel]
    params += [
        SettingParam("color_temperature", "colorTemperature", "Color Temperature",
                     "str", choices=[t.value for t in ColorTemperature], category="advanced"),
        SettingParam("gamma",            "gamma",             "Gamma",
                     "str", choices=[g.value for g in GammaPreset], category="advanced"),
        SettingParam("dynamic_contrast", "dynamicContrast",   "Dynamic Contrast",
                     "str", choices=dyn_levels, category="advanced"),
        SettingParam("dynamic_color",    "dynamicColor",      "Dynamic Color",
                     "str", choices=dyn_levels, category="advanced"),
        SettingParam("peak_brightness",  "peakBrightness",    "Peak Brightness",
                     "str", choices=dyn_levels, category="advanced"),
        SettingParam("super_resolution", "superResolution",   "Super Resolution",
                     "str", choices=dyn_levels, category="advanced"),
        SettingParam("noise_reduction",  "noiseReduction",    "Noise Reduction",
                     "str", choices=dyn_levels + ["auto"], category="advanced"),
        SettingParam("smooth_gradation", "smoothGradation",   "Smooth Gradation",
                     "str", choices=dyn_levels, category="advanced"),
        SettingParam("black_level",      "blackLevel",        "Black Level",
                     "str", choices=["auto", "low", "high"], category="advanced"),
        SettingParam("eye_comfort_mode", "eyeComfortMode",    "Eye Comfort Mode",
                     "bool", category="advanced"),
        SettingParam("trumotion",        "truMotionMode",     "TruMotion",
                     "str", choices=["off", "cinemaClear", "natural", "smooth", "user"],
                     category="advanced"),
        SettingParam("real_cinema",      "realCinema",        "Real Cinema",
                     "bool", category="advanced"),
    ]

    # --- White Balance (2-point) ---
    for ch in WB_CHANNELS:
        for wt in WB_TYPES:
            name = f"wb_{ch}_{wt}"
            api_key = f"whiteBalance{ch.capitalize()}{wt.capitalize()}"
            label = f"WB {ch.upper()} {wt.capitalize()}"
            params.append(SettingParam(name, api_key, label,
                                       "int", -50, 50, category="wb"))

    # --- Color Management System (6축) ---
    # api_key는 webOS setSettingsValidKeySet의 picture 카테고리 형식을 따른다:
    #   colorManagement{Hue|Saturation|Luminance}{Red|Green|...}  (예: colorManagementHueRed)
    # 과거 'cms{Color}{Param}'(cmsRedHue)는 유효 키가 아니라 getSystemSettings가
    # "Some keys are not allowed"로 요청 전체를 거부했음.
    for clr in CMS_COLORS:
        for prm in CMS_PARAMS:
            name = f"cms_{clr}_{prm}"
            api_key = f"colorManagement{prm.capitalize()}{clr.capitalize()}"
            label = f"CMS {clr.upper()} {prm.capitalize()}"
            params.append(SettingParam(name, api_key, label,
                                       "int", -30, 30, category="cms"))

    return {p.name: p for p in params}


# 전역 레지스트리 (모듈 로드 시 1회 생성)
SETTING_PARAMS: Dict[str, SettingParam] = _build_setting_params()

# api_key → name 역매핑 (응답 파싱용)
_API_KEY_TO_NAME: Dict[str, str] = {p.api_key: p.name for p in SETTING_PARAMS.values()}


# ============================================================
#  TV Picture Settings 데이터 모델
# ============================================================

@dataclass
class PictureSettings:
    """현재 TV Picture 설정 상태."""
    # --- Picture Mode ---
    picture_mode: str = "cinema"
    
    # --- 기본 설정 (0~100) ---
    backlight: int = 50           # OLED Light / 백라이트
    contrast: int = 85
    brightness: int = 50
    sharpness: int = 10
    color: int = 50               # 색 농도
    tint: int = 0                 # 색조 (R←→G), -50~+50 또는 0~100
    
    # --- 고급 설정 ---
    color_temperature: str = "warm2"
    gamma: str = "medium"
    dynamic_contrast: str = "off"
    dynamic_color: str = "off"
    peak_brightness: str = "off"
    super_resolution: str = "off"
    noise_reduction: str = "off"
    mpeg_noise_reduction: str = "off"
    smooth_gradation: str = "off"
    black_level: str = "auto"      # Auto / Low / High
    eye_comfort_mode: bool = False
    trumotion: str = "off"         # Off / Cinema Clear / Natural / Smooth / User
    real_cinema: bool = True       # 24p 소스 시 풀다운 제거
    motion_eye_care: bool = False
    
    # --- White Balance (20pt / 2pt) ---
    # 2-point: gain/offset per channel, -50~+50
    wb_red_gain: int = 0
    wb_green_gain: int = 0
    wb_blue_gain: int = 0
    wb_red_offset: int = 0
    wb_green_offset: int = 0
    wb_blue_offset: int = 0
    
    # --- Color Management System (6축) ---
    # 각 -30~+30
    cms_red_hue: int = 0
    cms_red_saturation: int = 0
    cms_red_luminance: int = 0
    cms_green_hue: int = 0
    cms_green_saturation: int = 0
    cms_green_luminance: int = 0
    cms_blue_hue: int = 0
    cms_blue_saturation: int = 0
    cms_blue_luminance: int = 0
    cms_cyan_hue: int = 0
    cms_cyan_saturation: int = 0
    cms_cyan_luminance: int = 0
    cms_magenta_hue: int = 0
    cms_magenta_saturation: int = 0
    cms_magenta_luminance: int = 0
    cms_yellow_hue: int = 0
    cms_yellow_saturation: int = 0
    cms_yellow_luminance: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환."""
        from dataclasses import asdict
        return asdict(self)

    def get_wb(self, channel: str, wb_type: str) -> int:
        """White Balance 값 조회. channel: red/green/blue, wb_type: gain/offset."""
        return getattr(self, f"wb_{channel}_{wb_type}", 0)
    
    def set_wb(self, channel: str, wb_type: str, value: int):
        """White Balance 값 설정."""
        setattr(self, f"wb_{channel}_{wb_type}", value)

    def get_cms(self, color: str, param: str) -> int:
        """CMS 값 조회. color: red/green/.., param: hue/saturation/luminance."""
        return getattr(self, f"cms_{color}_{param}", 0)
    
    def set_cms(self, color: str, param: str, value: int):
        """CMS 값 설정."""
        setattr(self, f"cms_{color}_{param}", value)


# ============================================================
#  TV Control API (통신 인터페이스)
# ============================================================

class TVControlAPI:
    """
    LG TV Picture 설정 제어 API.
    
    실제 통신(WebSocket 등)은 외부에서 set_command_sender()로 주입.
    이 클래스는 명령 포맷 생성 + 상태 관리만 담당.
    
    Usage:
        api = TVControlAPI()
        api.set_command_sender(ws_client.send_request)
        api.set_picture_mode("cinema")
        api.set_brightness(55)
    """

    # WebOS TV 서비스 URI
    URI_PICTURE_SETTINGS = "luna://settings/setSystemSettings"
    URI_GET_SETTINGS = "luna://settings/getSystemSettings"
    # 실 TV(.8) 검증: 'externalpq' 별칭 서비스만 동작. 'com.webos.service.pqcontroller'
    # 형태는 404(no such service or method)로 거부됨 → 참조(wss/cannedMessages.js) 형태 사용.
    URI_EXTERNAL_PQ = "palm://externalpq/setExternalPqData"
    URI_GET_EXTERNAL_PQ = "palm://externalpq/getExternalPqData"

    def __init__(self):
        self._settings = PictureSettings()
        self._command_sender: Optional[Callable] = None
        self._response_handler: Optional[Callable] = None
        self._id_counter: int = 100   # command id (connect.py와 겹치지 않도록)
        self._on_change_callbacks: List[Callable] = []
        
        # 마지막으로 보낸 명령 (디버깅용)
        self._last_command: Optional[Dict] = None

        # 파라미터 레지스트리 참조
        self.params: Dict[str, SettingParam] = SETTING_PARAMS

        # 응답 대기 콜백 {request_id: callback(response_data)}
        self._pending_responses: Dict[int, Callable] = {}

        # 초기 로드 완료 이벤트
        import threading
        self._load_complete_event = threading.Event()
        self._load_errors: List[str] = []

        # ── 전송 계층 라우팅 (SSH / WS) ───────────────────────────────
        # 단일 API가 현재 연결 모드에 따라 자동으로 SSH 또는 WS로 보낸다.
        # 페이지는 이 API만 호출하고 SSH/WS 분기를 알 필요가 없다.
        self._ssh = None                  # TVSSHClient (SSH 경로)
        self._mode: str = "websocket"     # 'ssh' | 'websocket'
        self._ssh_workers: List[Any] = [] # 실행 중 SSH 스레드 ref 유지(GC 방지)

    # ----------------------------------------------------------
    #  통신 연결
    # ----------------------------------------------------------

    def set_command_sender(self, sender: Callable):
        """
        명령 전송 함수 등록.
        
        sender 시그니처: sender(request: dict) -> None
        기존 connect.py의 WebSocketClient.send_request 와 호환.
        """
        self._command_sender = sender
        if sender is not None:
            logger.info("[TVControlAPI] Command sender registered.")
        else:
            logger.info("[TVControlAPI] Command sender cleared.")

    # ----------------------------------------------------------
    #  전송 계층 라우팅 (SSH / WS) — device_page(연결 모듈)가 주입
    # ----------------------------------------------------------

    def attach_ssh(self, ssh) -> None:
        """SSH 전송 백엔드(TVSSHClient) 등록. SSH 모드일 때 _send / send_lut이
        이 클라이언트로 라우팅한다 (블로킹 호출이라 스레드에서 실행)."""
        self._ssh = ssh

    def set_mode(self, mode: str) -> None:
        """현재 연결 모드 설정 ('ssh' | 'websocket'). 연결 모듈이 모드 변경/
        연결 시 호출. 페이지는 모드를 몰라도 된다 — 라우팅은 여기서 한다."""
        self._mode = "ssh" if str(mode).lower() == "ssh" else "websocket"

    def _ssh_live(self) -> bool:
        """현재 SSH 모드 + SSH 연결됨? (TVSSHClient.is_connected는 property)"""
        return (self._mode == "ssh" and self._ssh is not None
                and bool(getattr(self._ssh, "is_connected", False)))

    def _run_ssh(self, fn: Callable, *args) -> bool:
        """블로킹 SSH 호출(luna-send)을 백그라운드 스레드에서 실행 — GUI 미블록.
        기존 SSHCommandWorker를 재사용한다."""
        if self._ssh is None:
            return False
        try:
            from device.device_tv_ssh_client import SSHCommandWorker
        except Exception as exc:
            logger.error("[TVControlAPI] SSHCommandWorker import 실패: %s", exc)
            return False
        w = SSHCommandWorker(self._ssh, getattr(fn, "__name__", "ssh"), fn, *args)
        w.finished.connect(
            lambda: self._ssh_workers.remove(w) if w in self._ssh_workers else None)
        self._ssh_workers.append(w)
        w.start()
        return True

    def is_busy(self) -> bool:
        """SSH 전송 스레드가 실행 중인가 (드래그 중 중복 LUT 전송 방지용).
        WS 경로는 fire-and-forget이라 항상 False."""
        return any(getattr(w, "isRunning", lambda: False)()
                   for w in self._ssh_workers)

    def _send_via_ssh(self, request: Dict) -> bool:
        """WS-스타일 request를 SSH 클라이언트 메서드로 매핑해 전송.
        설정/Cal/패턴(작은 payload)만 처리 — 대용량 3D LUT는 send_lut()이
        ssh.send_3d_lut(palmbus)로 따로 보낸다(ARG_MAX 회피)."""
        uri = request.get("uri", "") or ""
        payload = request.get("payload", {}) or {}
        if "setSystemSettings" in uri:
            return self._run_ssh(self._ssh.set_system_settings,
                                 payload.get("settings", {}))
        if "setExternalPqData" in uri:
            if payload.get("command") == "BT709_3D_LUT_DATA":
                # 대용량 LUT는 inline luna-send면 ARG_MAX → base64를 풀어
                # ssh.send_3d_lut(palmbus)로 보낸다. (send_lut(raw)을 거치면
                # 이 디코딩 없이 바로 가지만, send_3d_lut_data를 직접 호출한
                # 경로도 안전하게 동작하도록 폴백.)
                try:
                    raw = base64.b64decode(payload.get("data", ""))
                    arr = np.frombuffer(raw, dtype=np.uint16).reshape(-1, 3)
                except Exception as exc:
                    logger.error("[TVControlAPI] SSH용 LUT 디코딩 실패: %s", exc)
                    return False
                return self._run_ssh(self._ssh.send_3d_lut, arr,
                                     payload.get("picMode", "cinema"),
                                     payload.get("profileNo", 0))
            # CAL_START / CAL_END / 패턴 등 작은 payload → set_external_pq 그대로.
            return self._run_ssh(self._ssh.set_external_pq, payload)
        if "getSystemSettings" in uri:
            # 읽기는 device_page(연결 모듈)가 모드별로 직접 처리 + 캐시를 채운다.
            logger.debug("[TVControlAPI] getSystemSettings(SSH) — device_page가 처리")
            return False
        return self._run_ssh(self._ssh.luna_raw, uri, payload)

    def set_response_handler(self, handler: Callable):
        """
        응답 처리 함수 등록.
        
        handler 시그니처: handler(response: dict) -> None
        """
        self._response_handler = handler

    def on_change(self, callback: Callable):
        """설정 변경 시 호출할 콜백 등록. callback(setting_name, value)"""
        self._on_change_callbacks.append(callback)

    @property
    def settings(self) -> PictureSettings:
        """현재 설정 상태 (로컬 캐시)."""
        return self._settings

    @property
    def last_command(self) -> Optional[Dict]:
        return self._last_command

    @property
    def is_connected(self) -> bool:
        return self._command_sender is not None

    # ----------------------------------------------------------
    #  명령 전송 (내부)
    # ----------------------------------------------------------

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _send(self, request: Dict) -> bool:
        """명령 전송. 현재 모드에 따라 SSH(스레드) 또는 WS로 자동 라우팅."""
        request.setdefault("id", self._next_id())
        self._last_command = request

        # SSH 모드 → 기존 SSH 클라이언트로 라우팅 (블로킹이라 스레드 실행)
        if self._ssh_live():
            sent = self._send_via_ssh(request)
            logger.info(f"[TVControlAPI] Sent via SSH: {request.get('uri', '?')}")
            return sent

        if self._command_sender is not None:
            try:
                self._command_sender(request)
                logger.info(f"[TVControlAPI] Sent: {request.get('uri', request.get('type', '?'))}")
                return True
            except Exception as e:
                logger.error(f"[TVControlAPI] Send error: {e}")
                return False
        else:
            logger.warning(f"[TVControlAPI] No sender. Command queued: "
                          f"{json.dumps(request, indent=2, ensure_ascii=False)[:200]}")
            return False

    def _notify_change(self, name: str, value: Any):
        """변경 콜백 호출."""
        for cb in self._on_change_callbacks:
            try:
                cb(name, value)
            except Exception as e:
                logger.error(f"[TVControlAPI] Callback error: {e}")

    # ----------------------------------------------------------
    #  이름 기반 파라미터 접근 API
    # ----------------------------------------------------------

    def get_param_info(self, name: str) -> Optional[SettingParam]:
        """파라미터 메타데이터 조회.

        Args:
            name: 파라미터 이름 (예: "backlight", "wb_red_gain", "cms_red_hue")
        Returns:
            SettingParam 또는 None
        """
        return self.params.get(name)

    def get_all_param_names(self, category: Optional[str] = None) -> List[str]:
        """등록된 파라미터 이름 목록.

        Args:
            category: 필터 ("basic"/"advanced"/"wb"/"cms"). None이면 전체.
        """
        if category:
            return [n for n, p in self.params.items() if p.category == category]
        return list(self.params.keys())

    def get_by_name(self, name: str) -> Any:
        """이름으로 현재 설정값 읽기.

        Args:
            name: 파라미터 이름 (PictureSettings 필드명)
        Returns:
            현재 값 (int / str / bool)
        Raises:
            KeyError: 알 수 없는 이름
        """
        if name not in self.params:
            raise KeyError(f"Unknown setting: '{name}'. "
                           f"Use get_all_param_names() to see available names.")
        return getattr(self._settings, name)

    def set_by_name(self, name: str, value: Any, send: bool = True) -> bool:
        """이름으로 설정값 변경 + (선택) TV 전송.

        Args:
            name:  파라미터 이름
            value: 설정할 값 (자동 타입 변환 / 클램핑)
            send:  True면 즉시 TV에 전송, False면 로컬만 변경
        Returns:
            send=True일 때 전송 성공 여부. send=False면 항상 True.
        """
        if name not in self.params:
            raise KeyError(f"Unknown setting: '{name}'")

        param = self.params[name]

        # 타입 변환 + 클램핑
        if param.value_type == "int":
            value = max(param.min_val, min(param.max_val, int(value)))
        elif param.value_type == "bool":
            if isinstance(value, str):
                value = value.lower() in ("true", "on", "1", "yes")
            else:
                value = bool(value)
        else:  # str
            value = str(value)

        # 로컬 설정 저장
        setattr(self._settings, name, value)
        self._notify_change(name, value)

        if not send:
            return True

        # WebOS 명령 빌드
        if param.value_type == "bool":
            api_val = "on" if value else "off"
        else:
            api_val = str(value)

        request = self._build_system_settings_request({param.api_key: api_val})
        return self._send(request)

    def set_multiple_by_name(self, settings: Dict[str, Any], send: bool = True) -> bool:
        """여러 파라미터를 한 번에 변경.

        Args:
            settings: {name: value, ...}
            send: True면 하나의 WebOS 요청으로 일괄 전송.
        """
        api_payload = {}
        for name, value in settings.items():
            if name not in self.params:
                logger.warning(f"[TVControlAPI] Unknown setting skipped: '{name}'")
                continue
            param = self.params[name]

            if param.value_type == "int":
                value = max(param.min_val, min(param.max_val, int(value)))
            elif param.value_type == "bool":
                if isinstance(value, str):
                    value = value.lower() in ("true", "on", "1", "yes")
                else:
                    value = bool(value)
            else:
                value = str(value)

            setattr(self._settings, name, value)
            self._notify_change(name, value)

            if send:
                if param.value_type == "bool":
                    api_payload[param.api_key] = "on" if value else "off"
                else:
                    api_payload[param.api_key] = str(value)

        if send and api_payload:
            request = self._build_system_settings_request(api_payload)
            return self._send(request)
        return True

    # ----------------------------------------------------------
    #  TV에서 초기 설정 로드
    # ----------------------------------------------------------

    def load_all_settings(self, callback: Optional[Callable] = None,
                           timeout: float = 5.0) -> bool:
        """TV에서 현재 Picture 설정 전체를 읽어와 로컬에 반영.

        응답은 handle_response()를 통해 비동기로 처리됩니다.
        connect.py의 on_message에서 handle_response()를 호출해야 합니다.

        Args:
            callback: 로드 완료 시 호출할 콜백. callback(success: bool, settings: dict)
            timeout:  응답 대기 타임아웃(초). 0이면 대기 안 함.
        Returns:
            요청 전송 성공 여부.
        """
        self._load_complete_event.clear()
        self._load_errors.clear()

        # 읽어올 API 키 목록 (SETTING_PARAMS에서 자동 생성)
        api_keys = [p.api_key for p in self.params.values()]
        if not api_keys:
            if callback:
                callback(True, self._settings.to_dict())
            return True

        # LG getSystemSettings는 keys 중 하나라도 읽기 실패(예: 현재 모드에 값이
        # 없는 truMotionMode → "no matched result from DB")면 배치 전체를 500으로
        # 거부한다. 그래서 키별 개별 요청을 보내고, 실패 키는 건너뛴 뒤 모두
        # 수신되면 callback을 호출한다.
        progress = {"remaining": len(api_keys), "ok": 0}

        def _make_handler(key: str):
            def _on_response(response_data):
                try:
                    if response_data:
                        self._apply_response_by_registry(response_data)
                        progress["ok"] += 1
                    else:
                        self._load_errors.append(key)
                except Exception as e:
                    logger.error(f"[TVControlAPI] Load error ({key}): {e}")
                    self._load_errors.append(key)
                finally:
                    progress["remaining"] -= 1
                    if progress["remaining"] <= 0:
                        logger.info(
                            f"[TVControlAPI] Settings loaded "
                            f"({progress['ok']}/{len(api_keys)} keys, "
                            f"{len(self._load_errors)} skipped)")
                        self._load_complete_event.set()
                        if callback:
                            callback(True, self._settings.to_dict())
            return _on_response

        sent_any = False
        for key in api_keys:
            request = self._build_get_settings_request([key])
            self._pending_responses[request["id"]] = _make_handler(key)
            if self._send(request):
                sent_any = True
            else:
                # 전송 실패 시 즉시 실패 처리(완료 카운트 유지)
                self._pending_responses.pop(request["id"], None)
                _make_handler(key)(None)
        return sent_any

    def load_settings_by_names(self, names: List[str],
                                callback: Optional[Callable] = None) -> bool:
        """지정한 이름의 파라미터만 TV에서 읽어오기.

        Args:
            names: 파라미터 이름 리스트 (예: ["backlight", "contrast"])
            callback: 완료 콜백. callback(success, {name: value})
        """
        api_keys = []
        for n in names:
            if n in self.params:
                api_keys.append(self.params[n].api_key)
            else:
                logger.warning(f"[TVControlAPI] Unknown name skipped: '{n}'")

        if not api_keys:
            if callback:
                callback(False, {})
            return False

        request = self._build_get_settings_request(api_keys)
        req_id = request["id"]

        def _on_response(response_data):
            if response_data:
                self._apply_response_by_registry(response_data)
            result = {n: getattr(self._settings, n) for n in names
                      if hasattr(self._settings, n)}
            if callback:
                callback(bool(response_data), result)

        self._pending_responses[req_id] = _on_response
        return self._send(request)

    def handle_response(self, message: str):
        """WebSocket 응답 메시지 처리.

        connect.py의 on_message 콜백에서 이 메서드를 호출하면
        대기 중인 load 요청에 응답을 라우팅합니다.

        Args:
            message: JSON 문자열 (WebSocket raw message)
        """
        try:
            data = json.loads(message) if isinstance(message, str) else message
        except json.JSONDecodeError:
            return

        # response(성공)/error(실패) 모두 pending 콜백으로 라우팅한다. error를
        # 흘려버리면 per-key 로더(load_all_settings)의 완료 카운트가 안 맞아
        # 완료 콜백이 영영 발화하지 않는다.
        msg_type = data.get("type")
        if msg_type not in ("response", "error"):
            return

        req_id = data.get("id")
        if req_id is not None and req_id in self._pending_responses:
            cb = self._pending_responses.pop(req_id)
            if msg_type == "error":
                settings = None   # 이 키 읽기 실패 신호
            else:
                payload = data.get("payload", {})
                settings = payload.get("settings", payload)
            try:
                cb(settings)
            except Exception as e:
                logger.error(f"[TVControlAPI] Response callback error: {e}")
            return

        if msg_type != "response":
            # 매칭되는 pending이 없는 error 응답 — 주로 set 실패.
            # 조용히 버리면 원인을 알 수 없으므로 로그 + UI 콜백으로 노출한다.
            payload = data.get("payload", {})
            err = data.get("error") or payload.get("errorText") or payload or data
            logger.warning(f"[TVControlAPI] TV error response: {err}")
            self._notify_change("tv_error", err)
            return
        # pending이 아닌 일반 응답도 자동 반영 시도
        payload = data.get("payload", {})
        settings = payload.get("settings", payload)
        if isinstance(settings, dict) and settings:
            self._apply_response_by_registry(settings)

    def _apply_response_by_registry(self, response_data: Dict):
        """응답 데이터를 SETTING_PARAMS 레지스트리 기반으로 로컬에 반영."""
        applied = 0
        for api_key, raw_val in response_data.items():
            name = _API_KEY_TO_NAME.get(api_key)
            if name is None:
                continue
            param = self.params[name]

            try:
                if param.value_type == "int":
                    val = int(raw_val)
                    val = max(param.min_val, min(param.max_val, val))
                elif param.value_type == "bool":
                    if isinstance(raw_val, bool):
                        val = raw_val
                    else:
                        val = str(raw_val).lower() in ("true", "on", "1", "yes")
                else:
                    val = str(raw_val)

                setattr(self._settings, name, val)
                applied += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"[TVControlAPI] Parse error for {name}: {e}")

        if applied:
            logger.info(f"[TVControlAPI] Applied {applied} settings from response.")

    # ----------------------------------------------------------
    #  ssap:// 기반 시스템 설정 명령 빌더
    # ----------------------------------------------------------

    def _build_system_settings_request(self, settings: Dict[str, Any]) -> Dict:
        """ssap://settings/setSystemSettings 형식 명령 생성."""
        return {
            "type": "request",
            "id": self._next_id(),
            "uri": self.URI_PICTURE_SETTINGS,
            "payload": {
                "category": "picture",
                "settings": settings
            }
        }

    def _build_get_settings_request(self, keys: List[str]) -> Dict:
        """ssap://settings/getSystemSettings 형식 조회 명령."""
        return {
            "type": "request",
            "id": self._next_id(),
            "uri": self.URI_GET_SETTINGS,
            "payload": {
                "category": "picture",
                "keys": keys
            }
        }

    def _build_external_pq_request(self, payload: Dict) -> Dict:
        """palm://externalpq/setExternalPqData 형식 명령 생성."""
        return {
            "type": "request",
            "id": self._next_id(),
            "uri": self.URI_EXTERNAL_PQ,
            "payload": payload
        }

    # ----------------------------------------------------------
    #  Picture Mode
    # ----------------------------------------------------------

    def set_picture_mode(self, mode: str) -> bool:
        """화질 모드 변경. mode: vivid/standard/cinema/game/..."""
        self._settings.picture_mode = mode
        request = self._build_system_settings_request({"pictureMode": mode})
        self._notify_change("picture_mode", mode)
        return self._send(request)

    def get_picture_mode(self) -> str:
        return self._settings.picture_mode

    # ----------------------------------------------------------
    #  기본 Picture 설정 (0~100 범위 슬라이더)
    # ----------------------------------------------------------

    def set_backlight(self, value: int) -> bool:
        """백라이트 / OLED Light (0~100)."""
        value = max(0, min(100, value))
        self._settings.backlight = value
        request = self._build_system_settings_request({"backlight": str(value)})
        self._notify_change("backlight", value)
        return self._send(request)

    def set_contrast(self, value: int) -> bool:
        """명암 (0~100)."""
        value = max(0, min(100, value))
        self._settings.contrast = value
        request = self._build_system_settings_request({"contrast": str(value)})
        self._notify_change("contrast", value)
        return self._send(request)

    def set_brightness(self, value: int) -> bool:
        """밝기 (0~100)."""
        value = max(0, min(100, value))
        self._settings.brightness = value
        request = self._build_system_settings_request({"brightness": str(value)})
        self._notify_change("brightness", value)
        return self._send(request)

    def set_sharpness(self, value: int) -> bool:
        """선명도 (0~50)."""
        value = max(0, min(50, value))
        self._settings.sharpness = value
        request = self._build_system_settings_request({"sharpness": str(value)})
        self._notify_change("sharpness", value)
        return self._send(request)

    def set_color(self, value: int) -> bool:
        """색 농도 (0~100)."""
        value = max(0, min(100, value))
        self._settings.color = value
        request = self._build_system_settings_request({"color": str(value)})
        self._notify_change("color", value)
        return self._send(request)

    def set_tint(self, value: int) -> bool:
        """색조 - Tint (R←→G). 범위 -50~+50."""
        value = max(-50, min(50, value))
        self._settings.tint = value
        request = self._build_system_settings_request({"tint": str(value)})
        self._notify_change("tint", value)
        return self._send(request)

    # ----------------------------------------------------------
    #  고급 Picture 설정
    # ----------------------------------------------------------

    def set_color_temperature(self, temp: str) -> bool:
        """색온도 프리셋. temp: warm3/warm2/warm1/natural/cool1/cool2."""
        self._settings.color_temperature = temp
        request = self._build_system_settings_request({"colorTemperature": temp})
        self._notify_change("color_temperature", temp)
        return self._send(request)

    def set_gamma(self, gamma: str) -> bool:
        """감마 프리셋. gamma: low/medium/high1/high2/mediumHavingLevel."""
        self._settings.gamma = gamma
        request = self._build_system_settings_request({"gamma": gamma})
        self._notify_change("gamma", gamma)
        return self._send(request)

    def set_dynamic_contrast(self, level: str) -> bool:
        """Dynamic Contrast. level: off/low/medium/high."""
        self._settings.dynamic_contrast = level
        request = self._build_system_settings_request({"dynamicContrast": level})
        self._notify_change("dynamic_contrast", level)
        return self._send(request)

    def set_dynamic_color(self, level: str) -> bool:
        """Dynamic Color. level: off/low/medium/high."""
        self._settings.dynamic_color = level
        request = self._build_system_settings_request({"dynamicColor": level})
        self._notify_change("dynamic_color", level)
        return self._send(request)

    def set_peak_brightness(self, level: str) -> bool:
        """Peak Brightness. level: off/low/medium/high."""
        self._settings.peak_brightness = level
        request = self._build_system_settings_request({"peakBrightness": level})
        self._notify_change("peak_brightness", level)
        return self._send(request)

    def set_super_resolution(self, level: str) -> bool:
        """Super Resolution. level: off/low/medium/high."""
        self._settings.super_resolution = level
        request = self._build_system_settings_request({"superResolution": level})
        self._notify_change("super_resolution", level)
        return self._send(request)

    def set_noise_reduction(self, level: str) -> bool:
        """Noise Reduction. level: off/low/medium/high/auto."""
        self._settings.noise_reduction = level
        request = self._build_system_settings_request({"noiseReduction": level})
        self._notify_change("noise_reduction", level)
        return self._send(request)

    def set_smooth_gradation(self, level: str) -> bool:
        """Smooth Gradation (밴딩 제거). level: off/low/medium/high."""
        self._settings.smooth_gradation = level
        request = self._build_system_settings_request({"smoothGradation": level})
        self._notify_change("smooth_gradation", level)
        return self._send(request)

    def set_black_level(self, level: str) -> bool:
        """Black Level (HDMI). level: auto/low/high."""
        self._settings.black_level = level
        request = self._build_system_settings_request({"blackLevel": level})
        self._notify_change("black_level", level)
        return self._send(request)

    def set_eye_comfort_mode(self, enabled: bool) -> bool:
        """Eye Comfort Mode (블루라이트 감소)."""
        self._settings.eye_comfort_mode = enabled
        request = self._build_system_settings_request({
            "eyeComfortMode": "on" if enabled else "off"
        })
        self._notify_change("eye_comfort_mode", enabled)
        return self._send(request)

    def set_trumotion(self, mode: str) -> bool:
        """TruMotion. mode: off/cinemaClear/natural/smooth/user."""
        self._settings.trumotion = mode
        request = self._build_system_settings_request({"truMotionMode": mode})
        self._notify_change("trumotion", mode)
        return self._send(request)

    def set_real_cinema(self, enabled: bool) -> bool:
        """Real Cinema (24p 풀다운 제거)."""
        self._settings.real_cinema = enabled
        request = self._build_system_settings_request({
            "realCinema": "on" if enabled else "off"
        })
        self._notify_change("real_cinema", enabled)
        return self._send(request)

    # ----------------------------------------------------------
    #  White Balance (2-point)
    # ----------------------------------------------------------

    def set_white_balance(self, channel: str, wb_type: str, value: int) -> bool:
        """
        White Balance 조정.
        
        Args:
            channel: "red" / "green" / "blue"
            wb_type: "gain" / "offset"
            value: -50 ~ +50
        """
        value = max(-50, min(50, value))
        self._settings.set_wb(channel, wb_type, value)
        
        # WebOS key 형식: whiteBalanceRedGain, whiteBalanceBlueOffset, ...
        key = f"whiteBalance{channel.capitalize()}{wb_type.capitalize()}"
        request = self._build_system_settings_request({key: str(value)})
        self._notify_change(f"wb_{channel}_{wb_type}", value)
        return self._send(request)

    def set_white_balance_all(self, gains: Tuple[int, int, int],
                               offsets: Tuple[int, int, int]) -> bool:
        """White Balance 일괄 설정. gains=(R,G,B), offsets=(R,G,B)."""
        settings = {}
        for i, ch in enumerate(WB_CHANNELS):
            g = max(-50, min(50, gains[i]))
            o = max(-50, min(50, offsets[i]))
            self._settings.set_wb(ch, "gain", g)
            self._settings.set_wb(ch, "offset", o)
            settings[f"whiteBalance{ch.capitalize()}Gain"] = str(g)
            settings[f"whiteBalance{ch.capitalize()}Offset"] = str(o)
        
        request = self._build_system_settings_request(settings)
        self._notify_change("white_balance_all", {"gains": gains, "offsets": offsets})
        return self._send(request)

    # ----------------------------------------------------------
    #  Color Management System (6축 색상 관리)
    # ----------------------------------------------------------

    def set_cms(self, color: str, param: str, value: int) -> bool:
        """
        CMS 6축 색상 관리.
        
        Args:
            color: "red"/"green"/"blue"/"cyan"/"magenta"/"yellow"
            param: "hue"/"saturation"/"luminance"
            value: -30 ~ +30
        """
        value = max(-30, min(30, value))
        self._settings.set_cms(color, param, value)
        
        key = f"cms{color.capitalize()}{param.capitalize()}"
        request = self._build_system_settings_request({key: str(value)})
        self._notify_change(f"cms_{color}_{param}", value)
        return self._send(request)

    # ----------------------------------------------------------
    #  Calibration 전용 (com.webos.service.pqcontroller)
    # ----------------------------------------------------------

    def send_cal_start(self, pic_mode: str = "cinema", profile: int = 0) -> bool:
        """캘리브레이션 모드 시작."""
        request = self._build_external_pq_request({
            "command": "CAL_START",
            "programID": 1,
            "picMode": pic_mode,
            "profileNo": profile,
            "dataOpt": 1,
            "dataType": "float",
            "dataCount": 9,
            "data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C2QO36MOb19P4U/"
        })
        self._notify_change("cal_start", {"picMode": pic_mode, "profile": profile})
        return self._send(request)

    def send_cal_end(self, pic_mode: str = "cinema", profile: int = 0) -> bool:
        """캘리브레이션 모드 종료."""
        request = self._build_external_pq_request({
            "command": "CAL_END",
            "programID": 1,
            "picMode": pic_mode,
            "profileNo": profile,
            "dataOpt": 1,
            "dataType": "float",
            "dataCount": 9,
            "data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C2QO36MOb19P4U/"
        })
        self._notify_change("cal_end", {"picMode": pic_mode, "profile": profile})
        return self._send(request)

    def send_3d_lut_data(self, encoded_data: str, data_count: int,
                          pic_mode: str = "cinema", profile: int = 0) -> bool:
        """3D LUT 데이터 전송 (Base64 인코딩된 데이터)."""
        request = self._build_external_pq_request({
            "command": "BT709_3D_LUT_DATA",
            "programID": 1,
            "picMode": pic_mode,
            "profileNo": profile,
            "dataOpt": 1,
            "dataType": "unsigned integer16",
            "dataCount": data_count,
            "data": encoded_data
        })
        self._notify_change("3d_lut_upload", {"dataCount": data_count})
        return self._send(request)

    def send_lut(self, lut_data, pic_mode: str = "cinema", profile: int = 0) -> bool:
        """3D LUT 전송 — 전송 계층(SSH/WS) 무관. 페이지는 raw LUT만 넘기면 된다.

        SSH: ssh.send_3d_lut(raw array, palmbus) — ARG_MAX 없음, 스레드 실행.
        WS : float[0,1]→uint16[0,4095] 인코딩 후 send_3d_lut_data로 전송.
        """
        if self._ssh_live():
            return self._run_ssh(self._ssh.send_3d_lut, lut_data, pic_mode, profile)
        arr = np.asarray(lut_data, dtype=np.float64)
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 4095.0
        arr = np.clip(np.round(arr), 0, 4095).astype(np.uint16)
        raw = arr.tobytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        return self.send_3d_lut_data(encoded, len(raw) // 2, pic_mode, profile)

    def get_3d_lut_data(self, callback: Optional[Callable] = None,
                         pic_mode: str = "cinema", profile: int = 0) -> bool:
        """TV로부터 3D LUT 데이터 가져오기.
        
        Args:
            callback: callback(success: bool, lut_array: np.ndarray | None)
                      WS 스레드에서 호출되므로 GUI 위젯은 Signal로 넘길 것.
            pic_mode: Picture Mode (기본 "cinema")
            profile: Profile 번호 (기본 0)
        Returns:
            요청 전송 성공 여부.
        """
        print("=" * 70)
        print("[GET LUT] ★★★ TVControlAPI.get_3d_lut_data CALLED ★★★")
        print("=" * 70)
        
        req_id = self._next_id()
        print(f"[GET LUT] Request ID: {req_id}")
        
        # GET 명령은 palm://externalpq/getExternalPqData 사용
        request = {
            "type": "request",
            "id": req_id,
            "uri": self.URI_GET_EXTERNAL_PQ,  # palm://externalpq/getExternalPqData
            "payload": {
                "command": "GET_3D_LUT_DATA",
                "programID": 1,
                "picMode": pic_mode,
                "profileNo": profile,
                "dataOpt": 1
            }
        }
        
        print(f"[GET LUT] Request built: {request}")
        logging.info(f"[GET LUT] Sending request to TV with id={req_id}")
        
        def _parse_response(settings):
            """응답 콜백 - settings는 이미 파싱된 dict 또는 None (error 응답)."""
            try:
                print("=" * 70)
                print("[GET LUT] ★★★ RESPONSE RECEIVED ★★★")
                print("=" * 70)
                logging.info("[GET LUT] Response received from TV, parsing...")
                
                if not settings:
                    print("[GET LUT] ERROR: TV returned error response (settings is None)")
                    print(f"Error received: {settings}")
                    logging.error(f"[GET LUT] TV returned error response")
                    if callback:
                        callback(False, None)
                    return
                
                print(f"[GET LUT] Settings keys: {list(settings.keys())}")
                logging.info("[GET LUT] Response ID matched, extracting payload...")
                
                # palm://externalpq/getExternalPqData 응답은 payload에 직접 데이터가 들어있음
                # settings = payload (handle_response에서 payload.get("settings", payload) 처리)
                
                # returnValue 확인
                if not settings.get("returnValue"):
                    print(f"[GET LUT] ERROR: returnValue is False")
                    logging.error(f"[GET LUT] TV returned returnValue=False")
                    if callback:
                        callback(False, None)
                    return
                
                # 직접 payload에서 데이터 읽기 (getExternalPqData 중첩 없음)
                b64_data = settings.get("data")
                data_count = settings.get("dataCount")
                data_type = settings.get("dataType")
                
                print(f"[GET LUT] Data metadata - type={data_type}, count={data_count}, b64_len={len(b64_data) if b64_data else 0}")
                logging.info(f"[GET LUT] Data metadata - type={data_type}, count={data_count}, b64_len={len(b64_data) if b64_data else 0}")
                
                if not b64_data or not data_count:
                    print(f"[GET LUT] ERROR: Missing data - b64_data={bool(b64_data)}, data_count={data_count}")
                    logging.error(f"[GET LUT] Missing data in response - b64_data={bool(b64_data)}, data_count={data_count}")
                    if callback:
                        callback(False, None)
                    return
                
                # Base64 디코딩
                print("[GET LUT] Decoding Base64 data...")
                logging.info("[GET LUT] Decoding Base64 data...")
                raw_bytes = base64.b64decode(b64_data)
                print(f"[GET LUT] Decoded {len(raw_bytes)} bytes")
                logging.info(f"[GET LUT] Decoded {len(raw_bytes)} bytes")
                
                # uint16 배열로 변환
                uint16_array = np.frombuffer(raw_bytes, dtype=np.uint16)
                print(f"[GET LUT] Converted to uint16 array: {len(uint16_array)} values")
                logging.info(f"[GET LUT] Converted to uint16 array: {len(uint16_array)} values")
                
                # LUT size 계산 (33x33x33x3 = 107811)
                total_values = len(uint16_array)
                lut_size = round((total_values / 3) ** (1/3))
                print(f"[GET LUT] Calculated LUT size: {lut_size}^3 ({total_values} values / 3 = {total_values/3:.0f} points)")
                logging.info(f"[GET LUT] Calculated LUT size: {lut_size}^3 ({total_values} values / 3 = {total_values/3:.0f} points)")
                
                # Reshape to (lut_size, lut_size, lut_size, 3)
                lut_array = uint16_array.reshape(lut_size, lut_size, lut_size, 3)
                print(f"[GET LUT] Reshaped to {lut_array.shape}")
                logging.info(f"[GET LUT] Reshaped to {lut_array.shape}")
                
                # 0~4095 범위를 0~1로 정규화
                lut_array = lut_array.astype(np.float64) / 4095.0
                print(f"[GET LUT] Normalized to [0,1]: min={lut_array.min():.4f}, max={lut_array.max():.4f}, mean={lut_array.mean():.4f}")
                logging.info(f"[GET LUT] Normalized to [0,1]: min={lut_array.min():.4f}, max={lut_array.max():.4f}, mean={lut_array.mean():.4f}")
                
                print(f"[GET LUT] SUCCESS: Parsed {lut_size}^3 LUT from TV")
                logging.info(f"[GET LUT] Successfully parsed {lut_size}^3 LUT from TV")
                if callback:
                    callback(True, lut_array)
                    
            except Exception as exc:
                print(f"[GET LUT] FATAL EXCEPTION: {exc}")
                logging.exception(f"[GET LUT] FATAL: Exception during response parsing")
                if callback:
                    callback(False, None)
        
        if callback:
            self._pending_responses[req_id] = _parse_response
        
        print(f"[GET LUT] Calling _send(request)...")
        result = self._send(request)
        print(f"[GET LUT] _send returned: {result}")
        logging.info(f"[GET LUT] Request sent, result={result}")
        return result

    # ----------------------------------------------------------
    #  스크린샷 캡처 (/tv/executeOneShot)
    # ----------------------------------------------------------

    def capture_screenshot(
        self,
        callback: Optional[Callable] = None,
        width: int = 1280,
        height: int = 720,
        method: Optional[str] = None,
        fmt: str = "PNG",
    ) -> bool:
        """TV 화면 스크린샷 캡처.

        Args:
            callback: callback(success: bool, data: bytes | None, image_uri: str | None)
                      WS 스레드에서 호출되므로 GUI 위젯은 Signal로 넘길 것.
            width:    캡처 이미지 너비 (기본 1280)
            height:   캡처 이미지 높이 (기본 720)
            method:   None → After (처리 후 출력), "SOURCE" → Before (원본 입력 신호)
            fmt:      이미지 포맷 (기본 "PNG"). UHD 분석용엔 "PNG" 권장.
        Returns:
            요청 전송 성공 여부.
        """
        import urllib.request as _urllib

        req_id = self._next_id()
        payload: Dict[str, Any] = {"width": width, "height": height, "format": fmt}
        if method:
            payload["method"] = method
        request = {
            "type": "request",
            "id": req_id,
            "uri": "luna://tv/executeOneShot",
            "payload": payload,
        }

        def _on_response(response_data):
            if response_data is None:
                logger.error("[TVControlAPI] Screenshot: no response")
                if callback:
                    callback(False, None, None)
                return
            image_uri = response_data.get("imageUri") if isinstance(response_data, dict) else None
            if not image_uri:
                logger.error("[TVControlAPI] Screenshot: no imageUri in %r", response_data)
                if callback:
                    callback(False, None, None)
                return
            try:
                import ssl as _ssl
                import time as _time
                _ctx = _ssl._create_unverified_context()
                _t0 = _time.perf_counter()
                with _urllib.urlopen(image_uri, timeout=15, context=_ctx) as resp:  # noqa: S310
                    data = resp.read()
                _dt = (_time.perf_counter() - _t0) * 1000.0
                logger.info("[TVControlAPI] Screenshot downloaded: %d bytes  %.0fms", len(data), _dt)
                if callback:
                    callback(True, data, image_uri)
            except Exception as exc:
                logger.error("[TVControlAPI] Screenshot download error: %s", exc)
                if callback:
                    callback(False, None, image_uri)

        self._pending_responses[req_id] = _on_response
        return self._send(request)

    # ----------------------------------------------------------
    #  패턴 제어 (com.webos.service.pqcontroller)
    # ----------------------------------------------------------

    def send_pattern_box(self, win_id: int, x: int, y: int,
                          w: int, h: int, r: int, g: int, b: int) -> bool:
        """패턴 윈도우 박스 설정."""
        request = self._build_external_pq_request({
            "command": "PTN_SINGLE_WINBOX_ATTR",
            "programID": 0,
            "winId": win_id,
            "startX": x, "startY": y,
            "width": w, "height": h,
            "fillR": r, "fillG": g, "fillB": b
        })
        return self._send(request)

    def send_pattern_enable(self, enable: bool, num_boxes: int = 2) -> bool:
        """패턴 표시 On/Off."""
        request = self._build_external_pq_request({
            "command": "PTN_CTRL",
            "programID": 0,
            "enable": "true" if enable else "false",
            "ptnType": 0,
            "numOfBox": num_boxes
        })
        self._notify_change("pattern_enable", enable)
        return self._send(request)

    # ----------------------------------------------------------
    #  조회 명령
    # ----------------------------------------------------------

    def request_all_picture_settings(self) -> bool:
        """현재 Picture 설정 전체 조회 요청 (SETTING_PARAMS 기반)."""
        keys = [p.api_key for p in self.params.values()]
        request = self._build_get_settings_request(keys)
        return self._send(request)

    def apply_response_settings(self, response_data: Dict):
        """조회 응답 데이터를 로컬 settings에 반영. (하위 호환 래퍼)

        Args:
            response_data: 서버 응답의 settings 딕셔너리
        """
        self._apply_response_by_registry(response_data)

    # ----------------------------------------------------------
    #  유틸리티
    # ----------------------------------------------------------

    def reset_to_default(self) -> PictureSettings:
        """설정을 기본값으로 초기화 (로컬만)."""
        self._settings = PictureSettings()
        self._notify_change("reset", None)
        return self._settings

    def export_settings_json(self) -> str:
        """현재 설정을 JSON 문자열로 내보내기."""
        return json.dumps(self._settings.to_dict(), indent=2, ensure_ascii=False)

    def import_settings_json(self, json_str: str):
        """JSON 문자열에서 설정 가져오기."""
        data = json.loads(json_str)
        for k, v in data.items():
            if hasattr(self._settings, k):
                setattr(self._settings, k, v)
        logger.info("[TVControlAPI] Settings imported from JSON.")


# ============================================================
#  TV Control Panel (Tkinter GUI)
# ============================================================

class TVControlPanel:
    """
    LG TV Picture 설정 제어 패널 (Tkinter).
    
    섹션 구성:
      1. Picture Mode 선택
      2. 기본 설정 슬라이더 (Backlight, Contrast, Brightness, Sharpness, Color, Tint)
      3. 색온도 / 감마
      4. 고급 설정 (Dynamic Contrast/Color, Super Resolution, ...)
      5. White Balance (2-point)
      6. Color Management (6축)
      7. Calibration 제어 (CAL Start/End, 3D LUT Upload)
    """

    def __init__(self, api: Optional[TVControlAPI] = None,
                 master: Optional[tk.Tk] = None):
        """
        Args:
            api: TVControlAPI 인스턴스. None이면 새로 생성.
            master: tkinter 루트. None이면 새로 생성.
        """
        self.api = api or TVControlAPI()
        
        self._own_root = master is None
        if self._own_root:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(master)
        
        self.root.title("LG TV Picture Control")
        self.root.geometry("520x920")
        self.root.resizable(True, True)
        
        # 슬라이더 참조 저장 (나중에 값 동기화용)
        self._sliders: Dict[str, tk.Scale] = {}
        self._combo_vars: Dict[str, tk.StringVar] = {}
        self._check_vars: Dict[str, tk.BooleanVar] = {}
        
        # 로그 텍스트
        self._log_var = tk.StringVar(value="Ready")
        
        # 초기화 중 슬라이더 콜백 억제
        self._initializing = True
        
        # 슬라이더 디바운스 타이머 (드래그 중 명령 폭주 방지)
        self._debounce_timers: Dict[str, str] = {}
        self._debounce_delay = 300  # ms
        
        self._build_ui()
        self._sync_ui_from_settings()
        
        # tkinter Scale의 command 콜백은 event loop에서 지연 실행됨.
        # mainloop 시작 후 모든 초기 이벤트가 처리된 다음에 _initializing을 해제해야
        # set() 호출로 인한 불필요한 API 호출을 방지할 수 있음.
        self.root.after(500, self._finish_init)
    
    def _finish_init(self):
        """초기화 완료 — 이후부터 슬라이더 조작이 실제 API 호출로 연결됨."""
        self._initializing = False
        self._log("Ready")

    # ----------------------------------------------------------
    #  UI 구축
    # ----------------------------------------------------------

    def _build_ui(self):
        """전체 UI 구축."""
        # 스크롤 가능한 메인 프레임
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.main_frame = ttk.Frame(canvas)
        
        self.main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 마우스 휠 스크롤
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- 섹션 빌드 ---
        self._build_connection_section()
        self._build_picture_mode_section()
        self._build_basic_section()
        self._build_color_temp_gamma_section()
        self._build_advanced_section()
        self._build_white_balance_section()
        self._build_cms_section()
        self._build_calibration_section()
        self._build_status_bar()

    # --- 연결 상태 ---
    def _build_connection_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Connection", padding=5)
        frame.pack(fill="x", padx=5, pady=(5, 2))
        
        status_text = "● Connected" if self.api.is_connected else "○ Not Connected"
        status_color = "green" if self.api.is_connected else "gray"
        self._conn_label = tk.Label(frame, text=status_text, fg=status_color,
                                     font=("Segoe UI", 9, "bold"))
        self._conn_label.pack(side="left", padx=5)
        
        btn_refresh = ttk.Button(frame, text="🔄 Sync Settings", width=16,
                                  command=self._on_sync_settings)
        btn_refresh.pack(side="right", padx=5)

        btn_load = ttk.Button(frame, text="📥 Load from TV", width=16,
                               command=self._on_load_from_tv)
        btn_load.pack(side="right", padx=2)

    # --- Picture Mode ---
    def _build_picture_mode_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Picture Mode", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        modes = [m.value for m in PictureMode]
        var = tk.StringVar(value=self.api.settings.picture_mode)
        self._combo_vars["picture_mode"] = var
        
        combo = ttk.Combobox(frame, textvariable=var, values=modes,
                              state="readonly", width=25)
        combo.pack(side="left", padx=5)
        combo.bind("<<ComboboxSelected>>",
                    lambda e: self._on_combo_change("picture_mode", var.get()))
        
        ttk.Button(frame, text="Apply", width=8,
                    command=lambda: self.api.set_picture_mode(var.get())).pack(side="right", padx=5)

    # --- 기본 설정 슬라이더 ---
    def _build_basic_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Basic Picture Settings", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        sliders = [
            ("backlight",  "OLED Light",  0, 100, self.api.set_backlight),
            ("contrast",   "Contrast",    0, 100, self.api.set_contrast),
            ("brightness", "Brightness",  0, 100, self.api.set_brightness),
            ("sharpness",  "Sharpness",   0,  50, self.api.set_sharpness),
            ("color",      "Color",       0, 100, self.api.set_color),
            ("tint",       "Tint (R↔G)", -50, 50, self.api.set_tint),
        ]
        
        for key, label, lo, hi, setter in sliders:
            self._add_slider(frame, key, label, lo, hi, setter)

    # --- 색온도 / 감마 ---
    def _build_color_temp_gamma_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Color Temperature / Gamma", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        # Color Temperature
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Color Temp:", width=14).pack(side="left")
        temps = [t.value for t in ColorTemperature]
        var_temp = tk.StringVar(value=self.api.settings.color_temperature)
        self._combo_vars["color_temperature"] = var_temp
        combo_temp = ttk.Combobox(row1, textvariable=var_temp, values=temps,
                                   state="readonly", width=12)
        combo_temp.pack(side="left", padx=5)
        combo_temp.bind("<<ComboboxSelected>>",
                         lambda e: self.api.set_color_temperature(var_temp.get()))
        
        # Gamma
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Gamma:", width=14).pack(side="left")
        gammas = [g.value for g in GammaPreset]
        var_gamma = tk.StringVar(value=self.api.settings.gamma)
        self._combo_vars["gamma"] = var_gamma
        combo_gamma = ttk.Combobox(row2, textvariable=var_gamma, values=gammas,
                                    state="readonly", width=12)
        combo_gamma.pack(side="left", padx=5)
        combo_gamma.bind("<<ComboboxSelected>>",
                          lambda e: self.api.set_gamma(var_gamma.get()))

    # --- 고급 설정 ---
    def _build_advanced_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Advanced Settings", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        levels = [d.value for d in DynamicLevel]
        
        combos = [
            ("dynamic_contrast",  "Dynamic Contrast", levels, self.api.set_dynamic_contrast),
            ("dynamic_color",     "Dynamic Color",    levels, self.api.set_dynamic_color),
            ("peak_brightness",   "Peak Brightness",  levels, self.api.set_peak_brightness),
            ("super_resolution",  "Super Resolution", levels, self.api.set_super_resolution),
            ("noise_reduction",   "Noise Reduction",  levels + ["auto"], self.api.set_noise_reduction),
            ("smooth_gradation",  "Smooth Gradation", levels, self.api.set_smooth_gradation),
            ("black_level",       "Black Level",      ["auto", "low", "high"], self.api.set_black_level),
        ]
        
        for key, label, values, setter in combos:
            self._add_combo_row(frame, key, label, values, setter)
        
        # 체크박스들
        checks = [
            ("eye_comfort_mode", "Eye Comfort Mode", self.api.set_eye_comfort_mode),
            ("real_cinema",      "Real Cinema",      self.api.set_real_cinema),
        ]
        
        check_frame = ttk.Frame(frame)
        check_frame.pack(fill="x", pady=2)
        for key, label, setter in checks:
            self._add_check(check_frame, key, label, setter)
        
        # TruMotion
        trumotion_vals = ["off", "cinemaClear", "natural", "smooth", "user"]
        self._add_combo_row(frame, "trumotion", "TruMotion", trumotion_vals,
                             self.api.set_trumotion)

    # --- White Balance (2-point) ---
    def _build_white_balance_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="White Balance (2-Point)", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        for ch in WB_CHANNELS:
            ch_frame = ttk.Frame(frame)
            ch_frame.pack(fill="x")
            
            color_hex = {"red": "#D44", "green": "#4A4", "blue": "#44D"}[ch]
            ttk.Label(ch_frame, text=f"  {ch.upper()}", width=8,
                      foreground=color_hex).pack(side="left")
            
            for wb_type in WB_TYPES:
                key = f"wb_{ch}_{wb_type}"
                sub = ttk.Frame(ch_frame)
                sub.pack(side="left", expand=True, fill="x")
                ttk.Label(sub, text=wb_type.capitalize(), width=6).pack(side="left")
                
                slider = tk.Scale(
                    sub, from_=-50, to=50, orient="horizontal", length=130,
                    command=lambda val, k=key, c=ch, t=wb_type: (
                        None if self._initializing else
                        self._on_slider_change(k, int(val),
                            lambda v, _c=c, _t=t: self.api.set_white_balance(_c, _t, v))
                    )
                )
                slider.set(self.api.settings.get_wb(ch, wb_type))
                slider.pack(side="left", fill="x", expand=True)
                self._sliders[key] = slider

    # --- Color Management System (6축) ---
    def _build_cms_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Color Management (6-Axis)", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        # 접기/펼치기
        self._cms_expanded = tk.BooleanVar(value=False)
        toggle_btn = ttk.Checkbutton(frame, text="Expand CMS Controls",
                                      variable=self._cms_expanded,
                                      command=lambda: self._toggle_cms())
        toggle_btn.pack(anchor="w")
        
        self._cms_inner = ttk.Frame(frame)
        # 초기에는 숨김
        
        color_hex_map = {
            "red": "#D44", "green": "#4A4", "blue": "#44D",
            "cyan": "#4BB", "magenta": "#B4B", "yellow": "#BB4"
        }
        
        for color in CMS_COLORS:
            color_frame = ttk.LabelFrame(self._cms_inner, text=color.upper(), padding=2)
            color_frame.pack(fill="x", pady=1)
            
            for param in CMS_PARAMS:
                key = f"cms_{color}_{param}"
                row = ttk.Frame(color_frame)
                row.pack(fill="x")
                ttk.Label(row, text=f"  {param.capitalize()}", width=12).pack(side="left")
                
                slider = tk.Scale(
                    row, from_=-30, to=30, orient="horizontal", length=180,
                    command=lambda val, k=key, c=color, p=param: (
                        None if self._initializing else
                        self._on_slider_change(k, int(val),
                            lambda v, _c=c, _p=p: self.api.set_cms(_c, _p, v))
                    )
                )
                slider.set(self.api.settings.get_cms(color, param))
                slider.pack(side="left", fill="x", expand=True)
                self._sliders[key] = slider

    def _toggle_cms(self):
        if self._cms_expanded.get():
            self._cms_inner.pack(fill="x")
        else:
            self._cms_inner.pack_forget()

    # --- Calibration ---
    def _build_calibration_section(self):
        frame = ttk.LabelFrame(self.main_frame, text="Calibration", padding=5)
        frame.pack(fill="x", padx=5, pady=2)
        
        # PicMode / Profile 선택
        row_mode = ttk.Frame(frame)
        row_mode.pack(fill="x", pady=2)
        ttk.Label(row_mode, text="PicMode:", width=10).pack(side="left")
        self._cal_picmode_var = tk.StringVar(value="cinema")
        ttk.Combobox(row_mode, textvariable=self._cal_picmode_var,
                      values=["cinema", "expert1", "expert2", "filmMaker", "game"],
                      state="readonly", width=14).pack(side="left", padx=5)
        
        ttk.Label(row_mode, text="Profile:").pack(side="left", padx=(10, 0))
        self._cal_profile_var = tk.IntVar(value=0)
        tk.Spinbox(row_mode, from_=0, to=10, textvariable=self._cal_profile_var,
                   width=5).pack(side="left", padx=5)
        
        # 버튼들
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="▶ CAL START", width=14,
                    command=self._on_cal_start).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="■ CAL END", width=14,
                    command=self._on_cal_end).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="📤 Upload 3D LUT", width=16,
                    command=self._on_lut_upload).pack(side="left", padx=3)
        
        # 패턴 제어
        ptn_frame = ttk.LabelFrame(frame, text="Pattern Control", padding=3)
        ptn_frame.pack(fill="x", pady=2)
        
        ttk.Button(ptn_frame, text="Pattern ON", width=12,
                    command=lambda: self.api.send_pattern_enable(True)).pack(side="left", padx=3)
        ttk.Button(ptn_frame, text="Pattern OFF", width=12,
                    command=lambda: self.api.send_pattern_enable(False)).pack(side="left", padx=3)

    # --- 상태 바 ---
    def _build_status_bar(self):
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x")
        
        self._status_label = ttk.Label(bar, textvariable=self._log_var,
                                        relief="sunken", anchor="w", padding=(5, 2))
        self._status_label.pack(fill="x")
        
        # 하단 버튼
        btn_bar = ttk.Frame(self.root)
        btn_bar.pack(side="bottom", fill="x", pady=3)
        
        ttk.Button(btn_bar, text="Reset All", width=10,
                    command=self._on_reset).pack(side="left", padx=5)
        ttk.Button(btn_bar, text="Export JSON", width=12,
                    command=self._on_export).pack(side="left", padx=5)
        ttk.Button(btn_bar, text="Import JSON", width=12,
                    command=self._on_import).pack(side="left", padx=5)

    # ----------------------------------------------------------
    #  위젯 헬퍼
    # ----------------------------------------------------------

    def _add_slider(self, parent, key: str, label: str,
                    lo: int, hi: int, setter: Callable):
        """슬라이더 + 라벨 + 값 표시 한 줄 추가."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        
        ttk.Label(row, text=label, width=14).pack(side="left")
        
        slider = tk.Scale(
            row, from_=lo, to=hi, orient="horizontal", length=250,
            command=lambda val, s=setter, k=key: self._on_slider_change(k, int(val), s)
        )
        current_val = getattr(self.api.settings, key, (lo + hi) // 2)
        slider.set(current_val)
        slider.pack(side="left", fill="x", expand=True, padx=5)
        self._sliders[key] = slider

    def _add_combo_row(self, parent, key: str, label: str,
                        values: list, setter: Callable):
        """콤보박스 한 줄 추가."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        
        ttk.Label(row, text=label, width=18).pack(side="left")
        
        current_val = getattr(self.api.settings, key, values[0])
        var = tk.StringVar(value=current_val)
        self._combo_vars[key] = var
        
        combo = ttk.Combobox(row, textvariable=var, values=values,
                              state="readonly", width=12)
        combo.pack(side="left", padx=5)
        combo.bind("<<ComboboxSelected>>",
                    lambda e, s=setter, v=var: s(v.get()))

    def _add_check(self, parent, key: str, label: str, setter: Callable):
        """체크박스 추가."""
        current_val = getattr(self.api.settings, key, False)
        var = tk.BooleanVar(value=current_val)
        self._check_vars[key] = var
        
        cb = ttk.Checkbutton(parent, text=label, variable=var,
                              command=lambda s=setter, v=var: s(v.get()))
        cb.pack(side="left", padx=10)

    # ----------------------------------------------------------
    #  이벤트 핸들러
    # ----------------------------------------------------------

    def _on_slider_change(self, key: str, value: int, setter: Callable):
        """슬라이더 변경 시 디바운스 후 API 호출 (드래그 중 명령 폭주 방지)."""
        if self._initializing:
            return
        
        # 기존 타이머 취소
        if key in self._debounce_timers:
            self.root.after_cancel(self._debounce_timers[key])
        
        # 새 타이머 등록 (300ms 후 실행)
        self._debounce_timers[key] = self.root.after(
            self._debounce_delay,
            lambda: self._execute_slider_command(key, value, setter)
        )

    def _execute_slider_command(self, key: str, value: int, setter: Callable):
        """디바운스 타이머 만료 후 실제 명령 전송."""
        self._debounce_timers.pop(key, None)
        setter(value)
        self._log(f"{key} → {value}")

    def _on_combo_change(self, key: str, value: str):
        """콤보 변경 (Picture Mode 등)."""
        self._log(f"{key} → {value}")

    def _on_sync_settings(self):
        """TV에서 현재 설정 조회 요청 (기존 방식 호환)."""
        self.api.request_all_picture_settings()
        self._log("Settings sync requested (legacy)")

    def _on_load_from_tv(self):
        """TV에서 현재 설정 전체 로드 (SETTING_PARAMS 기반)."""
        def _on_loaded(success: bool, settings_dict: dict):
            # WebSocket 스레드에서 호출되므로 tkinter는 after()로 UI 업데이트
            if success:
                self.root.after(0, self._after_load_success)
            else:
                self.root.after(0, lambda: self._log("❌ Load from TV failed"))

        sent = self.api.load_all_settings(callback=_on_loaded)
        if sent:
            self._log("📥 Loading settings from TV...")
        else:
            self._log("❌ Cannot send load request (no sender?)")

    def _after_load_success(self):
        """TV 설정 로드 성공 후 UI 동기화."""
        old_init = self._initializing
        self._initializing = True  # 슬라이더 콜백 억제
        try:
            self._sync_ui_from_settings()
            self._log("✅ Settings loaded from TV")
        finally:
            self._initializing = old_init

    def _on_cal_start(self):
        mode = self._cal_picmode_var.get()
        profile = self._cal_profile_var.get()
        self.api.send_cal_start(mode, profile)
        self._log(f"CAL START (mode={mode}, profile={profile})")

    def _on_cal_end(self):
        mode = self._cal_picmode_var.get()
        profile = self._cal_profile_var.get()
        self.api.send_cal_end(mode, profile)
        self._log(f"CAL END (mode={mode}, profile={profile})")

    def _on_lut_upload(self):
        """3D LUT 파일 선택 후 업로드."""
        import base64
        from tkinter import filedialog as fd
        
        file_path = fd.askopenfilename(
            title="Select LUT binary file",
            filetypes=[("Binary files", "*.bin *.dat"), ("All files", "*.*")],
            initialdir="."
        )
        if not file_path:
            return
        
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode("utf-8")
            count = len(data) // 2  # unsigned int16
            
            mode = self._cal_picmode_var.get()
            profile = self._cal_profile_var.get()
            self.api.send_3d_lut_data(encoded, count, mode, profile)
            self._log(f"3D LUT uploaded ({count} values)")
        except Exception as e:
            messagebox.showerror("Upload Error", str(e))

    def _on_reset(self):
        """설정 초기화."""
        if messagebox.askyesno("Reset", "Reset all settings to default?"):
            self.api.reset_to_default()
            self._sync_ui_from_settings()
            self._log("Settings reset to default")

    def _on_export(self):
        """설정 JSON 내보내기."""
        from tkinter import filedialog as fd
        path = fd.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.api.export_settings_json())
            self._log(f"Exported to {path}")

    def _on_import(self):
        """설정 JSON 가져오기."""
        from tkinter import filedialog as fd
        path = fd.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json")]
        )
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.api.import_settings_json(f.read())
            self._sync_ui_from_settings()
            self._log(f"Imported from {path}")

    # ----------------------------------------------------------
    #  UI ↔ Settings 동기화
    # ----------------------------------------------------------

    def _sync_ui_from_settings(self):
        """API settings → UI 위젯 동기화."""
        s = self.api.settings
        
        # 슬라이더
        for key, slider in self._sliders.items():
            val = getattr(s, key, None)
            if val is not None:
                slider.set(val)
        
        # 콤보박스
        for key, var in self._combo_vars.items():
            val = getattr(s, key, None)
            if val is not None:
                var.set(val)
        
        # 체크박스
        for key, var in self._check_vars.items():
            val = getattr(s, key, None)
            if val is not None:
                var.set(val)

    def sync_from_tv_response(self, response_data: Dict):
        """TV 응답 데이터로 UI 업데이트 (외부에서 호출)."""
        self.api.apply_response_settings(response_data)
        self._sync_ui_from_settings()
        self._log("Synced from TV")

    def update_connection_status(self, connected: bool):
        """연결 상태 UI 업데이트."""
        if connected:
            self._conn_label.config(text="● Connected", fg="green")
        else:
            self._conn_label.config(text="○ Not Connected", fg="gray")

    # ----------------------------------------------------------
    #  유틸리티
    # ----------------------------------------------------------

    def _log(self, msg: str):
        """상태 바 메시지 업데이트."""
        self._log_var.set(msg)
        logger.info(f"[TVPanel] {msg}")

    def show(self):
        """패널 표시 (독립 실행 시 mainloop)."""
        if self._own_root:
            self.root.mainloop()
        else:
            self.root.deiconify()

    def hide(self):
        """패널 숨기기."""
        self.root.withdraw()

    def destroy(self):
        """패널 완전 종료."""
        self.root.destroy()


# ============================================================
#  독립 실행
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    
    print("=" * 50)
    print("LG TV Picture Control Panel")
    print("=" * 50)
    print("NOTE: No WebSocket connected. Commands will be logged only.")
    print()
    
    api = TVControlAPI()
    
    # 디버깅용: 모든 명령 출력
    api.on_change(lambda name, val: print(f"  [API] {name} = {val}"))
    
    panel = TVControlPanel(api=api)
    panel.show()
