"""
core/pattern_source.py — 통합 패턴 표시 API (Unified pattern-display API).

하나의 인터페이스(show_color/show_image/last_color/showing)로 요청한 색을
선택된 백엔드에 라우팅한다. 호출부(캘리브레이션 런너 / 측정 루프)는 어느
디스플레이로 나가는지 모른 채 show_color()만 부르면 되고, 모듈 내부에서
'내부' / '외부' 백엔드를 결정한다.

용어 (사용자 정의)
    내부 패턴 (INTERNAL) : 앱이 별도로 띄우는 풀스크린 Window. (이미 구현돼 있던 것 —
                            core.pattern_display.PatternDisplayProxy 를 위임)
    외부 패턴 (EXTERNAL) : DaVinci 같은 외부 소스 제어. solid PNG → remote_control.upload
                            → Mac DaVinci → HDMI → TV.

이 duck-type 인터페이스는 PatternDisplayProxy 와 동일하므로
CalibrationRunner(pattern_proxy=...) / CalibrationWorkflow(pattern_window=...) 에
그대로 꽂힌다 — 런너/엔진 수정 불필요.

측정(sensor)과 TV 제어(webOS 설정/LUT)는 *별개 모듈*이다
(sensor_module / core.sensor_manager / core.tv_control). 이 모듈은 "알려진 색을
디스플레이에 띄우고, 화면에 떠서 안정될 때까지 블로킹"만 책임진다.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

# export_pkg 를 import 경로에 추가 (davinci 패키지 접근). 문서화된 사용법:
# sys.path.insert(0, ".../export_pkg")
_EXPORT_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "export_pkg")
if os.path.isdir(_EXPORT_PKG) and _EXPORT_PKG not in sys.path:
    sys.path.insert(0, _EXPORT_PKG)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


class PatternTarget(str, Enum):
    """패턴이 실제로 표시되는 대상."""
    INTERNAL = "internal"   # 내부 패턴: 앱이 띄우는 별도 Window
    EXTERNAL = "external"   # 외부 패턴: DaVinci 등 외부 소스 → HDMI → TV


# ─────────────────────────────────────────────────────────────
#  백엔드 (Strategy)
# ─────────────────────────────────────────────────────────────
class PatternBackend(ABC):
    """패턴 표시 백엔드 공통 계약.

    show_color() 는 *블로킹*이어야 한다 — 반환 시점에 패턴이 화면에 떠서
    settle 까지 끝났다고 호출부(측정 루프)가 가정한다.
    """
    name: str = "base"

    @abstractmethod
    def show_color(self, r: float, g: float, b: float) -> None:
        """0..1 float RGB 단색 패치를 표시하고 화면에 뜰 때까지 블로킹."""

    def show_image(self, img) -> None:  # 선택적
        raise NotImplementedError(f"{self.name} backend does not support show_image")

    def set_patch_size(self, fraction: float) -> None:  # 선택적
        pass

    def close(self) -> None:  # 선택적
        pass


class InternalWindowBackend(PatternBackend):
    """내부 패턴: 앱이 띄우는 별도 풀스크린 Qt Window.

    기존 core.pattern_display.PatternDisplayProxy 를 그대로 위임한다.
    proxy.show_color 는 워커 스레드에서 BlockingQueuedConnection 으로
    repaint 완료까지 블로킹하므로 settle 보장은 proxy 가 처리한다.
    """
    name = "internal"

    def __init__(self, proxy):
        # proxy: core.pattern_display.PatternDisplayProxy (또는 동일 duck-type)
        self._proxy = proxy

    def show_color(self, r: float, g: float, b: float) -> None:
        self._proxy.show_color(r, g, b)

    def show_image(self, img) -> None:
        if hasattr(self._proxy, "show_image"):
            self._proxy.show_image(img)

    def set_patch_size(self, fraction: float) -> None:
        if hasattr(self._proxy, "set_patch_size"):
            self._proxy.set_patch_size(fraction)

    def close(self) -> None:
        if hasattr(self._proxy, "close"):
            self._proxy.close()


class DavinciBackend(PatternBackend):
    """외부 패턴: DaVinci/Mac → HDMI → TV (외부 소스 체인 전체를 측정 대상으로).

    단색 PNG 를 생성해 remote_control.upload() 로 Mac 에 보낸다. 첫 프레임은
    import+append, 이후는 replace=True(ReplaceClip, 제자리 갱신)로 실시간 루프.
    16-bit PNG 는 10-bit 코드값을 full-range 스케일(round(code*65535/1023))로
    써서 float 정규화 NLE 체인에서 정확히 round-trip 되게 한다(생성기와 동일 관례).

    host/port 기본값은 remote_control 의 기본(192.168.0.3:7777)과 동일. host 를
    넘기면 remote_control 모듈 전역(HOST/PORT/BASE)을 갱신해 대상 Mac 을 바꾼다.
    """
    name = "davinci"

    def __init__(self, resolution: Tuple[int, int] = (256, 256), bit_depth: int = 16,
                 settle: float = 0.5, tmp_path: Optional[str] = None,
                 host: str = "192.168.0.3", port: int = 7777):
        self._w, self._h = resolution
        self._bit = bit_depth
        self._settle = settle
        self._tmp = tmp_path or os.path.join(tempfile.gettempdir(), "pattern_patch.png")
        self._uploaded = False
        from davinci import remote_control as _rc  # export_pkg
        # upload() 가 읽는 모듈 전역을 직접 갱신 (BASE 는 import 시점 고정이므로).
        _rc.HOST = host
        _rc.PORT = int(port)
        _rc.BASE = f"http://{host}:{int(port)}"
        self._rc = _rc

    def _write_solid(self, r: float, g: float, b: float) -> None:
        import numpy as np
        import cv2
        if self._bit >= 16:
            def code(v):  # 0..1 → 10-bit → 16-bit full-range
                return int(round(round(_clamp01(v) * 1023) * 65535 / 1023))
            bgr = (code(b), code(g), code(r))
            img = np.empty((self._h, self._w, 3), dtype=np.uint16)
        else:
            def code(v):
                return int(round(_clamp01(v) * 255))
            bgr = (code(b), code(g), code(r))
            img = np.empty((self._h, self._w, 3), dtype=np.uint8)
        img[:, :] = bgr  # BGR (OpenCV 순서)
        cv2.imwrite(self._tmp, img)

    def show_color(self, r: float, g: float, b: float) -> None:
        self._write_solid(r, g, b)
        self._send(self._tmp)
        time.sleep(self._settle)

    def show_image(self, img) -> None:
        """임의 패턴 numpy 배열(BGR)을 그대로 TV 로 전송."""
        import cv2
        cv2.imwrite(self._tmp, img)
        self._send(self._tmp)
        time.sleep(self._settle)

    def _send(self, path: str) -> None:
        status, resp = self._rc.upload(path, replace=self._uploaded)
        if status == 200 and isinstance(resp, dict) and resp.get("ok"):
            self._uploaded = True
        else:
            logger.warning("[davinci] upload failed: status=%s resp=%s", status, resp)


# ─────────────────────────────────────────────────────────────
#  통합 API
# ─────────────────────────────────────────────────────────────
class PatternSource(QObject):
    """하나의 패턴 표시 API. 활성 백엔드로 show_color 를 라우팅한다.

    PatternDisplayProxy 와 동일한 duck-type(show_color/last_color/showing)을
    노출하므로 캘리브레이션 런너/워크플로우에 그대로 주입할 수 있다.

    사용:
        ps = build_pattern_source(internal_proxy=proxy, davinci_host="192.168.0.3")
        runner = CalibrationRunner(sensor_manager, pattern_proxy=ps, mode="measurement")
        ps.set_active(PatternTarget.INTERNAL)   # 내부 패턴(앱 Window)으로 전환
        ps.set_active(PatternTarget.EXTERNAL)   # 외부 패턴(DaVinci)으로 전환
    """

    # 색 요청 즉시(백엔드가 실제로 그리기 전에) 방출 → UI 스와치 lock-step.
    showing = Signal(tuple)         # (r, g, b)
    target_changed = Signal(str)    # 활성 대상 변경

    def __init__(self, backends: Dict[PatternTarget, PatternBackend],
                 active: PatternTarget):
        super().__init__()
        if not backends:
            raise ValueError("PatternSource requires at least one backend")
        if active not in backends:
            raise ValueError(f"active target {active} not in backends {list(backends)}")
        self._backends: Dict[PatternTarget, PatternBackend] = dict(backends)
        self._active = active
        self._last_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── 대상 선택 ────────────────────────────────────────────
    @property
    def active(self) -> PatternTarget:
        return self._active

    @property
    def backend(self) -> PatternBackend:
        return self._backends[self._active]

    @property
    def targets(self):
        return tuple(self._backends.keys())

    def set_active(self, target: PatternTarget) -> None:
        if target not in self._backends:
            raise ValueError(f"unknown pattern target: {target}")
        if target != self._active:
            self._active = target
            self.target_changed.emit(target.value)

    # ── PatternDisplayProxy 호환 인터페이스 ──────────────────
    def show_color(self, r: float, g: float, b: float) -> None:
        self._last_color = (float(r), float(g), float(b))
        self.showing.emit(self._last_color)
        self.backend.show_color(float(r), float(g), float(b))

    @property
    def last_color(self) -> Tuple[float, float, float]:
        return self._last_color

    def show_gray(self, level: float) -> None:
        self.show_color(level, level, level)

    def show_image(self, img) -> None:
        self.backend.show_image(img)

    def set_patch_size(self, fraction: float) -> None:
        self.backend.set_patch_size(fraction)

    def close(self) -> None:
        for b in self._backends.values():
            try:
                b.close()
            except Exception as e:  # 백엔드 하나 실패가 나머지 정리를 막지 않게
                logger.warning("[pattern_source] backend %s close failed: %s",
                               getattr(b, "name", "?"), e)


def build_pattern_source(internal_proxy=None, *,
                         davinci_host: str = "192.168.0.3", davinci_port: int = 7777,
                         davinci_kwargs: Optional[dict] = None,
                         active: PatternTarget = PatternTarget.EXTERNAL) -> PatternSource:
    """내부(앱 Window) + 외부(DaVinci) 백엔드를 묶은 PatternSource 를 만든다.

    internal_proxy: core.pattern_display.PatternDisplayProxy (없으면 내부 백엔드 생략).
    davinci_host/port: 외부 소스(Mac) 대상. 기본 192.168.0.3:7777.
    """
    backends: Dict[PatternTarget, PatternBackend] = {}
    if internal_proxy is not None:
        backends[PatternTarget.INTERNAL] = InternalWindowBackend(internal_proxy)
    backends[PatternTarget.EXTERNAL] = DavinciBackend(
        host=davinci_host, port=davinci_port, **(davinci_kwargs or {}))
    if active not in backends:
        active = next(iter(backends))
    return PatternSource(backends, active=active)
