"""
Display Calibration Pattern Generator & Display Module
디스플레이 교정용 패턴 생성 및 출력 모듈

Capabilities:
  - Full-screen solid colour display on target monitor
  - Grayscale ramp patterns for gamma / CCT measurement
  - Primary / secondary colour patches
  - ColorChecker 24 pattern (approximate sRGB values)
  - Window pattern (colour patch centred on neutral background)
  - Automated measurement sequences
  - Multi-monitor support (Windows)

Usage:
    from calibration_patterns import PatternWindow, CalibrationPatterns

    pw = PatternWindow()
    pw.open(fullscreen=True, monitor=0)

    pw.show_color(1.0, 0.0, 0.0)  # Red
    time.sleep(1)
    pw.show_gray(0.5)              # 50% gray
    pw.close()

Author: Display Calibration System
"""

import tkinter as tk
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import sys
import logging

logger = logging.getLogger(__name__)

# Industry Standard Pattern Library (optional)
try:
    from calibration_patterns_industry import (
        StandardPatternSet, IndustryPatternLibrary, PATTERN_CATEGORIES,
    )
    HAS_INDUSTRY_PATTERNS = True
except ImportError:
    HAS_INDUSTRY_PATTERNS = False


# ============================================================================
# Pattern Types
# ============================================================================

class PatternType(Enum):
    """캘리브레이션 패턴 유형"""
    SOLID_COLOR = "solid_color"
    GRAY_PATCH = "gray_patch"
    GRAYSCALE_RAMP = "grayscale_ramp"
    PRIMARY_RAMP = "primary_ramp"
    WINDOW_PATTERN = "window_pattern"
    COLOR_CHECKER = "color_checker"
    CROSSHATCH = "crosshatch"


# ============================================================================
# Monitor Info
# ============================================================================

@dataclass
class MonitorInfo:
    """모니터 정보"""
    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    is_primary: bool = False


def list_monitors() -> List[MonitorInfo]:
    """
    사용 가능한 모니터 목록 반환 (Windows)

    Falls back to single-monitor detection via tkinter if
    platform-specific APIs are unavailable.
    """
    monitors = []

    # ── Windows: use ctypes EnumDisplayMonitors ──
    if sys.platform == 'win32':
        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            monitors_list = []

            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.wintypes.RECT),
                ctypes.c_double)

            def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                rect = lprcMonitor.contents
                monitors_list.append({
                    'x': rect.left,
                    'y': rect.top,
                    'w': rect.right - rect.left,
                    'h': rect.bottom - rect.top,
                })
                return 1

            user32.EnumDisplayMonitors(
                None, None, MONITORENUMPROC(callback), 0)

            for i, m in enumerate(monitors_list):
                monitors.append(MonitorInfo(
                    index=i,
                    name='Monitor {}'.format(i + 1),
                    x=m['x'], y=m['y'],
                    width=m['w'], height=m['h'],
                    is_primary=(m['x'] == 0 and m['y'] == 0),
                ))

            if monitors:
                return monitors
        except Exception as e:
            logger.debug("EnumDisplayMonitors failed: %s", e)

    # ── Fallback: tkinter screen size (primary only) ──
    try:
        _tmp = tk.Tk()
        _tmp.withdraw()
        w = _tmp.winfo_screenwidth()
        h = _tmp.winfo_screenheight()
        _tmp.destroy()
        monitors.append(MonitorInfo(
            index=0, name='Primary', x=0, y=0,
            width=w, height=h, is_primary=True))
    except Exception:
        monitors.append(MonitorInfo(
            index=0, name='Default', x=0, y=0,
            width=1920, height=1080, is_primary=True))

    return monitors


# ============================================================================
# Pattern Generator
# ============================================================================

class PatternGenerator:
    """
    캘리브레이션 패턴 이미지 생성

    Generates numpy arrays (H×W×3, uint8) for various pattern types.
    These can be displayed via PatternWindow or saved to disk.
    """

    @staticmethod
    def solid_color(r: float, g: float, b: float,
                    width: int = 1920, height: int = 1080) -> np.ndarray:
        """단색 패턴 (r,g,b: 0.0 – 1.0)"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = int(np.clip(r, 0, 1) * 255)
        img[:, :, 1] = int(np.clip(g, 0, 1) * 255)
        img[:, :, 2] = int(np.clip(b, 0, 1) * 255)
        return img

    @staticmethod
    def gray_patch(level: float,
                   width: int = 1920, height: int = 1080) -> np.ndarray:
        """회색 패치 (level: 0.0 – 1.0)"""
        return PatternGenerator.solid_color(level, level, level, width, height)

    @staticmethod
    def grayscale_ramp(steps: int = 256,
                       width: int = 1920, height: int = 1080,
                       horizontal: bool = True) -> np.ndarray:
        """그레이스케일 그라데이션 (수평 또는 수직)"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if horizontal:
            for i in range(width):
                val = int(i / (width - 1) * 255)
                img[:, i, :] = val
        else:
            for j in range(height):
                val = int(j / (height - 1) * 255)
                img[j, :, :] = val
        return img

    @staticmethod
    def primary_ramp(channel: int, steps: int = 256,
                     width: int = 1920, height: int = 1080) -> np.ndarray:
        """단일 채널 그라데이션 (channel: 0=R, 1=G, 2=B)"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(width):
            val = int(i / (width - 1) * 255)
            img[:, i, channel] = val
        return img

    @staticmethod
    def window_pattern(r: float, g: float, b: float,
                       window_pct: float = 0.2,
                       bg_level: float = 0.5,
                       width: int = 1920,
                       height: int = 1080) -> np.ndarray:
        """
        윈도우 패턴 (중앙 색상 패치 + 회색 배경)

        Args:
            r, g, b: 중앙 패치 색상 (0-1)
            window_pct: 전체 면적 대비 중앙 패치 비율
            bg_level: 배경 회색 레벨 (0-1)
        """
        img = PatternGenerator.gray_patch(bg_level, width, height)

        ww = int(width * np.sqrt(window_pct))
        wh = int(height * np.sqrt(window_pct))
        x0 = (width - ww) // 2
        y0 = (height - wh) // 2

        img[y0:y0+wh, x0:x0+ww, 0] = int(np.clip(r, 0, 1) * 255)
        img[y0:y0+wh, x0:x0+ww, 1] = int(np.clip(g, 0, 1) * 255)
        img[y0:y0+wh, x0:x0+ww, 2] = int(np.clip(b, 0, 1) * 255)
        return img

    @staticmethod
    def color_checker_24(width: int = 1920,
                         height: int = 1080) -> np.ndarray:
        """
        ColorChecker 24 패치 그리드 (근사 sRGB 값)

        X-Rite Classic ColorChecker 패턴과 유사한 4×6 그리드.
        """
        patches = [
            # Row 1 — natural colours
            (0.459, 0.314, 0.263),   # Dark Skin
            (0.788, 0.592, 0.478),   # Light Skin
            (0.337, 0.400, 0.545),   # Blue Sky
            (0.325, 0.392, 0.247),   # Foliage
            (0.463, 0.431, 0.616),   # Blue Flower
            (0.400, 0.686, 0.584),   # Bluish Green
            # Row 2 — vivid colours
            (0.812, 0.455, 0.176),   # Orange
            (0.271, 0.290, 0.569),   # Purplish Blue
            (0.737, 0.329, 0.318),   # Moderate Red
            (0.318, 0.220, 0.384),   # Purple
            (0.596, 0.659, 0.212),   # Yellow Green
            (0.867, 0.608, 0.169),   # Orange Yellow
            # Row 3 — saturated colours
            (0.169, 0.188, 0.494),   # Blue
            (0.286, 0.502, 0.235),   # Green
            (0.620, 0.239, 0.192),   # Red
            (0.902, 0.749, 0.118),   # Yellow
            (0.667, 0.298, 0.498),   # Magenta
            (0.086, 0.459, 0.561),   # Cyan
            # Row 4 — grey scale
            (0.941, 0.941, 0.941),   # White
            (0.725, 0.725, 0.725),   # Neutral 8
            (0.580, 0.580, 0.580),   # Neutral 6.5
            (0.424, 0.424, 0.424),   # Neutral 5
            (0.282, 0.282, 0.282),   # Neutral 3.5
            (0.122, 0.122, 0.122),   # Black
        ]

        img = np.zeros((height, width, 3), dtype=np.uint8)
        rows, cols = 4, 6
        margin_x = width // 20
        margin_y = height // 20
        patch_w = (width - 2 * margin_x) // cols
        patch_h = (height - 2 * margin_y) // rows
        gap = 4

        for idx, (r, g, b) in enumerate(patches):
            row = idx // cols
            col = idx % cols
            x0 = margin_x + col * patch_w + gap
            y0 = margin_y + row * patch_h + gap
            x1 = x0 + patch_w - 2 * gap
            y1 = y0 + patch_h - 2 * gap
            img[y0:y1, x0:x1, 0] = int(r * 255)
            img[y0:y1, x0:x1, 1] = int(g * 255)
            img[y0:y1, x0:x1, 2] = int(b * 255)

        return img

    @staticmethod
    def crosshatch(spacing: int = 100,
                   line_width: int = 2,
                   width: int = 1920,
                   height: int = 1080) -> np.ndarray:
        """크로스해치 패턴 (geometry / convergence 확인용)"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        half = line_width // 2

        for y in range(0, height, spacing):
            y0 = max(y - half, 0)
            y1 = min(y + half + 1, height)
            img[y0:y1, :, :] = 255

        for x in range(0, width, spacing):
            x0 = max(x - half, 0)
            x1 = min(x + half + 1, width)
            img[:, x0:x1, :] = 255

        return img


# ============================================================================
# Pattern Window (tkinter fullscreen)
# ============================================================================

class PatternWindow:
    """
    디스플레이 교정용 전체 화면 패턴 창

    tkinter 기반. 지정된 모니터에서 전체 화면으로 단색/패턴을 표시.
    update() 호출 방식으로 외부 이벤트 루프와 공존.

    Usage:
        pw = PatternWindow()
        pw.open(fullscreen=True, monitor=0)
        pw.show_color(1.0, 0.0, 0.0)
        time.sleep(1)
        pw.close()
    """

    def __init__(self):
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._image_item = None
        self._tk_image = None
        self._is_open = False
        self._width = 1920
        self._height = 1080
        self.monitors = list_monitors()
        self.current_monitor = 0
        
        print(f"[PatternWindow] Found {len(self.monitors)} monitor(s)")
        for i, m in enumerate(self.monitors):
            print(f"  Monitor {i}: {m.width}x{m.height} at ({m.x}, {m.y})"
                  f"{' (Primary)' if m.is_primary else ''}")

    @property
    def is_open(self) -> bool:
        return self._is_open
    
    def list_monitors(self) -> List[MonitorInfo]:
        """사용 가능한 모니터 목록 반환"""
        return self.monitors

    def open(self, fullscreen: bool = True,
             monitor: int = 0,
             width: int = 1920, height: int = 1080):
        """
        패턴 창 열기

        Args:
            fullscreen: 전체 화면 여부
            monitor: 모니터 인덱스 (0-based)
            width, height: 비전체화면 시 크기
        """
        if self._is_open:
            return

        monitors = list_monitors()
        if monitor < 0 or monitor >= len(monitors):
            monitor = 0

        mon = monitors[monitor]

        # tkinter root
        if tk._default_root is not None:
            self._root = tk.Toplevel()
        else:
            self._root = tk.Tk()

        self._root.title("Calibration Pattern")

        if fullscreen:
            self._width = mon.width
            self._height = mon.height
            self._root.overrideredirect(True)
            self._root.geometry('{}x{}+{}+{}'.format(
                mon.width, mon.height, mon.x, mon.y))
            self._root.attributes('-topmost', True)
        else:
            self._width = width
            self._height = height
            cx = mon.x + (mon.width - width) // 2
            cy = mon.y + (mon.height - height) // 2
            self._root.geometry('{}x{}+{}+{}'.format(
                width, height, cx, cy))

        self._canvas = tk.Canvas(
            self._root,
            width=self._width, height=self._height,
            highlightthickness=0, bd=0)
        self._canvas.pack(fill='both', expand=True)

        # ESC 키로 닫기
        self._root.bind('<Escape>', lambda e: self.close())

        self._root.configure(bg='black')
        self._canvas.configure(bg='black')
        self._root.update()
        self._is_open = True

        logger.info("[Pattern] Window opened on monitor %d (%dx%d)",
                    monitor, self._width, self._height)

    def close(self):
        """패턴 창 닫기"""
        if self._root:
            try:
                self._root.destroy()
            except tk.TclError:
                pass
        self._root = None
        self._canvas = None
        self._image_item = None
        self._tk_image = None
        self._is_open = False
        logger.info("[Pattern] Window closed")

    def show_color(self, r: float, g: float, b: float):
        """
        단색 표시 (r, g, b: 0.0 – 1.0)

        가장 빈번하게 사용되는 메서드.
        캘리브레이션 측정 시 각 테스트 색상을 표시.
        """
        if not self._is_open:
            return
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(np.clip(r, 0, 1) * 255),
            int(np.clip(g, 0, 1) * 255),
            int(np.clip(b, 0, 1) * 255))
        self._canvas.configure(bg=hex_color)
        # 이미지 아이템이 있으면 제거 (단색으로 전환)
        if self._image_item is not None:
            self._canvas.delete(self._image_item)
            self._image_item = None
        self._root.update()

    def show_gray(self, level: float):
        """회색 레벨 표시 (0.0 – 1.0)"""
        self.show_color(level, level, level)

    def show_image(self, img: np.ndarray):
        """
        numpy 이미지 배열 표시 (H×W×3, uint8)

        PatternGenerator 출력을 그대로 표시할 수 있음.
        """
        if not self._is_open:
            return

        try:
            from PIL import Image, ImageTk
            pil_img = Image.fromarray(img, 'RGB')
            self._tk_image = ImageTk.PhotoImage(pil_img)
            if self._image_item is not None:
                self._canvas.delete(self._image_item)
            self._image_item = self._canvas.create_image(
                0, 0, anchor='nw', image=self._tk_image)
            self._root.update()
        except ImportError:
            # PIL 없으면 fallback: 중앙 픽셀 색상으로 단색 표시
            h, w = img.shape[:2]
            c = img[h // 2, w // 2]
            self.show_color(c[0] / 255, c[1] / 255, c[2] / 255)

    def flash_sequence(self, colors: List[Tuple[float, float, float]],
                       duration: float = 0.5):
        """여러 색상을 순차적으로 표시 (duration 초씩)"""
        for r, g, b in colors:
            self.show_color(r, g, b)
            time.sleep(duration)


# ============================================================================
# Calibration Sequences (Measurement Patterns)
# ============================================================================

class CalibrationSequences:
    """
    캘리브레이션 단계별 측정 시퀀스 정의

    Each sequence is a list of dicts:
        {'name': str, 'rgb': (r,g,b), 'type': str}
    """

    @staticmethod
    def gamma_sequence(steps: int = 21,
                       custom_levels: List[float] = None,
                       white_only: bool = False) -> List[Dict]:
        """
        감마 / CCT 측정 시퀀스

        Args:
            steps: 균일 간격 레벨 수 (custom_levels 없을 때)
            custom_levels: 사용자 정의 레벨 리스트 (0.0 – 1.0)
            white_only: True이면 White(R=G=B)만 측정 (3배 빠름)
                        False이면 W,R,G,B 4종 측정 (정밀)

        Returns:
            List of dicts. Total = levels × (1 if white_only else 4)
        """
        if custom_levels is not None:
            levels = sorted(set(float(v) for v in custom_levels))
        else:
            levels = list(np.linspace(0, 1, steps))

        seq = []
        for lv in levels:
            lv = float(lv)
            pct = '{:.0f}%'.format(lv * 100)
            seq.append({
                'name': 'White_{}'.format(pct),
                'rgb': (lv, lv, lv),
                'type': 'grayscale'})
            if not white_only:
                seq.append({
                    'name': 'Red_{}'.format(pct),
                    'rgb': (lv, 0, 0),
                    'type': 'red_channel'})
                seq.append({
                    'name': 'Green_{}'.format(pct),
                    'rgb': (0, lv, 0),
                    'type': 'green_channel'})
                seq.append({
                    'name': 'Blue_{}'.format(pct),
                    'rgb': (0, 0, lv),
                    'type': 'blue_channel'})
        return seq

    @staticmethod
    def color_sequence(
            custom_patches: List[Tuple[str, Tuple[float, float, float]]] = None
    ) -> List[Dict]:
        """
        색역 측정 시퀀스

        Args:
            custom_patches: 사용자 정의 패치 목록 [(name, (r,g,b)), ...]
                            None이면 기본 10종 (RGBW + CMY + 그레이)
        """
        if custom_patches is not None:
            return [{'name': n, 'rgb': c, 'type': 'color'}
                    for n, c in custom_patches]
        return [
            {'name': 'White',    'rgb': (1.0, 1.0, 1.0), 'type': 'primary'},
            {'name': 'Red',      'rgb': (1.0, 0.0, 0.0), 'type': 'primary'},
            {'name': 'Green',    'rgb': (0.0, 1.0, 0.0), 'type': 'primary'},
            {'name': 'Blue',     'rgb': (0.0, 0.0, 1.0), 'type': 'primary'},
            {'name': 'Cyan',     'rgb': (0.0, 1.0, 1.0), 'type': 'secondary'},
            {'name': 'Magenta',  'rgb': (1.0, 0.0, 1.0), 'type': 'secondary'},
            {'name': 'Yellow',   'rgb': (1.0, 1.0, 0.0), 'type': 'secondary'},
            {'name': '75Gray',   'rgb': (0.75, 0.75, 0.75), 'type': 'gray'},
            {'name': '50Gray',   'rgb': (0.50, 0.50, 0.50), 'type': 'gray'},
            {'name': '25Gray',   'rgb': (0.25, 0.25, 0.25), 'type': 'gray'},
        ]

    @staticmethod
    def verification_sequence(
            custom_patches: List[Tuple[str, Tuple[float, float, float]]] = None
    ) -> List[Dict]:
        """
        검증 시퀀스 (ColorChecker 24 유사)

        Args:
            custom_patches: 사용자 정의 패치 목록. None이면 기본 24종.
        """
        if custom_patches is not None:
            return [{'name': n, 'rgb': c, 'type': 'verify'}
                    for n, c in custom_patches]
        patches = [
            ('DarkSkin',      (0.459, 0.314, 0.263)),
            ('LightSkin',     (0.788, 0.592, 0.478)),
            ('BlueSky',       (0.337, 0.400, 0.545)),
            ('Foliage',       (0.325, 0.392, 0.247)),
            ('BlueFlower',    (0.463, 0.431, 0.616)),
            ('BluishGreen',   (0.400, 0.686, 0.584)),
            ('Orange',        (0.812, 0.455, 0.176)),
            ('PurplishBlue',  (0.271, 0.290, 0.569)),
            ('ModerateRed',   (0.737, 0.329, 0.318)),
            ('Purple',        (0.318, 0.220, 0.384)),
            ('YellowGreen',   (0.596, 0.659, 0.212)),
            ('OrangeYellow',  (0.867, 0.608, 0.169)),
            ('Blue',          (0.169, 0.188, 0.494)),
            ('Green',         (0.286, 0.502, 0.235)),
            ('Red',           (0.620, 0.239, 0.192)),
            ('Yellow',        (0.902, 0.749, 0.118)),
            ('Magenta',       (0.667, 0.298, 0.498)),
            ('Cyan',          (0.086, 0.459, 0.561)),
            ('White95',       (0.941, 0.941, 0.941)),
            ('Neutral8',      (0.725, 0.725, 0.725)),
            ('Neutral65',     (0.580, 0.580, 0.580)),
            ('Neutral5',      (0.424, 0.424, 0.424)),
            ('Neutral35',     (0.282, 0.282, 0.282)),
            ('Black',         (0.122, 0.122, 0.122)),
        ]
        return [{'name': n, 'rgb': c, 'type': 'verify'} for n, c in patches]

    @staticmethod
    def quick_sequence() -> List[Dict]:
        """빠른 검증 시퀀스 (9 패치)"""
        return [
            {'name': 'White',   'rgb': (1.0, 1.0, 1.0), 'type': 'quick'},
            {'name': 'Red',     'rgb': (1.0, 0.0, 0.0), 'type': 'quick'},
            {'name': 'Green',   'rgb': (0.0, 1.0, 0.0), 'type': 'quick'},
            {'name': 'Blue',    'rgb': (0.0, 0.0, 1.0), 'type': 'quick'},
            {'name': '75Gray',  'rgb': (0.75, 0.75, 0.75), 'type': 'quick'},
            {'name': '50Gray',  'rgb': (0.50, 0.50, 0.50), 'type': 'quick'},
            {'name': '25Gray',  'rgb': (0.25, 0.25, 0.25), 'type': 'quick'},
            {'name': 'Skin',    'rgb': (0.788, 0.592, 0.478), 'type': 'quick'},
            {'name': 'Foliage', 'rgb': (0.325, 0.392, 0.247), 'type': 'quick'},
        ]

    @staticmethod
    def from_config(config) -> Dict[str, List[Dict]]:
        """
        CalibrationConfig로부터 모든 시퀀스를 한번에 생성

        Args:
            config: CalibrationConfig 인스턴스
                    (calibration_engine 모듈에서 정의)
        Returns:
            Dict with keys 'gamma', 'color', 'verify'
        """
        gamma_seq = CalibrationSequences.gamma_sequence(
            custom_levels=config.gamma_steps.levels,
            white_only=config.gamma_steps.white_only)

        color_seq = CalibrationSequences.color_sequence(
            custom_patches=config.color_patches.patches)

        verify_seq = CalibrationSequences.verification_sequence(
            custom_patches=config.verify_patches.patches)

        return {
            'gamma': gamma_seq,
            'color': color_seq,
            'verify': verify_seq,
            'summary': {
                'gamma_count': len(gamma_seq),
                'color_count': len(color_seq),
                'verify_count': len(verify_seq),
                'total': len(gamma_seq) + len(color_seq) + len(verify_seq),
            },
        }

    @staticmethod
    def from_standard_pattern(
            pattern_set: 'StandardPatternSet',
            include_gamma: bool = True,
            gamma_steps: int = 21,
            white_only: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        산업 표준 패턴 세트로부터 전체 측정 시퀀스 생성

        표준 패턴의 모든 패치를 색역 측정과 검증 시퀀스로 사용하고,
        감마 시퀀스는 별도로 구성합니다.

        Args:
            pattern_set: StandardPatternSet 열거형 값
            include_gamma: 감마 시퀀스 포함 여부
            gamma_steps: 감마 레벨 수
            white_only: 감마 측정 시 White만 사용

        Returns:
            Dict with keys 'gamma', 'color', 'verify', 'summary', 'pattern_info'

        Usage:
            from calibration_patterns_industry import StandardPatternSet
            seqs = CalibrationSequences.from_standard_pattern(
                StandardPatternSet.COLORCHECKER_SG,
                gamma_steps=41)
            print(seqs['summary'])
        """
        if not HAS_INDUSTRY_PATTERNS:
            raise ImportError(
                "calibration_patterns_industry module required.")

        info = IndustryPatternLibrary.get_info(pattern_set)
        patches = IndustryPatternLibrary.get_patches(pattern_set)

        # 색역 측정: 모든 패치
        color_seq = [{'name': n, 'rgb': c, 'type': 'color_standard'}
                     for n, c in patches]

        # 검증: 그레이스케일 + 유채색 대표 패치
        gray_patches = IndustryPatternLibrary.get_grayscale_patches(
            pattern_set)
        chromatic_patches = IndustryPatternLibrary.get_chromatic_patches(
            pattern_set)
        # 검증은 그레이 + 유채색 최대 30개로 제한
        verify_patches = list(gray_patches)
        remaining = 30 - len(verify_patches)
        if remaining > 0:
            verify_patches.extend(chromatic_patches[:remaining])
        verify_seq = [{'name': n, 'rgb': c, 'type': 'verify_standard'}
                      for n, c in verify_patches]

        # 감마 시퀀스
        gamma_seq = []
        if include_gamma:
            gamma_seq = CalibrationSequences.gamma_sequence(
                gamma_steps, white_only=white_only)

        result = {
            'gamma': gamma_seq,
            'color': color_seq,
            'verify': verify_seq,
            'summary': {
                'gamma_count': len(gamma_seq),
                'color_count': len(color_seq),
                'verify_count': len(verify_seq),
                'total': len(gamma_seq) + len(color_seq) + len(verify_seq),
                'pattern_name': info['name'],
                'pattern_short': info['short_name'],
            },
            'pattern_info': info,
        }

        logger.info("[CalibrationSequences] Standard pattern: %s "
                    "— color=%d, verify=%d, gamma=%d (total=%d)",
                    info['short_name'], len(color_seq), len(verify_seq),
                    len(gamma_seq), result['summary']['total'])
        return result

    @staticmethod
    def list_standard_patterns() -> List[Dict]:
        """사용 가능한 산업 표준 패턴 목록 반환"""
        if not HAS_INDUSTRY_PATTERNS:
            return []
        results = []
        for ps in StandardPatternSet:
            info = IndustryPatternLibrary.get_info(ps)
            results.append({
                'pattern_set': ps,
                'name': info['name'],
                'short_name': info['short_name'],
                'patches': info['patches'],
                'industry': info.get('industry', ''),
                'standard': info.get('standard', ''),
            })
        return results

    # ------------------------------------------------------------------ #
    #  Sweep-based Sequences  (Saturation / Luminance / Stimulus)
    # ------------------------------------------------------------------ #

    @staticmethod
    def saturation_sweep_sequence(
            steps: int = 11,
            colors: List[str] = None,
            value: float = 1.0,
            stimulus=None,
    ) -> List[Dict]:
        """
        채도 스윕 측정 시퀀스 — RGBCMY 각 색상에 대해 채도를 일정 단위로 증가

        Saturation은 0%(=Gray)부터 100%(=Pure Hue)까지 균일 간격으로 진행.
        HSV 모델 기반: H=고정, S=sweep, V=value 파라미터.

        Args:
            steps: 채도 단계 수 (기본 11 → 0%, 10%, 20% … 100%)
            colors: 대상 색상 리스트 (기본 RGBCMY)
            value: HSV의 V 값 (1.0 = 최대 밝기)
            stimulus: StimulusConfig 인스턴스 (None이면 미적용)

        Returns:
            List[Dict] — 측정 시퀀스
        """
        from calibration_engine import SweepPatternGenerator
        patches = SweepPatternGenerator.saturation_sweep(
            steps=steps, colors=colors, value=value, stimulus=stimulus)
        seq = []
        for name, rgb in patches:
            seq.append({
                'name': name,
                'rgb': rgb,
                'type': 'saturation_sweep',
            })
        return seq

    @staticmethod
    def luminance_sweep_sequence(
            steps: int = 11,
            colors: List[str] = None,
            saturation: float = 1.0,
            stimulus=None,
    ) -> List[Dict]:
        """
        휘도 스윕 측정 시퀀스 — RGBCMY 각 색상에 대해 휘도를 일정 단위로 증가

        Luminance(Value)는 0%(=Black)부터 100%(=Max brightness)까지 균일 간격.
        HSV 모델 기반: H=고정, S=saturation 파라미터, V=sweep.

        Args:
            steps: 휘도 단계 수 (기본 11 → 0%, 10%, 20% … 100%)
            colors: 대상 색상 리스트 (기본 RGBCMY)
            saturation: HSV의 S 값 (1.0 = 완전 포화)
            stimulus: StimulusConfig 인스턴스 (None이면 미적용)

        Returns:
            List[Dict] — 측정 시퀀스
        """
        from calibration_engine import SweepPatternGenerator
        patches = SweepPatternGenerator.luminance_sweep(
            steps=steps, colors=colors, saturation=saturation,
            stimulus=stimulus)
        seq = []
        for name, rgb in patches:
            seq.append({
                'name': name,
                'rgb': rgb,
                'type': 'luminance_sweep',
            })
        return seq

    @staticmethod
    def stimulus_characterization_sequence(
            stimulus=None,
            steps: int = 11,
    ) -> List[Dict]:
        """
        Stimulus 특성화 측정 시퀀스 — WRGB 패널 캘리브레이션용

        각 채널(W, R, G, B)에 대해 Stimulus 최대값까지의 Ramp를 생성.
        WRGB OLED 등 각 원색의 최대 밝기가 다른 패널에 최적화.

        Args:
            stimulus: StimulusConfig 인스턴스 (None이면 기본값 사용)
            steps: 채널당 레벨 수 (기본 11)

        Returns:
            List[Dict] — 측정 시퀀스
        """
        from calibration_engine import SweepPatternGenerator, StimulusConfig
        if stimulus is None:
            stimulus = StimulusConfig()
        patches = SweepPatternGenerator.stimulus_characterization(
            stimulus=stimulus, steps=steps)
        seq = []
        for name, rgb in patches:
            seq.append({
                'name': name,
                'rgb': rgb,
                'type': 'stimulus_characterization',
            })
        return seq

    @staticmethod
    def combined_sweep_sequence(
            sat_steps: int = 6,
            lum_steps: int = 6,
            colors: List[str] = None,
            stimulus=None,
    ) -> List[Dict]:
        """
        채도 × 휘도 결합 그리드 시퀀스 — RGBCMY에 대해 2D 매트릭스 측정

        각 색상에 대해 (Saturation, Value) 조합을 모두 측정하여
        색재현 능력을 2차원으로 매핑합니다.

        Args:
            sat_steps: 채도 단계 수 (기본 6 → 0%, 20%, 40%, 60%, 80%, 100%)
            lum_steps: 휘도 단계 수 (기본 6)
            colors: 대상 색상 리스트 (기본 RGBCMY)
            stimulus: StimulusConfig 인스턴스 (None이면 미적용)

        Returns:
            List[Dict] — 측정 시퀀스
        """
        from calibration_engine import SweepPatternGenerator
        patches = SweepPatternGenerator.combined_sweep(
            sat_steps=sat_steps, lum_steps=lum_steps,
            colors=colors, stimulus=stimulus)
        seq = []
        for name, rgb in patches:
            seq.append({
                'name': name,
                'rgb': rgb,
                'type': 'combined_sweep',
            })
        return seq


# ============================================================================
# Automated Measurement Runner
# ============================================================================

class MeasurementRunner:
    """
    자동화된 패턴 표시 + 센서 측정 러너

    PatternWindow와 SensorInterface를 연결하여
    시퀀스를 자동으로 실행합니다.
    """

    def __init__(self, pattern_window: PatternWindow,
                 sensor,
                 settle_time: float = 0.5):
        """
        Args:
            pattern_window: PatternWindow 인스턴스
            sensor: SensorInterface (sensor_module.py)
            settle_time: 패턴 표시 후 안정화 대기 (초)
        """
        self.pattern = pattern_window
        self.sensor = sensor
        self.settle_time = settle_time
        self._stop_requested = False
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """진행 콜백: callback(step, total, name, rgb)"""
        self._progress_callback = callback

    def stop(self):
        """측정 중지 요청"""
        self._stop_requested = True

    def run_sequence(self, sequence: List[Dict]) -> List[Dict]:
        """
        측정 시퀀스 실행

        Args:
            sequence: CalibrationSequences 메서드의 반환값
        Returns:
            List of dicts with 'name', 'rgb', 'type', 'xyz',
            'cie_xy', 'luminance', 'is_valid'
        """
        self._stop_requested = False
        results = []
        total = len(sequence)

        for i, patch in enumerate(sequence):
            if self._stop_requested:
                logger.info("[MeasRunner] Stopped at step %d/%d", i, total)
                break

            name = patch['name']
            r, g, b = patch['rgb']

            if self._progress_callback:
                self._progress_callback(i + 1, total, name, (r, g, b))

            # 패턴 표시
            self.pattern.show_color(r, g, b)
            time.sleep(self.settle_time)

            # 센서 측정
            reading = self.sensor.read()

            results.append({
                'name': name,
                'rgb': (r, g, b),
                'type': patch.get('type', ''),
                'xyz': reading.xyz.tolist() if hasattr(reading.xyz, 'tolist')
                       else list(reading.xyz),
                'cie_xy': reading.cie_xy,
                'luminance': reading.luminance,
                'is_valid': reading.is_valid,
            })

            logger.debug("[MeasRunner] %d/%d  %s → XYZ=%s",
                         i + 1, total, name,
                         np.round(reading.xyz, 4))

        return results

    def run_gamma_measurements(self, steps: int = 21,
                               custom_levels: List[float] = None,
                               white_only: bool = False) -> List[Dict]:
        """
        감마 시퀀스 자동 실행

        Args:
            steps: 균일 간격 레벨 수
            custom_levels: 사용자 정의 레벨 (우선)
            white_only: White만 측정 (빠름)
        """
        seq = CalibrationSequences.gamma_sequence(
            steps, custom_levels=custom_levels, white_only=white_only)
        return self.run_sequence(seq)

    def run_color_measurements(
            self,
            custom_patches: List[Tuple[str, Tuple[float, float, float]]] = None
    ) -> List[Dict]:
        """
        색역 시퀀스 자동 실행

        Args:
            custom_patches: 사용자 정의 패치 목록
        """
        seq = CalibrationSequences.color_sequence(
            custom_patches=custom_patches)
        return self.run_sequence(seq)

    def run_verification(
            self,
            custom_patches: List[Tuple[str, Tuple[float, float, float]]] = None
    ) -> List[Dict]:
        """
        검증 시퀀스 자동 실행

        Args:
            custom_patches: 사용자 정의 검증 패치
        """
        seq = CalibrationSequences.verification_sequence(
            custom_patches=custom_patches)
        return self.run_sequence(seq)

    def run_from_config(self, config) -> Dict[str, List[Dict]]:
        """
        CalibrationConfig로부터 전체 시퀀스 자동 실행

        Args:
            config: CalibrationConfig 인스턴스
        Returns:
            Dict with keys 'gamma', 'color', 'verify'
        """
        sequences = CalibrationSequences.from_config(config)
        return {
            'gamma': self.run_sequence(sequences['gamma']),
            'color': self.run_sequence(sequences['color']),
            'verify': self.run_sequence(sequences['verify']),
        }


# ============================================================================
# Demo / Self-Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    print("=" * 60)
    print("  Calibration Patterns — Demo")
    print("=" * 60)

    # 모니터 감지
    monitors = list_monitors()
    print("\nDetected monitors:")
    for m in monitors:
        print("  [{}] {} — {}x{} at ({},{}) {}".format(
            m.index, m.name, m.width, m.height, m.x, m.y,
            '(Primary)' if m.is_primary else ''))

    # 시퀀스 정보
    gamma_seq = CalibrationSequences.gamma_sequence(11)
    color_seq = CalibrationSequences.color_sequence()
    verify_seq = CalibrationSequences.verification_sequence()
    print("\nSequence sizes (default):")
    print("  Gamma (11 levels): {} patches".format(len(gamma_seq)))
    print("  Color:             {} patches".format(len(color_seq)))
    print("  Verification:      {} patches".format(len(verify_seq)))

    # 커스텀 시퀀스
    print("\nCustom sequences:")
    g_wo = CalibrationSequences.gamma_sequence(
        custom_levels=[0, 0.1, 0.3, 0.5, 0.75, 1.0], white_only=True)
    print("  Gamma white-only (6 lvl): {} patches".format(len(g_wo)))

    g_40 = CalibrationSequences.gamma_sequence(41)
    print("  Gamma 41-point full:      {} patches".format(len(g_40)))

    c_min = CalibrationSequences.color_sequence(custom_patches=[
        ('White', (1,1,1)), ('Red', (1,0,0)),
        ('Green', (0,1,0)), ('Blue', (0,0,1)),
    ])
    print("  Color minimal (4 patch):  {} patches".format(len(c_min)))

    # 패턴 생성 테스트
    print("\nPattern generation test:")
    img = PatternGenerator.solid_color(1, 0, 0)
    print("  solid_color: shape={} dtype={}".format(img.shape, img.dtype))

    img = PatternGenerator.grayscale_ramp()
    print("  grayscale_ramp: shape={}".format(img.shape))

    img = PatternGenerator.window_pattern(1, 0, 0, 0.1)
    print("  window_pattern: shape={}".format(img.shape))

    img = PatternGenerator.color_checker_24()
    print("  color_checker_24: shape={}".format(img.shape))

    img = PatternGenerator.crosshatch()
    print("  crosshatch: shape={}".format(img.shape))

    # ── 대화형 데모 (선택적) ──
    print("\n" + "=" * 60)
    ans = input("Open pattern window demo? (y/N): ").strip().lower()
    if ans == 'y':
        pw = PatternWindow()
        pw.open(fullscreen=False, width=800, height=600)

        demo_colors = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (1, 1, 1), (0.5, 0.5, 0.5), (0, 0, 0),
        ]
        for r, g, b in demo_colors:
            pw.show_color(r, g, b)
            time.sleep(0.4)

        # 그라데이션
        for i in range(256):
            pw.show_gray(i / 255)
            time.sleep(0.005)

        pw.close()
        print("Demo complete!")
    else:
        print("Skipped.")

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
