"""
Display Calibration UI — Professional Calibration Workflow
전문 디스플레이 캘리브레이션 워크플로우 UI

Features:
  ✓ 3-Phase Automatic Calibration (Grayscale → Color Gamut → Verification)
  ✓ Industry Standard Pattern Sets (ColorChecker, SMPTE, etc.)
  ✓ Real-time Progress Monitoring
  ✓ Before/After Comparison
  ✓ LUT Export (.cube, .csv)
  ✓ Comprehensive Report Generation

Shared Modules:
  - sensor_module: 센서 인터페이스
  - calibration_engine: 캘리브레이션 알고리즘
  - calibration_patterns_industry: 산업 표준 패턴
  - display_components: 공통 UI 컴포넌트

Usage:
    from calibration_ui import CalibrationUI
    
    # Option 1: 독립 실행
    cal_ui = CalibrationUI()
    cal_ui.show()
    
    # Option 2: 기존 센서 전달
    from sensor_module import CRColorimeterSensor
    sensor = CRColorimeterSensor(port='COM3')
    cal_ui = CalibrationUI(sensor=sensor)
    cal_ui.show()

Author: Display Calibration System
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import threading
from queue import Queue
import logging
import json

# 공통 모듈
from sensor_module import VirtualSensor, CRColorimeterSensor, SensorReading
from calibration_patterns import PatternWindow  # 패턴 윈도우 제어
from calibration_engine import (
    CalibrationWorkflow, CalibrationConfig, CalibrationPreset,
    WorkflowConfig, WorkflowPhase, StandardPatternSet,
    SignalRange, ColorEncoding, LUT3DGammaMode,
    DeltaE, ColorScience,
    GammaCalibrator, ColorGamutCalibrator,
    CalibrationAnalyzer, LUTExporter,
    CalibrationResult, ColorPatchMeasurement,
    TARGET_STANDARDS,
)

# Industry Pattern Library
try:
    from calibration_patterns_industry import (
        StandardPatternSet as PatternSet,
        IndustryPatternLibrary,
        PATTERN_CATEGORIES,
        PATTERN_METADATA
    )
    HAS_INDUSTRY_PATTERNS = True
except ImportError:
    HAS_INDUSTRY_PATTERNS = False
    print("[Warning] calibration_patterns_industry not available")

logger = logging.getLogger(__name__)


# ============================================================================
# Calibration State & Progress
# ============================================================================

class CalibrationState(Enum):
    """캘리브레이션 상태"""
    IDLE = "idle"
    SETUP = "setup"
    MEASURING = "measuring"
    COMPUTING = "computing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MeasurementProgress:
    """측정 진행 상황"""
    current_phase: WorkflowPhase = WorkflowPhase.PHASE1_GRAYSCALE
    total_patches: int = 0
    measured_patches: int = 0
    current_patch_name: str = ""
    current_rgb: Tuple[float, float, float] = (0, 0, 0)
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        if self.total_patches == 0:
            return 0.0
        return (self.measured_patches / self.total_patches) * 100


# ============================================================================
# Pattern Gallery Window
# ============================================================================

class PatternGalleryWindow:
    """패턴 선택 갤러리 창 - 작은 미리보기로 패턴 선택"""
    
    def __init__(self, parent_ui, available_patterns):
        self.parent_ui = parent_ui
        self.available_patterns = available_patterns
        self.selected_pattern = None
        
        logger.info("[Pattern Gallery] =" * 30)
        logger.info("[Pattern Gallery] Opening gallery with %d patterns", len(available_patterns))
        logger.debug("[Pattern Gallery] Patterns: %s", available_patterns)
        
        try:
            # 컴팩트한 창 크기 (14x8) + 스크롤 지원
            self.fig = plt.figure(figsize=(14, 8))
            self.fig.canvas.manager.set_window_title('Pattern Library - Select Pattern')
            self.fig.suptitle('Pattern Selection Gallery — Click to Open Details',
                             fontsize=13, fontweight='bold', y=0.96)
            
            # 설명 텍스트
            self.fig.text(0.5, 0.92, 'Click any pattern to see individual levels (Scroll if more patterns)',
                         ha='center', fontsize=9, style='italic', color='gray')
            
            # 스크롤 가능한 그리드 (4 rows × 4 cols = 16개씩 표시)
            self.scroll_offset = 0  # 스크롤 오프셋
            self.patterns_per_page = 16  # 한 페이지에 표시할 패턴 수
            
            # Grid 레이아웃 (4 rows × 4 cols) - 컴팩트하게
            gs = gridspec.GridSpec(4, 4, figure=self.fig,
                                  left=0.05, right=0.95, top=0.86, bottom=0.16,
                                  hspace=0.30, wspace=0.18)
            
            self.pattern_axes = []
            self.pattern_buttons = []
            
            # 전체 패턴 수와 페이지 정보
            total_patterns = len(available_patterns)
            total_pages = (total_patterns + self.patterns_per_page - 1) // self.patterns_per_page
            logger.debug("[Pattern Gallery] Total patterns: %d, Pages: %d", total_patterns, total_pages)
            
            logger.debug("[Pattern Gallery] Creating %d pattern previews...", min(self.patterns_per_page, len(available_patterns)))
            
            # 현재 페이지 패턴 렌더링
            self._render_pattern_page(gs, available_patterns)
            
            # 클릭 이벤트 연결
            self.fig.canvas.mpl_connect('pick_event', self.on_pattern_clicked)
            
            # 네비게이션 버튼 (패턴이 한 페이지보다 많을 때)
            total_patterns = len(available_patterns)
            if total_patterns > self.patterns_per_page:
                # Previous 버튼
                ax_prev = plt.axes([0.20, 0.06, 0.12, 0.04])
                self.btn_prev = Button(ax_prev, '<< Previous', color='lightblue', hovercolor='blue')
                self.btn_prev.label.set_fontsize(9)
                self.btn_prev.on_clicked(lambda e: self._scroll_patterns(-1, gs, available_patterns))
                
                # Next 버튼
                ax_next = plt.axes([0.68, 0.06, 0.12, 0.04])
                self.btn_next = Button(ax_next, 'Next >>', color='lightblue', hovercolor='blue')
                self.btn_next.label.set_fontsize(9)
                self.btn_next.on_clicked(lambda e: self._scroll_patterns(1, gs, available_patterns))
                
                # 페이지 정보 표시
                current_page = self.scroll_offset // self.patterns_per_page + 1
                total_pages = (total_patterns + self.patterns_per_page - 1) // self.patterns_per_page
                self.page_text = self.fig.text(0.5, 0.08, f'Page {current_page} / {total_pages}',
                                               ha='center', fontsize=9, fontweight='bold')
                logger.debug("[Pattern Gallery] Navigation enabled - Pages: %d", total_pages)
            
            # Close 버튼
            ax_close = plt.axes([0.42, 0.02, 0.16, 0.04])
            self.btn_close = Button(ax_close, 'Close Gallery', color='lightcoral', hovercolor='red')
            self.btn_close.label.set_fontsize(10)
            self.btn_close.on_clicked(self.on_close_clicked)
            
            logger.info("[Pattern Gallery] Gallery window created successfully with %d patterns", len(self.pattern_axes))
            logger.debug("[Pattern Gallery] Window size: 16x10, Grid: 3x4")
            
        except Exception as e:
            logger.error("[Pattern Gallery] Failed to create gallery: %s", str(e), exc_info=True)
            raise
    
    def _render_pattern_page(self, gs, available_patterns):
        """현재 스크롤 위치의 패턴 페이지 렌더링"""
        # 기존 패턴 axes 초기화
        for artist, _ in self.pattern_axes:
            if hasattr(artist, 'axes'):
                artist.axes.clear()
                artist.axes.set_visible(False)
        self.pattern_axes = []
        
        # 현재 페이지 범위
        start_idx = self.scroll_offset
        end_idx = min(start_idx + self.patterns_per_page, len(available_patterns))
        
        logger.debug("[Pattern Gallery] Rendering patterns %d-%d", start_idx, end_idx-1)
        
        # 패턴 렌더링
        for page_idx, global_idx in enumerate(range(start_idx, end_idx)):
            pattern_name = available_patterns[global_idx]
            
            row = page_idx // 4
            col = page_idx % 4
            ax = self.fig.add_subplot(gs[row, col])
            
            # 패턴 미리보기 렌더링 (artist 반환)
            artist = self._render_pattern_preview(ax, pattern_name, page_idx)
            
            # 클릭 이벤트 - artist에 picker 설정
            if artist:
                artist.set_picker(True)
                self.pattern_axes.append((artist, pattern_name))
            else:
                ax.set_picker(True)
                self.pattern_axes.append((ax, pattern_name))
        
        self.fig.canvas.draw_idle()
        logger.debug("[Pattern Gallery] Page rendered with %d patterns", len(self.pattern_axes))
    
    def _scroll_patterns(self, direction, gs, available_patterns):
        """패턴 페이지 스크롤 (direction: -1=prev, 1=next)"""
        total_patterns = len(available_patterns)
        total_pages = (total_patterns + self.patterns_per_page - 1) // self.patterns_per_page
        current_page = self.scroll_offset // self.patterns_per_page
        
        # 새 페이지 계산
        new_page = current_page + direction
        if 0 <= new_page < total_pages:
            self.scroll_offset = new_page * self.patterns_per_page
            logger.info("[Pattern Gallery] Scrolling to page %d/%d", new_page + 1, total_pages)
            
            # 페이지 다시 렌더링
            self._render_pattern_page(gs, available_patterns)
            
            # 페이지 번호 업데이트
            if hasattr(self, 'page_text'):
                self.page_text.set_text(f'Page {new_page + 1} / {total_pages}')
                self.fig.canvas.draw_idle()
        else:
            logger.debug("[Pattern Gallery] Already at boundary (page %d)", current_page + 1)
    
    def _render_pattern_preview(self, ax, pattern_name, idx):
        """패턴 미리보기 렌더링 - 클릭 가능한 artist 반환"""
        logger.debug("[Pattern Gallery] Rendering preview for: %s (index %d)", pattern_name, idx)
        
        artist = None
        try:
            # 제목 - 더 크고 명확하게
            ax.set_title(pattern_name, fontsize=10, fontweight='bold', pad=6)
            ax.axis('off')
            
            # 패턴 종류에 따라 미리보기 생성
            logger.debug("[Pattern Gallery] Generating preview data for: %s", pattern_name)
            preview_data = self._generate_preview(pattern_name)
            
            if preview_data is not None:
                artist = ax.imshow(preview_data, aspect='auto', interpolation='bilinear')
                logger.debug("[Pattern Gallery] Preview image rendered for: %s (shape: %s)", 
                           pattern_name, preview_data.shape)
            else:
                # 텍스트 표시
                logger.warning("[Pattern Gallery] No preview data for: %s, using text", pattern_name)
                artist = ax.text(0.5, 0.5, pattern_name.replace(' ', '\n'),
                                ha='center', va='center', fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # 테두리 - 더 두껑게
            for spine in ax.spines.values():
                spine.set_edgecolor('darkgray')
                spine.set_linewidth(2.0)
            
            logger.debug("[Pattern Gallery] Preview rendered successfully for: %s", pattern_name)
            
        except Exception as e:
            logger.error("[Pattern Gallery] Failed to render preview for %s: %s", 
                        pattern_name, str(e), exc_info=True)
            # 오류 표시
            artist = ax.text(0.5, 0.5, f"Error\n{pattern_name}", ha='center', va='center',
                            fontsize=8, color='red')
        
        return artist
    
    def _generate_preview(self, pattern_name):
        """패턴 미리보기 이미지 생성"""
        import numpy as np
        
        size = 100
        
        if 'Grayscale' in pattern_name or 'Gray' in pattern_name:
            # Grayscale ramp
            preview = np.linspace(0, 1, size).reshape(1, -1)
            preview = np.repeat(preview, size, axis=0)
            return np.stack([preview, preview, preview], axis=2)
        
        elif 'ColorChecker' in pattern_name or 'Checker' in pattern_name:
            # ColorChecker 24 patches (4x6)
            preview = np.zeros((size, size, 3))
            colors = [
                [0.4, 0.2, 0.1], [0.8, 0.6, 0.5], [0.3, 0.4, 0.6],
                [0.2, 0.3, 0.2], [0.5, 0.5, 0.8], [0.4, 0.8, 0.7],
            ]
            for i in range(2):
                for j in range(3):
                    y1, y2 = i * size // 2, (i + 1) * size // 2
                    x1, x2 = j * size // 3, (j + 1) * size // 3
                    preview[y1:y2, x1:x2] = colors[i * 3 + j]
            return preview
        
        elif 'SMPTE' in pattern_name:
            # SMPTE bars
            preview = np.zeros((size, size, 3))
            colors = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0],
                     [1, 0, 1], [1, 0, 0], [0, 0, 1]]
            for i, color in enumerate(colors):
                x1 = i * size // 7
                x2 = (i + 1) * size // 7
                preview[:, x1:x2] = color
            return preview
        
        elif 'Rec.709' in pattern_name or '709' in pattern_name:
            # RGB gradient
            preview = np.zeros((size, size, 3))
            x = np.linspace(0, 1, size)
            preview[:size//3, :, 0] = x
            preview[size//3:2*size//3, :, 1] = x
            preview[2*size//3:, :, 2] = x
            return preview
        
        else:
            # Default: Gray
            preview = np.ones((size, size, 3)) * 0.5
            return preview
    
    def on_pattern_clicked(self, event):
        """패턴 클릭 이벤트 - artist 기반"""
        logger.info("[Pattern Gallery] Click event triggered")
        logger.debug("[Pattern Gallery] Event artist type: %s", type(event.artist).__name__)
        
        try:
            for artist, pattern_name in self.pattern_axes:
                if event.artist == artist:
                    self.selected_pattern = pattern_name
                    logger.info("[Pattern Gallery] ===================================================")
                    logger.info("[Pattern Gallery] Pattern clicked: '%s'", pattern_name)
                    
                    # 선택 강조 표시 (artist의 axes를 통해 테두리 변경)
                    try:
                        ax_clicked = artist.axes
                        
                        # 모든 패턴의 테두리를 기본으로
                        logger.debug("[Pattern Gallery] Updating visual selection...")
                        for a, _ in self.pattern_axes:
                            ax = a.axes if hasattr(a, 'axes') else a
                            for spine in ax.spines.values():
                                spine.set_edgecolor('darkgray')
                                spine.set_linewidth(2.0)
                        
                        # 선택된 패턴만 강조
                        for spine in ax_clicked.spines.values():
                            spine.set_edgecolor('lime')
                            spine.set_linewidth(4.5)
                        
                        self.fig.canvas.draw_idle()
                        logger.debug("[Pattern Gallery] Visual selection updated - GREEN border")
                    except Exception as vis_err:
                        logger.error("[Pattern Gallery] Visual update failed: %s", str(vis_err), exc_info=True)
                    
                    # 세부 레벨 창 열기
                    try:
                        logger.info("[Pattern Gallery] Opening detail window for '%s'...", pattern_name)
                        if self.parent_ui:
                            logger.debug("[Pattern Gallery] Calling parent_ui.on_pattern_detail_requested()")
                            self.parent_ui.on_pattern_detail_requested(pattern_name)
                            logger.info("[Pattern Gallery] Detail window request sent successfully")
                        else:
                            logger.warning("[Pattern Gallery] No parent_ui - cannot open detail window")
                    except Exception as detail_err:
                        logger.error("[Pattern Gallery] Detail window failed: %s", str(detail_err), exc_info=True)
                        print(f"[Error] Failed to open detail window: {detail_err}")
                    
                    # 부모 UI 업데이트
                    try:
                        if self.parent_ui:
                            self.parent_ui.on_pattern_selected_from_gallery(pattern_name)
                            logger.debug("[Pattern Gallery] Parent UI updated")
                    except Exception as update_err:
                        logger.error("[Pattern Gallery] Parent UI update failed: %s", str(update_err), exc_info=True)
                    
                    break
        except Exception as e:
            logger.error("[Pattern Gallery] on_pattern_clicked failed: %s", str(e), exc_info=True)
            print(f"[Error] Pattern click handler failed: {e}")
    
    def on_close_clicked(self, event):
        """갤러리 닫기 버튼 클릭"""
        logger.info("[Pattern Gallery] Close button clicked")
        logger.debug("[Pattern Gallery] Closing gallery window...")
        self.close()
    
    def close(self):
        """갤러리 창 닫기"""
        logger.info("[Pattern Gallery] Closing gallery window")
        try:
            plt.close(self.fig)
            logger.debug("[Pattern Gallery] Gallery window closed successfully")
        except Exception as e:
            logger.error("[Pattern Gallery] Failed to close window: %s", str(e), exc_info=True)
    
    def show(self):
        """갤러리 창 표시"""
        logger.info("[Pattern Gallery] Displaying gallery window (non-blocking)")
        print("[Pattern Gallery] Calling plt.show(block=False)...")
        
        try:
            # Figure를 명시적으로 표시
            self.fig.show()
            plt.show(block=False)
            
            # 창 활성화
            if hasattr(self.fig.canvas.manager, 'window'):
                try:
                    self.fig.canvas.manager.window.raise_()
                    self.fig.canvas.manager.window.activateWindow()
                    logger.info("[Pattern Gallery] Window raised and activated")
                except Exception as e:
                    logger.warning(f"[Pattern Gallery] Could not raise window: {e}")
            
            # 강제로 캔버스 그리기
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            logger.info("[Pattern Gallery] Gallery window displayed successfully")
            print("[Pattern Gallery] Window display complete!")
            
        except Exception as e:
            logger.error(f"[Pattern Gallery] Failed to display window: {e}", exc_info=True)
            print(f"[ERROR] Window display failed: {e}")


# ============================================================================
# Pattern Detail Window (세부 레벨 선택)
# ============================================================================

class PatternDetailWindow:
    """
    패턴 세부 레벨 선택 창
    패턴 모음(Grayscale 11) 클릭 시 개별 레벨(Gray_0, Gray_20, ...) 선택
    """
    
    def __init__(self, parent_ui, pattern_name, pattern_levels):
        self.parent_ui = parent_ui
        self.pattern_name = pattern_name
        self.pattern_levels = pattern_levels  # [(name, rgb), ...]
        self.selected_level = None
        self.auto_display = True  # 자동 표시 모드
        
        logger.info("[Pattern Detail] " + "="*50)
        logger.info("[Pattern Detail] Opening detail window for '%s' with %d levels", 
                   pattern_name, len(pattern_levels))
        logger.debug("[Pattern Detail] Auto-display mode: %s", self.auto_display)
        
        try:
            # 더 큰 창
            self.fig = plt.figure(figsize=(12, 9))
            self.fig.canvas.manager.set_window_title(f'Pattern Detail - {pattern_name}')
            self.fig.suptitle(f'{pattern_name} — Click Level to Display on Monitor',
                             fontsize=13, fontweight='bold', y=0.96)
            
            # 설명 텍스트
            self.fig.text(0.5, 0.92, 'Click any level to instantly display on PatternWindow',
                         ha='center', fontsize=10, style='italic', color='blue')
            
            # Grid 레이아웃 (최대 8 columns)
            n_levels = len(pattern_levels)
            n_cols = min(8, n_levels)
            n_rows = (n_levels + n_cols - 1) // n_cols
            
            gs = gridspec.GridSpec(n_rows, n_cols, figure=self.fig,
                                  left=0.05, right=0.95, top=0.90, bottom=0.12,
                                  hspace=0.30, wspace=0.15)
            
            self.level_axes = []
            
            # 레벨 미리보기 생성
            for idx, (level_name, rgb) in enumerate(pattern_levels):
                row = idx // n_cols
                col = idx % n_cols
                ax = self.fig.add_subplot(gs[row, col])
                
                # 레벨 미리보기 (artist 반환)
                artist = self._render_level_preview(ax, level_name, rgb)
                if artist:
                    artist.set_picker(True)
                    self.level_axes.append((artist, level_name, rgb))
                else:
                    ax.set_picker(True)
                    self.level_axes.append((ax, level_name, rgb))
        
            # 클릭 이벤트
            self.fig.canvas.mpl_connect('pick_event', self.on_level_clicked)
            
            # 버튼
            ax_display = plt.axes([0.35, 0.04, 0.15, 0.05])
            self.btn_display = Button(ax_display, 'Display on Monitor',
                                      color='lightgreen', hovercolor='green')
            self.btn_display.on_clicked(self.on_display_selected)
            
            ax_close = plt.axes([0.52, 0.04, 0.15, 0.05])
            self.btn_close = Button(ax_close, 'Close', color='lightgray')
            self.btn_close.on_clicked(self.on_close_clicked)
            
            logger.debug("[Pattern Detail] Created %d level previews in %dx%d grid", 
                        n_levels, n_rows, n_cols)
            logger.info("[Pattern Detail] Detail window created successfully")
            
        except Exception as e:
            logger.error("[Pattern Detail] Failed to create detail window: %s", str(e), exc_info=True)
            raise
    
    def _render_level_preview(self, ax, level_name, rgb):
        """레벨 미리보기 렌더링 - 클릭 가능한 artist 반환"""
        ax.set_title(level_name, fontsize=8, pad=3)
        ax.axis('off')
        
        # RGB 색상 박스
        r, g, b = rgb
        preview = np.ones((50, 50, 3))
        preview[:, :] = [r, g, b]
        artist = ax.imshow(preview, aspect='auto')
        
        # RGB 값 텍스트
        rgb_text = f"({int(r*255)},{int(g*255)},{int(b*255)})"
        ax.text(0.5, -0.15, rgb_text, transform=ax.transAxes,
               ha='center', fontsize=7, family='monospace')
        
        # 테두리
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.5)
        
        return artist
    
    def on_level_clicked(self, event):
        """레벨 클릭 이벤트 - artist 기반, 즉시 PatternWindow에 표시"""
        logger.info("[Pattern Detail] Level click event triggered")
        logger.debug("[Pattern Detail] Event artist type: %s", type(event.artist).__name__)
        
        try:
            for artist, level_name, rgb in self.level_axes:
                if event.artist == artist:
                    logger.info("[Pattern Detail] " + "="*50)
                    logger.info("[Pattern Detail] Level clicked: %s", level_name)
                    logger.debug("[Pattern Detail] RGB values: (%.3f, %.3f, %.3f)", *rgb)
                    
                    self.selected_level = (level_name, rgb)
                    
                    # 선택 강조 (artist의 axes를 통해 테두리 변경)
                    logger.debug("[Pattern Detail] Updating visual selection...")
                    ax_clicked = artist.axes
                    
                    for a, _, _ in self.level_axes:
                        ax = a.axes if hasattr(a, 'axes') else a
                        for spine in ax.spines.values():
                            spine.set_edgecolor('gray')
                            spine.set_linewidth(1.5)
                    
                    for spine in ax_clicked.spines.values():
                        spine.set_edgecolor('lime')
                        spine.set_linewidth(4.0)
                    
                    try:
                        self.fig.canvas.draw_idle()
                        logger.debug("[Pattern Detail] Visual selection updated")
                    except Exception as draw_err:
                        logger.error("[Pattern Detail] Canvas draw failed: %s", str(draw_err))
                    
                    # 자동 표시 모드인 경우 즉시 PatternWindow에 표시
                    if self.auto_display:
                        logger.info("[Pattern Detail] Auto-display enabled - displaying immediately")
                        self._display_current_selection()
                    else:
                        logger.debug("[Pattern Detail] Auto-display disabled - waiting for button click")
                    
                    break
                    
        except Exception as e:
            logger.error("[Pattern Detail] Level click failed: %s", str(e), exc_info=True)
            print(f"[Error] Level click failed: {e}")
    
    def _display_current_selection(self):
        """현재 선택된 레벨을 PatternWindow에 표시"""
        if not self.selected_level:
            logger.warning("[Pattern Detail] No level selected - cannot display")
            return
        
        level_name, rgb = self.selected_level
        logger.info("[Pattern Detail] Displaying '%s' on PatternWindow...", level_name)
        
        if self.parent_ui:
            try:
                self.parent_ui.display_pattern_on_window(level_name, rgb)
                logger.info("[Pattern Detail] Display request sent successfully")
            except Exception as e:
                logger.error("[Pattern Detail] Display failed: %s", str(e), exc_info=True)
                print(f"[Error] Failed to display: {e}")
        else:
            logger.error("[Pattern Detail] No parent UI - cannot display")
    
    def on_display_selected(self, event):
        """선택한 레벨을 PatternWindow에 표시 (버튼 클릭용)"""
        logger.info("[Pattern Detail] Display button clicked")
        self._display_current_selection()
    
    def on_close_clicked(self, event):
        """닫기 버튼 클릭"""
        logger.info("[Pattern Detail] Close button clicked for '%s'", self.pattern_name)
        self.close()
    
    def close(self):
        """창 닫기"""
        logger.info("[Pattern Detail] Closing detail window for '%s'", self.pattern_name)
        try:
            plt.close(self.fig)
            logger.debug("[Pattern Detail] Detail window closed successfully")
        except Exception as e:
            logger.error("[Pattern Detail] Failed to close window: %s", str(e), exc_info=True)
    
    def show(self):
        """창 표시"""
        plt.show()


# ============================================================================
# Calibration UI
# ============================================================================

class CalibrationUI:
    """
    전문 캘리브레이션 UI
    
    Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Professional Display Calibration System                         │
    ├──────────────────┬───────────────────────────────────────────────┤
    │ Setup Panel      │ Progress & Preview                            │
    │                  │                                               │
    │ [Sensor]         │ Current Pattern: [    ]                       │
    │  COM3 Connected  │ Phase: Grayscale (1/3)                        │
    │                  │ Progress: ████████░░░░ 45/100 (45%)           │
    │ [Target]         │                                               │
    │  Gamma: 2.2      │ [Preview Window]                              │
    │  CCT: 6500K      │                                               │
    │  Standard: 709   │                                               │
    │                  │                                               │
    │ [Pattern Set]    │                                               │
    │  ○ Grayscale 11  │                                               │
    │  ○ ColorChecker  │                                               │
    │  ○ SMPTE Bars    │                                               │
    │  ○ Rec709 Sweep  │                                               │
    │                  │                                               │
    │ [Phase Select]   │                                               │
    │  ☑ Grayscale     │                                               │
    │  ☑ Color Gamut   │                                               │
    │  [X] Verification│                                               │
    │                  │                                               │
    │ [Start] [Stop]   │                                               │
    ├──────────────────┴───────────────────────────────────────────────┤
    │ Results & Report                                                 │
    │                                                                  │
    │ Before: dE2000 avg=5.2, max=12.4                                 │
    │ After:  dE2000 avg=0.8, max=2.1  [Excellent!]                    │
    │                                                                  │
    │ [Export LUT] [Save Report] [View Detailed Results]              │
    └──────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, sensor=None, parent_window=None):
        """
        Args:
            sensor: 기존 센서 인스턴스 (None이면 새로 생성)
            parent_window: 부모 윈도우 (color_analysis_main.py에서 호출 시)
        """
        self.sensor = sensor or VirtualSensor(
            noise_level=0.02, display_colorspace='BT.2020',
            max_luminance=100.0, black_level=0.05, native_gamma=2.2)
        self.parent_window = parent_window
        
        # Calibration 설정
        self.config = CalibrationConfig.from_preset(CalibrationPreset.STANDARD)
        self.workflow_config = WorkflowConfig()
        
        # 상태 관리
        self.state = CalibrationState.IDLE
        self.progress = MeasurementProgress()
        self.calibration_result = None
        
        # 비동기 작업 관리
        self.task_queue = Queue()
        self.calibration_thread = None
        self.is_running = False
        
        # UI 컴포넌트
        self.fig = None
        self.ui_timer = None
        
        # 패턴 윈도우 (별도 모니터 출력)
        self.pattern_window = None
        self.pattern_window_open = False
        self.current_monitor = 0  # 기본 모니터
        
        # 패턴 표시 방법
        self.pattern_display_mode = "slider"  # "slider" or "window"
        
        # 센서 제어 변수 (color_analysis_main.py와 동일)
        self.sensor_type = 'virtual'  # 'virtual' or 'cr'
        self.selected_port = None
        self.available_ports = []
        self.sensor_connected = True  # 초기 Virtual Sensor는 연결됨
        
        # Phase 선택 상태 (Workflow Preview 업데이트용)
        self.selected_phases = [True, True, True]  # [Grayscale, Color Gamut, Verification]
        
        # Pattern Preview용
        self.current_pattern_rgb = (0.5, 0.5, 0.5)
        self.current_pattern_name = "Mid Gray"

        # ── 캘리브레이션 엔진 연동 ──────────────────────────────
        self.gamma_cal: GammaCalibrator = None       # Phase 1 → 1D LUT
        self.gamut_cal: ColorGamutCalibrator = None  # Phase 2 → 3x3 / 3D LUT
        self.analyzer = CalibrationAnalyzer(
            getattr(self.config, 'target_standard', 'BT.709'))
        self.engine_result: CalibrationResult = None  # 최종 엔진 결과
        self.verify_patches: list = []               # Phase 3 측정 누적 리스트

        logger.info("[Calibration UI] Instance created - sensor type: %s", type(sensor).__name__)
        
    def show(self):
        """UI 표시"""
        self.setup_gui()
        plt.show()
    
    def setup_gui(self):
        """GUI 구성 - CalMAN/LightSpace 스타일 레이아웃"""
        self.fig = plt.figure(figsize=(22, 12))
        self.fig.canvas.manager.set_window_title('Professional Display Calibration')
        self.fig.suptitle('Display Calibration System — Industry Standard Workflow',
                         fontsize=15, fontweight='bold', y=0.97)
        
        # GridSpec: 산업 표준 레이아웃 (3열 구조)
        # ┌─────────────┬───────────────────────┬──────────────┐
        # │   Setup     │   Workflow Status     │   Controls   │
        # │   (18%)     │      (52%)            │   (30%)      │
        # ├─────────────┼───────────────────────┼──────────────┤
        # │   Pattern   │   Progress Chart      │   Results    │
        # │   Preview   │      (52%)            │   (30%)      │
        # └─────────────┴───────────────────────┴──────────────┘
        
        gs_main = gridspec.GridSpec(2, 3, figure=self.fig,
                                    left=0.03, right=0.98, top=0.93, bottom=0.04,
                                    hspace=0.20, wspace=0.18,
                                    width_ratios=[1.0, 2.8, 1.6],
                                    height_ratios=[2.2, 1.0])
        
        # ════════════════════════════════════════════════════════════
        # Row 0: Setup | Workflow Status | Controls
        # ════════════════════════════════════════════════════════════
        
        self.ax_setup = self.fig.add_subplot(gs_main[0, 0])
        self.ax_setup.set_title('Setup & Target', fontsize=11, fontweight='bold', pad=6)
        self.ax_setup.axis('off')
        
        self.ax_workflow = self.fig.add_subplot(gs_main[0, 1])
        self.ax_workflow.set_title('Workflow Status', fontsize=11, fontweight='bold', pad=6)
        self.ax_workflow.axis('off')
        
        self.ax_controls = self.fig.add_subplot(gs_main[0, 2])
        self.ax_controls.set_title('Phase & Control', fontsize=11, fontweight='bold', pad=6)
        self.ax_controls.axis('off')
        
        # ════════════════════════════════════════════════════════════
        # Row 1: Pattern Preview | Progress Chart | Results
        # ════════════════════════════════════════════════════════════
        
        self.ax_pattern_preview = self.fig.add_subplot(gs_main[1, 0])
        self.ax_pattern_preview.set_title('Pattern Preview', fontsize=10, fontweight='bold', pad=6)
        self.ax_pattern_preview.axis('off')
        
        self.ax_progress = self.fig.add_subplot(gs_main[1, 1])
        self.ax_progress.set_title('Measurement Progress', fontsize=10, fontweight='bold', pad=6)
        self.ax_progress.axis('off')
        
        self.ax_results = self.fig.add_subplot(gs_main[1, 2])
        self.ax_results.set_title('Calibration Results', fontsize=10, fontweight='bold', pad=6)
        self.ax_results.axis('off')
        
        # UI 컴포넌트 생성
        self.setup_setup_panel()
        self.setup_workflow_panel()
        self.setup_controls_panel()
        self.setup_action_buttons()
        
        # 초기 상태 표시
        self.update_workflow_display()
        self.update_progress_display()
        self.update_results_display()
        self.update_pattern_preview()
        
        # UI 업데이트 타이머
        self._start_ui_timer()
        
        logger.info("[Calibration UI] GUI initialized - Industry Standard Layout")
        print("[Calibration UI] Initialized - Industry Standard Layout")
    
    def setup_setup_panel(self):
        """Setup 패널 구성 (왼쪽) - 센서 제어, Calibration Method, Color 설정"""
        
        # ═══════════════════════════════════════════════════════════════
        # 섹션 1: 센서 제어 (Scan, Select, Connect)
        # ═══════════════════════════════════════════════════════════════
        
        # Scan Ports 버튼
        ax_scan = plt.axes([0.04, 0.88, 0.06, 0.025])
        self.btn_scan = Button(ax_scan, 'Scan', color='lightyellow', hovercolor='gold')
        self.btn_scan.label.set_fontsize(7)
        self.btn_scan.on_clicked(self.on_scan_ports)
        
        # Connect 버튼
        ax_connect = plt.axes([0.10, 0.88, 0.06, 0.025])
        self.btn_connect = Button(ax_connect, 'Connect', color='lightgreen', hovercolor='limegreen')
        self.btn_connect.label.set_fontsize(7)
        self.btn_connect.on_clicked(self.on_connect_sensor)
        
        # COM Port 선택 (RadioButtons)
        self.ax_port_radio = plt.axes([0.04, 0.79, 0.13, 0.08])
        self.ax_port_radio.set_title('● Sensor', fontsize=8, fontweight='bold', pad=2)
        self._port_labels = ['Virtual']
        self.radio_port = RadioButtons(self.ax_port_radio, self._port_labels,
                                       activecolor='dodgerblue')
        for lbl in self.radio_port.labels:
            lbl.set_fontsize(7)
        self.radio_port.on_clicked(self.on_port_selected)
        
        # ═══════════════════════════════════════════════════════════════
        # 섹션 2: Calibration Method 선택 (NEW!)
        # ═══════════════════════════════════════════════════════════════
        
        ax_method = plt.axes([0.04, 0.62, 0.13, 0.15])
        ax_method.set_title('● Calibration Method', fontsize=8, fontweight='bold', pad=2)
        method_labels = [
            'Quick\n(5 min)',
            'Standard\n(10 min)',
            'Professional\n(20 min)',
            'Broadcast',
            'Cinema',
            'Custom...'  # Custom 추가!
        ]
        self.radio_method = RadioButtons(ax_method, method_labels,
                                         activecolor='purple')
        for lbl in self.radio_method.labels:
            lbl.set_fontsize(6)
        
        def method_changed_with_log(label):
            logger.info(f"[UI Event] >>>>>> Method RadioButton CLICKED: {label} <<<<<<")
            print(f"\n[UI Event] Calibration Method RadioButton clicked: {label}")
            self.on_method_changed(label)
        
        self.radio_method.on_clicked(method_changed_with_log)
        
        # ═══════════════════════════════════════════════════════════════
        # 섹션 3: Target Settings (Gamma, CCT, Standard)
        # ═══════════════════════════════════════════════════════════════
        
        # Gamma 선택
        ax_gamma = plt.axes([0.04, 0.50, 0.13, 0.10])
        ax_gamma.set_title('● Target Gamma', fontsize=8, fontweight='bold', pad=2)
        self.radio_gamma = RadioButtons(ax_gamma, ['2.2', '2.4', 'BT.1886'],
                                        activecolor='forestgreen')
        for lbl in self.radio_gamma.labels:
            lbl.set_fontsize(7)
        self.radio_gamma.on_clicked(self.on_gamma_changed)
        
        # CCT 선택
        ax_cct = plt.axes([0.04, 0.40, 0.13, 0.08])
        ax_cct.set_title('● Target CCT', fontsize=8, fontweight='bold', pad=2)
        self.radio_cct = RadioButtons(ax_cct, ['5500K', '6500K', '9300K'],
                                      activecolor='orange')
        for lbl in self.radio_cct.labels:
            lbl.set_fontsize(7)
        self.radio_cct.on_clicked(self.on_cct_changed)
        
        # Color Standard 선택
        ax_standard = plt.axes([0.04, 0.30, 0.13, 0.08])
        ax_standard.set_title('● Color Standard', fontsize=8, fontweight='bold', pad=2)
        self.radio_standard = RadioButtons(ax_standard, ['BT.709', 'DCI-P3', 'BT.2020'],
                                           activecolor='royalblue')
        for lbl in self.radio_standard.labels:
            lbl.set_fontsize(7)
        self.radio_standard.on_clicked(self.on_standard_changed)
        
        # ═══════════════════════════════════════════════════════════════
        # 섹션 4: Pattern Gallery 버튼
        # ═══════════════════════════════════════════════════════════════
        
        ax_show_patterns = plt.axes([0.04, 0.24, 0.13, 0.04])
        self.btn_show_patterns = Button(ax_show_patterns, 'Pattern Gallery',
                                        color='lightcyan', hovercolor='cyan')
        self.btn_show_patterns.label.set_fontsize(8)
        
        def pattern_gallery_with_log(event):
            logger.info("[UI Event] >>>>>> Pattern Gallery button CLICKED <<<<<<")
            print("\n" + "="*70)
            print("[UI Event] Pattern Gallery button clicked!")
            print("="*70)
            self.on_show_pattern_gallery(event)
        
        self.btn_show_patterns.on_clicked(pattern_gallery_with_log)
        
        logger.debug("[Setup Panel] Setup panel initialized with sensor controls, method selection, and target settings")
        logger.info("[Setup Panel] Sensor: %s, Method: Standard, Gamma: %.1f, CCT: %dK, Standard: %s",
                   self.sensor_type, self.config.target_gamma, self.config.target_cct, self.config.target_standard)
    
    def setup_workflow_panel(self):
        """Workflow Status 패널 구성 (중앙) - 실시간 진행 상황 + Workflow 상세 정보"""
        logger.debug("[Workflow Panel] Initializing workflow status panel with details")
        
        # 초기 Workflow Detail 표시
        workflow_detail = (
            "=== Calibration Workflow Detail ===\n\n"
            "Current Method: STANDARD (10 min, 52 patches)\n\n"
            "Phase 1: Grayscale Calibration\n"
            "  - Purpose: Gamma curve & grayscale tracking\n"
            "  - Patterns: 11-step White ladder (0%, 10%, ..., 100%)\n"
            "  - Measurements: 11 patches\n"
            "  - Output: 1D LUT (1024 points x 3 channels)\n"
            "  - Standard: ITU-R BT.1886\n\n"
            "Phase 2: Color Gamut Calibration\n"
            "  - Purpose: Primary/Secondary color accuracy\n"
            "  - Patterns: RGBCMYW (7 colors at 100%)\n"
            "  - Measurements: 7 patches\n"
            "  - Output: 3x3 Matrix + 3D LUT (33^3 cube)\n"
            "  - Standard: BT.709 / DCI-P3 / BT.2020\n\n"
            "Phase 3: Verification\n"
            "  - Purpose: dE accuracy validation\n"
            "  - Patterns: ColorChecker Classic 24\n"
            "  - Measurements: 24 patches\n"
            "  - Output: dE2000 report (avg, max, min)\n"
            "  - Standard: ISO 17321-1\n\n"
            "Total: 42 patches, ~10 minutes\n"
        )
        
        self.workflow_detail_text = self.ax_workflow.text(
            0.02, 0.98, workflow_detail,
            fontsize=7, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        logger.info("[Workflow Panel] Workflow detail panel initialized with STANDARD method")
    
    def setup_controls_panel(self):
        """Controls 패널 구성 (오른쪽) - Phase 선택 + 제어 버튼"""
        y_pos = 0.95
        
        # Phase 선택 체크박스
        self.ax_controls.text(0.05, y_pos, '● Select Phases',
                             fontsize=9, fontweight='bold')
        
        ax_phase = plt.axes([0.835, 0.68, 0.14, 0.16])
        self.check_phases = CheckButtons(
            ax_phase,
            ['Grayscale\n(1D LUT)', 'Color Gamut\n(3x3 + 3D)', 'Verification\n(dE Report)'],
            [True, True, True])
        for label in self.check_phases.labels:
            label.set_fontsize(8)
        self.check_phases.on_clicked(self.on_phase_checked)
        
        # 제어 버튼은 setup_action_buttons()에서 처리
    
    def setup_action_buttons(self):
        """제어 버튼 (오른쪽 패널 하단)"""
        btn_x = 0.835
        btn_width = 0.14
        btn_height = 0.05
        
        # === Pattern Window 제어 섹션 ===
        self.ax_controls.text(0.05, 0.62, '● Pattern Window',
                             fontsize=9, fontweight='bold')
        
        # Open Pattern Window 버튼
        ax_open_pw = plt.axes([btn_x, 0.56, btn_width, 0.04])
        self.btn_open_pattern = Button(ax_open_pw, 'Open Pattern Window',
                                       color='lightyellow', hovercolor='yellow')
        self.btn_open_pattern.label.set_fontsize(9)
        self.btn_open_pattern.on_clicked(self.on_open_pattern_window)
        
        # === Workflow 확인 섹션 (NEW!) ===
        self.ax_controls.text(0.05, 0.49, '● Workflow',
                             fontsize=9, fontweight='bold')
        
        # Show Workflow 버튼
        ax_show_workflow = plt.axes([btn_x, 0.43, btn_width, 0.04])
        self.btn_show_workflow = Button(ax_show_workflow, 'Show Workflow Detail',
                                        color='lightcyan', hovercolor='cyan')
        self.btn_show_workflow.label.set_fontsize(9)
        self.btn_show_workflow.on_clicked(self.on_show_workflow_detail)
        
        # === Calibration 제어 섹션 ===
        
        # Start 버튼
        ax_start = plt.axes([btn_x, 0.35, btn_width, btn_height])
        self.btn_start = Button(ax_start, 'START CALIBRATION',
                               color='lightgreen', hovercolor='green')
        self.btn_start.label.set_fontsize(10)
        self.btn_start.label.set_weight('bold')
        self.btn_start.on_clicked(self.on_start_calibration)
        
        # Stop 버튼
        ax_stop = plt.axes([btn_x, 0.28, btn_width, btn_height])
        self.btn_stop = Button(ax_stop, 'STOP',
                              color='lightsalmon', hovercolor='red')
        self.btn_stop.label.set_fontsize(10)
        self.btn_stop.on_clicked(self.on_stop_calibration)
        
        # Export LUT 버튼
        ax_export = plt.axes([btn_x, 0.15, btn_width, 0.04])
        self.btn_export = Button(ax_export, 'Export LUT',
                                 color='lightblue', hovercolor='cyan')
        self.btn_export.label.set_fontsize(9)
        self.btn_export.on_clicked(self.on_export_lut)
        
        # Save Report 버튼
        ax_report = plt.axes([btn_x, 0.09, btn_width, 0.04])
        self.btn_report = Button(ax_report, 'Save Report',
                                 color='lightyellow', hovercolor='yellow')
        self.btn_report.label.set_fontsize(9)
        self.btn_report.on_clicked(self.on_save_report)
    
    def _update_workflow_preview(self):
        """Workflow 미리보기 업데이트 (DEPRECATED - use update_workflow_display instead)"""
        # Legacy method - no longer updates old workflow_preview text
        # Workflow status is now shown in ax_workflow panel via update_workflow_display()
        logger.debug("[Workflow] Legacy _update_workflow_preview() called - skipping (use update_workflow_display)")
        pass
    
    def _start_ui_timer(self):
        """UI 업데이트 타이머"""
        def check_queue():
            try:
                while not self.task_queue.empty():
                    task = self.task_queue.get_nowait()
                    task_type = task.get('type')
                    
                    if task_type == 'progress_update':
                        self.progress = task['progress']
                        self.update_progress_display()
                        self.update_workflow_display()  # Workflow 실시간 업데이트 추가!
                    elif task_type == 'display_pattern':  # 패턴 표시 (메인 스레드에서 처리)
                        self._display_pattern_on_window_main_thread(
                            task['pattern_name'], task['rgb'])
                        # 백그라운드 스레드에 표시 완료 신호 전송
                        evt = task.get('event')
                        if evt is not None:
                            evt.set()
                    elif task_type == 'calibration_complete':
                        # 잔류 display_pattern 태스크 드레인 (완료 후 한꺼번에 표시되는 문제 방지)
                        while not self.task_queue.empty():
                            try:
                                leftover = self.task_queue.get_nowait()
                                leftover_evt = leftover.get('event')
                                if leftover_evt is not None:
                                    leftover_evt.set()  # 대기 중인 스레드 해제
                            except Exception:
                                break
                        self.calibration_result = task['result']
                        self.state = CalibrationState.COMPLETED
                        self.update_results_display()
                        self.update_workflow_display()  # 완료 상태 업데이트
                        self.btn_start.label.set_text('Start Calibration')
                        self.btn_start.color = 'lightgreen'
                        print("[Calibration] Completed!")
                        # 패턴 창 초기화 및 닫기
                        if self.pattern_window_open and self.pattern_window:
                            try:
                                self.pattern_window.show_color(0.0, 0.0, 0.0)  # 검정으로 페이드
                                self.pattern_window.close()
                                self.pattern_window_open = False
                                print("[Pattern Window] Closed after calibration")
                            except Exception as _pe:
                                self.pattern_window_open = False
                    elif task_type == 'calibration_error':
                        self.state = CalibrationState.ERROR
                        self.update_workflow_display()  # 에러 상태 업데이트
                        print(f"[Calibration] Error: {task['error']}")
                        self.btn_start.label.set_text('Start Calibration')
                        self.btn_start.color = 'lightcoral'
            except:
                pass
        
        self.ui_timer = self.fig.canvas.new_timer(interval=100)
        self.ui_timer.add_callback(check_queue)
        self.ui_timer.start()
    
    def update_progress_display(self):
        """진행 상황 표시 업데이트"""
        self.ax_progress.clear()
        self.ax_progress.set_title('Progress & Preview', fontsize=12, fontweight='bold', pad=8)
        self.ax_progress.axis('off')
        self.ax_progress.set_xlim(0, 1)
        self.ax_progress.set_ylim(0, 1)
        
        if self.state == CalibrationState.IDLE:
            self.ax_progress.text(0.5, 0.5, 'Ready to start calibration',
                                 ha='center', va='center', fontsize=12,
                                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            # Phase 정보
            phase_text = f"Phase: {self.progress.current_phase.value}"
            self.ax_progress.text(0.05, 0.95, phase_text,
                                 fontsize=10, fontweight='bold', va='top')
            
            # 진행률
            progress_text = f"Progress: {self.progress.measured_patches}/{self.progress.total_patches} ({self.progress.progress_percent:.0f}%)"
            self.ax_progress.text(0.05, 0.88, progress_text, fontsize=9, va='top')
            
            # Progress Bar
            bar_y = 0.78
            bar_height = 0.05
            rect_bg = Rectangle((0.05, bar_y), 0.90, bar_height,
                               facecolor='lightgray', edgecolor='black', linewidth=1)
            self.ax_progress.add_patch(rect_bg)
            
            if self.progress.total_patches > 0:
                progress_width = 0.90 * (self.progress.measured_patches / self.progress.total_patches)
                rect_progress = Rectangle((0.05, bar_y), progress_width, bar_height,
                                         facecolor='lightgreen', edgecolor='black', linewidth=1)
                self.ax_progress.add_patch(rect_progress)
            
            # 현재 패치 정보
            patch_text = f"Current: {self.progress.current_patch_name}"
            self.ax_progress.text(0.05, 0.70, patch_text, fontsize=9, va='top')
            
            # 현재 RGB 프리뷰
            if self.progress.current_rgb != (0, 0, 0):
                r, g, b = self.progress.current_rgb
                rect_preview = Rectangle((0.35, 0.30), 0.30, 0.35,
                                        facecolor=(r, g, b), edgecolor='black', linewidth=2)
                self.ax_progress.add_patch(rect_preview)
                
                rgb_text = f"RGB: ({r:.3f}, {g:.3f}, {b:.3f})"
                self.ax_progress.text(0.50, 0.25, rgb_text, ha='center', fontsize=8,
                                     family='monospace')
            
            # 시간 정보
            if self.progress.elapsed_time > 0:
                time_text = f"Elapsed: {self.progress.elapsed_time:.1f}s"
                if self.progress.estimated_remaining > 0:
                    time_text += f"  |  Remaining: ~{self.progress.estimated_remaining:.1f}s"
                self.ax_progress.text(0.05, 0.08, time_text, fontsize=8, style='italic')
        
        self.fig.canvas.draw_idle()
    
    def update_workflow_display(self):
        """Workflow Status 패널 업데이트 - 실시간 진행 상황"""
        self.ax_workflow.clear()
        self.ax_workflow.set_title('Workflow Status', fontsize=11, fontweight='bold', pad=6)
        self.ax_workflow.axis('off')
        self.ax_workflow.set_xlim(0, 1)
        self.ax_workflow.set_ylim(0, 1)
        
        if self.state == CalibrationState.IDLE:
            status_text = (
                "IDLE - Ready to Start\n\n"
                "Click 'START CALIBRATION' to begin\n\n"
                "Selected Phases:\n"
            )
            phase_status = self.check_phases.get_status() if hasattr(self, 'check_phases') else [True, True, True]
            if phase_status[0]:
                status_text += "  [OK] Phase 1: Grayscale (11 levels)\n"
            if phase_status[1]:
                status_text += "  [OK] Phase 2: Color Gamut (RGBCMYW)\n"
            if phase_status[2]:
                status_text += "  [OK] Phase 3: Verification (ColorChecker)\n"
            
            self.ax_workflow.text(0.05, 0.95, status_text, fontsize=9, va='top',
                                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        elif self.state == CalibrationState.MEASURING:
            # 실시간 측정 상태
            phase_name = self.progress.current_phase.value if self.progress.current_phase else "Unknown"
            status_text = (
                f"RUNNING - {phase_name}\n\n"
                f"Patch: {self.progress.current_patch_name}\n"
                f"Progress: {self.progress.measured_patches}/{self.progress.total_patches}\n"
                f"Time: {self.progress.elapsed_time:.1f}s\n"
            )
            self.ax_workflow.text(0.05, 0.95, status_text, fontsize=9, va='top',
                                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            # Timeline visualization
            y_start = 0.60
            y_height = 0.05
            phases = [
                ('Phase 1: Grayscale', WorkflowPhase.PHASE1_GRAYSCALE),
                ('Phase 2: Color Gamut', WorkflowPhase.PHASE2_COLOR),
                ('Phase 3: Verification', WorkflowPhase.PHASE3_VERIFY)
            ]
            
            for i, (name, phase) in enumerate(phases):
                y = y_start - i * 0.12
                # Background
                rect = Rectangle((0.05, y), 0.90, y_height,
                               facecolor='lightgray', edgecolor='black', linewidth=1)
                self.ax_workflow.add_patch(rect)
                
                # Current phase highlight
                if self.progress.current_phase == phase:
                    rect_active = Rectangle((0.05, y), 0.90, y_height,
                                           facecolor='yellow', edgecolor='black', linewidth=2)
                    self.ax_workflow.add_patch(rect_active)
                
                self.ax_workflow.text(0.05, y - 0.01, name, fontsize=8, va='top')
        
        elif self.state == CalibrationState.COMPLETED:
            status_text = (
                "COMPLETED\n\n"
                "Calibration finished successfully!\n"
                "Check Results panel for details.\n"
            )
            self.ax_workflow.text(0.05, 0.95, status_text, fontsize=9, va='top',
                                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        elif self.state == CalibrationState.ERROR:
            status_text = (
                "ERROR\n\n"
                "Calibration encountered an error.\n"
                "Check console for details.\n"
            )
            self.ax_workflow.text(0.05, 0.95, status_text, fontsize=9, va='top',
                                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        self.fig.canvas.draw_idle()
    
    def update_pattern_preview(self):
        """Pattern Preview 패널 업데이트"""
        self.ax_pattern_preview.clear()
        self.ax_pattern_preview.set_title('Current Pattern', fontsize=10, fontweight='bold', pad=6)
        self.ax_pattern_preview.axis('off')
        self.ax_pattern_preview.set_xlim(0, 1)
        self.ax_pattern_preview.set_ylim(0, 1)
        
        # Pattern color preview
        r, g, b = self.current_pattern_rgb
        rect = Rectangle((0.1, 0.3), 0.8, 0.5,
                        facecolor=(r, g, b), edgecolor='black', linewidth=2)
        self.ax_pattern_preview.add_patch(rect)
        
        # Pattern name
        self.ax_pattern_preview.text(0.5, 0.15, self.current_pattern_name,
                                    ha='center', fontsize=9, fontweight='bold')
        
        # RGB values
        rgb_text = f"RGB: ({int(r*255)}, {int(g*255)}, {int(b*255)})"
        self.ax_pattern_preview.text(0.5, 0.05, rgb_text,
                                    ha='center', fontsize=8, family='monospace')
        
        self.fig.canvas.draw_idle()
    
    def update_results_display(self):
        """결과 표시 업데이트"""
        self.ax_results.clear()
        self.ax_results.set_title('Calibration Results & Report', fontsize=12, fontweight='bold', pad=8)
        self.ax_results.axis('off')
        self.ax_results.set_xlim(0, 1)
        self.ax_results.set_ylim(0, 1)

        if self.calibration_result is None:
            self.ax_results.text(0.5, 0.5, 'No calibration results yet',
                                ha='center', va='center', fontsize=11, color='gray')
        else:
            summary = (self.engine_result.summary
                       if self.engine_result and self.engine_result.summary
                       else {})
            before = summary.get('before', {})
            after  = summary.get('after',  {})

            lines = ["Calibration Completed Successfully!\n"]
            if before and after:
                b_mean = before.get('mean_dE2000', 0)
                b_max  = before.get('max_dE2000',  0)
                a_mean = after.get('mean_dE2000',  0)
                a_max  = after.get('max_dE2000',   0)
                lines.append(f"Before: dE2000 avg={b_mean:.2f}  max={b_max:.2f}")
                lines.append(f"After:  dE2000 avg={a_mean:.2f}   max={a_max:.2f}")
                imp = summary.get('improvement_pct', {})
                if imp:
                    lines.append(f"Improvement:  {imp.get('mean_dE2000', 0):.1f}% reduction")
                grade = 'Excellent!' if a_mean < 1.0 else ('Good' if a_mean < 2.0 else 'Fair')
                lines.append(f"Quality: {grade}")
                bg_color = 'lightgreen' if a_mean < 2.0 else 'lightyellow'
            else:
                lines.append("Before: dE2000 avg=N/A")
                lines.append("After:  dE2000 avg=N/A")
                bg_color = 'lightyellow'

            lines.append("")
            lines.append("Generated LUTs:")
            lut_1d = getattr(self.engine_result, 'lut_1d', None) if self.engine_result else None
            mat    = getattr(self.engine_result, 'matrix_3x3', None) if self.engine_result else None
            lut_3d = getattr(self.engine_result, 'lut_3d', None) if self.engine_result else None
            if lut_1d:
                lines.append(f"  1D LUT : {lut_1d.size} pts x3 ch")
            if mat:
                lines.append(f"  3x3 Matrix : color gamut correction")
            if lut_3d:
                lines.append(f"  3D LUT : {lut_3d.size}\u00b3 cube")
            if not lut_1d and not mat and not lut_3d:
                lines.append("  (none generated)")

            result_text = "\n".join(lines)
            self.ax_results.text(0.05, 0.95, result_text,
                                fontsize=10, va='top', family='monospace',
                                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))

        self.fig.canvas.draw_idle()
    
    # ════════════════════════════════════════════════════════════
    # Event Handlers
    # ════════════════════════════════════════════════════════════
    
    def on_start_calibration(self, event):
        """캘리브레이션 시작"""
        logger.info("[UI Event] " + "="*60)
        logger.info("[UI Event] START CALIBRATION button clicked")
        
        if self.is_running:
            logger.warning("[Calibration] Cannot start - already running")
            print("[Calibration] Already running")
            return
        
        logger.info("[Calibration] Starting workflow with config: Gamma=%.1f, CCT=%dK, Standard=%s",
                   self.config.target_gamma, self.config.target_cct, self.config.target_standard)
        print("[Calibration] Starting...")
        self.state = CalibrationState.MEASURING
        self.is_running = True
        
        self.btn_start.label.set_text('Running...')
        self.btn_start.color = 'yellow'
        
        # Workflow Status 업데이트
        self.update_workflow_display()
        self.fig.canvas.draw_idle()

        # 패턴 창을 메인 스레드에서 미리 열기 (Tkinter는 생성 스레드에서만 접근 가능)
        if not self.pattern_window_open:
            logger.info("[Calibration] Opening Pattern Window on main thread before background start")
            try:
                self.on_open_pattern_window(None)
                logger.info("[Calibration] Pattern Window opened on main thread")
            except Exception as e:
                logger.error("[Calibration] Failed to open Pattern Window: %s", str(e))
                print(f"[Warning] Pattern Window open failed: {e}")

        # 별도 스레드에서 실행
        self.calibration_thread = threading.Thread(
            target=self._run_calibration_workflow,
            daemon=True)
        self.calibration_thread.start()
        
        logger.info("[Calibration] Background thread started")
    
    def on_stop_calibration(self, event):
        """캘리브레이션 중지"""
        logger.info("[UI Event] " + "="*60)
        logger.info("[UI Event] STOP button clicked")
        
        if not self.is_running:
            logger.debug("[Calibration] Not running - nothing to stop")
            print("[Calibration] Not running")
            return
        
        logger.info("[Calibration] Stopping workflow...")
        print("[Calibration] Stopping...")
        self.is_running = False
        self.state = CalibrationState.IDLE
        self.btn_start.label.set_text('Start Calibration')
        self.btn_start.color = 'lightgreen'
        
        # Workflow Status 업데이트
        self.update_workflow_display()
        self.fig.canvas.draw_idle()
        
        logger.info("[Calibration] Workflow stopped by user")
    
    def on_gamma_changed(self, label):
        """Target Gamma 변경"""
        logger.info("[UI Event] Gamma changed to: %s", label)
        gamma_map = {'2.2': 2.2, '2.4': 2.4, 'BT.1886': 2.4, 'PQ': 2.2}
        self.config.target_gamma = gamma_map.get(label, 2.2)
        logger.debug("[Settings] Gamma value set to: %.1f", self.config.target_gamma)
        print(f"[Settings] Gamma changed to {label}")
        self._update_workflow_preview()  # Workflow 미리보기 업데이트
    
    def on_method_changed(self, label):
        """Calibration Method 변경 (Custom 설정 포함)"""
        logger.info("[UI Event] " + "="*60)
        logger.info("[UI Event] Calibration Method changed to: %s", label)
        
        # Custom 선택 시 설정 창 열기
        if label == 'Custom...':
            logger.info("[UI Event] Opening Custom Settings window")
            print("\n" + "="*70)
            print("CUSTOM CALIBRATION SETTINGS")
            print("="*70)
            try:
                self._show_custom_settings_window()
            except Exception as e:
                logger.error(f"[Custom] Failed to open settings: {e}", exc_info=True)
                print(f"[ERROR] Failed to open Custom Settings: {e}")
            return
            print("")
            print("Default Custom Settings:")
            print("  Phase 1: Grayscale")
            print("    - 21-step ladder (RGBW separation)")
            print("    - Measurements: 84 patches (21x4)")
            print("  Phase 2: Color Gamut")
            print("    - Extended 17 patches")
            print("    - 3D LUT: 33^3 cube")
            print("  Phase 3: Verification")
            print("    - ColorChecker 24")
            print("")
            print("Estimated time: ~15 minutes")
            print("="*70 + "\n")
            
            # Custom preset으로 workflow 업데이트
            custom_detail = (
                "=== CUSTOM Calibration (~15 min, 125 patches) ===\n\n"
                "Phase 1: Grayscale\n"
                "  - 21-step White + RGBW separation\n"
                "  - Measurements: 84 patches\n\n"
                "Phase 2: Color Gamut\n"
                "  - Extended gamut (17 patches)\n"
                "  - Measurements: 17 patches\n\n"
                "Phase 3: Verification\n"
                "  - ColorChecker 24\n"
                "  - Measurements: 24 patches\n\n"
                "Total: 125 patches, ~15 minutes"
            )
            
            if hasattr(self, 'workflow_detail_text'):
                self.workflow_detail_text.set_text(custom_detail)
                self.fig.canvas.draw_idle()
            
            logger.info("[Custom] Configuration applied: 125 patches, ~15 min")
            return
        
        method_map = {
            'Quick\n(5 min)': CalibrationPreset.QUICK,
            'Standard\n(10 min)': CalibrationPreset.STANDARD,
            'Professional\n(20 min)': CalibrationPreset.PROFESSIONAL,
            'Broadcast': CalibrationPreset.BROADCAST,
            'Cinema': CalibrationPreset.CINEMA,
        }
        
        preset = method_map.get(label, CalibrationPreset.STANDARD)
        
        # Workflow Detail 업데이트
        workflow_details = {
            CalibrationPreset.QUICK: (
                "=== QUICK Calibration (5 min, 25 patches) ===\n\n"
                "Optimized for: Fast turnaround\n"
                "Accuracy: Good (dE < 3.0)\n\n"
                "Phase 1: Grayscale\n"
                "  - 11-step White ladder\n"
                "  - Measurements: 11 patches\n"
                "  - White-only (no RGB separation)\n\n"
                "Phase 2: Color Gamut\n"
                "  - Primary colors only (RGB)\n"
                "  - Measurements: 3 patches\n\n"
                "Phase 3: Verification\n"
                "  - Grayscale patches only\n"
                "  - Measurements: 11 patches\n\n"
                "Total: 25 patches, ~5 minutes"
            ),
            CalibrationPreset.STANDARD: (
                "=== STANDARD Calibration (10 min, 52 patches) ===\n\n"
                "Optimized for: Balanced speed/accuracy\n"
                "Accuracy: Excellent (dE < 1.5)\n\n"
                "Phase 1: Grayscale\n"
                "  - 11-step White ladder\n"
                "  - Measurements: 11 patches\n"
                "  - Output: 1D LUT (1024 points)\n"
                "  - Standard: ITU-R BT.1886\n\n"
                "Phase 2: Color Gamut\n"
                "  - RGBCMYW (7 colors)\n"
                "  - Measurements: 7 patches\n"
                "  - Output: 3x3 Matrix + 3D LUT\n\n"
                "Phase 3: Verification\n"
                "  - ColorChecker 24\n"
                "  - Measurements: 24 patches\n"
                "  - Report: dE2000 statistics\n\n"
                "Total: 52 patches, ~10 minutes"
            ),
            CalibrationPreset.PROFESSIONAL: (
                "=== PROFESSIONAL Calibration (20 min, 125 patches) ===\n\n"
                "Optimized for: Maximum accuracy\n"
                "Accuracy: Reference (dE < 0.5)\n\n"
                "Phase 1: Grayscale\n"
                "  - 21-step White + RGB separation\n"
                "  - Measurements: 84 patches (21x4)\n"
                "  - Output: High-resolution 1D LUT\n"
                "  - Standard: ITU-R BT.1886\n\n"
                "Phase 2: Color Gamut\n"
                "  - Extended gamut test\n"
                "  - Measurements: 17 patches\n"
                "  - Output: 65^3 3D LUT\n\n"
                "Phase 3: Verification\n"
                "  - ColorChecker SG 140\n"
                "  - Measurements: 24 patches\n"
                "  - Report: dE2000 + dEITP\n\n"
                "Total: 125 patches, ~20 minutes"
            ),
            CalibrationPreset.BROADCAST: (
                "=== BROADCAST Calibration (EBU Tech 3320) ===\n\n"
                "Standard: EBU Tech 3320, ITU-R BT.1886\n"
                "Application: Broadcast production/grading\n\n"
                "Phase 1: Grayscale\n"
                "  - 21-step EBU ladder\n"
                "  - Target: BT.1886 EOTF\n"
                "  - Black level: 0.05 cd/m^2\n"
                "  - Peak white: 100 cd/m^2\n\n"
                "Phase 2: Color Gamut\n"
                "  - BT.709 primaries\n"
                "  - D65 white point (6500K)\n\n"
                "Phase 3: EBU Bars verification\n"
                "  - 75% EBU Bars\n"
                "  - PLUGE pattern\n\n"
                "Total: ~15 minutes"
            ),
            CalibrationPreset.CINEMA: (
                "=== CINEMA Calibration (SMPTE RP 431-2) ===\n\n"
                "Standard: SMPTE RP 431-2, DCI-P3\n"
                "Application: Digital cinema projection\n\n"
                "Phase 1: Grayscale\n"
                "  - Gamma 2.6 EOTF\n"
                "  - Peak white: 48 cd/m^2\n"
                "  - Sequential contrast\n\n"
                "Phase 2: DCI-P3 Gamut\n"
                "  - P3 primaries\n"
                "  - D65 white point\n\n"
                "Phase 3: SMPTE Test patterns\n"
                "  - RP 219-1 patterns\n"
                "  - Color bars verification\n\n"
                "Total: ~18 minutes"
            ),
            CalibrationPreset.HDR_REFERENCE: (
                "=== HDR REFERENCE (ITU-R BT.2111-2) ===\n\n"
                "Standard: ITU-R BT.2111, SMPTE ST.2084\n"
                "Application: HDR mastering/grading\n\n"
                "Phase 1: PQ EOTF\n"
                "  - 31-step PQ ladder\n"
                "  - 0.001 - 1000 nits\n"
                "  - ST.2084 curve\n\n"
                "Phase 2: BT.2020 Gamut\n"
                "  - Wide color gamut\n"
                "  - D65 white point\n\n"
                "Phase 3: HDR10 patterns\n"
                "  - HDR10 test patterns\n"
                "  - dEITP metric\n\n"
                "Total: ~25 minutes"
            )
        }
        
        detail_text = workflow_details.get(preset, workflow_details[CalibrationPreset.STANDARD])
        
        if hasattr(self, 'workflow_detail_text'):
            self.workflow_detail_text.set_text(detail_text)
            self.fig.canvas.draw_idle()
        
        logger.info("[Settings] Calibration method changed to: %s", preset.value)
        print(f"[Settings] Method changed to {preset.value}")
    
    def on_cct_changed(self, label):
        """Target CCT 변경"""
        logger.info("[UI Event] CCT changed to: %s", label)
        cct_map = {'5500K': 5500, '6500K': 6500, '9300K': 9300, 'Native': 0}
        self.config.target_cct = cct_map.get(label, 6500)
        logger.debug("[Settings] CCT value set to: %dK", self.config.target_cct)
        print(f"[Settings] CCT changed to {label}")
        self._update_workflow_preview()
    
    def on_standard_changed(self, label):
        """Color Standard 변경"""
        logger.info("[UI Event] Color Standard changed to: %s", label)
        self.config.target_standard = label
        logger.debug("[Settings] Color Standard updated")
        print(f"[Settings] Standard changed to {label}")
        self._update_workflow_preview()
    
    def on_scan_ports(self, event):
        """COM 포트 스캔"""
        logger.info("[UI Event] Scan Ports button clicked")
        print("[Sensor] Scanning COM ports...")
        
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            self.available_ports = [p.device for p in ports]
            
            logger.info("[Sensor] Found %d ports: %s", len(ports), self.available_ports)
            print(f"[Sensor] Found {len(ports)} ports: {self.available_ports}")
            
            # RadioButton 업데이트
            self._port_labels = ['Virtual'] + self.available_ports
            self.ax_port_radio.clear()
            self.ax_port_radio.set_title('● Sensor', fontsize=9, fontweight='bold', pad=2)
            self.radio_port = RadioButtons(self.ax_port_radio, self._port_labels,
                                           activecolor='dodgerblue')
            for lbl in self.radio_port.labels:
                lbl.set_fontsize(8)
            self.radio_port.on_clicked(self.on_port_selected)
            self.fig.canvas.draw_idle()
            
        except ImportError:
            logger.warning("[Sensor] pyserial not installed")
            print("[Sensor] pyserial not installed - using Virtual Sensor")
    
    def on_port_selected(self, label):
        """COM 포트 선택"""
        logger.info("[UI Event] Port selected: %s", label)
        
        if label == 'Virtual':
            self.sensor_type = 'virtual'
            self.selected_port = None
        else:
            self.sensor_type = 'cr'
            self.selected_port = label
        
        logger.debug("[Sensor] Selected: type=%s, port=%s", self.sensor_type, self.selected_port)
        print(f"[Sensor] Selected: {label}")
    
    def on_connect_sensor(self, event):
        """센서 연결"""
        logger.info("[UI Event] Connect button clicked")
        
        # 기존 센서 해제
        if self.sensor:
            try:
                self.sensor.disconnect()
            except:
                pass
        
        # 새 센서 연결
        try:
            if self.sensor_type == 'virtual':
                logger.info("[Sensor] Connecting to Virtual Sensor...")
                print("[Sensor] Connecting to Virtual Sensor...")
                self.sensor = VirtualSensor(
                    noise_level=0.02, display_colorspace='BT.2020',
                    max_luminance=100.0, black_level=0.05, native_gamma=2.2)
                self.sensor.connect()
                self.sensor_connected = True
                print("[Sensor] Virtual Sensor connected successfully!")
            else:
                logger.info("[Sensor] Connecting to CR Colorimeter on %s...", self.selected_port)
                print(f"[Sensor] Connecting to {self.selected_port}...")
                self.sensor = CRColorimeterSensor(port=self.selected_port)
                self.sensor.connect()
                self.sensor_connected = True
                print(f"[Sensor] CR Colorimeter connected on {self.selected_port}!")
            
            logger.info("[Sensor] Connection successful - type: %s", self.sensor_type)
            
        except Exception as e:
            logger.error("[Sensor] Connection failed: %s", str(e), exc_info=True)
            print(f"[Error] Sensor connection failed: {e}")
            self.sensor_connected = False
            
            # Fallback to Virtual (BT.2020 wide gamut)
            self.sensor = VirtualSensor(
                noise_level=0.02, display_colorspace='BT.2020',
                max_luminance=100.0, black_level=0.05, native_gamma=2.2)
            self.sensor.connect()
            self.sensor_type = 'virtual'
            self.sensor_connected = True
            print("[Sensor] Fallback to Virtual Sensor (BT.2020)")
    
    def on_range_changed(self, label):
        """Signal Range 변경"""
        logger.info("[UI Event] Signal Range changed to: %s", label)
        if 'Full' in label:
            self.config.signal_range = SignalRange.FULL
        else:
            self.config.signal_range = SignalRange.LIMITED
        logger.debug("[Settings] Signal Range set to: %s", self.config.signal_range.value)
        print(f"[Settings] Signal Range changed to {label}")
    
    def on_export_lut(self, event):
        """LUT 내보내기"""
        logger.info("[UI Event] Export LUT button clicked")

        if self.engine_result is None or (
                self.engine_result.lut_1d is None
                and self.engine_result.matrix_3x3 is None
                and self.engine_result.lut_3d is None):
            print("[Export] No LUT data available. Run calibration first.")
            return

        # 파일 저장 경로 선택
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            base_path = filedialog.asksaveasfilename(
                title="Export Calibration LUT",
                defaultextension=".cube",
                filetypes=[("CUBE LUT", "*.cube"), ("All Files", "*.*")])
            root.destroy()
        except Exception as _dial_err:
            logger.warning("[Export] File dialog failed: %s", _dial_err)
            base_path = f"calibration_export_{time.strftime('%Y%m%d_%H%M%S')}.cube"

        if not base_path:
            print("[Export] Cancelled.")
            return

        import os
        stem = os.path.splitext(base_path)[0]
        exported = []

        lut_1d = self.engine_result.lut_1d
        if lut_1d is not None:
            try:
                path = stem + "_1d.cube"
                LUTExporter.export_1d_cube(lut_1d, path)
                exported.append(path)
            except Exception as _e:
                print(f"[Export] 1D LUT export failed: {_e}")

        mat = self.engine_result.matrix_3x3
        if mat is not None:
            try:
                path = stem + "_matrix.json"
                LUTExporter.export_3x3_matrix(mat, path)
                exported.append(path)
            except Exception as _e:
                print(f"[Export] 3x3 matrix export failed: {_e}")

        lut_3d = self.engine_result.lut_3d
        if lut_3d is not None:
            try:
                path = stem + "_3d.cube"
                LUTExporter.export_3d_cube(lut_3d, path)
                exported.append(path)
            except Exception as _e:
                print(f"[Export] 3D LUT export failed: {_e}")

        if exported:
            print(f"[Export] Exported {len(exported)} file(s):")
            for p in exported:
                print(f"  {p}")
        else:
            print("[Export] Nothing exported.")
        logger.info("[Export] Done: %d file(s)", len(exported))
    
    def on_save_report(self, event):
        """리포트 저장"""
        logger.info("[UI Event] Save Report button clicked")

        summary = (self.engine_result.summary
                   if self.engine_result and self.engine_result.summary
                   else {})
        if not summary:
            print("[Report] No calibration summary available. Run calibration first.")
            return

        # 파일 저장 경로 선택
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            save_path = filedialog.asksaveasfilename(
                title="Save Calibration Report",
                defaultextension=".txt",
                filetypes=[("Text Report", "*.txt"),
                           ("JSON", "*.json"),
                           ("All Files", "*.*")])
            root.destroy()
        except Exception as _dial_err:
            logger.warning("[Report] File dialog failed: %s", _dial_err)
            save_path = f"calibration_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"

        if not save_path:
            print("[Report] Cancelled.")
            return

        try:
            if save_path.lower().endswith('.json'):
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, default=str)
            else:
                text = CalibrationAnalyzer.format_report(summary)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            print(f"[Report] Saved to: {save_path}")
            logger.info("[Report] Saved to: %s", save_path)
        except Exception as _e:
            print(f"[Report] Save failed: {_e}")
            logger.error("[Report] Save failed: %s", _e)
    
    def _show_custom_settings_window(self):
        """Custom Calibration 설정 창 표시 (matplotlib 위젯 사용)"""
        logger.info("[Custom Settings] Opening Custom Settings window")
        print("[Custom Settings] Creating interactive settings window...")
        
        try:
            from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
            
            # matplotlib Figure로 설정 창 생성
            settings_fig = plt.figure(figsize=(10, 8))
            settings_fig.canvas.manager.set_window_title("Custom Calibration Settings")
            settings_fig.suptitle('Custom Calibration Configuration', 
                                 fontsize=14, fontweight='bold')
            
            # 설정 초기값
            config = {
                'grayscale_steps': 21,
                'white_only': False,
                'color_preset': 'Standard (RGBCMYW, 7 patches)',
                'lut_size': 33,
                'verify_preset': 'ColorChecker 24',
                'settle_time': 0.5,
                'averaging': 1
            }
            
            # Phase 1: Grayscale Settings
            ax_gray_label = plt.axes([0.05, 0.85, 0.4, 0.03])
            ax_gray_label.axis('off')
            ax_gray_label.text(0, 0.5, 'Phase 1: Grayscale Settings', 
                             fontsize=11, fontweight='bold', va='center')
            
            # Grayscale Steps Slider
            ax_steps = plt.axes([0.15, 0.78, 0.3, 0.03])
            slider_steps = Slider(ax_steps, 'Steps', 5, 101, valinit=21, valstep=2)
            
            # White-only CheckBox
            ax_white = plt.axes([0.05, 0.72, 0.4, 0.05])
            check_white = CheckButtons(ax_white, ['White-only (RGBW separated)'], [False])
            
            # Phase 2: Color Gamut Settings
            ax_color_label = plt.axes([0.05, 0.63, 0.4, 0.03])
            ax_color_label.axis('off')
            ax_color_label.text(0, 0.5, 'Phase 2: Color Gamut Settings', 
                              fontsize=11, fontweight='bold', va='center')
            
            # Color Preset RadioButtons
            ax_color = plt.axes([0.05, 0.45, 0.4, 0.16])
            radio_color = RadioButtons(ax_color, [
                'Minimal (RGB only, 3 patches)',
                'Standard (RGBCMYW, 7 patches)',
                'Extended (17 patches)',
                'Professional (26 patches)'
            ], active=1)
            
            # 3D LUT Size RadioButtons
            ax_lut_label = plt.axes([0.55, 0.63, 0.4, 0.03])
            ax_lut_label.axis('off')
            ax_lut_label.text(0, 0.5, '3D LUT Size', fontsize=10, fontweight='bold', va='center')
            
            ax_lut = plt.axes([0.55, 0.50, 0.15, 0.12])
            radio_lut = RadioButtons(ax_lut, ['9', '17', '33', '65'], active=2)
            
            # Phase 3: Verification Settings
            ax_verify_label = plt.axes([0.05, 0.36, 0.4, 0.03])
            ax_verify_label.axis('off')
            ax_verify_label.text(0, 0.5, 'Phase 3: Verification Settings', 
                               fontsize=11, fontweight='bold', va='center')
            
            ax_verify = plt.axes([0.05, 0.24, 0.4, 0.10])
            radio_verify = RadioButtons(ax_verify, [
                'Quick (11 grayscale)',
                'ColorChecker 24',
                'ColorChecker SG 140'
            ], active=1)
            
            # Measurement Settings
            ax_meas_label = plt.axes([0.55, 0.36, 0.4, 0.03])
            ax_meas_label.axis('off')
            ax_meas_label.text(0, 0.5, 'Measurement Settings', 
                             fontsize=11, fontweight='bold', va='center')
            
            # Settle Time Slider
            ax_settle = plt.axes([0.65, 0.30, 0.25, 0.03])
            slider_settle = Slider(ax_settle, 'Settle (s)', 0.1, 5.0, valinit=0.5, valfmt='%.1f')
            
            # Averaging Slider
            ax_avg = plt.axes([0.65, 0.25, 0.25, 0.03])
            slider_avg = Slider(ax_avg, 'Averaging', 1, 10, valinit=1, valstep=1, valfmt='%d')
            
            # 예상 시간 표시
            ax_estimate = plt.axes([0.05, 0.14, 0.9, 0.06])
            ax_estimate.axis('off')
            estimate_text = ax_estimate.text(0.5, 0.5, '', fontsize=12, fontweight='bold',
                                            ha='center', va='center', color='blue')
            
            # 예상 시간 계산 함수
            def calculate_estimate():
                steps = int(slider_steps.val)
                white_only = check_white.get_status()[0]
                
                if white_only:
                    gray_patches = steps
                else:
                    gray_patches = steps * 4  # R, G, B, W
                
                color_map = {
                    'Minimal (RGB only, 3 patches)': 3,
                    'Standard (RGBCMYW, 7 patches)': 7,
                    'Extended (17 patches)': 17,
                    'Professional (26 patches)': 26
                }
                color_patches = color_map.get(config['color_preset'], 7)
                
                verify_map = {
                    'Quick (11 grayscale)': 11,
                    'ColorChecker 24': 24,
                    'ColorChecker SG 140': 140
                }
                verify_patches = verify_map.get(config['verify_preset'], 24)
                
                total = gray_patches + color_patches + verify_patches
                time_min = total * (slider_settle.val + 0.3) * slider_avg.val / 60.0
                
                estimate_text.set_text(f'Total: {total} patches, Estimated Time: ~{time_min:.1f} minutes')
                settings_fig.canvas.draw_idle()
                
                return total, time_min
            
            # 위젯 변경 이벤트 연결
            def update_estimate(val):
                calculate_estimate()
            
            slider_steps.on_changed(update_estimate)
            slider_settle.on_changed(update_estimate)
            slider_avg.on_changed(update_estimate)
            
            def on_white_changed(label):
                calculate_estimate()
            check_white.on_clicked(on_white_changed)
            
            def on_color_changed(label):
                config['color_preset'] = label
                calculate_estimate()
            radio_color.on_clicked(on_color_changed)
            
            def on_lut_changed(label):
                config['lut_size'] = int(label)
                calculate_estimate()
            radio_lut.on_clicked(on_lut_changed)
            
            def on_verify_changed(label):
                config['verify_preset'] = label
                calculate_estimate()
            radio_verify.on_clicked(on_verify_changed)
            
            # 초기 예상 시간 계산
            calculate_estimate()
            
            # Apply 버튼
            ax_apply = plt.axes([0.25, 0.03, 0.2, 0.06])
            btn_apply = Button(ax_apply, 'Apply Settings', color='lightgreen', hovercolor='green')
            
            def on_apply(event):
                total, time_min = calculate_estimate()
                
                logger.info("[Custom Settings] Settings applied")
                print(f"\n[Custom] Custom configuration applied: {total} patches, ~{time_min:.1f} min")
                
                # Workflow Detail 업데이트
                steps = int(slider_steps.val)
                white_only = check_white.get_status()[0]
                gray_patches = steps if white_only else steps * 4
                
                color_map = {
                    'Minimal (RGB only, 3 patches)': 3,
                    'Standard (RGBCMYW, 7 patches)': 7,
                    'Extended (17 patches)': 17,
                    'Professional (26 patches)': 26
                }
                color_patches = color_map.get(config['color_preset'], 7)
                
                verify_map = {
                    'Quick (11 grayscale)': 11,
                    'ColorChecker 24': 24,
                    'ColorChecker SG 140': 140
                }
                verify_patches = verify_map.get(config['verify_preset'], 24)
                
                custom_detail = (
                    f"=== CUSTOM Calibration (~{time_min:.0f} min, {total} patches) ===\n\n"
                    f"Phase 1: Grayscale\n"
                    f"  - {steps}-step {'White-only' if white_only else 'RGBW separation'}\n"
                    f"  - Measurements: {gray_patches} patches\n\n"
                    f"Phase 2: Color Gamut\n"
                    f"  - {config['color_preset']}\n"
                    f"  - 3D LUT: {config['lut_size']}^3 cube\n"
                    f"  - Measurements: {color_patches} patches\n\n"
                    f"Phase 3: Verification\n"
                    f"  - {config['verify_preset']}\n"
                    f"  - Measurements: {verify_patches} patches\n\n"
                    f"Measurement Settings:\n"
                    f"  - Settle time: {slider_settle.val:.1f}s\n"
                    f"  - Averaging: {int(slider_avg.val)}x\n\n"
                    f"Total: {total} patches, ~{time_min:.1f} minutes"
                )
                
                if hasattr(self, 'workflow_detail_text'):
                    self.workflow_detail_text.set_text(custom_detail)
                    self.fig.canvas.draw_idle()
                
                plt.close(settings_fig)
                logger.info("[Custom Settings] Window closed after apply")
            
            btn_apply.on_clicked(on_apply)
            
            # Cancel 버튼
            ax_cancel = plt.axes([0.55, 0.03, 0.2, 0.06])
            btn_cancel = Button(ax_cancel, 'Cancel', color='lightcoral', hovercolor='red')
            
            def on_cancel(event):
                logger.info("[Custom Settings] Settings cancelled")
                print("[Custom] Cancelled")
                plt.close(settings_fig)
            
            btn_cancel.on_clicked(on_cancel)
            
            # 창 표시
            plt.show(block=False)
            
            # 창 활성화
            if hasattr(settings_fig.canvas.manager, 'window'):
                try:
                    settings_fig.canvas.manager.window.raise_()
                    settings_fig.canvas.manager.window.activateWindow()
                except:
                    pass
            
            logger.info("[Custom Settings] Interactive window displayed successfully")
            print("[Custom Settings] Configure your settings and click Apply")
            
        except Exception as e:
            logger.error(f"[Custom Settings] Failed to create window: {e}", exc_info=True)
            print(f"[ERROR] Custom Settings window failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 기존 Tkinter 대화상자 메서드는 제거됨 (위의 matplotlib 버전으로 대체)
    
    def on_pattern_radio_changed(self, label):
        if not hasattr(self, '_tk_root') or self._tk_root is None:
            try:
                self._tk_root = tk.Tk()
                self._tk_root.withdraw()  # root 창 숨기기
                logger.info("[Custom Settings] Created Tkinter root window")
            except Exception as e:
                logger.error(f"[Custom Settings] Failed to create Tk root: {e}")
                print(f"[ERROR] Failed to open Custom Settings dialog: {e}")
                return
        
        # Tkinter Dialog 생성
        try:
            dialog = tk.Toplevel(self._tk_root)
            dialog.title("Custom Calibration Settings")
            dialog.geometry("500x650")
            dialog.resizable(False, False)
            
            # 대화상자를 최상위로
            dialog.lift()
            dialog.attributes('-topmost', True)
            dialog.after_idle(dialog.attributes, '-topmost', False)
            
        except Exception as e:
            logger.error(f"[Custom Settings] Failed to create dialog: {e}")
            print(f"[ERROR] Failed to open dialog: {e}")
            return
        
        # 스타일
        try:
            style = ttk.Style()
            style.theme_use('clam')
        except:
            pass
        
        # 메인 프레임
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 타이틀
        title = ttk.Label(main_frame, text="Custom Calibration Settings", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 20))
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Grayscale Settings
        # ═══════════════════════════════════════════════════════════════
        
        phase1_frame = ttk.LabelFrame(main_frame, text="Phase 1: Grayscale", padding="10")
        phase1_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(phase1_frame, text="Grayscale Levels:").grid(row=0, column=0, sticky=tk.W, pady=5)
        grayscale_steps = tk.IntVar(value=11)
        grayscale_spinbox = ttk.Spinbox(phase1_frame, from_=5, to=101, increment=2,
                                        textvariable=grayscale_steps, width=10)
        grayscale_spinbox.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(phase1_frame, text="(5-101 steps, default: 11)").grid(row=0, column=2, sticky=tk.W)
        
        white_only = tk.BooleanVar(value=False)
        white_only_check = ttk.Checkbutton(phase1_frame, text="White-only (빠름, 3배 단축)",
                                           variable=white_only)
        white_only_check.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Color Gamut Settings
        # ═══════════════════════════════════════════════════════════════
        
        phase2_frame = ttk.LabelFrame(main_frame, text="Phase 2: Color Gamut", padding="10")
        phase2_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(phase2_frame, text="Color Patch Set:").grid(row=0, column=0, sticky=tk.W, pady=5)
        color_preset = tk.StringVar(value="Standard")
        color_combo = ttk.Combobox(phase2_frame, textvariable=color_preset,
                                   values=["Minimal (RGB only, 3 patches)",
                                          "Standard (RGBCMYW, 7 patches)",
                                          "Extended (17 patches)",
                                          "Professional (26 patches)"],
                                   width=30, state='readonly')
        color_combo.current(1)
        color_combo.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(phase2_frame, text="3D LUT Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        lut_size = tk.IntVar(value=17)
        lut_combo = ttk.Combobox(phase2_frame, textvariable=lut_size,
                                values=[9, 17, 33, 65],
                                width=10, state='readonly')
        lut_combo.current(1)
        lut_combo.grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(phase2_frame, text="(9=빠름, 65=정밀)").grid(row=1, column=2, sticky=tk.W)
        
        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Verification Settings
        # ═══════════════════════════════════════════════════════════════
        
        phase3_frame = ttk.LabelFrame(main_frame, text="Phase 3: Verification", padding="10")
        phase3_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(phase3_frame, text="Verification Patches:").grid(row=0, column=0, sticky=tk.W, pady=5)
        verify_preset = tk.StringVar(value="ColorChecker 24")
        verify_combo = ttk.Combobox(phase3_frame, textvariable=verify_preset,
                                    values=["Quick (11 grayscale)",
                                           "ColorChecker 24",
                                           "ColorChecker SG 140"],
                                    width=25, state='readonly')
        verify_combo.current(1)
        verify_combo.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # ═══════════════════════════════════════════════════════════════
        # Measurement Settings
        # ═══════════════════════════════════════════════════════════════
        
        meas_frame = ttk.LabelFrame(main_frame, text="Measurement Settings", padding="10")
        meas_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(meas_frame, text="Settle Time (sec):").grid(row=0, column=0, sticky=tk.W, pady=5)
        settle_time = tk.DoubleVar(value=0.5)
        settle_spin = ttk.Spinbox(meas_frame, from_=0.1, to=5.0, increment=0.1,
                                 textvariable=settle_time, width=10, format="%.1f")
        settle_spin.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(meas_frame, text="(패턴 안정화 대기 시간)").grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(meas_frame, text="Averaging Count:").grid(row=1, column=0, sticky=tk.W, pady=5)
        averaging = tk.IntVar(value=1)
        avg_spin = ttk.Spinbox(meas_frame, from_=1, to=10, increment=1,
                              textvariable=averaging, width=10)
        avg_spin.grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(meas_frame, text="(측정 평균 횟수)").grid(row=1, column=2, sticky=tk.W)
        
        # ═══════════════════════════════════════════════════════════════
        # 예상 시간 계산
        # ═══════════════════════════════════════════════════════════════
        
        estimate_frame = ttk.Frame(main_frame)
        estimate_frame.pack(fill=tk.X, pady=10)
        
        estimate_label = ttk.Label(estimate_frame, text="",
                                   font=('Arial', 10, 'bold'), foreground='blue')
        estimate_label.pack()
        
        def update_estimate():
            gray = grayscale_steps.get()
            if white_only.get():
                gray_patches = gray
            else:
                gray_patches = gray * 4  # R, G, B, W
            
            color_map = {
                "Minimal (RGB only, 3 patches)": 3,
                "Standard (RGBCMYW, 7 patches)": 7,
                "Extended (17 patches)": 17,
                "Professional (26 patches)": 26
            }
            color_patches = color_map.get(color_preset.get(), 7)
            
            verify_map = {
                "Quick (11 grayscale)": 11,
                "ColorChecker 24": 24,
                "ColorChecker SG 140": 140
            }
            verify_patches = verify_map.get(verify_preset.get(), 24)
            
            total_patches = gray_patches + color_patches + verify_patches
            est_time = total_patches * (settle_time.get() + 0.3) * averaging.get() / 60.0
            
            estimate_label.config(
                text=f"Total: {total_patches} patches, ~{est_time:.1f} minutes"
            )
        
        # 초기 예상 시간 계산
        update_estimate()
        
        # 값 변경 시 예상 시간 업데이트 (안전한 콜백 처리)
        def safe_update(*args):
            try:
                update_estimate()
            except Exception as e:
                logger.warning(f"[Custom Settings] Estimate update failed: {e}")
        
        for var in [grayscale_steps, white_only, color_preset, verify_preset, 
                   settle_time, averaging, lut_size]:
            try:
                if isinstance(var, (tk.IntVar, tk.DoubleVar, tk.BooleanVar, tk.StringVar)):
                    var.trace('w', safe_update)
            except Exception as e:
                logger.warning(f"[Custom Settings] Trace setup failed: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # 버튼
        # ═══════════════════════════════════════════════════════════════
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        def on_ok():
            logger.info("[Custom Settings] Custom settings confirmed")
            logger.info("[Custom Settings] Grayscale: %d steps, White-only: %s",
                       grayscale_steps.get(), white_only.get())
            logger.info("[Custom Settings] Color: %s", color_preset.get())
            logger.info("[Custom Settings] Verify: %s", verify_preset.get())
            logger.info("[Custom Settings] Settle: %.1fs, Averaging: %d",
                       settle_time.get(), averaging.get())
            
            # Custom Workflow Detail 업데이트
            gray = grayscale_steps.get()
            gray_patches = gray if white_only.get() else gray * 4
            color_patches = {
                "Minimal (RGB only, 3 patches)": 3,
                "Standard (RGBCMYW, 7 patches)": 7,
                "Extended (17 patches)": 17,
                "Professional (26 patches)": 26
            }.get(color_preset.get(), 7)
            verify_patches = {
                "Quick (11 grayscale)": 11,
                "ColorChecker 24": 24,
                "ColorChecker SG 140": 140
            }.get(verify_preset.get(), 24)
            
            total = gray_patches + color_patches + verify_patches
            est_time = total * (settle_time.get() + 0.3) * averaging.get() / 60.0
            
            custom_detail = (
                f"=== CUSTOM Calibration ({est_time:.1f} min, {total} patches) ===\\n\\n"
                f"Custom Configuration\\n\\n"
                f"Phase 1: Grayscale\\n"
                f"  - {gray}-step ladder\\n"
                f"  - White-only: {'Yes' if white_only.get() else 'No'}\\n"
                f"  - Measurements: {gray_patches} patches\\n\\n"
                f"Phase 2: Color Gamut\\n"
                f"  - {color_preset.get()}\\n"
                f"  - 3D LUT: {lut_size.get()}³ cube\\n"
                f"  - Measurements: {color_patches} patches\\n\\n"
                f"Phase 3: Verification\\n"
                f"  - {verify_preset.get()}\\n"
                f"  - Measurements: {verify_patches} patches\\n\\n"
                f"Measurement Settings:\\n"
                f"  - Settle time: {settle_time.get():.1f}s\\n"
                f"  - Averaging: {averaging.get()}x\\n\\n"
                f"Total: {total} patches, ~{est_time:.1f} minutes"
            )
            
            if hasattr(self, 'workflow_detail_text'):
                self.workflow_detail_text.set_text(custom_detail)
                self.fig.canvas.draw_idle()
            
            print(f"[Settings] Custom calibration configured: {total} patches, ~{est_time:.1f} min")
            dialog.destroy()
        
        def on_cancel():
            logger.info("[Custom Settings] Custom settings cancelled")
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=15).pack(side=tk.LEFT, padx=5)
        
        # 먼저 모든 위젯 레이아웃 계산
        dialog.update_idletasks()
        
        # Dialog 중앙 정렬
        try:
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            dialog_width = dialog.winfo_reqwidth()
            dialog_height = dialog.winfo_reqheight()
            x = (screen_width // 2) - (dialog_width // 2)
            y = (screen_height // 2) - (dialog_height // 2)
            dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
            logger.info(f"[Custom Settings] Dialog positioned at {x},{y} size {dialog_width}x{dialog_height}")
        except Exception as e:
            logger.warning(f"[Custom Settings] Failed to center dialog: {e}")
        
        # Dialog 표시
        try:
            dialog.deiconify()
            dialog.lift()
            dialog.focus_force()
            dialog.grab_set()
            logger.info("[Custom Settings] Dialog displayed and modal")
        except Exception as e:
            logger.warning(f"[Custom Settings] Display setup failed: {e}")
        
        # Tkinter 이벤트를 처리하기 위한 타이머 (Qt Timer 사용)
        from PyQt6.QtCore import QTimer
        
        self._tk_dialog = dialog
        self._tk_timer = QTimer()
        
        def pump_events():
            """Tkinter 이벤트 펌프 - Qt 타이머에서 안전하게 호출"""
            try:
                if self._tk_dialog and self._tk_dialog.winfo_exists():
                    # Tkinter 이벤트 큐 처리
                    self._tk_dialog.update()
                else:
                    # 대화상자 닫힘
                    if hasattr(self, '_tk_timer') and self._tk_timer:
                        self._tk_timer.stop()
                        self._tk_timer = None
                    self._tk_dialog = None
            except Exception as e:
                logger.warning(f"[Custom Settings] Event pump error: {e}")
                if hasattr(self, '_tk_timer') and self._tk_timer:
                    self._tk_timer.stop()
        
        self._tk_timer.timeout.connect(pump_events)
        self._tk_timer.start(50)  # 50ms마다 Tkinter 이벤트 처리
        
        logger.info("[Custom Settings] Event pump started with Qt Timer")
    
    def on_pattern_radio_changed(self, label):
        """Pattern RadioButton 변경 이벤트"""
        logger.info("[UI Event] Pattern radio button changed to: %s", label)
        logger.debug("[Pattern Selection] Selected pattern: %s", label)
    
    def on_phase_checked(self, label):
        """Phase 체크박스 변경 이벤트"""
        logger.info("[UI Event] Phase checkbox toggled: %s", label)
        self._update_workflow_preview()
        logger.debug("[Workflow] Phase selection updated")
    
    def on_show_pattern_gallery(self, event):
        """Show Pattern Gallery 버튼 클릭"""
        logger.info("[UI Event] Show Pattern Gallery button clicked")
        
        # 사용 가능한 패턴 목록 (산업 표준 기반)
        # References:
        #   - X-Rite ColorChecker: ISO 17321-1
        #   - SMPTE: ECR 1-1978 (75%), RP 219-1:2014 (100%)
        #   - EBU: Tech 3373 (PAL/Europe)
        #   - ITU-R: BT.709, BT.1886, BT.2020, BT.2111-2
        #   - SMPTE ST.2084: PQ EOTF (HDR)
        #   - CIE 015:2018: Colorimetry
        if HAS_INDUSTRY_PATTERNS:
            patterns = [
                'Grayscale 11',            # ITU-R BT.1886 (11-step)
                'Grayscale 21',            # CalMAN Standard (21-step, 5%)
                'ColorChecker Classic',    # X-Rite 24 patches (ISO 17321-1)
                'ColorChecker Digital',    # X-Rite Digital variant
                'SMPTE Bars 75%',          # SMPTE ECR 1-1978 (NTSC)
                'SMPTE Bars 100%',         # SMPTE RP 219-1:2014 (HD)
                'EBU Bars 75%',            # EBU Tech 3373 (PAL/Europe)
                'Rec.709 Saturation',      # ITU-R BT.709-6 (33 levels)
                'Rec.2020 Gamut',          # ITU-R BT.2020-2 (Wide Gamut)
                'RGB Primary',             # CIE 015:2018 Primary Colors
                'RGBCMY + White',          # Basic Color Set
                'Skin Tone Patches',       # ITU-R BT.2111 Appendix 1
                'HDR PQ Ramp'              # SMPTE ST.2084 (0.1~10000 nits)
            ]
        else:
            patterns = [
                'Grayscale 11',
                'RGB Basic',
                'RGBCMY Extended',
                'White Point',
                'Black Level'
            ]
        
        logger.debug("[Pattern Gallery] Opening gallery with %d patterns", len(patterns))
        print(f"[Pattern Gallery] Opening gallery with {len(patterns)} patterns")
        
        # Pattern Gallery 창 열기 - 참조 유지 필수!
        try:
            logger.info("[Pattern Gallery] Creating PatternGalleryWindow instance...")
            print("[Pattern Gallery] Step 1: Creating window...")
            
            self._gallery_window = PatternGalleryWindow(self, patterns)
            
            logger.info("[Pattern Gallery] PatternGalleryWindow created successfully")
            print("[Pattern Gallery] Step 2: Window created successfully")
            
            logger.info("[Pattern Gallery] Calling gallery.show()...")
            print("[Pattern Gallery] Step 3: Showing window...")
            
            self._gallery_window.show()
            
            logger.info("[Pattern Gallery] Gallery window opened and displayed")
            print("[Pattern Gallery] Step 4: Window should be visible now!")
            print(f"[Pattern Gallery] Figure ID: {self._gallery_window.fig.number}")
            print(f"[Pattern Gallery] Figure visible: {self._gallery_window.fig.get_visible()}")
            print("[Pattern Gallery] >> If you don't see the window, check your taskbar or other monitors")
            
        except Exception as e:
            logger.error(f"[Pattern Gallery] Failed to open gallery: {e}", exc_info=True)
            print(f"[ERROR] Failed to open Pattern Gallery: {e}")
            import traceback
            traceback.print_exc()
    
    def on_pattern_selected_from_gallery(self, pattern_name):
        """Pattern Gallery에서 패턴 선택 시 호출"""
        logger.info("[Pattern Selection] Pattern '%s' selected from gallery", pattern_name)
        
        # RadioButton 업데이트 (매칭되는 항목이 있으면)
        for i, label in enumerate(self.radio_pattern.labels):
            if pattern_name in label.get_text() or label.get_text() in pattern_name:
                self.radio_pattern.set_active(i)
                logger.debug("[Pattern Selection] RadioButton updated to index %d (%s)", i, label.get_text())
                break
        
        # UI 새로고침
        self.fig.canvas.draw_idle()
        logger.debug("[Pattern Selection] UI refreshed with new pattern selection")
        print(f"[Pattern] Selected: {pattern_name}")
    
    def on_pattern_detail_requested(self, pattern_name):
        """패턴 세부 레벨 창 열기 요청"""
        logger.info("[Pattern Detail] " + "="*50)
        logger.info("[Pattern Detail] Detail window requested for: %s", pattern_name)
        logger.debug("[Pattern Detail] Starting detail window creation process...")
        
        try:
            # 패턴별 세부 레벨 정의
            logger.debug("[Pattern Detail] Step 1: Getting pattern levels for: %s", pattern_name)
            pattern_levels = self._get_pattern_levels(pattern_name)
            logger.debug("[Pattern Detail] Step 1 completed: _get_pattern_levels() returned %d levels", 
                        len(pattern_levels) if pattern_levels else 0)
            
            if not pattern_levels:
                logger.warning("[Pattern Detail] No levels defined for pattern: %s", pattern_name)
                print(f"[Pattern Detail] No detail levels for {pattern_name}")
                return
            
            logger.info("[Pattern Detail] Found %d levels for pattern '%s'", len(pattern_levels), pattern_name)
            logger.debug("[Pattern Detail] Step 2: Creating PatternDetailWindow instance...")
            
            detail_window = PatternDetailWindow(self, pattern_name, pattern_levels)
            logger.debug("[Pattern Detail] Step 2 completed: PatternDetailWindow created")
            
            logger.debug("[Pattern Detail] Step 3: Calling detail_window.show()...")
            detail_window.show()
            logger.debug("[Pattern Detail] Step 3 completed: show() called")
            
            logger.info("[Pattern Detail] OK - Detail window opened successfully for '%s'", pattern_name)
            print(f"[Pattern Detail] OK - Opened detail window for '{pattern_name}' with {len(pattern_levels)} levels")
            
        except Exception as e:
            logger.error("[Pattern Detail] FAILED - Failed to open detail window: %s", str(e), exc_info=True)
            print(f"[Error] Failed to open detail window for {pattern_name}: {e}")
    
    def _get_pattern_levels(self, pattern_name):
        """패턴 이름에 따라 세부 레벨 반환"""
        logger.debug("[Pattern Levels] " + "="*50)
        logger.debug("[Pattern Levels] Getting levels for pattern: '%s'", pattern_name)
        
        try:
            if 'Grayscale 11' in pattern_name:
                # 11-step grayscale
                levels = []
                for i in range(11):
                    val = i / 10.0
                    levels.append((f"Gray_{int(val*255)}", (val, val, val)))
                logger.info("[Pattern Levels] Generated %d grayscale levels (11-step)", len(levels))
                return levels
            
            elif 'Grayscale 21' in pattern_name or 'Grayscale' in pattern_name:
                # 21-step grayscale
                levels = []
                for i in range(21):
                    val = i / 20.0
                    levels.append((f"Gray_{int(val*255)}", (val, val, val)))
                logger.info("[Pattern Levels] Generated %d grayscale levels (21-step)", len(levels))
                return levels
            
            elif 'ColorChecker' in pattern_name or 'Checker' in pattern_name:
                # ColorChecker 24 patches
                levels = [
                    ("Dark Skin", (0.45, 0.32, 0.25)),
                    ("Light Skin", (0.77, 0.58, 0.49)),
                    ("Blue Sky", (0.35, 0.44, 0.64)),
                    ("Foliage", (0.33, 0.42, 0.29)),
                    ("Blue Flower", (0.47, 0.47, 0.67)),
                    ("Bluish Green", (0.40, 0.73, 0.67)),
                    ("Orange", (0.84, 0.46, 0.28)),
                    ("Purplish Blue", (0.31, 0.35, 0.63)),
                    ("Moderate Red", (0.76, 0.33, 0.37)),
                    ("Purple", (0.36, 0.25, 0.40)),
                    ("Yellow Green", (0.60, 0.73, 0.35)),
                    ("Orange Yellow", (0.85, 0.60, 0.27)),
                    ("Blue", (0.20, 0.25, 0.56)),
                    ("Green", (0.32, 0.59, 0.36)),
                    ("Red", (0.70, 0.24, 0.26)),
                    ("Yellow", (0.85, 0.76, 0.30)),
                    ("Magenta", (0.76, 0.33, 0.56)),
                    ("Cyan", (0.24, 0.51, 0.67)),
                    ("White", (0.96, 0.96, 0.96)),
                    ("Neutral 8", (0.78, 0.78, 0.78)),
                    ("Neutral 6.5", (0.59, 0.59, 0.59)),
                    ("Neutral 5", (0.39, 0.39, 0.39)),
                    ("Neutral 3.5", (0.20, 0.20, 0.20)),
                    ("Black", (0.03, 0.03, 0.03))
                ]
                logger.info("[Pattern Levels] Generated %d ColorChecker patches", len(levels))
                return levels
            
            elif 'SMPTE' in pattern_name:
                # SMPTE Bars
                # SMPTE ECR 1-1978 (75%) / SMPTE RP 219-1:2014 (100% HD)
                levels = [
                    ("White 100%", (1.0, 1.0, 1.0)),
                    ("Yellow 75%", (0.75, 0.75, 0.0)),
                    ("Cyan 75%", (0.0, 0.75, 0.75)),
                    ("Green 75%", (0.0, 0.75, 0.0)),
                    ("Magenta 75%", (0.75, 0.0, 0.75)),
                    ("Red 75%", (0.75, 0.0, 0.0)),
                    ("Blue 75%", (0.0, 0.0, 0.75))
                ]
                logger.info("[Pattern Levels] Generated %d SMPTE color bars", len(levels))
                return levels
            
            elif 'EBU' in pattern_name:
                # EBU Color Bars (European Broadcasting Union)
                # EBU Tech 3373, IBA Colour Bars (PAL standard)
                levels = [
                    ("White 100%", (1.0, 1.0, 1.0)),
                    ("Yellow 75%", (0.75, 0.75, 0.0)),
                    ("Cyan 75%", (0.0, 0.75, 0.75)),
                    ("Green 75%", (0.0, 0.75, 0.0)),
                    ("Magenta 75%", (0.75, 0.0, 0.75)),
                    ("Red 75%", (0.75, 0.0, 0.0)),
                    ("Blue 75%", (0.0, 0.0, 0.75)),
                    ("Black", (0.0, 0.0, 0.0))
                ]
                logger.info("[Pattern Levels] Generated %d EBU color bars (PAL)", len(levels))
                return levels
            
            elif 'Rec.709' in pattern_name or '709' in pattern_name or 'Saturation' in pattern_name:
                # RGB Saturation Sweep - 더 세밀하게
                levels = []
                logger.debug("[Pattern Levels] Creating Rec.709 saturation sweep...")
                
                # Red saturation
                for i in range(11):
                    val = i / 10.0
                    levels.append((f"Red_{int(val*100)}%", (val, 0, 0)))
                
                # Green saturation
                for i in range(11):
                    val = i / 10.0
                    levels.append((f"Green_{int(val*100)}%", (0, val, 0)))
                
                # Blue saturation
                for i in range(11):
                    val = i / 10.0
                    levels.append((f"Blue_{int(val*100)}%", (0, 0, val)))
                
                logger.info("[Pattern Levels] Generated %d Rec.709 saturation levels (RGB: 11+11+11)", len(levels))
                return levels
            
            elif 'Rec.2020' in pattern_name or '2020' in pattern_name:
                # Rec.2020 Gamut - wide color gamut
                levels = [
                    ("Red 100%", (1.0, 0.0, 0.0)),
                    ("Green 100%", (0.0, 1.0, 0.0)),
                    ("Blue 100%", (0.0, 0.0, 1.0)),
                    ("Cyan 100%", (0.0, 1.0, 1.0)),
                    ("Magenta 100%", (1.0, 0.0, 1.0)),
                    ("Yellow 100%", (1.0, 1.0, 0.0)),
                    ("White 100%", (1.0, 1.0, 1.0)),
                    ("Red 75%", (0.75, 0.0, 0.0)),
                    ("Green 75%", (0.0, 0.75, 0.0)),
                    ("Blue 75%", (0.0, 0.0, 0.75)),
                ]
                logger.info("[Pattern Levels] Generated %d Rec.2020 gamut levels", len(levels))
                return levels
            
            elif 'RGB' in pattern_name and 'Primary' in pattern_name:
                # RGB Primary colors
                levels = [
                    ("Red 100%", (1.0, 0.0, 0.0)),
                    ("Red 75%", (0.75, 0.0, 0.0)),
                    ("Red 50%", (0.5, 0.0, 0.0)),
                    ("Red 25%", (0.25, 0.0, 0.0)),
                    ("Green 100%", (0.0, 1.0, 0.0)),
                    ("Green 75%", (0.0, 0.75, 0.0)),
                    ("Green 50%", (0.0, 0.5, 0.0)),
                    ("Green 25%", (0.0, 0.25, 0.0)),
                    ("Blue 100%", (0.0, 0.0, 1.0)),
                    ("Blue 75%", (0.0, 0.0, 0.75)),
                    ("Blue 50%", (0.0, 0.0, 0.5)),
                    ("Blue 25%", (0.0, 0.0, 0.25)),
                ]
                logger.info("[Pattern Levels] Generated %d RGB primary levels", len(levels))
                return levels
            
            elif 'RGBCMY' in pattern_name or 'White' in pattern_name:
                # RGBCMY + White (7 colors)
                levels = [
                    ("Red", (1.0, 0.0, 0.0)),
                    ("Green", (0.0, 1.0, 0.0)),
                    ("Blue", (0.0, 0.0, 1.0)),
                    ("Cyan", (0.0, 1.0, 1.0)),
                    ("Magenta", (1.0, 0.0, 1.0)),
                    ("Yellow", (1.0, 1.0, 0.0)),
                    ("White", (1.0, 1.0, 1.0)),
                ]
                logger.info("[Pattern Levels] Generated %d RGBCMY+White levels", len(levels))
                return levels
            
            elif 'HDR' in pattern_name or 'PQ' in pattern_name:
                # HDR PQ Ramp - SMPTE ST.2084 PQ EOTF (Perceptual Quantizer)
                # ITU-R BT.2111-2 HDR Color Bar 기반
                levels = []
                logger.debug("[Pattern Levels] Creating HDR PQ ramp (SMPTE ST.2084)...")
                
                try:
                    from calibration_engine import ColorScience
                    import numpy as np
                    # 0.1 nits ~ 10000 nits (21 steps) - 비선형 분포
                    for i in range(21):
                        # 로그 분포: 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000 nits
                        if i < 10:
                            nits = 0.1 * (5.0 ** (i / 9.0))  # 0.1 ~ 0.5 nits
                        elif i < 15:
                            nits = 0.5 + (10.0 - 0.5) * ((i - 9) / 5.0)  # 0.5 ~ 10 nits
                        else:
                            nits = 10.0 + (10000.0 - 10.0) * ((i - 14) / 6.0) ** 2.4  # 10 ~ 10000 nits
                        
                        # PQ OETF: luminance (nits) → code value (0-1)
                        # pq_oetf는 linear L → non-linear V 변환
                        L_array = np.array([nits])
                        code_array = ColorScience.pq_oetf(L_array)
                        code = float(code_array[0])
                        levels.append((f"PQ_{nits:.1f}nits", (code, code, code)))
                        logger.debug("[Pattern Levels]   PQ level %d: %.1f nits → code %.4f", i, nits, code)
                except ImportError:
                    # Fallback: 단순 비선형 근사
                    logger.warning("[Pattern Levels] ColorScience not available, using approximation")
                    for i in range(21):
                        val = (i / 20.0) ** 2.4  # 감마 2.4 근사
                        levels.append((f"PQ_{int(val*100)}%", (val, val, val)))
                
                logger.info("[Pattern Levels] Generated %d HDR PQ ramp levels (SMPTE ST.2084)", len(levels))
                return levels
            
            elif 'Skin' in pattern_name or 'Tone' in pattern_name:
                # Skin Tone Patches - Various skin tones for accurate reproduction
                levels = [
                    ("Fair Skin 1", (0.95, 0.76, 0.68)),
                    ("Fair Skin 2", (0.92, 0.73, 0.65)),
                    ("Light Skin 1", (0.88, 0.68, 0.58)),
                    ("Light Skin 2", (0.85, 0.65, 0.55)),
                    ("Medium Skin 1", (0.75, 0.58, 0.48)),
                    ("Medium Skin 2", (0.70, 0.53, 0.43)),
                    ("Olive Skin 1", (0.68, 0.55, 0.42)),
                    ("Olive Skin 2", (0.62, 0.50, 0.38)),
                    ("Tan Skin 1", (0.58, 0.45, 0.35)),
                    ("Tan Skin 2", (0.52, 0.40, 0.30)),
                    ("Brown Skin 1", (0.48, 0.35, 0.28)),
                    ("Brown Skin 2", (0.42, 0.30, 0.24)),
                    ("Dark Skin 1", (0.35, 0.25, 0.20)),
                    ("Dark Skin 2", (0.28, 0.20, 0.16)),
                    ("Deep Skin 1", (0.22, 0.16, 0.13)),
                    ("Deep Skin 2", (0.18, 0.13, 0.10)),
                ]
                logger.info("[Pattern Levels] Generated %d skin tone patches", len(levels))
                return levels
            
            else:
                # 기본 RGB 패턴
                logger.warning("[Pattern Levels] Unknown pattern '%s', using basic RGB", pattern_name)
                levels = [
                    ("Red 100%", (1.0, 0.0, 0.0)),
                    ("Green 100%", (0.0, 1.0, 0.0)),
                    ("Blue 100%", (0.0, 0.0, 1.0)),
                    ("White 100%", (1.0, 1.0, 1.0)),
                    ("Black", (0.0, 0.0, 0.0))
                ]
                logger.info("[Pattern Levels] Generated %d basic RGB levels", len(levels))
                return levels
                
        except Exception as e:
            logger.error("[Pattern Levels] Failed to generate levels for '%s': %s", 
                        pattern_name, str(e), exc_info=True)
            return []
    
    def on_open_pattern_window(self, event):
        """Pattern Window 열기"""
        logger.info("[UI Event] Open Pattern Window button clicked")
        
        if self.pattern_window_open:
            logger.info("[Pattern Window] Window already open - bringing to front")
            print("[Pattern Window] Already open")
            return
        
        try:
            logger.info("[Pattern Window] Creating PatternWindow instance")
            self.pattern_window = PatternWindow()
            
            # 모니터 선택 (현재는 기본 모니터)
            logger.debug("[Pattern Window] Opening on monitor %d", self.current_monitor)
            self.pattern_window.open(fullscreen=False, monitor=self.current_monitor)
            
            # 초기 패턴 (Mid Gray)
            self.pattern_window.show_color(0.5, 0.5, 0.5)
            
            self.pattern_window_open = True
            logger.info("[Pattern Window] Window opened successfully on monitor %d", self.current_monitor)
            print(f"[Pattern Window] Opened on monitor {self.current_monitor}")
            
        except Exception as e:
            logger.error("[Pattern Window] Failed to open: %s", str(e), exc_info=True)
            print(f"[Pattern Window] Error: {e}")
    
    def on_show_workflow_detail(self, event):
        """Workflow Detail 표시 - matplotlib Figure 사용 (Tkinter 제거)"""
        logger.info("[UI Event] " + "="*60)
        logger.info("[UI Event] Show Workflow Detail button clicked")
        
        # matplotlib Figure로 상세 정보 표시 (Qt 충돌 방지)
        detail_fig = plt.figure(figsize=(12, 14))
        detail_fig.canvas.manager.set_window_title("Calibration Workflow Detail")
        
        # 텍스트 영역 생성
        ax = detail_fig.add_subplot(111)
        ax.axis('off')
        
        # 현재 Workflow Detail 텍스트 가져오기
        if hasattr(self, 'workflow_detail_text'):
            workflow_text = self.workflow_detail_text.get_text()
        else:
            workflow_text = "No workflow detail available"
        
        # 상태 정보
        status_map = {
            CalibrationState.IDLE: "[IDLE] - Ready to start",
            CalibrationState.MEASURING: "[RUNNING] - Calibration in progress",
            CalibrationState.COMPLETED: "[COMPLETED] - Calibration finished",
            CalibrationState.ERROR: "[ERROR] - Error occurred"
        }
        status_text = status_map.get(self.state, "[UNKNOWN]")
        
        # Phase별 패턴 상세 정보 생성
        separator = '='*70
        
        # 선택된 Phase 확인
        phase_status = self.check_phases.get_status() if hasattr(self, 'check_phases') else [True, True, True]
        
        # Phase 1: Grayscale 패턴 상세 정보
        phase1_info = ""
        if phase_status[0]:
            phase1_info = """
Phase 1: Grayscale Calibration (ENABLED)
{sep}
Purpose: Gamma curve calibration & grayscale tracking
Target Gamma: {gamma}
Target White Point: {cct}K

Pattern List (11-step ladder):
  1. Gray 0%   (0.000) - Black level
  2. Gray 10%  (0.100)
  3. Gray 20%  (0.200)
  4. Gray 30%  (0.300)
  5. Gray 40%  (0.400)
  6. Gray 50%  (0.500) - Mid gray
  7. Gray 60%  (0.600)
  8. Gray 70%  (0.700)
  9. Gray 80%  (0.800)
 10. Gray 90%  (0.900)
 11. Gray 100% (1.000) - Peak white

Output: 1D LUT (1024 points x 3 channels - R,G,B)
Standard: ITU-R BT.1886 (Reference EOTF)
""".format(sep='-'*70, gamma=self.config.target_gamma, cct=int(self.config.target_cct))
        else:
            phase1_info = "\nPhase 1: Grayscale Calibration (DISABLED)\n"
        
        # Phase 2: Color Gamut 패턴 상세 정보
        phase2_info = ""
        if phase_status[1]:
            phase2_info = """
Phase 2: Color Gamut Calibration (ENABLED)
{sep}
Purpose: Primary/Secondary color accuracy & gamut mapping
Target Standard: {standard}
Target White Point: {cct}K

Pattern List (RGBCMYW - 7 colors):
  1. Red     (1.0, 0.0, 0.0) - Primary Red
  2. Green   (0.0, 1.0, 0.0) - Primary Green
  3. Blue    (0.0, 0.0, 1.0) - Primary Blue
  4. Cyan    (0.0, 1.0, 1.0) - Secondary Cyan (G+B)
  5. Magenta (1.0, 0.0, 1.0) - Secondary Magenta (R+B)
  6. Yellow  (1.0, 1.0, 0.0) - Secondary Yellow (R+G)
  7. White   (1.0, 1.0, 1.0) - Peak White (R+G+B)

Output: 
  - 3x3 Color Matrix (RGB primaries correction)
  - 3D LUT (33^3 = 35,937 points) for residual correction
Standards: 
  - BT.709: HDTV standard (sRGB gamut)
  - DCI-P3: Digital Cinema (wider gamut)
  - BT.2020: UHDTV standard (widest gamut)
""".format(sep='-'*70, standard=self.config.target_standard, cct=int(self.config.target_cct))
        else:
            phase2_info = "\nPhase 2: Color Gamut Calibration (DISABLED)\n"
        
        # Phase 3: Verification 패턴 상세 정보
        phase3_info = ""
        if phase_status[2]:
            phase3_info = """
Phase 3: Verification & Accuracy Check (ENABLED)
{sep}
Purpose: Final accuracy validation with industry-standard patches
Color Difference Metric: CIE dE2000 (CIEDE2000)

Pattern List (ColorChecker Classic 24):
  Row 1: Skin tones & natural colors
    1. Dark Skin       11. Yellow Green
    2. Light Skin      12. Orange Yellow
    3. Blue Sky        13. Blue
    4. Foliage         14. Green
    5. Blue Flower     15. Red
    6. Bluish Green    16. Yellow
  
  Row 2: Primary & secondary colors
    7. Orange          17. Magenta
    8. Purplish Blue   18. Cyan
    9. Moderate Red    19. White (D50)
   10. Purple          20. Neutral 8 (80%)
  
  Row 3: Grayscale patches
   21. Neutral 6.5 (65%)
   22. Neutral 5 (50%)
   23. Neutral 3.5 (35%)
   24. Black (3%)

Output: dE2000 Report
  - Average dE2000 (color accuracy indicator)
  - Maximum dE2000 (worst-case error)
  - Pass/Fail criteria: dE < 3.0 (Good), < 1.5 (Excellent), < 0.5 (Reference)
Standard: ISO 17321-1 (ColorChecker)
""".format(sep='-'*70)
        else:
            phase3_info = "\nPhase 3: Verification (DISABLED)\n"
        
        # 전체 정보 통합
        additional_info = f"""
{separator}
CURRENT CONFIGURATION
{separator}

Status: {status_text}

Sensor: {self.sensor_type.upper()}
  - Connection: {'Connected' if self.sensor_connected else 'Disconnected'}
  - Measurements completed: {self.sensor.get_measurement_count() if hasattr(self.sensor, 'get_measurement_count') else 0}

Calibration Settings:
  - Target Gamma: {self.config.target_gamma}
  - Target CCT: {self.config.target_cct}K
  - Color Standard: {self.config.target_standard}
  - Signal Range: {self.config.signal_range.value} (0-255 vs 16-235)

{separator}
PATTERN DETAILS BY PHASE
{separator}
{phase1_info}
{phase2_info}
{phase3_info}
{separator}
SUMMARY
{separator}

Total Enabled Phases: {sum(phase_status)}/3
Estimated Measurements: {11 * phase_status[0] + 7 * phase_status[1] + 24 * phase_status[2]} patches
Estimated Time: ~{(11 * phase_status[0] + 7 * phase_status[1] + 24 * phase_status[2]) * 0.5 / 60:.1f} minutes

Note: Press 'Show Workflow Detail' button to refresh this information
      after changing calibration settings.

{separator}
"""
        
        full_text = workflow_text + additional_info
        
        # 텍스트 표시 (작은 폰트로 모든 내용 표시)
        ax.text(0.02, 0.98, full_text, 
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Print to console (ASCII-safe for Windows cp949)
        try:
            ascii_text = full_text.encode('ascii', errors='replace').decode('ascii')
            print("\n" + "="*70)
            print("WORKFLOW DETAIL")
            print("="*70)
            print(ascii_text)
            print("="*70 + "\n")
        except Exception as e:
            logger.warning(f"[Workflow Detail] Console print failed: {e}")
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking으로 표시
        
        logger.info("[Workflow Detail] Window opened successfully")

    
    def display_pattern_on_window(self, pattern_name, rgb):
        """PatternWindow에 패턴 표시 (DEPRECATED - use _display_pattern_on_window_main_thread via task_queue)"""
        logger.warning("[Pattern Display] display_pattern_on_window is deprecated - use task_queue instead")
        # task_queue를 통해 메인 스레드에서 처리
        self.task_queue.put({
            'type': 'display_pattern',
            'pattern_name': pattern_name,
            'rgb': rgb
        })
    
    def _display_pattern_on_window_main_thread(self, pattern_name, rgb):
        """PatternWindow에 패턴 표시 (메인 스레드 전용)"""
        logger.info("[Pattern Display] " + "="*50)
        logger.info("[Pattern Display] Display request: '%s'", pattern_name)
        logger.debug("[Pattern Display] RGB (normalized): (%.3f, %.3f, %.3f)", *rgb)
        logger.debug("[Pattern Display] RGB (0-255): (%d, %d, %d)", 
                    int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        # PatternWindow 열려있는지 확인
        if not self.pattern_window_open or not self.pattern_window:
            logger.warning("[Pattern Display] PatternWindow not open - opening now")
            try:
                self.on_open_pattern_window(None)
                logger.info("[Pattern Display] PatternWindow opened successfully")
            except Exception as e:
                logger.error("[Pattern Display] Failed to open PatternWindow: %s", str(e))
                print(f"[Error] Cannot open PatternWindow: {e}")
                return
        
        try:
            r, g, b = rgb
            logger.debug("[Pattern Display] Calling PatternWindow.show_color()...")
            
            self.pattern_window.show_color(r, g, b)
            
            logger.info("[Pattern Display] Pattern displayed successfully")
            logger.debug("[Pattern Display] Monitor updated with new color")
            print(f"[Pattern Display] {pattern_name}: RGB({int(r*255)},{int(g*255)},{int(b*255)})")
            
            # Pattern Preview 업데이트
            self.current_pattern_rgb = rgb
            self.current_pattern_name = pattern_name
            self.update_pattern_preview()
            
        except Exception as e:
            logger.error("[Pattern Display] Failed to display pattern: %s", str(e), exc_info=True)
            print(f"[Pattern Display] Error: {e}")
    
    # ════════════════════════════════════════════════════════════
    # Calibration Workflow (Background Thread)
    # ════════════════════════════════════════════════════════════
    
    def _run_calibration_workflow(self):
        """캘리브레이션 워크플로우 실행 (백그라운드 스레드)"""
        logger.info("[Workflow] " + "="*60)
        logger.info("[Workflow] Calibration workflow started in background thread")
        logger.info("[Workflow] Config: Gamma=%.1f, CCT=%dK, Standard=%s",
                   self.config.target_gamma, self.config.target_cct, self.config.target_standard)
        
        try:
            # ── 엔진 캘리브레이터 초기화 ──────────────────────
            std = getattr(self.config, 'target_standard', 'BT.709')
            gamma = getattr(self.config, 'target_gamma', 2.2)
            cct   = getattr(self.config, 'target_cct', 6500.0)
            sig   = getattr(self.config, 'signal_range', SignalRange.FULL)

            self.gamma_cal = GammaCalibrator(
                target_gamma=gamma, target_cct=cct, signal_range=sig)
            self.gamut_cal = ColorGamutCalibrator(target_standard=std)
            self.analyzer  = CalibrationAnalyzer(std)
            self.engine_result = CalibrationResult()
            self.verify_patches = []
            logger.info("[Workflow] Engine calibrators initialized"
                        " — gamma=%.2f  CCT=%dK  standard=%s", gamma, cct, std)

            # Phase 1: Grayscale
            if self.check_phases.get_status()[0]:
                logger.info("[Workflow] Phase 1 (Grayscale) enabled - starting...")
                self._run_grayscale_phase()
            else:
                logger.info("[Workflow] Phase 1 (Grayscale) skipped by user")
                logger.debug("[Workflow] Phase 1 (Grayscale) skipped")
            
            # Phase 2: Color Gamut
            if self.check_phases.get_status()[1]:
                logger.info("[Workflow] Phase 2 (Color Gamut) enabled - starting...")
                self._run_color_gamut_phase()
            else:
                logger.info("[Workflow] Phase 2 (Color Gamut) skipped by user")
                logger.debug("[Workflow] Phase 2 (Color Gamut) skipped")
            
            # Phase 3: Verification
            if self.check_phases.get_status()[2]:
                logger.info("[Workflow] Phase 3 (Verification) enabled - starting...")
                self._run_verification_phase()
            else:
                logger.info("[Workflow] Phase 3 (Verification) skipped by user")
                logger.debug("[Workflow] Phase 3 (Verification) skipped")
            
            # 완료
            self.task_queue.put({
                'type': 'calibration_complete',
                'result': {'status': 'success'}
            })
            logger.info("[Workflow] Calibration workflow completed successfully")
            
        except Exception as e:
            logger.error("[Workflow] Calibration workflow error: %s", str(e), exc_info=True)
            self.task_queue.put({
                'type': 'calibration_error',
                'error': str(e)
            })
        finally:
            self.is_running = False
            logger.debug("[Workflow] Workflow thread terminated")
    
    def _run_grayscale_phase(self):
        """Grayscale Phase — Calman-style Hierarchical + Adaptive Sampling"""
        logger.info("[Phase 1] " + "="*60)
        logger.info("[Phase 1] Grayscale calibration (Hierarchical + Adaptive)")
        print("[Phase 1] Grayscale calibration started")

        from calibration_engine import (HierarchicalGammaCalibrator,
                                         HierarchicalGammaConfig)

        # ── Preset별 설정 ─────────────────────────────────────────────
        preset_name = getattr(self.config, '_preset_name', 'standard')
        preset_map = {
            'express':      dict(max_steps=5,  adaptive=False, per_channel=False,
                                 max_adaptive_points=0, adaptive_threshold=1.0),
            'standard':     dict(max_steps=9,  adaptive=True,  per_channel=True,
                                 max_adaptive_points=4, adaptive_threshold=0.05),
            'high':         dict(max_steps=13, adaptive=True,  per_channel=True,
                                 max_adaptive_points=6, adaptive_threshold=0.03),
            'professional': dict(max_steps=17, adaptive=True,  per_channel=True,
                                 max_adaptive_points=8, adaptive_threshold=0.02),
        }
        p = preset_map.get(preset_name, preset_map['standard'])

        cfg = HierarchicalGammaConfig(
            target_gamma=self.config.target_gamma,
            target_cct=self.config.target_cct,
            lut_size=1024,
            bit_depth=getattr(self.config, 'bit_depth', 10),
            gamma_aware=True,
            per_channel=p['per_channel'],
            adaptive=p['adaptive'],
            adaptive_threshold=p['adaptive_threshold'],
            max_adaptive_points=p['max_adaptive_points'],
            damping=0.85,           # 안정성을 위해 약간 감쇠
            noise_floor_Y=0.005,
        )
        hier = HierarchicalGammaCalibrator(config=cfg)

        max_steps = p['max_steps']
        base_sequence = list(HierarchicalGammaCalibrator.MEASUREMENT_SEQUENCE[:max_steps])

        print(f"[Phase 1] Config: preset={preset_name}  steps={max_steps}  "
              f"gamma_aware={cfg.gamma_aware}  per_ch={cfg.per_channel}  "
              f"adaptive={cfg.adaptive}(thr={cfg.adaptive_threshold:.0%})  "
              f"damping={cfg.damping}")
        logger.info("[Phase 1] Config: %s", cfg)

        measurement_log = []
        total_measured = 0

        # ── 측정 루프 (Base + Adaptive 포인트를 동적으로 처리) ────────
        # pending: 아직 측정하지 않은 포인트 큐 (Base + Adaptive 합산)
        pending = list(base_sequence)  # 초기에는 Base만

        step_idx = 0
        while pending and self.is_running:
            level = pending.pop(0)
            step_idx += 1

            # display code: 채널별 독립 (per_channel=True인 경우 R≠G≠B 가능)
            r_c, g_c, b_c = hier.get_display_code(level)
            r_c = float(np.clip(r_c, 0.0, 1.0))
            g_c = float(np.clip(g_c, 0.0, 1.0))
            b_c = float(np.clip(b_c, 0.0, 1.0))
            rgb_display = (r_c, g_c, b_c)

            # 단계 레이블
            if abs(level - 1.0) < 1e-6:
                step_label = "White 100% [Global Gain]"
                is_adaptive = False
            elif abs(level - 0.0) < 1e-6:
                step_label = "Black 0% [Lb record]"
                is_adaptive = False
            else:
                anchors_now = sorted(hier.get_anchors().keys())
                L_lo_s = max(a for a in anchors_now if a <= level)
                L_hi_s = min(a for a in anchors_now if a >= level)
                is_adaptive = level not in base_sequence
                tag = "[Adaptive]" if is_adaptive else ""
                step_label = (f"{int(round(level*100))}% "
                              f"[Tent {int(L_lo_s*100)}%~{int(L_hi_s*100)}%]{tag}")

            patch_name = f"[{'A' if is_adaptive else 'B'}{step_idx}] {step_label}"
            total_in_q = len(pending) + step_idx
            print(f"[Phase 1] {step_idx}/{total_in_q} {step_label}  "
                  f"disp=({r_c:.4f},{g_c:.4f},{b_c:.4f})")

            self.progress.current_patch_name = patch_name
            self.progress.current_rgb = rgb_display
            self.progress.measured_patches = step_idx - 1
            self.progress.total_patches = total_in_q
            self.task_queue.put({'type': 'progress_update', 'progress': self.progress})

            display_event = threading.Event()
            self.task_queue.put({
                'type': 'display_pattern',
                'pattern_name': patch_name,
                'rgb': rgb_display,
                'event': display_event
            })
            display_event.wait(timeout=2.0)
            time.sleep(0.3)

            if hasattr(self.sensor, 'set_pattern_hint'):
                self.sensor.set_pattern_hint(rgb_display)

            reading = self.sensor.read()
            total_measured += 1

            print(f"  -> Y={reading.luminance:.3f} cd/m2  "
                  f"xy=({reading.cie_xy[0]:.4f},{reading.cie_xy[1]:.4f})")
            logger.info("[Phase 1] Step %d: level=%.4f Y=%.4f",
                        step_idx, level, reading.luminance)

            # HierarchicalGammaCalibrator 갱신
            result = hier.process_level(level, reading.xyz)

            if result and 'Y_target' in result:
                Y_target = result['Y_target']
                gain = result.get('gain_L', 1.0)
                gamma_e = result.get('gamma_est', cfg.target_gamma)
                seg_lo = result.get('L_lo', 0)
                seg_hi = result.get('L_hi', 1)
                adaptive_added = result.get('adaptive_added', [])

                print(f"  -> Y_target={Y_target:.3f}  gain={gain:.4f}  "
                      f"gamma_est={gamma_e:.2f}  "
                      f"seg=[{seg_lo:.3f},{seg_hi:.3f}]")
                if adaptive_added:
                    print(f"  -> Adaptive points added: "
                          f"{[f'{p*100:.1f}%' for p in adaptive_added]}")
                    # 새 Adaptive 포인트를 pending에 삽입 (정렬 유지)
                    for ap in adaptive_added:
                        if ap not in [x for x in pending]:
                            pending.insert(0, ap)  # 다음에 바로 측정

                measurement_log.append({
                    'step': step_idx, 'level': level,
                    'Y_meas': reading.luminance, 'Y_target': Y_target,
                    'gain': gain, 'gamma_est': gamma_e,
                    'is_adaptive': is_adaptive,
                })
            elif abs(level - 1.0) < 1e-6 and hier.Lw:
                Y_target = hier._bt1886_Y(1.0)
                print(f"  -> [White] Lw={hier.Lw:.3f}  Y_target={Y_target:.3f}")
                measurement_log.append({
                    'step': step_idx, 'level': level,
                    'Y_meas': reading.luminance, 'Y_target': Y_target,
                    'gain': Y_target / max(reading.luminance, 1e-6),
                    'gamma_est': cfg.target_gamma, 'is_adaptive': False,
                })

        # ── 최종 LUT 저장 ──────────────────────────────────────────────
        self.progress.measured_patches = total_measured
        self.task_queue.put({'type': 'progress_update', 'progress': self.progress})

        self.engine_result.lut_1d = hier.get_lut_1d()

        if self.gamma_cal is not None:
            if hier.Lw:
                self.gamma_cal.measured_Lw = hier.Lw
            if hier.Lb is not None:
                self.gamma_cal.measured_Lb = hier.Lb

        # ── 요약 출력 ────────────────────────────────────────────────
        summary = hier.get_summary()
        anchors = hier.get_anchors()
        print(f"\n[Phase 1] === Calibration Summary ===")
        print(f"  Measurements : {summary.get('n_measurements', 0)} "
              f"({summary.get('n_adaptive', 0)} adaptive)")
        print(f"  Anchors      : {summary.get('n_anchors', 0)} pts  "
              f"{[round(k*100) for k in sorted(anchors)]}%")
        print(f"  Lw={summary.get('Lw', 0):.2f} cd/m2  "
              f"Lb={summary.get('Lb', 0):.4f} cd/m2")
        print(f"  Mean residual: {summary.get('mean_residual', 0)*100:.2f}%  "
              f"Max: {summary.get('max_residual', 0)*100:.2f}%")
        print(f"  Config: gamma_aware={cfg.gamma_aware}  "
              f"per_ch={cfg.per_channel}  damping={cfg.damping}")
        print("\n  Step  Level  Y_meas   Y_target  Gain    GammaEst  Adaptive")
        print("  " + "-"*62)
        for m in measurement_log:
            err = (m['gain'] - 1.0) * 100
            print(f"  {m['step']:4d}  {int(m['level']*100):4d}%  "
                  f"{m['Y_meas']:7.3f}  {m['Y_target']:8.3f}  "
                  f"{m['gain']:6.4f}  {m['gamma_est']:8.2f}  "
                  f"{'[A]' if m['is_adaptive'] else ''}")

        logger.info("[Phase 1] Done: %d steps  Lw=%.2f  Lb=%.4f  "
                    "mean_res=%.3f  n_adaptive=%d",
                    total_measured,
                    summary.get('Lw', 0), summary.get('Lb', 0),
                    summary.get('mean_residual', 0),
                    summary.get('n_adaptive', 0))
        print("[Phase 1] Grayscale calibration completed")


        logger.info("[Phase 1] " + "="*60)
    def _run_color_gamut_phase(self):
        """Color Gamut Phase 실행"""
        logger.info("[Phase 2] " + "="*60)
        logger.info("[Phase 2] Color Gamut calibration started")
        print("[Phase 2] Color Gamut calibration started")
        
        # 엔진 설정에 정의된 다차원 컬러 패치 로드 (볼류메트릭 지원)
        if hasattr(self, 'config') and hasattr(self.config, 'color_patches') and self.config.color_patches.patches:
            colors = self.config.color_patches.patches
        else:
            colors = [
                ('Red', (1, 0, 0)),
                ('Green', (0, 1, 0)),
                ('Blue', (0, 0, 1)),
                ('Cyan', (0, 1, 1)),
                ('Magenta', (1, 0, 1)),
                ('Yellow', (1, 1, 0)),
                ('White', (1, 1, 1))
            ]
        
        logger.info("[Phase 2] Measuring %d color patches (RGBCMYW)", len(colors))
        
        self.progress = MeasurementProgress(
            current_phase=WorkflowPhase.PHASE2_COLOR,
            total_patches=len(colors),
            measured_patches=0
        )
        
        start_time = time.time()
        
        for i, (name, rgb) in enumerate(colors):
            if not self.is_running:
                logger.warning("[Phase 2] Calibration stopped by user at patch %d/%d", i+1, len(colors))
                break
            
            logger.info("[Phase 2] [%d/%d] Measuring: %s", i+1, len(colors), name)
            logger.debug("[Phase 2] RGB normalized: (%.3f, %.3f, %.3f)", *rgb)
            logger.debug("[Phase 2] RGB 8-bit: (%d, %d, %d)", int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            
            # 콘솔 진행 표시
            elapsed = time.time() - start_time
            print(f"[Phase 2] Progress: {i+1}/{len(colors)} - {name} RGB{rgb} - Elapsed: {elapsed:.1f}s")
            
            # 진행률 업데이트
            self.progress.current_patch_name = name
            self.progress.current_rgb = rgb
            self.progress.measured_patches = i
            self.progress.elapsed_time = elapsed
            self.task_queue.put({'type': 'progress_update', 'progress': self.progress})
            
            # 패턴 윈도우에 표시 - Event로 메인 스레드 표시 완료까지 동기화
            display_event = threading.Event()
            self.task_queue.put({
                'type': 'display_pattern',
                'pattern_name': name,
                'rgb': rgb,
                'event': display_event
            })
            # 메인 스레드에서 패턴이 실제 표시될 때까지 대기 (최대 2초)
            display_event.wait(timeout=2.0)
            logger.debug("[Phase 2] Pattern display confirmed")
            
            # 안정화 시간 (패턴 표시 확인 후 대기)
            time.sleep(0.4)
            
            # VirtualSensor에 패턴 힌트 전달
            if hasattr(self.sensor, 'set_pattern_hint'):
                self.sensor.set_pattern_hint(rgb)
                logger.debug("[Phase 2] Pattern hint set for VirtualSensor: %s (%.3f, %.3f, %.3f)", name, *rgb)
            
            # 센서 측정
            logger.debug("[Phase 2] Reading sensor...")
            reading = self.sensor.read()
            logger.info("[Phase 2] Measured %s: XYZ=(%.3f, %.3f, %.3f), xy=(%.4f, %.4f), L=%.2f cd/m^2",
                       name, reading.xyz[0], reading.xyz[1], reading.xyz[2],
                       reading.cie_xy[0], reading.cie_xy[1], reading.luminance)
            print(f"  -> {name}: L={reading.luminance:.2f} cd/m^2, xy=({reading.cie_xy[0]:.4f}, {reading.cie_xy[1]:.4f})")

            # ── 엔진 연동: ColorGamutCalibrator에 측정 데이터 등록 ──
            if self.gamut_cal is not None:
                try:
                    self.gamut_cal.add_measurement(
                        name, np.array(rgb, dtype=float), reading.xyz)
                    logger.debug("[Phase 2] gamut_cal measurement added: %s", name)
                except Exception as _e:
                    logger.warning("[Phase 2] gamut_cal.add_measurement failed: %s", _e)

        # 최종 진행률 100% 업데이트
        self.progress.measured_patches = len(colors)
        self.task_queue.put({'type': 'progress_update', 'progress': self.progress})

        # ── 3x3 행렬 및 3D LUT 생성 (엔진 연동) ────────────────────────
        if self.gamut_cal is not None and len(self.gamut_cal.measurements) >= 3:
            try:
                self.engine_result.matrix_3x3 = self.gamut_cal.calculate_3x3_matrix()
                logger.info("[Phase 2] 3x3 matrix calculated")
                print(f"[Phase 2] 3x3 color correction matrix calculated")
                m = self.engine_result.matrix_3x3.data
                print(f"  Matrix:\n    [{m[0,0]:.4f} {m[0,1]:.4f} {m[0,2]:.4f}]")
                print(f"           [{m[1,0]:.4f} {m[1,1]:.4f} {m[1,2]:.4f}]")
                print(f"           [{m[2,0]:.4f} {m[2,1]:.4f} {m[2,2]:.4f}]")
            except Exception as _e:
                logger.warning("[Phase 2] Matrix calculation failed: %s", _e)
                print(f"[Phase 2] Matrix calculation failed: {_e}")
            try:
                lut_size = getattr(self.config, 'lut_3d_size', 17)
                self.engine_result.lut_3d = self.gamut_cal.generate_3d_lut(
                    size=lut_size)
                logger.info("[Phase 2] 3D LUT generated: %d^3", lut_size)
                print(f"[Phase 2] 3D LUT generated ({lut_size}^3 cube)")

                # --- Phase 2b: Volumetric Refinement (RBF Convergence Loop) ---
                # Standard 모드(7패치)도 포함: 최소 3패치 이상이면 RBF 보간 유효
                if len(self.gamut_cal.measurements) > 3:
                    logger.info("[Phase 2b] Starting Volumetric Refinement loop...")
                    print("[Phase 2b] Refining 3D LUT volumetrically (convergence loop)...")

                    phase2b_max_iter = getattr(
                        getattr(self, 'workflow_config', None), 'phase2b_max_iterations', 3)
                    phase2b_de_threshold = getattr(
                        getattr(self, 'workflow_config', None), 'phase2b_de_threshold', 1.0)

                    from calibration_engine import CalibrationWorkflow as _CW2b

                    temp_wf = _CW2b(sensor=None)
                    temp_wf.gamut_cal = self.gamut_cal
                    temp_wf.config = self.config

                    # 타겟 XYZ 계산용 행렬 (M_target @ rgb_in)
                    std = TARGET_STANDARDS.get(
                        self.config.target_standard, TARGET_STANDARDS['BT.709'])
                    M_target_p2b = ColorScience.primaries_to_xyz_matrix(std)

                    for p2b_iter in range(phase2b_max_iter):
                        try:
                            residuals = []
                            M = self.engine_result.matrix_3x3.data
                            M_inv = np.linalg.pinv(M)
                            lut_3d_now = self.engine_result.lut_3d

                            # ΔE: 현재 LUT 보정 적용 후 예측 XYZ vs 목표 XYZ
                            # 매 iteration마다 LUT가 갱신되므로 dE가 실제로 감소함
                            de_values = []
                            for meas in self.gamut_cal.measurements:
                                rgb_in = meas.input_rgb

                                # 현재 LUT를 적용한 출력 RGB → 디스플레이가 실제로 내보내는 값
                                rgb_corrected = lut_3d_now.apply(rgb_in)
                                # 보정된 RGB로 예측되는 XYZ (VirtualSensor의 EOTF 역산)
                                # 여기서는 선형 근사: XYZ ≈ M_display @ rgb_corrected^gamma
                                predicted_xyz_lut = np.dot(M, rgb_corrected)

                                # 목표 XYZ
                                target_xyz = M_target_p2b @ rgb_in

                                # 잔차 = 측정 XYZ - 행렬 예측 XYZ (LUT 보정 미반영 기준)
                                # (RBF 보간용 잔차는 원 측정치 기준 유지)
                                residual_xyz = meas.measured_XYZ - np.dot(M, rgb_in)
                                residual_rgb = np.dot(M_inv, residual_xyz)
                                residuals.append((rgb_in, residual_rgb))

                                # dE: LUT 보정 후 예측 vs 목표
                                lab_pred = ColorScience.XYZ_to_Lab(predicted_xyz_lut)
                                lab_target = ColorScience.XYZ_to_Lab(target_xyz)
                                de_values.append(DeltaE.ciede2000(lab_pred, lab_target))

                            mean_de = float(np.mean(de_values))
                            max_de = float(np.max(de_values))
                            print(f"[Phase 2b] Iter {p2b_iter+1}/{phase2b_max_iter}: "
                                  f"mean dE2000={mean_de:.3f}  max={max_de:.3f}")
                            logger.info("[Phase 2b] Iter %d: mean_dE=%.3f  max_dE=%.3f",
                                        p2b_iter+1, mean_de, max_de)

                            # 수렴 판정
                            if mean_de < phase2b_de_threshold:
                                print(f"[Phase 2b] Converged at iteration {p2b_iter+1} "
                                      f"(mean dE={mean_de:.3f} < {phase2b_de_threshold})")
                                logger.info("[Phase 2b] CONVERGED at iter %d", p2b_iter+1)
                                break

                            # 잔차를 3D LUT에 합성 (damping 감소: 안정성)
                            damping = max(0.3, 0.8 - p2b_iter * 0.15)
                            temp_wf._apply_residuals_to_3d_lut(
                                self.engine_result.lut_3d, residuals, damping=damping)
                            logger.info("[Phase 2b] Iter %d: residuals applied (damping=%.2f)",
                                        p2b_iter+1, damping)

                        except Exception as e:
                            logger.error("[Phase 2b] Iter %d failed: %s", p2b_iter+1, e)
                            print(f"[Phase 2b] Iter {p2b_iter+1} failed: {e}")
                            break

                    print("[Phase 2b] Volumetric Refinement loop completed")
                    logger.info("[Phase 2b] Refinement loop completed")
            except Exception as _e:
                logger.warning("[Phase 2] 3D LUT generation failed: %s", _e)
                print(f"[Phase 2] 3D LUT generation skipped: {_e}")

        logger.info("[Phase 2] Color Gamut calibration completed - %d/%d patches measured", i+1, len(colors))
        print("[Phase 2] Color Gamut calibration completed")
    
    def _run_verification_phase(self):
        """Verification Phase 실행"""
        logger.info("[Phase 3] " + "="*60)
        logger.info("[Phase 3] Verification started")
        print("[Phase 3] Verification started")
        
        logger.info("[Phase 3] Measuring 24 ColorChecker patches for verification")
        
        # ColorChecker Classic 24 patches
        # RGB: sRGB 색공간 기준, XYZ: D50 illuminant 참조값 (ISO 17321-1)
        colorchecker_patches = [
            ('Dark Skin', (0.45, 0.31, 0.24), np.array([10.06, 9.04, 6.55])),
            ('Light Skin', (0.77, 0.57, 0.48), np.array([35.76, 34.99, 29.94])),
            ('Blue Sky', (0.35, 0.43, 0.57), np.array([19.30, 20.10, 31.71])),
            ('Foliage', (0.32, 0.37, 0.23), np.array([13.07, 15.04, 8.26])),
            ('Blue Flower', (0.47, 0.47, 0.67), np.array([24.27, 23.99, 43.70])),
            ('Bluish Green', (0.40, 0.67, 0.63), np.array([40.95, 49.74, 44.80])),
            ('Orange', (0.85, 0.48, 0.19), np.array([43.13, 35.37, 10.01])),
            ('Purplish Blue', (0.30, 0.33, 0.64), np.array([12.00, 11.99, 31.71])),
            ('Moderate Red', (0.74, 0.31, 0.35), np.array([29.83, 19.72, 15.62])),
            ('Purple', (0.35, 0.21, 0.36), np.array([10.39, 7.54, 16.14])),
            ('Yellow Green', (0.59, 0.73, 0.22), np.array([44.29, 53.48, 14.12])),
            ('Orange Yellow', (0.88, 0.60, 0.10), np.array([44.00, 42.74, 8.26])),
            ('Blue', (0.18, 0.24, 0.57), np.array([8.25, 8.08, 31.20])),
            ('Green', (0.22, 0.51, 0.30), np.array([23.04, 30.44, 17.91])),
            ('Red', (0.70, 0.20, 0.19), np.array([12.00, 6.62, 5.85])),
            ('Yellow', (0.89, 0.82, 0.11), np.array([59.10, 62.66, 13.57])),
            ('Magenta', (0.74, 0.31, 0.59), np.array([19.77, 13.06, 33.64])),
            ('Cyan', (0.00, 0.49, 0.64), np.array([19.29, 25.39, 36.20])),
            ('White', (0.96, 0.96, 0.96), np.array([90.01, 89.99, 66.89])),
            ('Neutral 8', (0.80, 0.80, 0.80), np.array([59.10, 59.12, 43.90])),
            ('Neutral 6.5', (0.66, 0.66, 0.66), np.array([36.20, 36.23, 26.91])),
            ('Neutral 5', (0.50, 0.50, 0.50), np.array([19.77, 19.79, 14.70])),
            ('Neutral 3.5', (0.35, 0.35, 0.35), np.array([9.00, 9.01, 6.69])),
            ('Black', (0.20, 0.20, 0.20), np.array([3.13, 3.13, 2.33])),
        ]
        
        self.progress = MeasurementProgress(
            current_phase=WorkflowPhase.PHASE3_VERIFY,
            total_patches=len(colorchecker_patches),
            measured_patches=0
        )
        
        start_time = time.time()
        
        for i, (patch_name, rgb, ref_xyz) in enumerate(colorchecker_patches):
            if not self.is_running:
                logger.warning("[Phase 3] Verification stopped by user at patch %d/%d", i+1, len(colorchecker_patches))
                break
            
            logger.info("[Phase 3] [%d/%d] Verifying: %s", i+1, len(colorchecker_patches), patch_name)
            logger.debug("[Phase 3] RGB normalized: (%.3f, %.3f, %.3f)", *rgb)
            logger.debug("[Phase 3] Reference XYZ (D50): (%.2f, %.2f, %.2f)", *ref_xyz)
            
            # 진행률 업데이트
            self.progress.current_patch_name = patch_name
            self.progress.current_rgb = rgb
            self.progress.measured_patches = i
            self.progress.elapsed_time = time.time() - start_time
            self.task_queue.put({'type': 'progress_update', 'progress': self.progress})
            
            # 패턴 윈도우에 표시 - Event로 메인 스레드 표시 완료까지 동기화
            display_event = threading.Event()
            self.task_queue.put({
                'type': 'display_pattern',
                'pattern_name': patch_name,
                'rgb': rgb,
                'event': display_event
            })
            # 메인 스레드에서 패턴이 실제 표시될 때까지 대기 (최대 2초)
            display_event.wait(timeout=2.0)
            logger.debug("[Phase 3] Pattern display confirmed")
            
            # 안정화 시간 (패턴 표시 확인 후 대기)
            time.sleep(0.3)
            
            # VirtualSensor에 패턴 힌트 전달
            if hasattr(self.sensor, 'set_pattern_hint'):
                self.sensor.set_pattern_hint(rgb)
                logger.debug("[Phase 3] Pattern hint set for VirtualSensor: %s (%.3f, %.3f, %.3f)", patch_name, *rgb)
            
            # 센서 측정
            logger.debug("[Phase 3] Reading sensor...")
            reading = self.sensor.read()
            logger.info("[Phase 3] Measured %s: XYZ=(%.3f, %.3f, %.3f), xy=(%.4f, %.4f)",
                       patch_name, reading.xyz[0], reading.xyz[1], reading.xyz[2],
                       reading.cie_xy[0], reading.cie_xy[1])

            # ── 엔진 연동: 검증 패치 누적 (참조 XYZ 값 포함) ──
            if self.verify_patches is not None:
                self.verify_patches.append(ColorPatchMeasurement(
                    name=patch_name,
                    input_rgb=np.array(rgb, dtype=float),
                    measured_XYZ=reading.xyz.copy(),
                    target_XYZ=ref_xyz.copy(),  # ColorChecker 표준 참조값 사용
                ))

        # 최종 진행률 100% 업데이트
        self.progress.measured_patches = len(colorchecker_patches)
        self.task_queue.put({'type': 'progress_update', 'progress': self.progress})

        # ── 검증 리포트 생성 (엔진 연동) ────────────────────────────────
        if self.verify_patches:
            try:
                report = self.analyzer.compare_before_after(
                    self.verify_patches,
                    matrix=getattr(self.engine_result, 'matrix_3x3', None),
                    lut_1d=getattr(self.engine_result, 'lut_1d', None),
                )
                self.engine_result.summary = report
                text = CalibrationAnalyzer.format_report(report)
                print(text)
                logger.info("[Phase 3] dE report: before mean=%.3f  after mean=%.3f",
                            report['before']['mean_dE2000'],
                            report['after']['mean_dE2000'])
            except Exception as _e:
                logger.warning("[Phase 3] Verification report failed: %s", _e)
                print(f"[Phase 3] Verification report failed: {_e}")

        logger.info("[Phase 3] Verification completed - %d/%d patches verified", i+1, len(colorchecker_patches))
        print("[Phase 3] Verification completed")


# ============================================================================
# Standalone Execution
# ============================================================================

def main():
    """독립 실행"""
    print("="*70)
    print("Professional Display Calibration System")
    print("="*70)
    print()
    
    # Virtual Sensor 사용 (BT.2020 wide gamut display simulation)
    sensor = VirtualSensor(
        noise_level=0.02, display_colorspace='BT.2020',
        max_luminance=100.0, black_level=0.05, native_gamma=2.2)
    sensor.connect()
    
    # Calibration UI 실행
    cal_ui = CalibrationUI(sensor=sensor)
    cal_ui.show()


if __name__ == '__main__':
    main()
