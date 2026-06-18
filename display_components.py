"""
Display Components Module
디스플레이 컴포넌트 추상화 및 모듈화

각 디스플레이 패널을 독립적인 컴포넌트로 관리하여
업데이트 로직의 일관성과 유지보수성을 향상
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class DisplayComponent(ABC):
    """디스플레이 컴포넌트 추상 베이스 클래스"""
    
    def __init__(self, ax):
        """
        Args:
            ax: matplotlib axes 객체
        """
        self.ax = ax
        self._initialized = False
    
    @abstractmethod
    def update(self, result: Dict[str, Any]):
        """
        분석 결과로 디스플레이 업데이트
        
        Args:
            result: ColorAnalyzerAdvanced.analyze_color()의 반환값
        """
        pass
    
    @abstractmethod
    def needs_redraw_on(self, event_type: str) -> bool:
        """
        특정 이벤트에서 업데이트 필요 여부
        
        Args:
            event_type: 'full', 'color_change', 'gamma_change', 
                       'standard_change', 'sensor_measurement', etc.
        
        Returns:
            True면 업데이트 필요, False면 스킵
        """
        pass
    
    def clear(self):
        """axes 초기화"""
        self.ax.clear()


class ColorSampleDisplay(DisplayComponent):
    """색상 샘플 디스플레이"""
    
    def update(self, result: Dict[str, Any]):
        self.ax.clear()
        self.ax.set_title('Color Sample', fontsize=12, fontweight='bold', pad=8)
        self.ax.axis('off')
        
        # sRGB 색상 가져오기
        srgb = result.get('srgb_color', np.array([1.0, 1.0, 1.0]))
        srgb_clipped = np.clip(srgb, 0, 1)
        
        # 색상 샘플 사각형
        rect = Rectangle((0.1, 0.1), 0.8, 0.8, 
                        facecolor=srgb_clipped, edgecolor='black', linewidth=2,
                        transform=self.ax.transAxes)
        self.ax.add_patch(rect)
    
    def needs_redraw_on(self, event_type: str) -> bool:
        # 색상 변경시 항상 업데이트
        return event_type in ['full', 'color_change', 'sensor_measurement', 
                             'gamma_change', 'brightness_change']


class AnalysisResultDisplay(DisplayComponent):
    """분석 결과 텍스트 디스플레이"""
    
    def update(self, result: Dict[str, Any]):
        self.ax.clear()
        self.ax.set_title('Analysis Results', fontsize=12, fontweight='bold', pad=8)
        self.ax.axis('off')
        
        # 결과 텍스트 구성
        lines = []
        
        # 기본 정보
        if 'cie_xy' in result:
            x, y = result['cie_xy']
            lines.append(f"CIE xy: ({x:.4f}, {y:.4f})")
        
        if 'cie_xyz' in result:
            X, Y, Z = result['cie_xyz']
            lines.append(f"CIE XYZ: ({X:.3f}, {Y:.3f}, {Z:.3f})")
        
        if 'luminance' in result:
            lines.append(f"Luminance: {result['luminance']:.2f} cd/m²")
        
        lines.append("")
        
        # RGB 정보
        if 'rgb_linear' in result:
            r, g, b = result['rgb_linear']
            lines.append(f"RGB Linear: ({r:.3f}, {g:.3f}, {b:.3f})")
        
        if 'srgb_color' in result:
            r, g, b = result['srgb_color']
            lines.append(f"sRGB: ({r:.3f}, {g:.3f}, {b:.3f})")
        
        lines.append("")
        
        # Gamma 정보
        if 'gamma_info' in result:
            gamma_type = result['gamma_info'].get('type', 'N/A')
            lines.append(f"Gamma: {gamma_type}")
        
        if 'color_standard' in result:
            lines.append(f"Standard: {result['color_standard']}")
        
        # 텍스트 표시
        text = "\n".join(lines)
        self.ax.text(0.05, 0.95, text, 
                    transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    family='monospace')
    
    def needs_redraw_on(self, event_type: str) -> bool:
        # 대부분의 변경사항에서 업데이트
        return event_type in ['full', 'color_change', 'sensor_measurement',
                             'gamma_change', 'standard_change', 'brightness_change']


class ChromaticityDisplay(DisplayComponent):
    """CIE 1931 색도도 디스플레이"""
    
    def __init__(self, ax):
        super().__init__(ax)
        self._setup_chromaticity_base()
    
    def _setup_chromaticity_base(self):
        """색도도 베이스 그리기 (스펙트럼 궤적, 그리드 등)"""
        # 이 부분은 한 번만 그리면 되는 고정 요소
        # 실제 구현시 스펙트럼 궤적 등을 그림
        pass
    
    def update(self, result: Dict[str, Any]):
        self.ax.clear()
        self.ax.set_title('CIE 1931 Chromaticity', fontsize=12, fontweight='bold', pad=8)
        self.ax.set_xlabel('CIE x', fontsize=10)
        self.ax.set_ylabel('CIE y', fontsize=10)
        self.ax.set_xlim(0, 0.8)
        self.ax.set_ylim(0, 0.9)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.tick_params(labelsize=9)
        
        # 색도도 그리기 (기존 로직)
        self._draw_chromaticity_diagram(result)
    
    def _draw_chromaticity_diagram(self, result: Dict[str, Any]):
        """색도도 그리기 로직"""
        # 기존 display_chromaticity_diagram 로직
        # 스펙트럼 궤적, 색 영역, 현재 포인트 등
        
        # Planckian locus
        cct_range = np.arange(1000, 25001, 100)
        planck_x, planck_y = [], []
        for cct in cct_range:
            x, y = self._cct_to_xy(cct)
            planck_x.append(x)
            planck_y.append(y)
        self.ax.plot(planck_x, planck_y, 'k--', alpha=0.3, linewidth=1, label='Planckian Locus')
        
        # 현재 색상 포인트
        if 'cie_xy' in result:
            x, y = result['cie_xy']
            srgb = result.get('srgb_color', [1, 1, 1])
            srgb_clipped = np.clip(srgb, 0, 1)
            self.ax.plot(x, y, 'o', markersize=10, 
                        markerfacecolor=srgb_clipped,
                        markeredgecolor='black', markeredgewidth=2)
        
        self.ax.legend(fontsize=8, loc='upper right')
    
    def _cct_to_xy(self, cct):
        """CCT를 CIE xy로 변환 (간단한 근사)"""
        # McCamy의 역변환 근사
        if cct < 4000:
            x = -0.2661239e9 / (cct**3) - 0.2343589e6 / (cct**2) + 0.8776956e3 / cct + 0.179910
        else:
            x = -3.0258469e9 / (cct**3) + 2.1070379e6 / (cct**2) + 0.2226347e3 / cct + 0.240390
        
        y = -3.000 * (x**2) + 2.870 * x - 0.275 if x < 0.5 else -1.000 * (x**2) + 1.640 * x - 0.180
        return x, y
    
    def needs_redraw_on(self, event_type: str) -> bool:
        # 색상이나 표준이 변경될 때만 업데이트
        return event_type in ['full', 'color_change', 'sensor_measurement', 'standard_change']


class RGBInfoDisplay(DisplayComponent):
    """RGB 상세 정보 디스플레이"""
    
    def update(self, result: Dict[str, Any]):
        self.ax.clear()
        self.ax.set_title('RGB Information', fontsize=12, fontweight='bold', pad=8)
        self.ax.axis('off')
        
        lines = []
        
        # RGB 채널별 정보
        if 'rgb_channels' in result:
            rgb = result['rgb_channels']
            lines.append("RGB Channels:")
            lines.append(f"  R: {rgb[0]:.4f}")
            lines.append(f"  G: {rgb[1]:.4f}")
            lines.append(f"  B: {rgb[2]:.4f}")
            lines.append("")
        
        # Ratio 정보
        if 'rgb_ratio' in result:
            ratio = result['rgb_ratio']
            lines.append("Normalized Ratio:")
            lines.append(f"  R: {ratio[0]:.4f}")
            lines.append(f"  G: {ratio[1]:.4f}")
            lines.append(f"  B: {ratio[2]:.4f}")
            lines.append("")
        
        # Brightness
        if 'brightness' in result:
            lines.append(f"Brightness: {result['brightness']:.4f}")
        
        text = "\n".join(lines)
        self.ax.text(0.05, 0.95, text,
                    transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    family='monospace')
    
    def needs_redraw_on(self, event_type: str) -> bool:
        return event_type in ['full', 'color_change', 'sensor_measurement', 
                             'brightness_change']


class EOTFDisplay(DisplayComponent):
    """EOTF 커브 디스플레이"""
    
    def update(self, result: Dict[str, Any]):
        self.ax.clear()
        self.ax.set_title('EOTF Curve (ST.2084)', fontsize=12, fontweight='bold', pad=8)
        self.ax.set_xlabel('Input Signal (0-1)', fontsize=10)
        self.ax.set_ylabel('Luminance (cd/m²)', fontsize=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.tick_params(labelsize=9)
        
        # EOTF 커브 그리기
        if 'eotf_data' in result:
            eotf = result['eotf_data']
            self.ax.plot(eotf['input'], eotf['output'], 'b-', linewidth=2, label='EOTF')
            
            # 현재 포인트
            if 'current_point' in eotf:
                pt = eotf['current_point']
                self.ax.plot(pt['x'], pt['y'], 'ro', markersize=8, label='Current')
            
            self.ax.legend(fontsize=8)
    
    def needs_redraw_on(self, event_type: str) -> bool:
        # EOTF는 gamma나 brightness 변경시 업데이트
        return event_type in ['full', 'gamma_change', 'brightness_change', 
                             'color_change', 'sensor_measurement']


class HDRInfoDisplay(DisplayComponent):
    """HDR 정보 디스플레이"""
    
    def update(self, result: Dict[str, Any]):
        self.ax.clear()
        self.ax.set_title('HDR Information', fontsize=12, fontweight='bold', pad=8)
        self.ax.axis('off')
        
        lines = []
        
        if 'hdr_info' in result:
            hdr = result['hdr_info']
            lines.append("HDR Metadata:")
            lines.append(f"  Max CLL: {hdr.get('max_cll', 0):.0f} cd/m²")
            lines.append(f"  Peak Luminance: {hdr.get('display_peak', 0):.0f} cd/m²")
            lines.append(f"  Roll-off: {hdr.get('roll_off', 0):.0f} cd/m²")
            lines.append("")
            
            if 'nits' in hdr:
                lines.append(f"Current: {hdr['nits']:.2f} nits")
        
        text = "\n".join(lines)
        self.ax.text(0.05, 0.95, text,
                    transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    family='monospace')
    
    def needs_redraw_on(self, event_type: str) -> bool:
        # HDR 정보는 brightness나 HDR 설정 변경시
        return event_type in ['full', 'brightness_change', 'hdr_change',
                             'sensor_measurement']


class MeasurementTableDisplay(DisplayComponent):
    """센서 측정 히스토리 테이블"""
    
    def __init__(self, ax):
        super().__init__(ax)
        self.measurement_history = []
    
    def add_measurement(self, reading):
        """측정 데이터 추가"""
        from datetime import datetime
        
        dt = datetime.fromtimestamp(reading.timestamp)
        time_str = dt.strftime("%H:%M:%S")
        
        self.measurement_history.append({
            'num': len(self.measurement_history) + 1,
            'time': time_str,
            'x': reading.cie_xy[0],
            'y': reading.cie_xy[1],
            'Y': reading.luminance
        })
        
        # 최대 20개로 제한
        if len(self.measurement_history) > 20:
            self.measurement_history.pop(0)
            # 번호 재정렬
            for i, entry in enumerate(self.measurement_history):
                entry['num'] = i + 1
    
    def clear_history(self):
        """히스토리 초기화"""
        self.measurement_history.clear()
    
    def update(self, result: Dict[str, Any]):
        """테이블 업데이트 (result는 사용하지 않음 - 내부 히스토리 사용)"""
        self.ax.clear()
        self.ax.set_title('Measurement History', fontsize=12, fontweight='bold', pad=8)
        self.ax.axis('off')
        
        if not self.measurement_history:
            self.ax.text(0.5, 0.5, 'No measurements yet',
                        ha='center', va='center', fontsize=10,
                        transform=self.ax.transAxes)
            return
        
        # 테이블 헤더
        header = "#   Time      x       y       Y(cd/m²)"
        lines = [header]
        lines.append("-" * 40)
        
        # 최근 측정부터 표시 (역순)
        for entry in reversed(self.measurement_history[-10:]):  # 최근 10개만
            line = "{:2d}  {}  {:.4f}  {:.4f}  {:6.1f}".format(
                entry['num'], entry['time'],
                entry['x'], entry['y'], entry['Y'])
            lines.append(line)
        
        text = "\n".join(lines)
        self.ax.text(0.05, 0.95, text,
                    ha='left', va='top', fontsize=8,
                    family='monospace',
                    transform=self.ax.transAxes)
    
    def needs_redraw_on(self, event_type: str) -> bool:
        # 센서 측정시에만 업데이트
        return event_type in ['sensor_measurement', 'measurement_added']


class DisplayManager:
    """디스플레이 컴포넌트 통합 관리자"""
    
    def __init__(self, axes_dict: Dict[str, Any]):
        """
        Args:
            axes_dict: {'color_sample': ax1, 'analysis': ax2, ...}
        """
        self.components = {}
        
        # 컴포넌트 초기화
        if 'color_sample' in axes_dict:
            self.components['color_sample'] = ColorSampleDisplay(axes_dict['color_sample'])
        
        if 'analysis' in axes_dict:
            self.components['analysis'] = AnalysisResultDisplay(axes_dict['analysis'])
        
        if 'chromaticity' in axes_dict:
            self.components['chromaticity'] = ChromaticityDisplay(axes_dict['chromaticity'])
        
        if 'rgb_info' in axes_dict:
            self.components['rgb_info'] = RGBInfoDisplay(axes_dict['rgb_info'])
        
        if 'eotf' in axes_dict:
            self.components['eotf'] = EOTFDisplay(axes_dict['eotf'])
        
        if 'hdr_info' in axes_dict:
            self.components['hdr_info'] = HDRInfoDisplay(axes_dict['hdr_info'])
        
        if 'measurement_table' in axes_dict:
            self.components['measurement_table'] = MeasurementTableDisplay(axes_dict['measurement_table'])
    
    def update(self, result: Dict[str, Any], event_type: str = 'full'):
        """
        이벤트 타입에 따라 필요한 컴포넌트만 업데이트
        
        Args:
            result: 분석 결과 딕셔너리
            event_type: 이벤트 타입 ('full', 'color_change', 'sensor_measurement', etc.)
        """
        for name, component in self.components.items():
            if component.needs_redraw_on(event_type):
                component.update(result)
    
    def get_component(self, name: str) -> Optional[DisplayComponent]:
        """특정 컴포넌트 가져오기"""
        return self.components.get(name)
    
    def add_measurement(self, reading):
        """측정 히스토리에 추가"""
        table = self.get_component('measurement_table')
        if table and isinstance(table, MeasurementTableDisplay):
            table.add_measurement(reading)
    
    def clear_measurements(self):
        """측정 히스토리 초기화"""
        table = self.get_component('measurement_table')
        if table and isinstance(table, MeasurementTableDisplay):
            table.clear_history()
