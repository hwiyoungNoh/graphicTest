"""
CalibrationWorkflow Integration Tests

3-Phase 캘리브레이션 워크플로우 검증:
  Phase 1:  Grayscale (Gamma + WP simultaneous) — ITERATIVE
  Phase 2:  Color Gamut — BATCH
  Phase 2b: 3D LUT Refinement — BATCH LOOP
  Phase 3:  Final Verification — ONE-SHOT

가상 센서(SimulatedSensor)와 가상 디스플레이 모델을 사용하여
실제 측정 없이 워크플로우 로직을 검증합니다.
"""

import numpy as np
import sys
import logging
from dataclasses import dataclass
from typing import Dict

# ── 로깅 ──
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s')

# ── 가상 센서 및 디스플레이 모델 ──

@dataclass
class FakeSensorReading:
    """센서 읽기 결과"""
    rgb: np.ndarray
    xyz: np.ndarray
    cie_xy: tuple
    luminance: float
    timestamp: float = 0.0
    is_valid: bool = True
    error_message: str = ""


class SimulatedDisplay:
    """
    가상 디스플레이 모델

    네이티브 감마 + 약간 어긋난 원색 + 비영 블랙레벨
    """

    def __init__(self,
                 native_gamma: float = 2.4,
                 Lw: float = 300.0,
                 Lb: float = 0.5,
                 primary_shift: float = 0.02):
        self.native_gamma = native_gamma
        self.Lw = Lw
        self.Lb = Lb

        # 약간 어긋난 원색 (BT.709 + shift)
        from calibration_engine import (
            ColorScience, TARGET_STANDARDS)
        std = TARGET_STANDARDS['BT.709']

        # shift: 원색 좌표를 약간 이동
        shifted = {
            'R': (std['R'][0] + primary_shift,
                  std['R'][1] - primary_shift * 0.5),
            'G': (std['G'][0] - primary_shift * 0.5,
                  std['G'][1] + primary_shift),
            'B': (std['B'][0] + primary_shift * 0.3,
                  std['B'][1] - primary_shift * 0.3),
            'W': std['W'],
        }
        self.M_display = ColorScience.primaries_to_xyz_matrix(shifted)

    def render(self, r: float, g: float, b: float) -> np.ndarray:
        """
        RGB 코드값 → 디스플레이 출력 XYZ

        L_ch = Lb + (Lw - Lb) × code^γ_native
        XYZ = M_display × [R_linear, G_linear, B_linear]
        """
        r, g, b = np.clip([r, g, b], 0, 1)

        # Power law EOTF
        R_lin = self.Lb + (self.Lw - self.Lb) * (r ** self.native_gamma)
        G_lin = self.Lb + (self.Lw - self.Lb) * (g ** self.native_gamma)
        B_lin = self.Lb + (self.Lw - self.Lb) * (b ** self.native_gamma)

        # Normalize to Lw
        R_norm = R_lin / self.Lw
        G_norm = G_lin / self.Lw
        B_norm = B_lin / self.Lw

        xyz = self.M_display @ np.array([R_norm, G_norm, B_norm])
        return xyz * self.Lw


class SimulatedSensor:
    """
    가상 센서: SimulatedDisplay 기반

    패턴 윈도우의 마지막 show_color 호출 값을 사용하여
    디스플레이 출력을 시뮬레이션합니다.
    """

    def __init__(self, display: SimulatedDisplay,
                 noise_level: float = 0.001):
        self.display = display
        self.noise_level = noise_level
        self._last_rgb = (0.0, 0.0, 0.0)

    def set_pattern(self, r, g, b):
        self._last_rgb = (float(r), float(g), float(b))

    def read(self) -> FakeSensorReading:
        r, g, b = self._last_rgb
        xyz = self.display.render(r, g, b)

        # 측정 노이즈
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, 3)
            xyz = xyz + noise * xyz.max()
            xyz = np.maximum(xyz, 0)

        from calibration_engine import ColorScience
        xy = ColorScience.XYZ_to_xy(xyz)
        lum = float(xyz[1])

        return FakeSensorReading(
            rgb=np.array([r, g, b]),
            xyz=xyz,
            cie_xy=tuple(xy),
            luminance=lum,
        )


class FakePatternWindow:
    """가상 패턴 윈도우: 센서에 현재 표시 값 전달"""

    def __init__(self, sensor: SimulatedSensor):
        self.sensor = sensor

    def show_color(self, r, g, b):
        self.sensor.set_pattern(r, g, b)


# ============================================================================
# Test Cases
# ============================================================================

def test_workflow_phase1_basic():
    """Phase 1: 기본 Grayscale 캘리브레이션 (단일 반복)"""
    print("\n[Test] Phase 1 — Basic Grayscale Calibration")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5)
    sensor = SimulatedSensor(display, noise_level=0)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.settle_time = 0.0

    wf_cfg = WorkflowConfig(phase1_max_iterations=1)

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)
    result = workflow.run_phase1_grayscale()

    assert result.iterations == 1, \
        "Expected 1 iteration, got {}".format(result.iterations)
    assert result.phase.value == 'phase1_grayscale'
    assert workflow.result.lut_1d is not None, "LUT should be generated"
    assert workflow.result.lut_1d.size == 1024, "LUT should be 1024 entries"

    # LUT 단조증가 확인
    lut = workflow.result.lut_1d
    r_mono = all(lut.r[i] <= lut.r[i+1] + 1e-6
                 for i in range(lut.size - 1))
    assert r_mono, "R channel should be monotonically increasing"

    print("  ✓ Phase 1 basic: PASS")
    return True


def test_workflow_phase1_iterative():
    """Phase 1: 반복 수렴 테스트"""
    print("\n[Test] Phase 1 — Iterative Convergence")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5,
                               primary_shift=0.03)
    sensor = SimulatedSensor(display, noise_level=0.0005)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.settle_time = 0.0

    wf_cfg = WorkflowConfig(
        phase1_max_iterations=3,
        phase1_dY_threshold=0.01,
        phase1_duv_threshold=0.005,
    )

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)
    result = workflow.run_phase1_grayscale()

    assert result.iterations >= 1, "Should run at least 1 iteration"
    assert result.iterations <= 3, "Should not exceed max iterations"

    # 반복 히스토리 존재 확인
    history = result.metrics.get('history', [])
    assert len(history) == result.iterations, \
        "History length should match iterations"

    print("  ✓ Phase 1 iterative: {} iterations, converged={}".format(
        result.iterations, result.converged))
    return True


def test_workflow_phase2_batch():
    """Phase 2: 색역 배치 캘리브레이션"""
    print("\n[Test] Phase 2 — Color Gamut Batch")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, ColorPatchTable,
        WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5,
                               primary_shift=0.02)
    sensor = SimulatedSensor(display, noise_level=0)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.color_patches = ColorPatchTable.standard()
    cfg.settle_time = 0.0

    wf_cfg = WorkflowConfig(phase1_max_iterations=1)

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)

    # Phase 1 먼저 실행
    workflow.run_phase1_grayscale()

    # Phase 2
    result = workflow.run_phase2_color()

    assert result.phase.value == 'phase2_color'
    assert result.iterations == 1
    assert workflow.result.matrix_3x3 is not None, \
        "Matrix should be generated"

    # 행렬 조건수: 합리적 범위
    cond = result.metrics.get('matrix_cond', 0)
    assert cond < 100, "Matrix condition number should be reasonable"

    # 행렬식: 비특이(non-singular) 확인
    det = result.metrics.get('matrix_det', 0)
    raw_det = float(np.linalg.det(workflow.result.matrix_3x3.data))
    assert raw_det > 0, \
        "Matrix determinant should be positive (raw={:.2e})".format(raw_det)

    print("  ✓ Phase 2 batch: det={:.4f}  cond={:.2f}".format(det, cond))
    return True


def test_workflow_phase2b_refinement():
    """Phase 2b: 3D LUT 잔차 보정"""
    print("\n[Test] Phase 2b — 3D LUT Refinement")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, ColorPatchTable,
        WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5,
                               primary_shift=0.02)
    sensor = SimulatedSensor(display, noise_level=0)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.color_patches = ColorPatchTable.standard()
    cfg.lut_3d_size = 9     # 작은 크기로 빠른 테스트
    cfg.settle_time = 0.0
    cfg.prefer_matrix = False

    wf_cfg = WorkflowConfig(
        phase1_max_iterations=1,
        phase2b_max_iterations=2,
        phase2b_de_threshold=2.0,
    )

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)

    # Phase 1 + 2
    workflow.run_phase1_grayscale()
    workflow.run_phase2_color()

    # Phase 2b
    result = workflow.run_phase2b_refinement()

    assert result.phase.value == 'phase2b_refinement'
    assert result.iterations >= 1

    print("  ✓ Phase 2b: {} iterations, converged={}".format(
        result.iterations, result.converged))
    return True


def test_workflow_phase3_verify():
    """Phase 3: 최종 검증"""
    print("\n[Test] Phase 3 — Final Verification")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5)
    sensor = SimulatedSensor(display, noise_level=0)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.settle_time = 0.0

    wf_cfg = WorkflowConfig(phase1_max_iterations=1)

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)

    # Phase 1 + 2 + 3
    workflow.run_phase1_grayscale()
    workflow.run_phase2_color()
    result = workflow.run_phase3_verify()

    assert result.phase.value == 'phase3_verify'
    assert 'mean_dE2000' in result.metrics
    assert 'grade' in result.metrics
    assert result.metrics['grade'] in [
        'REFERENCE', 'BROADCAST', 'PROFESSIONAL', 'CONSUMER']

    print("  ✓ Phase 3: grade={}, mean_ΔE={:.3f}".format(
        result.metrics['grade'],
        result.metrics['mean_dE2000']))
    return True


def test_workflow_full_run():
    """전체 워크플로우 (Phase 1 → 2 → 2b → 3)"""
    print("\n[Test] Full Workflow — All Phases")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, ColorPatchTable,
        WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5,
                               primary_shift=0.02)
    sensor = SimulatedSensor(display, noise_level=0.0003)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.color_patches = ColorPatchTable.standard()
    cfg.lut_3d_size = 9
    cfg.settle_time = 0.0
    cfg.prefer_matrix = False

    wf_cfg = WorkflowConfig(
        phase1_max_iterations=2,
        phase2b_max_iterations=1,
        phase2b_de_threshold=2.0,
    )

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)
    summary = workflow.run()

    # 모든 Phase 실행 확인
    assert 'phase1' in summary['phases'], "Phase 1 should be in summary"
    assert 'phase2' in summary['phases'], "Phase 2 should be in summary"
    assert 'phase2b' in summary['phases'], "Phase 2b should be in summary"
    assert 'phase3' in summary['phases'], "Phase 3 should be in summary"

    # CalibrationResult 존재 확인
    result = summary['calibration_result']
    assert result.lut_1d is not None, "1D LUT should exist"
    assert result.matrix_3x3 is not None, "Matrix should exist"

    # 워크플로우 리포트 생성
    report = workflow.get_workflow_report()
    assert 'Phase 1' in report
    assert 'Phase 2' in report
    assert 'Phase 3' in report

    print("  ✓ Full workflow: total_time={:.1f}s".format(
        summary['total_time_sec']))
    return True


def test_workflow_skip_phases():
    """Phase 건너뛰기 테스트"""
    print("\n[Test] Workflow — Skip Phases")

    from calibration_engine import (
        CalibrationConfig, CalibrationPreset,
        GammaStepTable, WorkflowConfig, CalibrationWorkflow,
    )

    display = SimulatedDisplay(native_gamma=2.4, Lw=300, Lb=0.5)
    sensor = SimulatedSensor(display, noise_level=0)
    pattern = FakePatternWindow(sensor)

    cfg = CalibrationConfig.from_preset(CalibrationPreset.QUICK)
    cfg.gamma_steps = GammaStepTable.uniform(11, white_only=True)
    cfg.settle_time = 0.0

    wf_cfg = WorkflowConfig(phase1_max_iterations=1)

    workflow = CalibrationWorkflow(
        sensor, pattern, config=cfg, workflow_config=wf_cfg)
    summary = workflow.run(skip_phases=['phase2b', 'phase3'])

    # Phase 2b, 3 건너뜀
    assert 'phase2b' not in summary['phases']
    assert 'phase3' not in summary['phases']
    assert 'phase1' in summary['phases']
    assert 'phase2' in summary['phases']

    print("  ✓ Skip phases: PASS")
    return True


def test_workflow_compose_1d_luts():
    """1D LUT 합성 테스트"""
    print("\n[Test] LUT Composition")

    from calibration_engine import (
        LUT1D, CalibrationWorkflow,
    )

    # Identity LUT
    lut_a = LUT1D()
    # 약간 수정된 LUT
    lut_b = LUT1D()
    lut_b.r = np.power(np.linspace(0, 1, 1024), 0.9)
    lut_b.g = np.power(np.linspace(0, 1, 1024), 0.9)
    lut_b.b = np.power(np.linspace(0, 1, 1024), 0.9)

    # Identity ∘ B = B
    composed = CalibrationWorkflow._compose_1d_luts(lut_a, lut_b)
    diff = np.max(np.abs(composed.r - lut_b.r))
    assert diff < 0.01, "Identity ∘ B should ≈ B (diff={:.4f})".format(diff)

    # B ∘ Identity = B
    composed2 = CalibrationWorkflow._compose_1d_luts(lut_b, lut_a)
    # lut_a is identity → lut_b applied first, then identity lookup
    diff2 = np.max(np.abs(composed2.r - lut_b.r))
    assert diff2 < 0.01, "B ∘ Identity should ≈ B (diff={:.4f})".format(diff2)

    print("  ✓ LUT composition: PASS")
    return True


def test_workflow_data_classes():
    """WorkflowPhase, WorkflowConfig, PhaseResult 데이터클래스"""
    print("\n[Test] Workflow Data Classes")

    from calibration_engine import (
        WorkflowPhase, WorkflowConfig, PhaseResult,
    )

    # WorkflowPhase enum
    assert WorkflowPhase.PHASE1_GRAYSCALE.value == 'phase1_grayscale'
    assert WorkflowPhase.PHASE2_COLOR.value == 'phase2_color'
    assert WorkflowPhase.PHASE2B_REFINEMENT.value == 'phase2b_refinement'
    assert WorkflowPhase.PHASE3_VERIFY.value == 'phase3_verify'

    # WorkflowConfig defaults
    wfc = WorkflowConfig()
    assert wfc.phase1_max_iterations == 3
    assert wfc.phase1_dY_threshold == 0.005
    assert wfc.phase1_duv_threshold == 0.002
    assert wfc.phase2b_max_iterations == 2
    assert wfc.phase2b_de_threshold == 1.0

    # PhaseResult
    pr = PhaseResult(
        phase=WorkflowPhase.PHASE1_GRAYSCALE,
        iterations=2, converged=True)
    assert pr.iterations == 2
    assert pr.converged is True

    print("  ✓ Data classes: PASS")
    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 72)
    print("  CalibrationWorkflow — Integration Tests")
    print("=" * 72)

    tests = [
        test_workflow_data_classes,
        test_workflow_compose_1d_luts,
        test_workflow_phase1_basic,
        test_workflow_phase1_iterative,
        test_workflow_phase2_batch,
        test_workflow_phase2b_refinement,
        test_workflow_phase3_verify,
        test_workflow_full_run,
        test_workflow_skip_phases,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print("  ✗ {} returned False".format(test.__name__))
        except Exception as e:
            failed += 1
            print("  ✗ {} FAILED: {}".format(test.__name__, e))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 72)
    print("  Results: {} passed, {} failed, {} total".format(
        passed, failed, len(tests)))
    print("=" * 72)

    sys.exit(0 if failed == 0 else 1)
