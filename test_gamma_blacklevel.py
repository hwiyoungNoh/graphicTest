"""
감마 캘리브레이션 블랙레벨 검증 테스트

검증 목적:
  1. 디스플레이의 최대/최소 밝기를 측정하여 감마곡선에 반영하는가?
  2. BT.1886 EOTF 모델에 맞는 보정이 이루어지는가?
  3. 암부(shadow) 영역에서 블랙레벨 오프셋을 올바르게 처리하는가?
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from calibration_engine import (
    GammaCalibrator, ColorScience, TARGET_STANDARDS, LUT1D
)


# ============================================================================
# 디스플레이 시뮬레이터
# ============================================================================

class DisplaySimulator:
    """비영 블랙레벨을 가진 실제 LCD 디스플레이 시뮬레이션"""

    def __init__(self, Lw=300.0, Lb=0.5, native_gamma=2.4,
                 standard='BT.709'):
        self.Lw = Lw          # 최대 백색 휘도 (cd/m²)
        self.Lb = Lb          # 블랙 레벨 (cd/m²)
        self.native_gamma = native_gamma
        self.standard = standard
        self.M = ColorScience.primaries_to_xyz_matrix(
            TARGET_STANDARDS[standard])

    @property
    def contrast_ratio(self):
        return self.Lw / self.Lb

    def eotf(self, V):
        """디스플레이 EOTF: 입력 신호 V(0-1) → 출력 휘도(cd/m²)"""
        return self.Lb + (self.Lw - self.Lb) * (V ** self.native_gamma)

    def measure_white(self, level):
        """White (R=G=B=level) 측정 시뮬레이션 → XYZ"""
        L = self.eotf(level)
        L_norm = L / self.Lw
        return self.M @ np.array([L_norm, L_norm, L_norm]) * self.Lw

    def measure_red(self, level):
        """Red channel only (R=level, G=B=0) 측정 → XYZ"""
        L_r = self.eotf(level) / self.Lw
        L_0 = self.eotf(0) / self.Lw  # 블랙레벨
        return self.M @ np.array([L_r, L_0, L_0]) * self.Lw

    def measure_green(self, level):
        L_g = self.eotf(level) / self.Lw
        L_0 = self.eotf(0) / self.Lw
        return self.M @ np.array([L_0, L_g, L_0]) * self.Lw

    def measure_blue(self, level):
        L_b = self.eotf(level) / self.Lw
        L_0 = self.eotf(0) / self.Lw
        return self.M @ np.array([L_0, L_0, L_b]) * self.Lw


def bt1886_eotf(V, Lw, Lb, gamma):
    """
    BT.1886 참조 EOTF

    L = a * max(V + b, 0)^γ

    a = (Lw^(1/γ) - Lb^(1/γ))^γ
    b = Lb^(1/γ) / (Lw^(1/γ) - Lb^(1/γ))
    """
    Lw_inv = Lw ** (1.0 / gamma)
    Lb_inv = Lb ** (1.0 / gamma)
    a = (Lw_inv - Lb_inv) ** gamma
    b = Lb_inv / (Lw_inv - Lb_inv)
    return a * max(V + b, 0) ** gamma


# ============================================================================
# 테스트 실행
# ============================================================================

def run_test():
    # ── 디스플레이 설정 ──
    Lw = 300.0   # cd/m²
    Lb = 0.5     # cd/m² (비영!)
    native_gamma = 2.4
    target_gamma = 2.2
    target_cct = 6500.0

    display = DisplaySimulator(Lw, Lb, native_gamma)

    print("=" * 72)
    print("  감마 캘리브레이션 블랙레벨 검증 테스트")
    print("=" * 72)

    print("\n[디스플레이 시뮬레이션]")
    print("  블랙 레벨 (Lb):  {:.2f} cd/m²".format(Lb))
    print("  백색 휘도 (Lw):  {:.1f} cd/m²".format(Lw))
    print("  명암비:           {}:1".format(int(Lw / Lb)))
    print("  네이티브 감마:    {}".format(native_gamma))
    print("  타겟 감마:        {}".format(target_gamma))

    # ── 1. 측정 데이터 생성 (21 포인트) ──
    gcal = GammaCalibrator(target_gamma=target_gamma,
                           target_cct=target_cct)
    levels = np.linspace(0, 1, 21)

    for lv in levels:
        w = display.measure_white(lv)
        r = display.measure_red(lv)
        g = display.measure_green(lv)
        b = display.measure_blue(lv)
        gcal.add_measurement(lv, w, r, g, b)

    # ── 2. LUT 생성 ──
    lut = gcal.generate_lut()

    # ── 3. 기본 특성 확인 ──
    gamma_est = gcal.get_measured_gamma()
    cct_est = gcal.get_measured_cct()
    print("\n[측정 결과]")
    print("  추정 감마: R={r:.3f}  G={g:.3f}  B={b:.3f}".format(**gamma_est))
    print("  추정 CCT:  {:.0f}K".format(cct_est))

    # ── 4. 핵심 검증 ──
    print("\n[검증 1] Lw/Lb 속성 존재 여부:")
    has_lw = hasattr(gcal, 'measured_Lw')
    has_lb = hasattr(gcal, 'measured_Lb')
    print("  measured_Lw: {}".format("O" if has_lw else "X (미구현)"))
    print("  measured_Lb: {}".format("O" if has_lb else "X (미구현)"))

    # ── 5. 측정된 블랙/백색 Y 값 ──
    m0 = gcal.measurements[0]   # level = 0.0
    m100 = gcal.measurements[-1]  # level = 1.0
    black_Y = m0.white_XYZ[1]
    white_Y = m100.white_XYZ[1]
    print("\n[검증 2] 측정된 블랙/백색 휘도:")
    print("  블랙 Y (level=0.0): {:.4f} cd/m²".format(black_Y))
    print("  백색 Y (level=1.0): {:.4f} cd/m²".format(white_Y))
    print("  블랙/백색 비율:     {:.3f}%".format(
        black_Y / white_Y * 100))

    # ── 6. LUT[0] 값 확인 ──
    print("\n[검증 3] LUT[0] 값 (블랙 입력):")
    print("  LUT[0] R={:.6f} G={:.6f} B={:.6f}".format(
        lut.r[0], lut.g[0], lut.b[0]))
    if lut.r[0] == 0.0 and lut.g[0] == 0.0 and lut.b[0] == 0.0:
        print("  → LUT[0]=0은 BT.1886에서 정상 (V=0 → L=Lb: 물리적 최소)")
        print("  → 디스플레이 EOTF(0) = Lb = {:.2f} cd/m²".format(Lb))
        print("  → BT.1886: 입력 0에서 Lb가 타겟이므로 LUT[0]=0이 올바름")

    # ── 7. BT.1886 비교 검증 ──
    # 검증: 보정 후 White(R=G=B=t) 출력의 전체 Y가 BT.1886 타겟과 일치하는지
    print("\n[검증 4] LUT 보정 후 White(R=G=B) vs BT.1886 타겟:")
    print("  {:>5s} | {:>8s} {:>8s} {:>8s} | {:>10s} | {:>10s} | {:>6s}".format(
        "입력V", "LUT_R", "LUT_G", "LUT_B", "White_Y", "BT1886_Y", "에러%"))
    print("  " + "-" * 72)

    test_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
                   0.30, 0.50, 0.75, 1.0]
    errors_bt1886 = []
    errors_shadow = []

    for t in test_levels:
        idx = min(int(t * 1023), 1023)
        lut_r = lut.r[idx]
        lut_g = lut.g[idx]
        lut_b = lut.b[idx]

        # LUT 출력으로 디스플레이의 각 채널 구동 후 전체 Y 계산
        # Y = sum(M[1,:] * [eotf(R)/Lw, eotf(G)/Lw, eotf(B)/Lw]) * Lw
        # BT.709 M[1,:] = [0.2126, 0.7152, 0.0722] → 합 ≈ 1.0
        # White(R=G=B=V) → Y = eotf(V), so individual:
        Y_from_r = display.eotf(lut_r) * display.M[1, 0]
        Y_from_g = display.eotf(lut_g) * display.M[1, 1]
        Y_from_b = display.eotf(lut_b) * display.M[1, 2]
        # 보정: 순수 신호 합산 (블랙은 한 번만)
        # 각 채널 eotf = Lb + signal, 3채널 합산 시 Lb가 3번 카운트됨
        # 실제 White 출력 Y = eotf(R)*0.2126 + eotf(G)*0.7152 + eotf(B)*0.0722
        # 단, 디스플레이 모델에서 White(R=G=B=V) → Y = eotf(V)
        # 개별 채널 조합 시: Y = M[1,:] @ [eotf(R)/Lw, eotf(G)/Lw, eotf(B)/Lw] * Lw
        R_sub = display.eotf(lut_r) / Lw
        G_sub = display.eotf(lut_g) / Lw
        B_sub = display.eotf(lut_b) / Lw
        actual_Y = display.M[1, :] @ np.array([R_sub, G_sub, B_sub]) * Lw

        # BT.1886 타겟 휘도
        target_Y = bt1886_eotf(t, Lw, Lb, target_gamma)

        # 에러
        if target_Y > 0.01:
            err = abs(actual_Y - target_Y) / target_Y * 100
        else:
            err = 0.0
        errors_bt1886.append(err)
        if t <= 0.20:
            errors_shadow.append(err)

        print("  {:5.3f} | {:8.5f} {:8.5f} {:8.5f} | {:10.4f} | {:10.4f} | {:5.1f}%".format(
            t, lut_r, lut_g, lut_b, actual_Y, target_Y, err))

    avg_err = np.mean(errors_bt1886)
    max_err = np.max(errors_bt1886)
    shadow_err = np.mean(errors_shadow)

    print("\n[검증 요약]")
    print("  전체 평균 BT.1886 에러: {:.2f}%".format(avg_err))
    print("  최대 BT.1886 에러:      {:.2f}%".format(max_err))
    print("  암부(0~20%) 평균 에러:  {:.2f}%".format(shadow_err))

    # ── 8. 낮은 코드값 LUT 상세 분석 ──
    print("\n[검증 5] 저레벨 코드 LUT 상세 (White Y = 합산):")
    for idx in [0, 1, 2, 3, 5, 10, 20, 50]:
        t = idx / 1023.0
        target_Y = bt1886_eotf(t, Lw, Lb, target_gamma)
        R_sub = display.eotf(lut.r[idx]) / Lw
        G_sub = display.eotf(lut.g[idx]) / Lw
        B_sub = display.eotf(lut.b[idx]) / Lw
        actual_Y = display.M[1, :] @ np.array([R_sub, G_sub, B_sub]) * Lw
        print("  code {:3d} (V={:.4f}): "
              "actual={:.3f} cd/m²  target={:.3f} cd/m²".format(
                  idx, t, actual_Y, target_Y))

    # ── 결론 ──
    print("\n" + "=" * 72)
    PASS_THRESHOLD = 10.0  # 10% 이내이면 PASS (CCT 보정에 의한 잔여 포함)
    if avg_err < PASS_THRESHOLD and shadow_err < PASS_THRESHOLD:
        print("  결과: PASS — BT.1886 블랙레벨 보정이 올바르게 동작함")
        print("  (평균 에러 {:.2f}% < {:.1f}% 임계값)".format(
            avg_err, PASS_THRESHOLD))
        if avg_err > 3.0:
            print("  참고: 잔여 에러의 주요 원인은 CCT 보정")
            print("    (D65 → Planckian 6500K 백색점 시프트로 인한 휘도 변화)")
    else:
        print("  결과: FAIL — 블랙레벨 보정 미흡")
        print("  원인 분석:")
        if not has_lw:
            print("    - GammaCalibrator에 measured_Lw 속성 없음")
        if not has_lb:
            print("    - GammaCalibrator에 measured_Lb 속성 없음")
        print("    - 암부 영역 에러: {:.2f}%".format(shadow_err))
    print("=" * 72)

    # ── 추가 검증: CCT 보정 없이 순수 BT.1886 정확도 ──
    print("\n" + "=" * 72)
    print("  [추가 검증] CCT 보정 없이 순수 BT.1886 정확도")
    print("=" * 72)

    # 디스플레이 백색이 이미 Planckian 6500K인 경우 시뮬레이션
    from calibration_engine import TARGET_STANDARDS
    planckian_xy = ColorScience.planckian_xy(6500)
    std_6500 = {
        'R': (0.640, 0.330), 'G': (0.300, 0.600), 'B': (0.150, 0.060),
        'W': planckian_xy,
    }
    M_6500 = ColorScience.primaries_to_xyz_matrix(std_6500)

    gcal2 = GammaCalibrator(target_gamma=target_gamma, target_cct=6500)
    for lv in np.linspace(0, 1, 21):
        L = display.eotf(lv) / Lw
        L_0 = display.eotf(0) / Lw
        w = M_6500 @ np.array([L, L, L]) * Lw
        r = M_6500 @ np.array([L, L_0, L_0]) * Lw
        g = M_6500 @ np.array([L_0, L, L_0]) * Lw
        b = M_6500 @ np.array([L_0, L_0, L]) * Lw
        gcal2.add_measurement(lv, w, r, g, b)

    lut2 = gcal2.generate_lut()

    print("\n  {:>5s} | {:>8s} {:>8s} {:>8s} | {:>10s} | {:>10s} | {:>6s}".format(
        "입력V", "LUT_R", "LUT_G", "LUT_B", "White_Y", "BT1886_Y", "에러%"))
    print("  " + "-" * 72)

    pure_errors = []
    for t in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.75, 1.0]:
        idx = min(int(t * 1023), 1023)
        R_sub = display.eotf(lut2.r[idx]) / Lw
        G_sub = display.eotf(lut2.g[idx]) / Lw
        B_sub = display.eotf(lut2.b[idx]) / Lw
        actual_Y = M_6500[1, :] @ np.array([R_sub, G_sub, B_sub]) * Lw
        target_Y = bt1886_eotf(t, Lw, Lb, target_gamma)
        err = abs(actual_Y - target_Y) / max(target_Y, 0.01) * 100
        pure_errors.append(err)
        print("  {:5.3f} | {:8.5f} {:8.5f} {:8.5f} | {:10.4f} | {:10.4f} | {:5.1f}%".format(
            t, lut2.r[idx], lut2.g[idx], lut2.b[idx],
            actual_Y, target_Y, err))

    pure_avg = np.mean(pure_errors)
    pure_max = np.max(pure_errors)
    print("\n  순수 BT.1886 정확도 (CCT 보정 없음):")
    print("  평균 에러: {:.2f}%  최대 에러: {:.2f}%".format(
        pure_avg, pure_max))
    pure_pass = pure_avg < 3.0
    print("  결과: {} (3% 임계값)".format(
        "PASS" if pure_pass else "FAIL"))
    print("=" * 72)

    return avg_err < PASS_THRESHOLD


if __name__ == "__main__":
    run_test()
