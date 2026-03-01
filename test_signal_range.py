"""
Signal Range & Color Encoding 검증 테스트

RGB / YCbCr Limited Range / Full Range 변환 및 캘리브레이션 보상 검증.

테스트 항목:
  1. QuantizationRange: Limited ↔ Full 변환 정확도 (8/10/12-bit)
  2. RGB ↔ YCbCr 변환 라운드트립 (BT.601/709/2020)
  3. Limited Range LUT 생성: 서브블랙/슈퍼화이트 처리
  4. Limited Range LUT 보정 정확도 (BT.1886)
  5. Pattern encoding 라운드트립
  6. CalibrationConfig 신호 범위 직렬화/역직렬화

참조:
  - ITU-R BT.601-7 §2.5.3  Quantization of Y, Cb, Cr
  - ITU-R BT.709-6 §4.4    Quantization
  - ITU-R BT.2020-2 §6     Narrow/Full range

Usage:
    python test_signal_range.py
"""

import numpy as np
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_engine import (
    SignalRange, ColorEncoding, QuantizationRange,
    GammaCalibrator, CalibrationConfig, CalibrationPreset,
    LUT1D, ColorScience, TARGET_STANDARDS, LUTExporter,
)


def test_quantization_range_8bit():
    """8-bit Limited Range 양자화 변환 정확도"""
    print("\n[Test 1] 8-bit QuantizationRange")
    qr = QuantizationRange(bit_depth=8)
    all_pass = True

    # Y/RGB Limited: code 16=0.0, code 235=1.0
    tests = [
        (0.0,  16),
        (0.25, 71),    # round(219*0.25 + 16) = round(70.75) = 71
        (0.5,  126),   # round(219*0.5 + 16) = round(125.5) = 126
        (0.75, 180),   # round(219*0.75 + 16) = round(180.25) = 180
        (1.0,  235),
    ]
    for norm, expected_code in tests:
        code = qr.to_limited_code_y(norm)
        back = qr.from_limited_code_y(code)
        ok = (code == expected_code) and (abs(back - norm) < 0.005)
        if not ok:
            all_pass = False
        print("  norm={:.2f} -> code={} (expect={}) -> back={:.4f}  [{}]".format(
            norm, code, expected_code, back, "PASS" if ok else "FAIL"))

    # Chroma Limited: code 16=min, 128=neutral, 240=max
    c_tests = [
        (0.0,  128),   # neutral
        (0.5,  240),   # max positive
        (-0.5, 16),    # max negative
    ]
    for norm, expected_code in c_tests:
        code = qr.to_limited_code_c(norm)
        back = qr.from_limited_code_c(code)
        ok = (code == expected_code) and (abs(back - norm) < 0.005)
        if not ok:
            all_pass = False
        print("  chroma norm={:+.2f} -> code={} (expect={}) -> back={:+.4f}  [{}]".format(
            norm, code, expected_code, back, "PASS" if ok else "FAIL"))

    print("  8-bit: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_quantization_range_10bit():
    """10-bit Limited Range 양자화 변환 정확도"""
    print("\n[Test 2] 10-bit QuantizationRange")
    qr = QuantizationRange(bit_depth=10)
    all_pass = True

    # 10-bit: Y offset=64, range=876, max=1023
    # code = round(876 * norm + 64)
    tests = [
        (0.0,  64),
        (0.5,  502),   # round(876*0.5+64) = round(502) = 502
        (1.0,  940),   # 64 + 876 = 940
    ]
    for norm, expected_code in tests:
        code = qr.to_limited_code_y(norm)
        back = qr.from_limited_code_y(code)
        ok = (code == expected_code) and (abs(back - norm) < 0.002)
        if not ok:
            all_pass = False
        print("  norm={:.2f} -> code={} (expect={}) -> back={:.4f}  [{}]".format(
            norm, code, expected_code, back, "PASS" if ok else "FAIL"))

    # Limited ↔ Full round-trip
    for ltd_norm in [0.0, 0.25, 0.5, 0.75, 1.0]:
        full = qr.limited_to_full(ltd_norm)
        back = qr.full_to_limited(full)
        ok = abs(back - ltd_norm) < 1e-10
        if not ok:
            all_pass = False
        print("  ltd={:.2f} -> full={:.6f} -> back={:.6f}  [{}]".format(
            ltd_norm, full, back, "PASS" if ok else "FAIL"))

    print("  10-bit: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_limited_full_boundary_values():
    """Limited Range 경계값 검증 (서브블랙/슈퍼화이트)"""
    print("\n[Test 3] Limited/Full Range 경계값")
    qr = QuantizationRange(bit_depth=8)
    all_pass = True

    # Full 0.0 → Limited 범위 바깥 (서브블랙)
    ltd_at_full_zero = qr.full_to_limited(0.0)
    # Full 0.0 = code 0, Limited에서는 (0-16)/219 = -0.073...
    expected = -16.0 / 219.0
    ok = abs(ltd_at_full_zero - expected) < 0.001
    if not ok:
        all_pass = False
    print("  Full 0.0 -> Limited {:.4f} (expect {:.4f})  [{}]".format(
        ltd_at_full_zero, expected, "PASS" if ok else "FAIL"))

    # Full 1.0 → Limited 범위 바깥 (슈퍼화이트)
    ltd_at_full_one = qr.full_to_limited(1.0)
    expected_sw = (255.0 - 16.0) / 219.0  # 1.0913...
    ok = abs(ltd_at_full_one - expected_sw) < 0.001
    if not ok:
        all_pass = False
    print("  Full 1.0 -> Limited {:.4f} (expect {:.4f})  [{}]".format(
        ltd_at_full_one, expected_sw, "PASS" if ok else "FAIL"))

    # Limited 0.0 → Full: should be 16/255 = 0.0627
    full_at_ltd_zero = qr.limited_to_full(0.0)
    expected_f = 16.0 / 255.0
    ok = abs(full_at_ltd_zero - expected_f) < 0.001
    if not ok:
        all_pass = False
    print("  Limited 0.0 -> Full {:.4f} (expect {:.4f})  [{}]".format(
        full_at_ltd_zero, expected_f, "PASS" if ok else "FAIL"))

    # Limited 1.0 → Full: should be 235/255 = 0.9216
    full_at_ltd_one = qr.limited_to_full(1.0)
    expected_w = 235.0 / 255.0
    ok = abs(full_at_ltd_one - expected_w) < 0.001
    if not ok:
        all_pass = False
    print("  Limited 1.0 -> Full {:.4f} (expect {:.4f})  [{}]".format(
        full_at_ltd_one, expected_w, "PASS" if ok else "FAIL"))

    print("  경계값: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_rgb_ycbcr_roundtrip():
    """RGB ↔ YCbCr 변환 라운드트립 정확도 (BT.601/709/2020)"""
    print("\n[Test 4] RGB <-> YCbCr round-trip")
    all_pass = True

    test_colors = {
        'White':   np.array([1.0, 1.0, 1.0]),
        'Black':   np.array([0.0, 0.0, 0.0]),
        'Red':     np.array([1.0, 0.0, 0.0]),
        'Green':   np.array([0.0, 1.0, 0.0]),
        'Blue':    np.array([0.0, 0.0, 1.0]),
        'Cyan':    np.array([0.0, 1.0, 1.0]),
        'Magenta': np.array([1.0, 0.0, 1.0]),
        'Yellow':  np.array([1.0, 1.0, 0.0]),
        '50%Gray': np.array([0.5, 0.5, 0.5]),
        '25%Gray': np.array([0.25, 0.25, 0.25]),
    }

    for std in ['BT.601', 'BT.709', 'BT.2020']:
        max_err = 0
        for name, rgb in test_colors.items():
            ycbcr = QuantizationRange.rgb_to_ycbcr(rgb, std)
            rgb_back = QuantizationRange.ycbcr_to_rgb(ycbcr, std)
            err = np.max(np.abs(rgb - rgb_back))
            max_err = max(max_err, err)

        # White: Y=1, Cb=Cr=0
        w_ycbcr = QuantizationRange.rgb_to_ycbcr(
            np.array([1.0, 1.0, 1.0]), std)
        y_ok = abs(w_ycbcr[0] - 1.0) < 1e-10
        cb_ok = abs(w_ycbcr[1]) < 1e-10
        cr_ok = abs(w_ycbcr[2]) < 1e-10

        # Gray: Y=level, Cb=Cr=0
        g_ycbcr = QuantizationRange.rgb_to_ycbcr(
            np.array([0.5, 0.5, 0.5]), std)
        g_y_ok = abs(g_ycbcr[0] - 0.5) < 1e-10
        g_cb_ok = abs(g_ycbcr[1]) < 1e-10

        ok = (max_err < 1e-10) and y_ok and cb_ok and cr_ok and g_y_ok
        if not ok:
            all_pass = False
        print("  {}: max_err={:.2e}  W.Y={:.6f}  W.Cb={:.2e}  "
              "Gray.Y={:.6f}  [{}]".format(
                  std, max_err, w_ycbcr[0], w_ycbcr[1],
                  g_ycbcr[0], "PASS" if ok else "FAIL"))

    print("  RGB<->YCbCr: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_ycbcr_coefficient_values():
    """YCbCr 계수가 ITU 표준과 일치하는지 검증"""
    print("\n[Test 5] YCbCr 계수 정확도")
    all_pass = True

    # ITU-R BT.709 정의: Kr=0.2126, Kb=0.0722
    expected = {
        'BT.601':  (0.299,  0.114),
        'BT.709':  (0.2126, 0.0722),
        'BT.2020': (0.2627, 0.0593),
    }

    for std, (kr_exp, kb_exp) in expected.items():
        coeff = QuantizationRange.YCBCR_COEFFICIENTS[std]
        kr_ok = abs(coeff['Kr'] - kr_exp) < 1e-10
        kb_ok = abs(coeff['Kb'] - kb_exp) < 1e-10
        ok = kr_ok and kb_ok
        if not ok:
            all_pass = False
        kg = 1.0 - coeff['Kr'] - coeff['Kb']
        print("  {}: Kr={:.4f}(exp={:.4f}) Kb={:.4f}(exp={:.4f}) "
              "Kg={:.4f}  [{}]".format(
                  std, coeff['Kr'], kr_exp, coeff['Kb'], kb_exp,
                  kg, "PASS" if ok else "FAIL"))

    print("  계수: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_limited_range_lut_generation():
    """Limited Range LUT 생성: 서브블랙/슈퍼화이트/유효범위 처리"""
    print("\n[Test 6] Limited Range LUT generation")
    all_pass = True

    # 가상 디스플레이 시뮬레이션
    Lb = 0.5    # cd/m2
    Lw = 300.0  # cd/m2
    native_gamma = 2.4
    target_gamma = 2.2

    gcal = GammaCalibrator(
        target_gamma=target_gamma, target_cct=6500,
        signal_range=SignalRange.LIMITED, bit_depth=8)

    M_sim = ColorScience.primaries_to_xyz_matrix(TARGET_STANDARDS['BT.709'])
    for i in range(21):
        lv = i / 20.0
        L_ch = Lb + (Lw - Lb) * (lv ** native_gamma)
        L_norm = L_ch / Lw
        L_0 = Lb / Lw
        w_xyz = M_sim @ np.array([L_norm, L_norm, L_norm]) * Lw
        r_xyz = M_sim @ np.array([L_norm, L_0, L_0]) * Lw
        g_xyz = M_sim @ np.array([L_0, L_norm, L_0]) * Lw
        b_xyz = M_sim @ np.array([L_0, L_0, L_norm]) * Lw
        gcal.add_measurement(lv, w_xyz, r_xyz, g_xyz, b_xyz)

    lut = gcal.generate_lut()
    qr = QuantizationRange(8)
    idx_black, idx_white = qr.get_lut_active_indices(
        SignalRange.LIMITED, lut.size)

    # Check 1: signal_range 메타데이터
    ok = lut.signal_range == SignalRange.LIMITED
    if not ok:
        all_pass = False
    print("  LUT signal_range = {} [{}]".format(
        lut.signal_range.value, "PASS" if ok else "FAIL"))

    # Check 2: 서브블랙 영역 (index 0 ~ idx_black)이 동일한 블랙 출력
    sub_black_consistent = True
    for i in range(0, min(idx_black + 1, lut.size)):
        if abs(lut.r[i] - lut.r[0]) > 1e-6:
            sub_black_consistent = False
            break
    if not sub_black_consistent:
        all_pass = False
    print("  Sub-black region consistent: [{}]".format(
        "PASS" if sub_black_consistent else "FAIL"))

    # Check 3: 슈퍼화이트 영역 (idx_white ~ 1023)이 동일한 화이트 출력
    super_white_consistent = True
    for i in range(idx_white, lut.size):
        if abs(lut.r[i] - lut.r[-1]) > 1e-6:
            super_white_consistent = False
            break
    if not super_white_consistent:
        all_pass = False
    print("  Super-white region consistent: [{}]".format(
        "PASS" if super_white_consistent else "FAIL"))

    # Check 4: 유효 범위 내 단조증가
    monotonic = True
    for i in range(idx_black + 1, idx_white):
        if lut.r[i] < lut.r[i-1] - 1e-10:
            monotonic = False
            break
    if not monotonic:
        all_pass = False
    print("  Monotonically increasing in active range: [{}]".format(
        "PASS" if monotonic else "FAIL"))

    # Check 5: 블랙 출력값이 limited range의 16/255에 근사
    expected_black = qr.limited_to_full(0.0)  # 16/255 = 0.0627
    black_ok = abs(lut.r[0] - expected_black) < 0.01
    if not black_ok:
        all_pass = False
    print("  Black output={:.4f} (expect ~{:.4f}): [{}]".format(
        lut.r[0], expected_black, "PASS" if black_ok else "FAIL"))

    # Check 6: 화이트 출력값이 limited range의 235/255에 근사
    expected_white = qr.limited_to_full(1.0)  # 235/255 = 0.9216
    # Note: CCT 보정 때문에 정확히 0.9216은 아닐 수 있음 (G채널 약간 낮음)
    white_ok = abs(lut.r[-1] - expected_white) < 0.02
    if not white_ok:
        all_pass = False
    print("  White output R={:.4f} (expect ~{:.4f}): [{}]".format(
        lut.r[-1], expected_white, "PASS" if white_ok else "FAIL"))

    print("  Limited LUT: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_full_range_lut_unchanged():
    """Full Range LUT는 기존 동작과 동일한지 검증"""
    print("\n[Test 7] Full Range LUT unchanged behavior")
    all_pass = True

    Lb = 0.5
    Lw = 300.0
    native_gamma = 2.4

    # Full Range 캘리브레이터
    gcal_full = GammaCalibrator(
        target_gamma=2.2, target_cct=6500,
        signal_range=SignalRange.FULL, bit_depth=8)

    # 동일한 측정 데이터
    M_sim = ColorScience.primaries_to_xyz_matrix(TARGET_STANDARDS['BT.709'])
    for i in range(21):
        lv = i / 20.0
        L_ch = Lb + (Lw - Lb) * (lv ** native_gamma)
        L_norm = L_ch / Lw
        L_0 = Lb / Lw
        w_xyz = M_sim @ np.array([L_norm, L_norm, L_norm]) * Lw
        r_xyz = M_sim @ np.array([L_norm, L_0, L_0]) * Lw
        g_xyz = M_sim @ np.array([L_0, L_norm, L_0]) * Lw
        b_xyz = M_sim @ np.array([L_0, L_0, L_norm]) * Lw
        gcal_full.add_measurement(lv, w_xyz, r_xyz, g_xyz, b_xyz)

    lut = gcal_full.generate_lut()

    # LUT[0] 은 0.0 근처
    ok_black = lut.r[0] < 0.01
    if not ok_black:
        all_pass = False
    print("  Full Range LUT[0]={:.6f} (should be ~0.0): [{}]".format(
        lut.r[0], "PASS" if ok_black else "FAIL"))

    # LUT[1023] 은 1.0 근처
    ok_white = lut.r[1023] > 0.99
    if not ok_white:
        all_pass = False
    print("  Full Range LUT[1023]={:.6f} (should be ~1.0): [{}]".format(
        lut.r[1023], "PASS" if ok_white else "FAIL"))

    # signal_range == FULL
    ok_range = lut.signal_range == SignalRange.FULL
    if not ok_range:
        all_pass = False
    print("  signal_range = {}: [{}]".format(
        lut.signal_range.value, "PASS" if ok_range else "FAIL"))

    print("  Full Range: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_lut_to_limited_conversion():
    """Full Range LUT → Limited Range 변환 메서드"""
    print("\n[Test 8] LUT1D.to_limited_range() conversion")
    all_pass = True

    # Identity LUT (pass-through)
    lut_full = LUT1D(size=1024, signal_range=SignalRange.FULL)
    lut_ltd = lut_full.to_limited_range(bit_depth=8)

    # 변환된 LUT 메타데이터 확인
    ok = lut_ltd.signal_range == SignalRange.LIMITED
    if not ok:
        all_pass = False
    print("  Converted signal_range = {}: [{}]".format(
        lut_ltd.signal_range.value, "PASS" if ok else "FAIL"))

    ok = lut_ltd.bit_depth == 8
    if not ok:
        all_pass = False
    print("  Converted bit_depth = {}: [{}]".format(
        lut_ltd.bit_depth, "PASS" if ok else "FAIL"))

    # LUT[0] (서브블랙) 출력이 limited 블랙에 근사
    qr = QuantizationRange(8)
    expected_min = qr.limited_to_full(0.0)
    ok = abs(lut_ltd.r[0] - expected_min) < 0.01
    if not ok:
        all_pass = False
    print("  LUT[0] = {:.4f} (expect ~{:.4f}): [{}]".format(
        lut_ltd.r[0], expected_min, "PASS" if ok else "FAIL"))

    print("  to_limited_range: {}".format(
        "ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_pattern_encoding():
    """패턴 인코딩/디코딩 라운드트립 (모든 비트깊이)"""
    print("\n[Test 9] Pattern encoding round-trip")
    all_pass = True

    for bd in [8, 10, 12]:
        qr = QuantizationRange(bit_depth=bd)
        max_err = 0
        for desired in np.linspace(0, 1, 101):
            encoded = qr.encode_pattern_value(desired, SignalRange.LIMITED)
            decoded = qr.decode_pattern_value(encoded, SignalRange.LIMITED)
            err = abs(decoded - desired)
            max_err = max(max_err, err)

        ok = max_err < 0.001
        if not ok:
            all_pass = False
        print("  {}-bit: max round-trip error = {:.6f}  [{}]".format(
            bd, max_err, "PASS" if ok else "FAIL"))

        # Full Range는 그대로 통과
        for desired in [0.0, 0.5, 1.0]:
            encoded = qr.encode_pattern_value(desired, SignalRange.FULL)
            ok_full = abs(encoded - desired) < 1e-10
            if not ok_full:
                all_pass = False

    print("  Pattern encoding: {}".format(
        "ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_config_signal_range_serialization():
    """CalibrationConfig 신호 범위 설정 JSON 직렬화"""
    print("\n[Test 10] Config signal range JSON serialization")
    all_pass = True

    cfg = CalibrationConfig.from_preset(
        CalibrationPreset.STANDARD,
        signal_range=SignalRange.LIMITED,
        color_encoding=ColorEncoding.YCBCR_444,
        bit_depth=10,
        ycbcr_standard='BT.2020',
        gpu_handles_range=False)

    # Save & Load
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False) as f:
        tmp_path = f.name
    try:
        cfg.to_json(tmp_path)
        cfg_loaded = CalibrationConfig.from_json(tmp_path)

        checks = [
            ('signal_range', cfg_loaded.signal_range, SignalRange.LIMITED),
            ('color_encoding', cfg_loaded.color_encoding,
             ColorEncoding.YCBCR_444),
            ('bit_depth', cfg_loaded.bit_depth, 10),
            ('ycbcr_standard', cfg_loaded.ycbcr_standard, 'BT.2020'),
            ('gpu_handles_range', cfg_loaded.gpu_handles_range, False),
        ]
        for name, actual, expected in checks:
            ok = actual == expected
            if not ok:
                all_pass = False
            print("  {}: {} == {} [{}]".format(
                name, actual, expected, "PASS" if ok else "FAIL"))
    finally:
        os.unlink(tmp_path)

    print("  Serialization: {}".format(
        "ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_lut_export_limited_domain():
    """Limited Range LUT 내보내기: DOMAIN 확인"""
    print("\n[Test 11] Limited Range LUT export domain")
    all_pass = True

    lut = LUT1D(size=256, signal_range=SignalRange.LIMITED, bit_depth=8)

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.cube', delete=False) as f:
        tmp_path = f.name
    try:
        LUTExporter.export_1d_cube(lut, tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # DOMAIN_MIN should be ~0.0627
        # DOMAIN_MAX should be ~0.9216
        qr = QuantizationRange(8)
        d_min, d_max = qr.get_lut_domain(SignalRange.LIMITED)

        has_min = "DOMAIN_MIN {:.6f}".format(d_min) in content
        has_max = "DOMAIN_MAX {:.6f}".format(d_max) in content
        has_range = "Range=limited" in content

        ok = has_min and has_max and has_range
        if not ok:
            all_pass = False
        print("  DOMAIN_MIN contains {:.4f}: [{}]".format(
            d_min, "PASS" if has_min else "FAIL"))
        print("  DOMAIN_MAX contains {:.4f}: [{}]".format(
            d_max, "PASS" if has_max else "FAIL"))
        print("  Range=limited in header: [{}]".format(
            "PASS" if has_range else "FAIL"))
    finally:
        os.unlink(tmp_path)

    print("  Export domain: {}".format(
        "ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_lut_active_indices():
    """LUT 유효 인덱스 범위 검증"""
    print("\n[Test 12] LUT active indices")
    all_pass = True

    for bd, expected_min_idx, expected_max_idx in [
        (8,  64, 942),    # 16/255*1023 ≈ 64, 235/255*1023 ≈ 942
        (10, 64, 940),    # 64/1023*1023 ≈ 64, 940/1023*1023 ≈ 940
    ]:
        qr = QuantizationRange(bit_depth=bd)
        idx_min, idx_max = qr.get_lut_active_indices(
            SignalRange.LIMITED, 1024)
        ok = (idx_min == expected_min_idx and idx_max == expected_max_idx)
        if not ok:
            all_pass = False
        print("  {}-bit: active [{}, {}] (expect [{}, {}])  [{}]".format(
            bd, idx_min, idx_max, expected_min_idx, expected_max_idx,
            "PASS" if ok else "FAIL"))

    # Full Range: 0 ~ 1023
    qr8 = QuantizationRange(8)
    idx_min, idx_max = qr8.get_lut_active_indices(SignalRange.FULL, 1024)
    ok = (idx_min == 0 and idx_max == 1023)
    if not ok:
        all_pass = False
    print("  Full Range: active [{}, {}] (expect [0, 1023])  [{}]".format(
        idx_min, idx_max, "PASS" if ok else "FAIL"))

    print("  Active indices: {}".format(
        "ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_ycbcr_primary_colors():
    """주요 색상의 YCbCr 값 검증 (ITU 표준 참조)"""
    print("\n[Test 13] YCbCr primary color values (BT.709)")
    all_pass = True

    # BT.709 정의: Kr=0.2126, Kb=0.0722, Kg=0.7152
    # Red (1,0,0): Y=0.2126, Cb=(0-0.2126)/(2*0.9278)=-0.1146,
    #              Cr=(1-0.2126)/(2*0.7874)=0.5
    red_ycbcr = QuantizationRange.rgb_to_ycbcr(
        np.array([1.0, 0.0, 0.0]), 'BT.709')
    ok_y = abs(red_ycbcr[0] - 0.2126) < 0.001
    ok_cr = abs(red_ycbcr[2] - 0.5) < 0.001
    ok = ok_y and ok_cr
    if not ok:
        all_pass = False
    print("  Red: Y={:.4f}(exp=0.2126) Cr={:.4f}(exp=0.5)  [{}]".format(
        red_ycbcr[0], red_ycbcr[2], "PASS" if ok else "FAIL"))

    # Blue (0,0,1): Y=0.0722, Cb=0.5, Cr=...
    blue_ycbcr = QuantizationRange.rgb_to_ycbcr(
        np.array([0.0, 0.0, 1.0]), 'BT.709')
    ok_y = abs(blue_ycbcr[0] - 0.0722) < 0.001
    ok_cb = abs(blue_ycbcr[1] - 0.5) < 0.001
    ok = ok_y and ok_cb
    if not ok:
        all_pass = False
    print("  Blue: Y={:.4f}(exp=0.0722) Cb={:.4f}(exp=0.5)  [{}]".format(
        blue_ycbcr[0], blue_ycbcr[1], "PASS" if ok else "FAIL"))

    # Green (0,1,0): Y=0.7152, Cb and Cr both negative
    green_ycbcr = QuantizationRange.rgb_to_ycbcr(
        np.array([0.0, 1.0, 0.0]), 'BT.709')
    ok_y = abs(green_ycbcr[0] - 0.7152) < 0.001
    ok_cb = green_ycbcr[1] < 0  # negative
    ok_cr = green_ycbcr[2] < 0  # negative
    ok = ok_y and ok_cb and ok_cr
    if not ok:
        all_pass = False
    print("  Green: Y={:.4f}(exp=0.7152) Cb={:+.4f}(<0) Cr={:+.4f}(<0)  "
          "[{}]".format(
              green_ycbcr[0], green_ycbcr[1], green_ycbcr[2],
              "PASS" if ok else "FAIL"))

    print("  YCbCr primaries: {}".format(
        "ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def test_12bit_quantization():
    """12-bit 양자화 변환"""
    print("\n[Test 14] 12-bit QuantizationRange")
    qr = QuantizationRange(bit_depth=12)
    all_pass = True

    # 12-bit: offset=256, range=3504, max=4095
    # Y code = round(3504 * norm + 256)
    tests = [
        (0.0,  256),
        (0.5,  2008),   # round(3504*0.5+256) = round(2008)
        (1.0,  3760),   # 256 + 3504 = 3760
    ]
    for norm, expected_code in tests:
        code = qr.to_limited_code_y(norm)
        ok = code == expected_code
        if not ok:
            all_pass = False
        print("  norm={:.2f} -> code={} (expect={})  [{}]".format(
            norm, code, expected_code, "PASS" if ok else "FAIL"))

    print("  12-bit: {}".format("ALL PASS" if all_pass else "SOME FAIL"))
    return all_pass


def run_all_tests():
    """전체 테스트 실행"""
    print("=" * 72)
    print("  Signal Range & Color Encoding Verification Tests")
    print("=" * 72)

    results = []
    results.append(("QuantizationRange 8-bit", test_quantization_range_8bit()))
    results.append(("QuantizationRange 10-bit", test_quantization_range_10bit()))
    results.append(("Limited/Full boundary", test_limited_full_boundary_values()))
    results.append(("RGB<->YCbCr roundtrip", test_rgb_ycbcr_roundtrip()))
    results.append(("YCbCr coefficients", test_ycbcr_coefficient_values()))
    results.append(("Limited Range LUT gen", test_limited_range_lut_generation()))
    results.append(("Full Range LUT unchanged", test_full_range_lut_unchanged()))
    results.append(("LUT to_limited_range()", test_lut_to_limited_conversion()))
    results.append(("Pattern encoding", test_pattern_encoding()))
    results.append(("Config serialization", test_config_signal_range_serialization()))
    results.append(("LUT export domain", test_lut_export_limited_domain()))
    results.append(("LUT active indices", test_lut_active_indices()))
    results.append(("YCbCr primary colors", test_ycbcr_primary_colors()))
    results.append(("12-bit quantization", test_12bit_quantization()))

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print("  {:40s}  [{}]".format(name, status))

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print("\n  Total: {}/{}  {}".format(
        passed_count, total,
        "ALL PASSED" if all_passed else "SOME FAILED"))
    print("=" * 72)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
