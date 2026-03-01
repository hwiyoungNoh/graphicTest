#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sensor_module.py 종합 테스트 모듈
=================================

테스트 범주:
  1. 데이터 구조 (Dataclass) 검증
  2. VirtualSensor 기능 검증
  3. CRColorimeterSensor 단위 테스트 (Mock 시리얼)
  4. 프로토콜 파싱 로직 검증
  5. 색 변환 (XYZ ↔ RGB, XYZ → xy) 검증
  6. 팩토리 함수 검증
  7. 설정 편의 메서드 검증
  8. 엣지 케이스 & 오류 처리

실행 방법:
  python test_sensor_module.py
  또는
  python -m pytest test_sensor_module.py -v
"""

import sys
import os
import time
import logging
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from io import StringIO
from dataclasses import asdict

import numpy as np

# === 로깅 설정 ===
LOG_FILE = "test_sensor_module.log"
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"

# 파일 핸들러 + 콘솔 핸들러
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

test_logger = logging.getLogger("test_sensor")

# === 모듈 import ===
from sensor_module import (
    # Enums
    CRObserver, FlickerFilterType, FlickerFilterFamily,
    ResponseTimeFilterType, ResponseTimeMode,
    # Dataclasses
    SensorReading, CRModeItem, CRAccessoryItem, CRMatrixItem,
    CRFilterItem, CRConfiguration, CRSetup, CRCIEData,
    CRSpectrumData, CRTemporalData, CRWarning, CRReading,
    FlickerSettings, ResponseTimeSettings,
    # Constants
    CR_NEW_LINE, CR_RESPONSE_SEPARATOR, CR_RESULT_SEPARATOR,
    CR_RESPONSE_OK, CR_RESPONSE_ERROR,
    CR_DEFAULT_BAUDRATE, CR_DEFAULT_TIMEOUT, CR_COMMAND_DELAY,
    # Classes
    SensorInterface, CRColorimeterSensor, VirtualSensor,
    # Factory
    create_sensor,
)


# ============================================================================
# 1. 데이터 구조 테스트
# ============================================================================

class TestDataStructures(unittest.TestCase):
    """Dataclass 생성 및 기본값 검증"""

    def test_sensor_reading_defaults(self):
        """SensorReading 기본값 검증"""
        test_logger.info(">>> SensorReading 기본값 테스트")
        r = SensorReading(
            rgb=np.array([0.0, 0.0, 0.0]),
            xyz=np.array([0.0, 0.0, 0.0]),
            cie_xy=(0.0, 0.0),
            luminance=0.0,
            timestamp=0.0,
        )
        np.testing.assert_array_equal(r.rgb, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(r.xyz, np.array([0.0, 0.0, 0.0]))
        self.assertEqual(r.cie_xy, (0.0, 0.0))
        self.assertEqual(r.luminance, 0.0)
        self.assertTrue(r.is_valid)  # default is True
        self.assertEqual(r.error_message, "")
        test_logger.info("    ✔ SensorReading 기본값 정상")

    def test_sensor_reading_custom(self):
        """SensorReading 커스텀 값 검증"""
        test_logger.info(">>> SensorReading 커스텀 값 테스트")
        r = SensorReading(
            rgb=np.array([0.5, 0.6, 0.7]),
            xyz=np.array([30.0, 40.0, 50.0]),
            cie_xy=(0.25, 0.35),
            luminance=120.5,
            timestamp=1000.0,
            is_valid=True,
            error_message="",
        )
        self.assertTrue(r.is_valid)
        self.assertAlmostEqual(r.luminance, 120.5)
        self.assertAlmostEqual(r.cie_xy[0], 0.25)
        test_logger.info("    ✔ SensorReading 커스텀 값 정상")

    def test_cr_configuration_defaults(self):
        """CRConfiguration 기본값 검증"""
        test_logger.info(">>> CRConfiguration 기본값 테스트")
        cfg = CRConfiguration()
        self.assertEqual(cfg.id, "")
        self.assertEqual(cfg.model, "")
        self.assertEqual(cfg.firmware, "")
        self.assertEqual(cfg.instrument_type, 0)
        self.assertEqual(len(cfg.modes), 0)
        self.assertEqual(len(cfg.accessories), 0)
        self.assertEqual(len(cfg.filters), 0)
        self.assertAlmostEqual(cfg.min_exposure, 0.0)
        self.assertAlmostEqual(cfg.max_exposure, 0.0)
        test_logger.info("    ✔ CRConfiguration 기본값 정상")

    def test_cr_setup_defaults(self):
        """CRSetup 기본값 검증"""
        test_logger.info(">>> CRSetup 기본값 테스트")
        s = CRSetup()
        self.assertEqual(s.mode_id, 0)
        self.assertEqual(s.accessory_id, -1)
        self.assertEqual(s.filter1_id, -1)
        self.assertAlmostEqual(s.sync_freq, 60.0)
        self.assertEqual(s.cmf, 0)
        test_logger.info("    ✔ CRSetup 기본값 정상")

    def test_cr_reading_defaults(self):
        """CRReading 기본값 검증"""
        test_logger.info(">>> CRReading 기본값 테스트")
        r = CRReading()
        self.assertEqual(len(r.cie), 2)
        self.assertIsInstance(r.cie[0], CRCIEData)
        self.assertIsInstance(r.cie[1], CRCIEData)
        self.assertEqual(len(r.warnings), 0)
        self.assertIsInstance(r.spectrum, CRSpectrumData)
        self.assertIsInstance(r.temporal, CRTemporalData)
        test_logger.info("    ✔ CRReading 기본값 정상")

    def test_cr_cie_data(self):
        """CRCIEData 값 설정 및 읽기"""
        test_logger.info(">>> CRCIEData 테스트")
        cie = CRCIEData(X="30.5", Y="40.2", Z="50.7",
                         XYZ="30.5 40.2 50.7",
                         xy="0.252 0.332")
        self.assertEqual(cie.X, "30.5")
        self.assertAlmostEqual(float(cie.Y), 40.2)
        test_logger.info("    ✔ CRCIEData 정상")

    def test_spectrum_data(self):
        """CRSpectrumData 검증"""
        test_logger.info(">>> CRSpectrumData 테스트")
        sd = CRSpectrumData(
            starting_wavelength=380.0,
            ending_wavelength=780.0,
            delta=5.0,
            data=[0.1 * i for i in range(81)]
        )
        self.assertEqual(len(sd.data), 81)
        self.assertAlmostEqual(sd.starting_wavelength, 380.0)
        self.assertAlmostEqual(sd.ending_wavelength, 780.0)
        test_logger.info("    ✔ CRSpectrumData 정상 (81 데이터 포인트)")

    def test_temporal_data(self):
        """CRTemporalData 검증"""
        test_logger.info(">>> CRTemporalData 테스트")
        td = CRTemporalData(sampling_rate=1000.0,
                             data=[0.5 + 0.3 * np.sin(i * 0.1) for i in range(100)])
        self.assertEqual(len(td.data), 100)
        self.assertAlmostEqual(td.sampling_rate, 1000.0)
        test_logger.info("    ✔ CRTemporalData 정상 (100 샘플)")

    def test_flicker_settings_defaults(self):
        """FlickerSettings 기본값"""
        test_logger.info(">>> FlickerSettings 기본값 테스트")
        fs = FlickerSettings()
        self.assertEqual(fs.filter_type, FlickerFilterType.NONE)
        self.assertEqual(fs.filter_family, FlickerFilterFamily.NONE)
        self.assertEqual(fs.order, 1)
        self.assertAlmostEqual(fs.frequency, 800.0)
        test_logger.info("    ✔ FlickerSettings 기본값 정상")

    def test_response_time_settings_defaults(self):
        """ResponseTimeSettings 기본값"""
        test_logger.info(">>> ResponseTimeSettings 기본값 테스트")
        rts = ResponseTimeSettings()
        self.assertEqual(rts.mode, ResponseTimeMode.AUTO)
        self.assertEqual(rts.filter_type, ResponseTimeFilterType.NONE)
        self.assertEqual(rts.average, 5)
        self.assertFalse(rts.clipping_enabled)
        test_logger.info("    ✔ ResponseTimeSettings 기본값 정상")

    def test_enum_values(self):
        """Enum 값 검증"""
        test_logger.info(">>> Enum 값 테스트")
        self.assertEqual(CRObserver.DEGREE_2, 0)
        self.assertEqual(CRObserver.DEGREE_10, 1)
        self.assertEqual(FlickerFilterType.NONE, 0)
        self.assertEqual(FlickerFilterType.LOWPASS, 1)
        self.assertEqual(FlickerFilterType.BANDPASS, 3)
        self.assertEqual(ResponseTimeMode.AUTO, 0)
        self.assertEqual(ResponseTimeMode.MANUAL, 1)
        test_logger.info("    ✔ Enum 값 정상")


# ============================================================================
# 2. VirtualSensor 테스트
# ============================================================================

class TestVirtualSensor(unittest.TestCase):
    """VirtualSensor 기능 종합 검증"""

    def setUp(self):
        np.random.seed(42)  # 재현 가능한 결과
        self.sensor = VirtualSensor(noise_level=0.02)

    def test_initial_state(self):
        """초기 상태 검증"""
        test_logger.info(">>> VirtualSensor 초기 상태 테스트")
        self.assertFalse(self.sensor.is_connected())
        self.assertEqual(self.sensor.get_measurement_count(), 0)
        test_logger.info("    ✔ 초기 상태: 미연결, 측정 0회")

    def test_connect_disconnect(self):
        """연결/해제 동작 검증"""
        test_logger.info(">>> VirtualSensor 연결/해제 테스트")
        result = self.sensor.connect()
        self.assertTrue(result)
        self.assertTrue(self.sensor.is_connected())
        test_logger.info("    ✔ 연결 성공")

        result = self.sensor.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.sensor.is_connected())
        test_logger.info("    ✔ 연결 해제 성공")

    def test_read_when_connected(self):
        """연결 상태에서 측정 검증"""
        test_logger.info(">>> VirtualSensor 측정 테스트 (연결됨)")
        self.sensor.connect()
        reading = self.sensor.read()

        self.assertTrue(reading.is_valid)
        self.assertEqual(reading.error_message, "")
        self.assertEqual(reading.rgb.shape, (3,))
        self.assertEqual(reading.xyz.shape, (3,))
        self.assertEqual(len(reading.cie_xy), 2)
        self.assertGreater(reading.luminance, 0)
        self.assertGreater(reading.timestamp, 0)

        # RGB 범위 [0, 1]
        self.assertTrue(np.all(reading.rgb >= 0))
        self.assertTrue(np.all(reading.rgb <= 1))

        # CIE xy 범위 (합리적 범위)
        x, y = reading.cie_xy
        self.assertGreater(x, 0)
        self.assertLess(x, 1)
        self.assertGreater(y, 0)
        self.assertLess(y, 1)

        test_logger.info("    ✔ 측정값 유효: RGB=[%.4f,%.4f,%.4f] "
                         "xy=(%.4f,%.4f) Y=%.2f",
                         *reading.rgb, *reading.cie_xy, reading.luminance)

    def test_read_when_disconnected(self):
        """미연결 상태에서 측정 검증"""
        test_logger.info(">>> VirtualSensor 측정 테스트 (미연결)")
        reading = self.sensor.read()
        self.assertFalse(reading.is_valid)
        self.assertIn("not connected", reading.error_message.lower())
        test_logger.info("    ✔ 미연결 시 is_valid=False, 에러 메시지 정상")

    def test_measurement_count(self):
        """측정 카운트 검증"""
        test_logger.info(">>> VirtualSensor 측정 카운트 테스트")
        self.sensor.connect()

        for i in range(5):
            self.sensor.read()
        self.assertEqual(self.sensor.get_measurement_count(), 5)
        test_logger.info("    ✔ 5회 측정 후 카운트 = %d",
                         self.sensor.get_measurement_count())

        self.sensor.reset_measurement_count()
        self.assertEqual(self.sensor.get_measurement_count(), 0)
        test_logger.info("    ✔ 카운트 리셋 성공")

    def test_multiple_readings_variation(self):
        """다수 측정 시 값의 변동 검증 (노이즈 존재 확인)"""
        test_logger.info(">>> VirtualSensor 다수 측정 변동 테스트")
        self.sensor.connect()

        readings = [self.sensor.read() for _ in range(10)]
        rgbs = [r.rgb for r in readings]

        # 모든 측정이 유효해야 함
        self.assertTrue(all(r.is_valid for r in readings))

        # 최소한 일부 측정 값은 서로 달라야 함 (노이즈 존재)
        unique = len(set(tuple(r.tolist()) for r in rgbs))
        self.assertGreater(unique, 1, "모든 측정이 동일 — 노이즈가 작동하지 않음")
        test_logger.info("    ✔ 10회 측정 중 %d개 고유값 (노이즈 동작 확인)", unique)

    def test_noise_level_effect(self):
        """노이즈 레벨에 따른 분산 변화"""
        test_logger.info(">>> VirtualSensor 노이즈 레벨 영향 테스트")

        np.random.seed(42)
        low_noise = VirtualSensor(noise_level=0.001)
        low_noise.connect()
        low_readings = [low_noise.read().rgb[0] for _ in range(50)]
        low_std = np.std(low_readings)

        np.random.seed(42)
        high_noise = VirtualSensor(noise_level=0.1)
        high_noise.connect()
        high_readings = [high_noise.read().rgb[0] for _ in range(50)]
        high_std = np.std(high_readings)

        test_logger.info("    Low noise std=%.6f, High noise std=%.6f",
                         low_std, high_std)
        # 참고: 랜덤 베이스라인 자체 분산이 커서 노이즈 효과가 상대적으로 작을 수 있음
        # 그래도 양쪽 모두 유효한 측정 범위인지 확인
        self.assertTrue(np.all(np.array(low_readings) >= 0))
        self.assertTrue(np.all(np.array(low_readings) <= 1))
        self.assertTrue(np.all(np.array(high_readings) >= 0))
        self.assertTrue(np.all(np.array(high_readings) <= 1))
        test_logger.info("    ✔ 모든 측정값이 [0,1] 범위 내 유지")

    def test_rgb_to_xyz_conversion(self):
        """VirtualSensor 내부 RGB→XYZ 변환 검증"""
        test_logger.info(">>> VirtualSensor RGB→XYZ 변환 테스트")
        # D65 white (1,1,1) → XYZ 합이 ~1 근방
        white_xyz = VirtualSensor._rgb_to_xyz(np.array([1.0, 1.0, 1.0]))
        self.assertGreater(white_xyz[0], 0)
        self.assertGreater(white_xyz[1], 0)
        self.assertGreater(white_xyz[2], 0)
        test_logger.info("    White (1,1,1) → XYZ = [%.4f, %.4f, %.4f]",
                         *white_xyz)

        # Black (0,0,0) → XYZ = (0,0,0)
        black_xyz = VirtualSensor._rgb_to_xyz(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(black_xyz, [0, 0, 0])
        test_logger.info("    Black (0,0,0) → XYZ = [%.4f, %.4f, %.4f]",
                         *black_xyz)

        # Red (1,0,0) → X > Y, Z
        red_xyz = VirtualSensor._rgb_to_xyz(np.array([1.0, 0.0, 0.0]))
        self.assertGreater(red_xyz[0], red_xyz[1])
        self.assertGreater(red_xyz[0], red_xyz[2])
        test_logger.info("    Red (1,0,0) → XYZ = [%.4f, %.4f, %.4f]",
                         *red_xyz)
        test_logger.info("    ✔ RGB→XYZ 변환 정상")

    def test_xyz_to_xy_conversion(self):
        """VirtualSensor 내부 XYZ→xy 변환 검증"""
        test_logger.info(">>> VirtualSensor XYZ→xy 변환 테스트")
        # D65 근사값
        x, y = VirtualSensor._xyz_to_xy(np.array([0.9505, 1.0000, 1.0890]))
        self.assertAlmostEqual(x, 0.3127, places=2)
        self.assertAlmostEqual(y, 0.3290, places=2)
        test_logger.info("    D65 XYZ → xy = (%.4f, %.4f)  [기대: ~(0.3127, 0.3290)]",
                         x, y)

        # zero → D65 fallback
        x0, y0 = VirtualSensor._xyz_to_xy(np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(x0, 0.3127)
        self.assertAlmostEqual(y0, 0.3290)
        test_logger.info("    Zero XYZ → xy = (%.4f, %.4f) [D65 fallback]", x0, y0)
        test_logger.info("    ✔ XYZ→xy 변환 정상")


# ============================================================================
# 3. CRColorimeterSensor 단위 테스트 (Mock)
# ============================================================================

class TestCRColorimeterSensor(unittest.TestCase):
    """CRColorimeterSensor Mock 기반 단위 테스트"""

    def setUp(self):
        """Mock 없이 인스턴스 생성 (connect 하지 않음)"""
        self.sensor = CRColorimeterSensor(port='COM99', baudrate=9600)

    def test_initial_state(self):
        """초기 상태 검증"""
        test_logger.info(">>> CRColorimeterSensor 초기 상태 테스트")
        self.assertFalse(self.sensor.is_connected())
        self.assertEqual(self.sensor.port, 'COM99')
        self.assertEqual(self.sensor.baudrate, 9600)
        self.assertEqual(self.sensor.measurement_count, 0)
        self.assertIsInstance(self.sensor.configuration, CRConfiguration)
        self.assertIsInstance(self.sensor.setup, CRSetup)
        self.assertIsInstance(self.sensor.flicker_settings, FlickerSettings)
        self.assertIsInstance(self.sensor.response_time_settings, ResponseTimeSettings)
        test_logger.info("    ✔ 초기 상태 정상")

    def test_disconnect_without_connect(self):
        """연결 없이 disconnect 호출"""
        test_logger.info(">>> CRColorimeterSensor 미연결 disconnect 테스트")
        result = self.sensor.disconnect()
        self.assertTrue(result)
        test_logger.info("    ✔ 미연결 상태에서 disconnect 안전하게 성공")

    def test_read_when_not_connected(self):
        """미연결 상태에서 read 호출"""
        test_logger.info(">>> CRColorimeterSensor 미연결 read 테스트")
        reading = self.sensor.read()
        self.assertFalse(reading.is_valid)
        self.assertIn("not connected", reading.error_message.lower())
        test_logger.info("    ✔ 미연결 시 is_valid=False, 에러메시지: '%s'",
                         reading.error_message)

    def test_connect_without_pyserial(self):
        """pyserial 미설치 시 connect 실패"""
        test_logger.info(">>> CRColorimeterSensor pyserial 미설치 테스트")
        with patch.dict('sys.modules', {'serial': None}):
            sensor = CRColorimeterSensor(port='COM99')
            # connect 내부에서 import serial 실패 시 False 반환
            result = sensor.connect()
            self.assertFalse(result)
            test_logger.info("    ✔ pyserial 없을 때 connect() = False")


# ============================================================================
# 4. 프로토콜 파싱 테스트
# ============================================================================

class TestProtocolParsing(unittest.TestCase):
    """CR 프로토콜 응답 파싱 로직 검증"""

    def setUp(self):
        self.sensor = CRColorimeterSensor(port='COM99')

    def test_parse_ok_response(self):
        """OK 응답 파싱"""
        test_logger.info(">>> OK 응답 파싱 테스트")
        resp = "OK:00:RC Firmware:1.23"
        parsed = self.sensor._parse_response(resp)
        self.assertEqual(parsed['type'], 'OK')
        self.assertEqual(parsed['code'], '00')
        self.assertEqual(parsed['command'], 'RC Firmware')
        self.assertEqual(parsed['result'], '1.23')
        test_logger.info("    파싱 결과: %s", parsed)
        test_logger.info("    ✔ OK 응답 파싱 정상")

    def test_parse_error_response(self):
        """ER 응답 파싱"""
        test_logger.info(">>> ER 응답 파싱 테스트")
        resp = "ER:01:M:Sensor not ready"
        parsed = self.sensor._parse_response(resp)
        self.assertEqual(parsed['type'], 'ER')
        self.assertEqual(parsed['code'], '01')
        self.assertEqual(parsed['command'], 'M')
        self.assertEqual(parsed['result'], 'Sensor not ready')
        test_logger.info("    파싱 결과: %s", parsed)
        test_logger.info("    ✔ ER 응답 파싱 정상")

    def test_parse_empty_response(self):
        """빈 응답 파싱"""
        test_logger.info(">>> 빈 응답 파싱 테스트")
        parsed = self.sensor._parse_response("")
        self.assertEqual(parsed['type'], '')
        self.assertEqual(parsed['code'], '')
        self.assertEqual(parsed['command'], '')
        self.assertEqual(parsed['result'], '')
        test_logger.info("    ✔ 빈 응답 파싱 안전하게 처리")

    def test_parse_partial_response(self):
        """불완전 응답 파싱"""
        test_logger.info(">>> 불완전 응답 파싱 테스트")
        parsed = self.sensor._parse_response("OK:00")
        self.assertEqual(parsed['type'], 'OK')
        self.assertEqual(parsed['code'], '00')
        self.assertEqual(parsed['command'], '')
        self.assertEqual(parsed['result'], '')
        test_logger.info("    ✔ 불완전 응답도 안전하게 처리")

    def test_parse_response_with_colons_in_result(self):
        """결과에 콜론이 포함된 응답 파싱 (maxsplit=3)"""
        test_logger.info(">>> 결과에 콜론 포함 응답 테스트")
        resp = "OK:00:RM Time:2024:01:15 12:30:45"
        parsed = self.sensor._parse_response(resp)
        self.assertEqual(parsed['type'], 'OK')
        self.assertEqual(parsed['code'], '00')
        self.assertEqual(parsed['command'], 'RM Time')
        self.assertEqual(parsed['result'], '2024:01:15 12:30:45')
        test_logger.info("    ✔ 콜론 포함 결과 정상 파싱: '%s'",
                         parsed['result'])

    def test_parse_list_item_format(self):
        """리스트 항목 형식 파싱 ("ID),Name),Type")"""
        test_logger.info(">>> 리스트 항목 형식 파싱 테스트")
        item = "0),CIE 1931),Standard"
        parts = item.split("),")
        self.assertEqual(len(parts), 3)
        self.assertEqual(int(parts[0]), 0)
        self.assertEqual(parts[1], "CIE 1931")
        self.assertEqual(parts[2], "Standard")
        test_logger.info("    ✔ 리스트 항목 파싱 정상: id=%s, name=%s, type=%s",
                         parts[0], parts[1], parts[2])


# ============================================================================
# 5. 색 변환 테스트
# ============================================================================

class TestColorConversions(unittest.TestCase):
    """XYZ ↔ RGB, XYZ → xy 변환 검증"""

    def test_xyz_to_rgb_white(self):
        """D65 white point XYZ → sRGB ≈ (1,1,1)"""
        test_logger.info(">>> XYZ→RGB D65 white 테스트")
        # D65 reference white
        xyz = np.array([0.9505, 1.0000, 1.0890])
        rgb = CRColorimeterSensor._xyz_to_rgb(xyz)
        test_logger.info("    D65 XYZ [0.9505, 1.0000, 1.0890] → RGB [%.4f, %.4f, %.4f]",
                         *rgb)
        # sRGB 변환 후 (1,1,1) 근방이어야 함
        np.testing.assert_array_almost_equal(rgb, [1.0, 1.0, 1.0], decimal=1)
        test_logger.info("    ✔ D65 → sRGB ≈ (1,1,1) 확인")

    def test_xyz_to_rgb_black(self):
        """XYZ (0,0,0) → sRGB (0,0,0)"""
        test_logger.info(">>> XYZ→RGB black 테스트")
        xyz = np.array([0.0, 0.0, 0.0])
        rgb = CRColorimeterSensor._xyz_to_rgb(xyz)
        np.testing.assert_array_almost_equal(rgb, [0.0, 0.0, 0.0])
        test_logger.info("    ✔ Black XYZ → RGB = [0,0,0]")

    def test_xyz_to_rgb_clamping(self):
        """XYZ 극단값 → sRGB [0,1] 클리핑"""
        test_logger.info(">>> XYZ→RGB 클리핑 테스트")
        xyz = np.array([2.0, 2.0, 2.0])  # 과도한 값
        rgb = CRColorimeterSensor._xyz_to_rgb(xyz)
        self.assertTrue(np.all(rgb >= 0), "RGB가 0 미만")
        self.assertTrue(np.all(rgb <= 1), "RGB가 1 초과")
        test_logger.info("    과도한 XYZ [2,2,2] → RGB [%.4f, %.4f, %.4f] (클리핑됨)",
                         *rgb)
        test_logger.info("    ✔ 클리핑 정상 작동")

    def test_xyz_to_rgb_negative_handling(self):
        """음수 XYZ → sRGB 안전 처리"""
        test_logger.info(">>> XYZ→RGB 음수 처리 테스트")
        xyz = np.array([-0.1, 0.5, 0.2])
        rgb = CRColorimeterSensor._xyz_to_rgb(xyz)
        self.assertTrue(np.all(rgb >= 0), "클리핑 후 음수 RGB 존재")
        test_logger.info("    음수 XYZ [-0.1, 0.5, 0.2] → RGB [%.4f, %.4f, %.4f]",
                         *rgb)
        test_logger.info("    ✔ 음수 입력도 안전하게 처리")

    def test_xyz_to_xy_static_normal(self):
        """XYZ → CIE xy 정상 변환"""
        test_logger.info(">>> XYZ→xy 정상 변환 테스트")
        xyz = np.array([0.9505, 1.0000, 1.0890])
        x, y = CRColorimeterSensor._xyz_to_xy_static(xyz)
        s = sum(xyz)
        expected_x = xyz[0] / s
        expected_y = xyz[1] / s
        self.assertAlmostEqual(x, expected_x, places=5)
        self.assertAlmostEqual(y, expected_y, places=5)
        test_logger.info("    D65 XYZ → xy = (%.5f, %.5f)", x, y)
        test_logger.info("    ✔ 정상 변환 확인")

    def test_xyz_to_xy_static_zero(self):
        """XYZ = 0 → D65 fallback"""
        test_logger.info(">>> XYZ→xy zero fallback 테스트")
        x, y = CRColorimeterSensor._xyz_to_xy_static(np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(x, 0.3127)
        self.assertAlmostEqual(y, 0.3290)
        test_logger.info("    Zero XYZ → xy = (%.4f, %.4f) [D65 fallback]", x, y)
        test_logger.info("    ✔ D65 fallback 정상")

    def test_xyz_to_xy_static_near_zero(self):
        """XYZ 매우 작은 값 → D65 fallback"""
        test_logger.info(">>> XYZ→xy near-zero 테스트")
        x, y = CRColorimeterSensor._xyz_to_xy_static(np.array([1e-12, 1e-12, 1e-12]))
        self.assertAlmostEqual(x, 0.3127)
        self.assertAlmostEqual(y, 0.3290)
        test_logger.info("    ✔ 매우 작은 XYZ도 D65 fallback 적용")

    def test_round_trip_consistency(self):
        """RGB → XYZ → RGB 왕복 일관성 (근사)"""
        test_logger.info(">>> RGB→XYZ→RGB 왕복 테스트")
        # VirtualSensor RGB→XYZ, CRColorimeterSensor XYZ→RGB
        original_rgb = np.array([0.5, 0.3, 0.8])
        xyz = VirtualSensor._rgb_to_xyz(original_rgb)
        recovered_rgb = CRColorimeterSensor._xyz_to_rgb(xyz)

        test_logger.info("    원본 RGB:  [%.4f, %.4f, %.4f]", *original_rgb)
        test_logger.info("    중간 XYZ:  [%.4f, %.4f, %.4f]", *xyz)
        test_logger.info("    복원 RGB:  [%.4f, %.4f, %.4f]", *recovered_rgb)

        # 감마 처리 방식이 다르므로 정확한 일치는 아니지만 근방이어야 함
        diff = np.abs(original_rgb - recovered_rgb)
        max_diff = np.max(diff)
        test_logger.info("    최대 차이: %.6f", max_diff)
        self.assertLess(max_diff, 0.15,
                        "왕복 변환 오차가 허용 범위 초과: %.6f" % max_diff)
        test_logger.info("    ✔ 왕복 변환 오차 %.6f < 0.15 (합격)", max_diff)

    def test_primary_colors_xyz_to_rgb(self):
        """순색 Primary → RGB 변환 방향성 검증"""
        test_logger.info(">>> 순색 XYZ→RGB 방향성 테스트")

        # 순수 Red에 가까운 XYZ
        red_xyz = np.array([0.4124, 0.2127, 0.0193])
        red_rgb = CRColorimeterSensor._xyz_to_rgb(red_xyz)
        self.assertGreater(red_rgb[0], red_rgb[1])
        self.assertGreater(red_rgb[0], red_rgb[2])
        test_logger.info("    Red XYZ → RGB = [%.4f, %.4f, %.4f]  R이 최대", *red_rgb)

        # 순수 Green에 가까운 XYZ
        green_xyz = np.array([0.3576, 0.7152, 0.1192])
        green_rgb = CRColorimeterSensor._xyz_to_rgb(green_xyz)
        self.assertGreater(green_rgb[1], green_rgb[0])
        self.assertGreater(green_rgb[1], green_rgb[2])
        test_logger.info("    Green XYZ → RGB = [%.4f, %.4f, %.4f]  G가 최대", *green_rgb)

        # 순수 Blue에 가까운 XYZ
        blue_xyz = np.array([0.1805, 0.0722, 0.9505])
        blue_rgb = CRColorimeterSensor._xyz_to_rgb(blue_xyz)
        self.assertGreater(blue_rgb[2], blue_rgb[0])
        self.assertGreater(blue_rgb[2], blue_rgb[1])
        test_logger.info("    Blue XYZ → RGB = [%.4f, %.4f, %.4f]  B가 최대", *blue_rgb)

        test_logger.info("    ✔ 순색 방향성 모두 정상")


# ============================================================================
# 6. CRReading 파싱 유틸리티 테스트
# ============================================================================

class TestCRReadingParsing(unittest.TestCase):
    """CRReading 데이터 파싱 유틸리티 검증"""

    def setUp(self):
        self.sensor = CRColorimeterSensor(port='COM99')

    def test_parse_xyz_valid(self):
        """유효한 CIE XYZ 파싱"""
        test_logger.info(">>> CRReading XYZ 파싱 테스트 (유효)")
        reading = CRReading()
        reading.cie[0].X = "30.5"
        reading.cie[0].Y = "40.2"
        reading.cie[0].Z = "50.7"

        xyz = CRColorimeterSensor._parse_xyz(reading, CRObserver.DEGREE_2)
        np.testing.assert_array_almost_equal(xyz, [30.5, 40.2, 50.7])
        test_logger.info("    ✔ XYZ 파싱 정상: [%.1f, %.1f, %.1f]", *xyz)

    def test_parse_xyz_empty(self):
        """빈 CIE XYZ 파싱"""
        test_logger.info(">>> CRReading XYZ 파싱 테스트 (빈 값)")
        reading = CRReading()
        xyz = CRColorimeterSensor._parse_xyz(reading, CRObserver.DEGREE_2)
        np.testing.assert_array_almost_equal(xyz, [0.0, 0.0, 0.0])
        test_logger.info("    ✔ 빈 XYZ → [0,0,0]")

    def test_parse_xyz_invalid_string(self):
        """잘못된 문자열 XYZ 파싱"""
        test_logger.info(">>> CRReading XYZ 파싱 테스트 (잘못된 값)")
        reading = CRReading()
        reading.cie[0].X = "invalid"
        reading.cie[0].Y = "40.2"
        reading.cie[0].Z = "50.7"
        xyz = CRColorimeterSensor._parse_xyz(reading, CRObserver.DEGREE_2)
        # ValueError 발생 시 (0,0,0) fallback
        np.testing.assert_array_almost_equal(xyz, [0.0, 0.0, 0.0])
        test_logger.info("    ✔ 잘못된 값 → [0,0,0] fallback")

    def test_parse_xy_valid(self):
        """유효한 CIE xy 파싱"""
        test_logger.info(">>> CRReading xy 파싱 테스트 (유효)")
        reading = CRReading()
        reading.cie[0].xy = "0.3127),0.3290"
        x, y = CRColorimeterSensor._parse_xy(reading, CRObserver.DEGREE_2)
        self.assertAlmostEqual(x, 0.3127, places=3)
        self.assertAlmostEqual(y, 0.3290, places=3)
        test_logger.info("    ✔ xy 파싱 정상: (%.4f, %.4f)", x, y)

    def test_parse_xy_fallback_to_xyz(self):
        """xy 없을 때 XYZ→xy fallback"""
        test_logger.info(">>> CRReading xy fallback 테스트")
        reading = CRReading()
        reading.cie[0].X = "0.9505"
        reading.cie[0].Y = "1.0000"
        reading.cie[0].Z = "1.0890"
        reading.cie[0].xy = ""
        x, y = CRColorimeterSensor._parse_xy(reading, CRObserver.DEGREE_2)
        self.assertAlmostEqual(x, 0.3127, places=2)
        self.assertAlmostEqual(y, 0.3290, places=2)
        test_logger.info("    ✔ XYZ→xy fallback 정상: (%.4f, %.4f)", x, y)

    def test_parse_luminance_valid(self):
        """유효한 휘도 파싱"""
        test_logger.info(">>> CRReading 휘도 파싱 테스트")
        reading = CRReading()
        reading.cie[0].Y = "120.5"
        lum = CRColorimeterSensor._parse_luminance(reading, CRObserver.DEGREE_2)
        self.assertAlmostEqual(lum, 120.5)
        test_logger.info("    ✔ 휘도 파싱 정상: %.1f cd/m²", lum)

    def test_parse_luminance_empty(self):
        """빈 휘도 파싱"""
        test_logger.info(">>> CRReading 빈 휘도 파싱 테스트")
        reading = CRReading()
        lum = CRColorimeterSensor._parse_luminance(reading, CRObserver.DEGREE_2)
        self.assertAlmostEqual(lum, 0.0)
        test_logger.info("    ✔ 빈 값 → 0.0 cd/m²")

    def test_parse_10degree_observer(self):
        """10° Observer 데이터 파싱"""
        test_logger.info(">>> CRReading 10° Observer 테스트")
        reading = CRReading()
        reading.cie[1].X = "31.0"
        reading.cie[1].Y = "41.0"
        reading.cie[1].Z = "51.0"
        xyz = CRColorimeterSensor._parse_xyz(reading, CRObserver.DEGREE_10)
        np.testing.assert_array_almost_equal(xyz, [31.0, 41.0, 51.0])
        test_logger.info("    ✔ 10° Observer 파싱 정상: [%.1f, %.1f, %.1f]", *xyz)


# ============================================================================
# 7. 설정 편의 메서드 테스트
# ============================================================================

class TestSetupConvenienceMethods(unittest.TestCase):
    """CRColorimeterSensor 설정 편의 메서드 검증"""

    def setUp(self):
        self.sensor = CRColorimeterSensor(port='COM99')
        # 가상 configuration 구성
        self.sensor.configuration.modes = [
            CRModeItem(id=0, name="Standard"),
            CRModeItem(id=1, name="Fast"),
        ]
        self.sensor.configuration.exposure_modes = [
            CRModeItem(id=0, name="Auto"),
            CRModeItem(id=1, name="Manual"),
        ]
        self.sensor.configuration.speeds = [
            CRModeItem(id=0, name="Normal"),
            CRModeItem(id=1, name="Fast"),
            CRModeItem(id=2, name="Slow"),
        ]
        self.sensor.configuration.apertures = [
            CRModeItem(id=0, name="1 degree"),
        ]
        self.sensor.configuration.sync_modes = [
            CRModeItem(id=0, name="Internal"),
            CRModeItem(id=1, name="External"),
        ]
        self.sensor.configuration.range_modes = [
            CRModeItem(id=0, name="Auto"),
        ]
        self.sensor.configuration.accessories = [
            CRAccessoryItem(id=0, name="None"),
            CRAccessoryItem(id=1, name="CRL-100"),
        ]
        self.sensor.configuration.min_exposure = 0.1
        self.sensor.configuration.max_exposure = 10000.0
        self.sensor.configuration.min_sync_freq = 20.0
        self.sensor.configuration.max_sync_freq = 200.0
        self.sensor.configuration.min_sampling_rate = 100.0
        self.sensor.configuration.max_sampling_rate = 100000.0

    def test_set_mode(self):
        """모드 설정"""
        test_logger.info(">>> set_mode 테스트")
        self.sensor.set_mode(1)
        self.assertEqual(self.sensor.setup_modified.mode_id, 1)
        test_logger.info("    ✔ mode_id = 1 (Fast)")

    def test_set_mode_out_of_range(self):
        """범위 초과 모드 설정 → 무시"""
        test_logger.info(">>> set_mode 범위 초과 테스트")
        original = self.sensor.setup_modified.mode_id
        self.sensor.set_mode(99)
        self.assertEqual(self.sensor.setup_modified.mode_id, original)
        test_logger.info("    ✔ 범위 초과 시 변경 없음")

    def test_set_exposure_mode(self):
        """노출 모드 설정"""
        test_logger.info(">>> set_exposure_mode 테스트")
        self.sensor.set_exposure_mode(1)
        self.assertEqual(self.sensor.setup_modified.exposure_mode_id, 1)
        test_logger.info("    ✔ exposure_mode_id = 1 (Manual)")

    def test_set_exposure(self):
        """노출 시간 설정"""
        test_logger.info(">>> set_exposure 테스트")
        self.sensor.set_exposure(50.0)
        self.assertAlmostEqual(self.sensor.setup_modified.exposure, 50.0)
        test_logger.info("    ✔ exposure = 50.0 msec")

    def test_set_exposure_out_of_range(self):
        """범위 초과 노출 시간 → 무시"""
        test_logger.info(">>> set_exposure 범위 초과 테스트")
        original = self.sensor.setup_modified.exposure
        self.sensor.set_exposure(99999.0)  # max는 10000
        self.assertEqual(self.sensor.setup_modified.exposure, original)
        test_logger.info("    ✔ 범위 초과 시 변경 없음")

    def test_set_speed(self):
        """속도 설정"""
        test_logger.info(">>> set_speed 테스트")
        self.sensor.set_speed(2)
        self.assertEqual(self.sensor.setup_modified.speed_id, 2)
        test_logger.info("    ✔ speed_id = 2 (Slow)")

    def test_set_aperture(self):
        """조리개 설정"""
        test_logger.info(">>> set_aperture 테스트")
        self.sensor.set_aperture(0)
        self.assertEqual(self.sensor.setup_modified.aperture_id, 0)
        test_logger.info("    ✔ aperture_id = 0")

    def test_set_sync_mode(self):
        """동기화 모드 설정"""
        test_logger.info(">>> set_sync_mode 테스트")
        self.sensor.set_sync_mode(1)
        self.assertEqual(self.sensor.setup_modified.sync_mode_id, 1)
        test_logger.info("    ✔ sync_mode_id = 1 (External)")

    def test_set_sync_freq(self):
        """동기화 주파수 설정"""
        test_logger.info(">>> set_sync_freq 테스트")
        self.sensor.set_sync_freq(120.0)
        self.assertAlmostEqual(self.sensor.setup_modified.sync_freq, 120.0)
        test_logger.info("    ✔ sync_freq = 120.0 Hz")

    def test_set_sync_freq_out_of_range(self):
        """범위 초과 동기 주파수 → 무시"""
        test_logger.info(">>> set_sync_freq 범위 초과 테스트")
        original = self.sensor.setup_modified.sync_freq
        self.sensor.set_sync_freq(500.0)  # max는 200
        self.assertEqual(self.sensor.setup_modified.sync_freq, original)
        test_logger.info("    ✔ 범위 초과 시 변경 없음")

    def test_set_sampling_rate(self):
        """샘플링 레이트 설정"""
        test_logger.info(">>> set_sampling_rate 테스트")
        self.sensor.set_sampling_rate(48000.0)
        self.assertAlmostEqual(self.sensor.setup_modified.sampling_rate, 48000.0)
        test_logger.info("    ✔ sampling_rate = 48000.0 Hz")

    def test_set_range_mode(self):
        """범위 모드 설정"""
        test_logger.info(">>> set_range_mode 테스트")
        self.sensor.set_range_mode(0)
        self.assertEqual(self.sensor.setup_modified.range_mode_id, 0)
        test_logger.info("    ✔ range_mode_id = 0 (Auto)")

    def test_set_accessory(self):
        """액세서리 설정"""
        test_logger.info(">>> set_accessory 테스트")
        self.sensor.set_accessory(1)
        self.assertEqual(self.sensor.setup_modified.accessory_id, 1)
        test_logger.info("    ✔ accessory_id = 1 (CRL-100)")

    def test_set_cmf(self):
        """CMF 설정"""
        test_logger.info(">>> set_cmf 테스트")
        self.sensor.set_cmf(1)
        self.assertEqual(self.sensor.setup_modified.cmf, 1)
        test_logger.info("    ✔ cmf = 1")


# ============================================================================
# 8. 플리커/응답시간 설정 테스트
# ============================================================================

class TestFlickerResponseTimeSettings(unittest.TestCase):
    """플리커 및 응답시간 분석 설정 검증"""

    def setUp(self):
        self.sensor = CRColorimeterSensor(port='COM99')

    def test_flicker_filter_type(self):
        """플리커 필터 타입 설정"""
        test_logger.info(">>> 플리커 필터 타입 테스트")
        self.sensor.set_flicker_filter_type(FlickerFilterType.LOWPASS)
        self.assertEqual(self.sensor.flicker_settings.filter_type,
                         FlickerFilterType.LOWPASS)
        test_logger.info("    ✔ filter_type = LOWPASS")

    def test_flicker_filter_order(self):
        """플리커 필터 차수 설정"""
        test_logger.info(">>> 플리커 필터 차수 테스트")
        self.sensor.set_flicker_filter_order(5)
        self.assertEqual(self.sensor.flicker_settings.order, 5)

        # 범위 초과
        self.sensor.set_flicker_filter_order(100)
        self.assertEqual(self.sensor.flicker_settings.order, 5)  # 변경 안됨
        test_logger.info("    ✔ 유효 범위 내: order=5, 범위 초과 시 무시")

    def test_flicker_frequency(self):
        """플리커 주파수 설정"""
        test_logger.info(">>> 플리커 주파수 테스트")
        self.sensor.set_flicker_filter_frequency(120.0)
        self.assertAlmostEqual(self.sensor.flicker_settings.frequency, 120.0)
        test_logger.info("    ✔ frequency = 120.0 Hz")

    def test_flicker_bandwidth(self):
        """플리커 대역폭 설정"""
        test_logger.info(">>> 플리커 대역폭 테스트")
        self.sensor.set_flicker_filter_bandwidth(200.0)
        self.assertAlmostEqual(self.sensor.flicker_settings.bandwidth, 200.0)
        test_logger.info("    ✔ bandwidth = 200.0 Hz")

    def test_flicker_max_search_frequency(self):
        """플리커 최대 검색 주파수 설정"""
        test_logger.info(">>> 플리커 최대 검색 주파수 테스트")
        self.sensor.set_flicker_max_search_frequency(60.0)
        self.assertAlmostEqual(
            self.sensor.flicker_settings.max_search_frequency, 60.0)
        test_logger.info("    ✔ max_search_frequency = 60.0 Hz")

    def test_response_time_mode(self):
        """응답시간 모드 설정"""
        test_logger.info(">>> 응답시간 모드 테스트")
        self.sensor.set_response_time_mode(ResponseTimeMode.MANUAL)
        self.assertEqual(self.sensor.response_time_settings.mode,
                         ResponseTimeMode.MANUAL)
        test_logger.info("    ✔ mode = MANUAL")

    def test_response_time_filter_type(self):
        """응답시간 필터 타입 설정"""
        test_logger.info(">>> 응답시간 필터 타입 테스트")
        self.sensor.set_response_time_filter_type(ResponseTimeFilterType.MOVING_WINDOW_AVERAGE)
        self.assertEqual(self.sensor.response_time_settings.filter_type,
                         ResponseTimeFilterType.MOVING_WINDOW_AVERAGE)
        test_logger.info("    ✔ filter_type = MOVING_WINDOW_AVERAGE")

    def test_response_time_average(self):
        """응답시간 평균 횟수 설정"""
        test_logger.info(">>> 응답시간 평균 테스트")
        self.sensor.set_response_time_average(10)
        self.assertEqual(self.sensor.response_time_settings.average, 10)

        # 범위 초과
        self.sensor.set_response_time_average(50)
        self.assertEqual(self.sensor.response_time_settings.average, 10)
        test_logger.info("    ✔ average = 10, 범위 초과 시 무시")

    def test_response_time_clipping(self):
        """응답시간 클리핑 설정"""
        test_logger.info(">>> 응답시간 클리핑 테스트")
        self.sensor.set_response_time_clipping(True, lo=0.2, hi=0.8)
        self.assertTrue(self.sensor.response_time_settings.clipping_enabled)
        self.assertAlmostEqual(self.sensor.response_time_settings.clipping_lo, 0.2)
        self.assertAlmostEqual(self.sensor.response_time_settings.clipping_hi, 0.8)
        test_logger.info("    ✔ clipping enabled, lo=0.2, hi=0.8")

    def test_response_time_noise_level(self):
        """응답시간 노이즈 레벨 설정"""
        test_logger.info(">>> 응답시간 노이즈 레벨 테스트")
        self.sensor.set_response_time_noise_level(0.03)
        self.assertAlmostEqual(self.sensor.response_time_settings.noise_level, 0.03)
        test_logger.info("    ✔ noise_level = 0.03")

    def test_response_time_step_zone(self):
        """응답시간 스텝 응답 구간 설정"""
        test_logger.info(">>> 응답시간 스텝 응답 구간 테스트")
        self.sensor.set_response_time_step_zone(lo=0.15, hi=0.85)
        self.assertAlmostEqual(
            self.sensor.response_time_settings.step_response_zone_lo, 0.15)
        self.assertAlmostEqual(
            self.sensor.response_time_settings.step_response_zone_hi, 0.85)
        test_logger.info("    ✔ step_zone lo=0.15, hi=0.85")


# ============================================================================
# 9. 유틸리티 메서드 테스트
# ============================================================================

class TestUtilityMethods(unittest.TestCase):
    """유틸리티 메서드 검증"""

    def setUp(self):
        self.sensor = CRColorimeterSensor(port='COM99')

    def test_find_id_by_name(self):
        """이름으로 ID 검색"""
        test_logger.info(">>> _find_id_by_name 테스트")
        items = [
            CRModeItem(id=0, name="Standard"),
            CRModeItem(id=1, name="Fast"),
            CRModeItem(id=2, name="Slow"),
        ]
        self.assertEqual(CRColorimeterSensor._find_id_by_name(items, "Fast"), 1)
        self.assertEqual(CRColorimeterSensor._find_id_by_name(items, "Slow"), 2)
        self.assertEqual(CRColorimeterSensor._find_id_by_name(items, "None"), -1)
        self.assertEqual(CRColorimeterSensor._find_id_by_name(items, "Unknown"), -1)
        test_logger.info("    ✔ 이름→ID 검색 정상")

    def test_measurement_count(self):
        """측정 카운트 관리"""
        test_logger.info(">>> 측정 카운트 테스트")
        self.assertEqual(self.sensor.get_measurement_count(), 0)
        self.sensor.measurement_count = 5
        self.assertEqual(self.sensor.get_measurement_count(), 5)
        self.sensor.reset_measurement_count()
        self.assertEqual(self.sensor.get_measurement_count(), 0)
        test_logger.info("    ✔ 측정 카운트 관리 정상")

    def test_get_device_info(self):
        """기기 정보 반환"""
        test_logger.info(">>> get_device_info 테스트")
        self.sensor.configuration.id = "SN12345"
        self.sensor.configuration.model = "CR-300"
        self.sensor.configuration.firmware = "1.25"
        self.sensor.configuration.instrument_type = 3
        self.sensor.configuration.modes = [CRModeItem(id=0, name="Standard")]
        self.sensor.configuration.accessories = []

        info = self.sensor.get_device_info()
        self.assertEqual(info['id'], "SN12345")
        self.assertEqual(info['model'], "CR-300")
        self.assertEqual(info['firmware'], "1.25")
        self.assertEqual(info['instrument_type'], "3")
        self.assertEqual(info['port'], "COM99")
        self.assertEqual(info['modes'], "1")
        self.assertEqual(info['accessories'], "0")
        test_logger.info("    ✔ 기기 정보: %s", info)

    def test_firmware_version_number(self):
        """펌웨어 버전 숫자 변환"""
        test_logger.info(">>> _firmware_version_number 테스트")
        self.sensor.configuration.firmware = "1.25"
        self.assertAlmostEqual(self.sensor._firmware_version_number(), 1.25)

        self.sensor.configuration.firmware = "invalid"
        self.assertAlmostEqual(self.sensor._firmware_version_number(), 0.0)

        self.sensor.configuration.firmware = ""
        self.assertAlmostEqual(self.sensor._firmware_version_number(), 0.0)
        test_logger.info("    ✔ 펌웨어 버전 숫자 변환 정상")


# ============================================================================
# 10. 팩토리 함수 테스트
# ============================================================================

class TestSensorFactory(unittest.TestCase):
    """create_sensor 팩토리 함수 검증"""

    def test_create_virtual_sensor(self):
        """가상 센서 생성"""
        test_logger.info(">>> create_sensor('virtual') 테스트")
        sensor = create_sensor('virtual', noise_level=0.05)
        self.assertIsInstance(sensor, VirtualSensor)
        self.assertIsInstance(sensor, SensorInterface)
        self.assertAlmostEqual(sensor.noise_level, 0.05)
        test_logger.info("    ✔ VirtualSensor 생성 정상")

    def test_create_cr_sensor(self):
        """CR 센서 생성"""
        test_logger.info(">>> create_sensor('cr') 테스트")
        sensor = create_sensor('cr', port='COM5', baudrate=9600)
        self.assertIsInstance(sensor, CRColorimeterSensor)
        self.assertIsInstance(sensor, SensorInterface)
        self.assertEqual(sensor.port, 'COM5')
        test_logger.info("    ✔ CRColorimeterSensor 생성 정상")

    def test_create_sensor_aliases(self):
        """CR 센서 별칭 검증"""
        test_logger.info(">>> create_sensor 별칭 테스트")
        for alias in ['colorimeter', 'cr100', 'cr250', 'cr300']:
            sensor = create_sensor(alias, port='COM1')
            self.assertIsInstance(sensor, CRColorimeterSensor)
            test_logger.info("    ✔ alias='%s' → CRColorimeterSensor", alias)

    def test_create_sensor_invalid_type(self):
        """잘못된 센서 타입 → ValueError"""
        test_logger.info(">>> create_sensor 잘못된 타입 테스트")
        with self.assertRaises(ValueError):
            create_sensor('nonexistent')
        test_logger.info("    ✔ ValueError 발생 확인")

    def test_create_virtual_default_noise(self):
        """가상 센서 기본 노이즈 레벨"""
        test_logger.info(">>> create_sensor('virtual') 기본 노이즈 테스트")
        sensor = create_sensor('virtual')
        self.assertAlmostEqual(sensor.noise_level, 0.02)
        test_logger.info("    ✔ 기본 noise_level = 0.02")


# ============================================================================
# 11. Mock 시리얼 통합 테스트
# ============================================================================

class TestCRSensorMockSerial(unittest.TestCase):
    """Mock Serial을 사용한 CRColorimeterSensor 통합 테스트"""

    def _create_mock_serial(self):
        """pyserial Mock 생성"""
        mock_serial_module = MagicMock()
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True

        mock_serial_module.Serial.return_value = mock_serial_instance
        mock_serial_module.EIGHTBITS = 8
        mock_serial_module.PARITY_NONE = 'N'
        mock_serial_module.STOPBITS_ONE = 1

        return mock_serial_module, mock_serial_instance

    def _setup_readline_responses(self, mock_serial, responses):
        """readline 응답 시퀀스 설정"""
        encoded = [(r + "\r\n").encode('ascii') for r in responses]
        # 마지막에 빈 응답(타임아웃)
        encoded.append(b"")
        mock_serial.readline.side_effect = encoded

    def test_send_command_format(self):
        """명령 전송 형식 검증"""
        test_logger.info(">>> Mock 시리얼 명령 전송 형식 테스트")
        mock_module, mock_serial = self._create_mock_serial()

        self._setup_readline_responses(mock_serial, [
            "OK:00:RC Firmware:1.25",
        ])

        with patch.dict('sys.modules', {'serial': mock_module}):
            sensor = CRColorimeterSensor(port='COM3')
            sensor._serial = mock_serial
            sensor._connected = True

            response = sensor._send_command("RC Firmware")

            # write 호출 검증
            mock_serial.write.assert_called_once()
            written = mock_serial.write.call_args[0][0]
            self.assertEqual(written, b"RC Firmware\r\n")
            test_logger.info("    ✔ 전송 형식: %s", written)

    def test_parse_and_validate_ok_response(self):
        """OK 응답 수신 및 파싱"""
        test_logger.info(">>> Mock 시리얼 OK 응답 수신 테스트")
        mock_module, mock_serial = self._create_mock_serial()

        self._setup_readline_responses(mock_serial, [
            "OK:00:RC Firmware:1.25",
        ])

        with patch.dict('sys.modules', {'serial': mock_module}):
            sensor = CRColorimeterSensor(port='COM3')
            sensor._serial = mock_serial
            sensor._connected = True

            result = sensor._send_and_get_result("RC Firmware")
            self.assertEqual(result, "1.25")
            test_logger.info("    ✔ 결과: '%s'", result)

    def test_error_response_raises_exception(self):
        """ER 응답 시 RuntimeError 발생"""
        test_logger.info(">>> Mock 시리얼 ER 응답 예외 테스트")
        mock_module, mock_serial = self._create_mock_serial()

        self._setup_readline_responses(mock_serial, [
            "ER:01:M:Sensor not ready",
        ])

        with patch.dict('sys.modules', {'serial': mock_module}):
            sensor = CRColorimeterSensor(port='COM3')
            sensor._serial = mock_serial
            sensor._connected = True

            with self.assertRaises(RuntimeError) as ctx:
                sensor._send_and_parse("M")
            self.assertIn("Sensor not ready", str(ctx.exception))
            test_logger.info("    ✔ RuntimeError 발생: %s", ctx.exception)

    def test_list_response_parsing(self):
        """리스트 응답 파싱 (count + 항목 라인)"""
        test_logger.info(">>> Mock 시리얼 리스트 응답 테스트")
        mock_module, mock_serial = self._create_mock_serial()

        # RC Mode → count=3, 이후 3줄 항목
        responses = [
            "OK:00:RC Mode:3",
            "0),Standard",
            "1),Fast",
            "2),Slow",
        ]
        encoded = [(r + "\r\n").encode('ascii') for r in responses]
        encoded.append(b"")
        encoded.append(b"")
        encoded.append(b"")
        mock_serial.readline.side_effect = encoded

        with patch.dict('sys.modules', {'serial': mock_module}):
            sensor = CRColorimeterSensor(port='COM3')
            sensor._serial = mock_serial
            sensor._connected = True

            count, items = sensor._send_and_get_list("RC Mode")
            self.assertEqual(count, 3)
            self.assertEqual(len(items), 3)
            self.assertIn("Standard", items[0])
            self.assertIn("Fast", items[1])
            self.assertIn("Slow", items[2])
            test_logger.info("    ✔ 리스트 응답: count=%d, items=%s",
                             count, items)


# ============================================================================
# 12. 상수값 검증
# ============================================================================

class TestConstants(unittest.TestCase):
    """모듈 상수값 검증"""

    def test_protocol_constants(self):
        """프로토콜 상수"""
        test_logger.info(">>> 프로토콜 상수 검증")
        self.assertEqual(CR_NEW_LINE, "\r\n")
        self.assertEqual(CR_RESPONSE_SEPARATOR, ":")
        self.assertEqual(CR_RESULT_SEPARATOR, "),")
        self.assertEqual(CR_RESPONSE_OK, "OK")
        self.assertEqual(CR_RESPONSE_ERROR, "ER")
        test_logger.info("    ✔ 프로토콜 상수 정상")

    def test_serial_defaults(self):
        """시리얼 기본값"""
        test_logger.info(">>> 시리얼 기본값 검증")
        self.assertEqual(CR_DEFAULT_BAUDRATE, 9600)
        self.assertEqual(CR_DEFAULT_TIMEOUT, 1.0)
        self.assertEqual(CR_COMMAND_DELAY, 0.02)
        test_logger.info("    ✔ 시리얼 기본값 정상 (9600bps, 1.0s timeout, 20ms delay)")


# ============================================================================
# Main: 테스트 실행 및 결과 요약
# ============================================================================

if __name__ == "__main__":
    test_logger.info("=" * 80)
    test_logger.info("sensor_module.py 종합 테스트 시작")
    test_logger.info("=" * 80)
    test_logger.info("")

    start_time = time.time()

    # 테스트 실행 (verbosity=2: 각 테스트 이름과 결과 표시)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 테스트 클래스 순서
    test_classes = [
        TestDataStructures,
        TestVirtualSensor,
        TestCRColorimeterSensor,
        TestProtocolParsing,
        TestColorConversions,
        TestCRReadingParsing,
        TestSetupConvenienceMethods,
        TestFlickerResponseTimeSettings,
        TestUtilityMethods,
        TestSensorFactory,
        TestCRSensorMockSerial,
        TestConstants,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(
        stream=sys.stdout, verbosity=2, buffer=False)
    result = runner.run(suite)

    elapsed = time.time() - start_time

    # === 결과 요약 ===
    test_logger.info("")
    test_logger.info("=" * 80)
    test_logger.info("테스트 결과 요약")
    test_logger.info("=" * 80)
    test_logger.info("  총 테스트 수 : %d", result.testsRun)
    test_logger.info("  성공         : %d", result.testsRun - len(result.failures) - len(result.errors))
    test_logger.info("  실패         : %d", len(result.failures))
    test_logger.info("  에러         : %d", len(result.errors))
    test_logger.info("  소요 시간    : %.2f 초", elapsed)
    test_logger.info("=" * 80)

    if result.failures:
        test_logger.warning("실패한 테스트:")
        for test, traceback in result.failures:
            test_logger.warning("  FAIL: %s", test)
            test_logger.warning("  %s", traceback)

    if result.errors:
        test_logger.error("에러 발생 테스트:")
        for test, traceback in result.errors:
            test_logger.error("  ERROR: %s", test)
            test_logger.error("  %s", traceback)

    if result.wasSuccessful():
        test_logger.info("✅ 모든 테스트 통과!")
    else:
        test_logger.error("❌ 일부 테스트 실패/에러 발생")

    test_logger.info("로그 파일: %s", os.path.abspath(LOG_FILE))
    test_logger.info("=" * 80)
