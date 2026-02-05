"""
Sensor Module for Color Calibration System
센서 인터페이스 및 가상 센서 구현
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import time

# ============================================================================
# Sensor Reading Data Structure
# ============================================================================

@dataclass
class SensorReading:
    """센서 측정 결과 데이터 구조"""
    rgb: np.ndarray              # 측정된 RGB 값 (0-1 범위)
    xyz: np.ndarray              # XYZ 삼자극값
    cie_xy: Tuple[float, float]  # CIE 1931 xy 좌표
    luminance: float             # 휘도 (cd/m2)
    timestamp: float             # 측정 시간 (Unix timestamp)
    is_valid: bool = True        # 측정 유효성
    error_message: str = ""      # 오류 메시지

# ============================================================================
# Sensor Interface (Abstract Base Class)
# ============================================================================

class SensorInterface(ABC):
    """
    센서 인터페이스 추상 클래스
    실제 센서 구현 시 이 클래스를 상속하여 구현
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        센서 연결
        Returns:
            bool: 연결 성공 여부
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        센서 연결 해제
        Returns:
            bool: 연결 해제 성공 여부
        """
        pass

    @abstractmethod
    def read(self) -> SensorReading:
        """
        센서 값 읽기
        Returns:
            SensorReading: 측정 결과
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        연결 상태 확인
        Returns:
            bool: 연결 상태
        """
        pass

# ============================================================================
# Virtual Sensor (Simulation)
# ============================================================================

class VirtualSensor(SensorInterface):
    """
    가상 센서 (시뮬레이션용)
    랜덤한 RGB 값을 생성하여 실제 센서 동작을 시뮬레이션
    """

    def __init__(self, noise_level: float = 0.02):
        """
        Args:
            noise_level (float): 노이즈 레벨 (0-1 범위, 기본값 2%)
        """
        self.connected = False
        self.noise_level = noise_level
        self.measurement_count = 0

        print("[Virtual Sensor] Initialized (Noise Level: {:.1f}%)".format(noise_level * 100))

    def connect(self) -> bool:
        """가상 센서 연결 시뮬레이션"""
        print("[Virtual Sensor] Connecting...")
        time.sleep(0.1)  # 연결 지연 시뮬레이션
        self.connected = True
        print("[Virtual Sensor] Connected successfully!")
        return True

    def disconnect(self) -> bool:
        """가상 센서 연결 해제"""
        print("[Virtual Sensor] Disconnecting...")
        self.connected = False
        print("[Virtual Sensor] Disconnected.")
        return True

    def is_connected(self) -> bool:
        """연결 상태 반환"""
        return self.connected

    def _generate_random_rgb(self) -> np.ndarray:
        """
        랜덤한 RGB 값 생성
        Returns:
            np.ndarray: 랜덤 RGB 값 (0-1 범위)
        """
        # 완전 랜덤 RGB 생성
        rgb = np.random.random(3)

        # 노이즈 추가
        noise = np.random.normal(0, self.noise_level, 3)
        rgb_with_noise = rgb + noise

        # 0-1 범위로 클리핑
        rgb_with_noise = np.clip(rgb_with_noise, 0, 1)

        return rgb_with_noise

    def _rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """
        RGB를 XYZ로 변환 (BT.709 기준)
        Args:
            rgb (np.ndarray): RGB 값 (0-1 범위)
        Returns:
            np.ndarray: XYZ 값
        """
        # BT.709 / sRGB RGB to XYZ 변환 행렬
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

        # Gamma correction (simplified sRGB)
        rgb_linear = np.power(rgb, 2.2)

        # XYZ 변환
        xyz = M @ rgb_linear

        return xyz

    def _xyz_to_xy(self, xyz: np.ndarray) -> Tuple[float, float]:
        """
        XYZ를 CIE xy로 변환
        Args:
            xyz (np.ndarray): XYZ 값
        Returns:
            Tuple[float, float]: CIE xy 좌표
        """
        X, Y, Z = xyz
        sum_xyz = X + Y + Z

        if sum_xyz < 1e-10:
            return (0.3127, 0.3290)  # D65 white point

        x = X / sum_xyz
        y = Y / sum_xyz

        return (x, y)

    def read(self) -> SensorReading:
        """
        센서 값 읽기 (랜덤 RGB 생성)
        Returns:
            SensorReading: 측정 결과
        """
        if not self.connected:
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0),
                luminance=0.0,
                timestamp=time.time(),
                is_valid=False,
                error_message="Sensor not connected"
            )

        print("[Virtual Sensor] Reading... (Measurement #{})".format(self.measurement_count + 1))
        time.sleep(0.2)  # 측정 시간 시뮬레이션

        # 랜덤 RGB 생성
        measured_rgb = self._generate_random_rgb()

        # XYZ 변환
        xyz = self._rgb_to_xyz(measured_rgb)

        # CIE xy 변환
        cie_xy = self._xyz_to_xy(xyz)

        # 휘도 계산 (Y 값 기준, 랜덤 스케일 적용)
        luminance = xyz[1] * np.random.uniform(80, 120)  # 80-120 cd/m2 범위

        self.measurement_count += 1

        print("[Virtual Sensor] Measured RGB: R={:.3f}, G={:.3f}, B={:.3f}".format(
            measured_rgb[0], measured_rgb[1], measured_rgb[2]))
        print("[Virtual Sensor] CIE xy: x={:.4f}, y={:.4f}".format(cie_xy[0], cie_xy[1]))
        print("[Virtual Sensor] Luminance: {:.2f} cd/m2".format(luminance))

        return SensorReading(
            rgb=measured_rgb,
            xyz=xyz,
            cie_xy=cie_xy,
            luminance=luminance,
            timestamp=time.time(),
            is_valid=True,
            error_message=""
        )

    def get_measurement_count(self) -> int:
        """총 측정 횟수 반환"""
        return self.measurement_count

    def reset_measurement_count(self):
        """측정 횟수 초기화"""
        self.measurement_count = 0
        print("[Virtual Sensor] Measurement count reset")

# ============================================================================
# Example: Real Sensor Implementation Template
# ============================================================================

class RealColorSensor(SensorInterface):
    """
    실제 센서 구현 예시 (템플릿)
    시리얼 통신 또는 USB 통신을 통해 실제 센서와 연결
    """

    def __init__(self, port: str = 'COM3', baudrate: int = 9600):
        """
        Args:
            port (str): 시리얼 포트 (예: 'COM3', '/dev/ttyUSB0')
            baudrate (int): 통신 속도
        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False

    def connect(self) -> bool:
        """
        실제 센서 연결
        TODO: 실제 시리얼 통신 구현
        """
        try:
            # import serial  # pyserial 필요
            # self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            # 센서 초기화 명령 전송
            # self.serial.write(b'INIT\n')
            # response = self.serial.readline()

            self.connected = True
            print("[Real Sensor] Connected to {}".format(self.port))
            return True
        except Exception as e:
            print("[Real Sensor] Connection failed: {}".format(e))
            return False

    def disconnect(self) -> bool:
        """실제 센서 연결 해제"""
        try:
            if self.serial:
                # self.serial.close()
                pass
            self.connected = False
            print("[Real Sensor] Disconnected")
            return True
        except Exception as e:
            print("[Real Sensor] Disconnection failed: {}".format(e))
            return False

    def is_connected(self) -> bool:
        """연결 상태"""
        return self.connected

    def read(self) -> SensorReading:
        """
        실제 센서에서 값 읽기
        TODO: 실제 센서 프로토콜에 맞게 구현
        """
        if not self.connected:
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0),
                luminance=0.0,
                timestamp=time.time(),
                is_valid=False,
                error_message="Sensor not connected"
            )

        try:
            # 센서에 측정 명령 전송
            # self.serial.write(b'MEASURE\n')

            # 센서로부터 데이터 수신
            # raw_data = self.serial.readline().decode('utf-8')

            # 데이터 파싱 (예시)
            # parts = raw_data.strip().split(',')
            # r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
            # x, y = float(parts[3]), float(parts[4])
            # lum = float(parts[5])

            # 실제 구현 필요
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0),
                luminance=0.0,
                timestamp=time.time(),
                is_valid=True,
                error_message=""
            )
        except Exception as e:
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0),
                luminance=0.0,
                timestamp=time.time(),
                is_valid=False,
                error_message="Read error: {}".format(str(e))
            )

# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Sensor Module Test")
    print("="*80)
    print()

    # 가상 센서 테스트
    sensor = VirtualSensor(noise_level=0.02)

    # 연결
    sensor.connect()
    print()

    # 3회 측정
    for i in range(3):
        print("\nMeasurement #{}:".format(i+1))
        print("-" * 60)
        reading = sensor.read()

        if reading.is_valid:
            print("  RGB: [{:.4f}, {:.4f}, {:.4f}]".format(
                reading.rgb[0], reading.rgb[1], reading.rgb[2]))
            print("  CIE xy: ({:.4f}, {:.4f})".format(
                reading.cie_xy[0], reading.cie_xy[1]))
            print("  Luminance: {:.2f} cd/m2".format(reading.luminance))
            print("  Timestamp: {:.3f}".format(reading.timestamp))
        else:
            print("  Error: {}".format(reading.error_message))
        print("-" * 60)

    print()
    print("Total measurements: {}".format(sensor.get_measurement_count()))

    # 연결 해제
    print()
    sensor.disconnect()

    print()
    print("="*80)
