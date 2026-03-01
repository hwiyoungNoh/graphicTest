"""
Sensor Module for Color Calibration System
Colorimetry Research (CR) 색채계 센서 제어 및 가상 센서 구현

CR-100 / CR-250 / CR-300 시리즈 센서를 시리얼(COM) 포트를 통해 제어합니다.
프로토콜은 VisualCPP/CRRemote SDK 기반으로 구현되었습니다.

통신 프로토콜 요약:
  - 시리얼: 9600bps, 8-N-1, No Flow Control
  - 명령 형식:  "<command>\\r\\n"
  - 응답 형식:  "OK:<code>:<command>:<result>\\r\\n"
             또는 "ER:<code>:<command>:<description>\\r\\n"
  - 명령 체계:
      RC <param>           : Read Configuration (구성 읽기)
      RS <param>           : Read Setup (설정 읽기)
      SM <param> <value>   : Set Measurement (설정 변경)
      RM <param>           : Read Measurement (측정 결과 읽기)
      M                    : Measure (측정 시작)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from enum import IntEnum
import time
import threading
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Enumerations (CR Colorimeter Protocol)
# ============================================================================

class CRObserver(IntEnum):
    """CIE 관찰자 유형"""
    DEGREE_2 = 0
    DEGREE_10 = 1


class FlickerFilterType(IntEnum):
    """플리커 필터 유형"""
    NONE = 0
    LOWPASS = 1
    HIGHPASS = 2
    BANDPASS = 3
    BANDSTOP = 4


class FlickerFilterFamily(IntEnum):
    """플리커 필터 패밀리"""
    NONE = 0
    BUTTERWORTH = 1


class ResponseTimeFilterType(IntEnum):
    """응답 시간 필터 유형"""
    NONE = 0
    MOVING_WINDOW_AVERAGE = 1


class ResponseTimeMode(IntEnum):
    """응답 시간 모드"""
    AUTO = 0
    MANUAL = 1


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SensorReading:
    """센서 측정 결과 데이터 구조 (표준화된 출력)"""
    rgb: np.ndarray              # 측정된 RGB 값 (0-1 범위)
    xyz: np.ndarray              # XYZ 삼자극값
    cie_xy: Tuple[float, float]  # CIE 1931 xy 좌표
    luminance: float             # 휘도 (cd/m²)
    timestamp: float             # 측정 시간 (Unix timestamp)
    is_valid: bool = True        # 측정 유효성
    error_message: str = ""      # 오류 메시지


@dataclass
class CRModeItem:
    """CR 센서 모드/설정 항목"""
    id: int = 0
    name: str = ""


@dataclass
class CRAccessoryItem:
    """CR 센서 액세서리 항목"""
    id: int = 0
    name: str = ""
    type: str = ""
    matrices: List['CRMatrixItem'] = field(default_factory=list)


@dataclass
class CRMatrixItem:
    """CR 센서 매트릭스(캘리브레이션) 항목"""
    id: int = 0
    name: str = ""
    calibration: List[float] = field(default_factory=list)


@dataclass
class CRFilterItem:
    """CR 센서 필터 항목"""
    id: int = 0
    name: str = ""
    type: str = ""


@dataclass
class CRConfiguration:
    """CR 센서 전체 구성 정보 (RC 명령으로 다운로드)"""
    id: str = ""
    model: str = ""
    firmware: str = ""
    instrument_type: int = 0

    modes: List[CRModeItem] = field(default_factory=list)
    accessories: List[CRAccessoryItem] = field(default_factory=list)
    filters: List[CRFilterItem] = field(default_factory=list)
    apertures: List[CRModeItem] = field(default_factory=list)
    exposure_modes: List[CRModeItem] = field(default_factory=list)
    range_modes: List[CRModeItem] = field(default_factory=list)
    ranges: List[CRModeItem] = field(default_factory=list)
    speeds: List[CRModeItem] = field(default_factory=list)
    sync_modes: List[CRModeItem] = field(default_factory=list)
    matrix_modes: List[CRModeItem] = field(default_factory=list)
    user_calib_modes: List[CRModeItem] = field(default_factory=list)
    match_set: List[CRModeItem] = field(default_factory=list)

    min_exposure: float = 0.0
    max_exposure: float = 0.0
    min_sync_freq: float = 0.0
    max_sync_freq: float = 0.0
    min_exposure_x: int = 0
    max_exposure_x: int = 0
    min_sampling_rate: float = 0.0
    max_sampling_rate: float = 0.0


@dataclass
class CRSetup:
    """CR 센서 측정 설정 (RS/SM 명령으로 읽기/쓰기)"""
    mode_id: int = 0
    accessory_id: int = -1
    filter1_id: int = -1
    filter2_id: int = -1
    filter3_id: int = -1
    aperture_id: int = 0
    range_mode_id: int = 0
    range_id: int = 0
    speed_id: int = 0
    exposure_mode_id: int = 0
    exposure: float = 0.0
    max_auto_exposure: float = 0.0
    sync_mode_id: int = 0
    sync_freq: float = 60.0
    exposure_x: int = 0
    matrix_mode_id: int = 0
    user_calib_mode_id: int = 0
    matrix_id: int = -1
    match_id: int = -1
    sampling_rate: float = 0.0
    cmf: int = 0


@dataclass
class CRCIEData:
    """CIE 색채 데이터 (RM 명령으로 읽기)"""
    X: str = ""
    Y: str = ""
    Z: str = ""
    XYZ: str = ""
    xy: str = ""
    uv: str = ""
    upvp: str = ""
    CCT: str = ""


@dataclass
class CRSpectrumData:
    """스펙트럼 데이터"""
    starting_wavelength: float = 0.0
    ending_wavelength: float = 0.0
    delta: float = 0.0
    data: List[float] = field(default_factory=list)


@dataclass
class CRTemporalData:
    """시간축 데이터 (Temporal)"""
    sampling_rate: float = 0.0
    data: List[float] = field(default_factory=list)


@dataclass
class CRWarning:
    """CR 센서 경고"""
    code: int = 0
    description: str = ""


@dataclass
class CRReading:
    """CR 센서 측정 결과 전체 (RM 명령 세트로 다운로드)"""
    id: str = ""
    model: str = ""
    time: str = ""
    mode: str = ""
    accessory: str = ""
    filter: str = ""
    aperture: str = ""
    exposure_mode: str = ""
    exposure: str = ""
    max_auto_exposure: str = ""
    range_mode: str = ""
    range: str = ""
    speed: str = ""
    sync_mode: str = ""
    sync_freq: str = ""
    exposure_x: str = ""
    matrix_mode: str = ""
    user_calib_mode: str = ""
    matrix_id: str = ""
    match_id: str = ""
    cmf: str = ""
    yv: str = ""
    radiometric: str = ""

    cie: List[CRCIEData] = field(default_factory=lambda: [CRCIEData(), CRCIEData()])
    warnings: List[CRWarning] = field(default_factory=list)
    all_warnings: str = ""
    spectrum: CRSpectrumData = field(default_factory=CRSpectrumData)
    temporal: CRTemporalData = field(default_factory=CRTemporalData)


@dataclass
class FlickerSettings:
    """플리커 측정 설정"""
    filter_type: int = FlickerFilterType.NONE
    filter_family: int = FlickerFilterFamily.NONE
    order: int = 1
    frequency: float = 800.0
    bandwidth: float = 800.0
    max_search_frequency: float = 120.0


@dataclass
class ResponseTimeSettings:
    """응답 시간 측정 설정"""
    mode: int = ResponseTimeMode.AUTO
    filter_type: int = ResponseTimeFilterType.NONE
    average: int = 5
    clipping_enabled: bool = False
    clipping_lo: float = 0.1
    clipping_hi: float = 0.9
    noise_level: float = 0.05
    step_response_zone_lo: float = 0.1
    step_response_zone_hi: float = 0.9


# ============================================================================
# Constants (CR Protocol)
# ============================================================================

CR_NEW_LINE = "\r\n"
CR_RESPONSE_SEPARATOR = ":"
CR_RESULT_SEPARATOR = "),"
CR_RESULT_SPACE = " "
CR_RESPONSE_OK = "OK"
CR_RESPONSE_ERROR = "ER"

# Serial Port Defaults
CR_DEFAULT_BAUDRATE = 9600
CR_DEFAULT_PARITY = 'N'       # NoParity
CR_DEFAULT_DATABITS = 8
CR_DEFAULT_STOPBITS = 1       # OneStopBit
CR_DEFAULT_TIMEOUT = 1.0      # seconds
CR_COMMAND_DELAY = 0.02       # 20ms delay after sending command


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
        """센서 연결"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """센서 연결 해제"""
        pass

    @abstractmethod
    def read(self) -> SensorReading:
        """센서 값 읽기"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        pass


# ============================================================================
# CR Colorimeter Sensor (Real Sensor - Serial Communication)
# ============================================================================

class CRColorimeterSensor(SensorInterface):
    """
    Colorimetry Research 색채계 센서 제어 클래스

    CR-100 / CR-250 / CR-300 시리즈를 시리얼 포트를 통해 제어합니다.
    프로토콜은 CRRemote SDK (VisualCPP)를 기반으로 Python으로 재구현하였습니다.

    사용법:
        sensor = CRColorimeterSensor(port='COM3')
        sensor.connect()
        reading = sensor.read()
        sensor.disconnect()
    """

    def __init__(self, port: str = 'COM3', baudrate: int = CR_DEFAULT_BAUDRATE,
                 timeout: float = CR_DEFAULT_TIMEOUT):
        """
        Args:
            port: 시리얼 포트 이름 (예: 'COM3', 'COM4')
            baudrate: 통신 속도 (기본값: 9600)
            timeout: 읽기 타임아웃 (초)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None
        self._connected = False
        self._lock = threading.Lock()

        # 센서 상태
        self.configuration = CRConfiguration()
        self.setup = CRSetup()
        self.setup_modified = CRSetup()
        self.last_reading = CRReading()
        self.flicker_settings = FlickerSettings()
        self.response_time_settings = ResponseTimeSettings()

        self.measurement_count = 0

        logger.info("[CR Sensor] Initialized (Port: %s, Baud: %d)", port, baudrate)

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        센서 연결 (시리얼 포트 Open → 펌웨어 확인 → 구성/설정 다운로드)

        Returns:
            bool: 연결 성공 여부
        """
        try:
            import serial
        except ImportError:
            logger.error("[CR Sensor] pyserial 패키지가 필요합니다. "
                         "'pip install pyserial'로 설치하세요.")
            return False

        try:
            logger.info("[CR Sensor] Connecting to %s ...", self.port)
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                write_timeout=self.timeout,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )

            # 포트 버퍼 초기화
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            time.sleep(0.1)

            self._connected = True
            logger.info("[CR Sensor] Serial port opened: %s", self.port)

            # 펌웨어 버전 → 구성 → 설정 순서로 다운로드
            self._download_version()
            self._download_configuration()
            self._download_setup()

            logger.info(
                "[CR Sensor] Connected! Model: %s, FW: %s, ID: %s",
                self.configuration.model,
                self.configuration.firmware,
                self.configuration.id,
            )
            return True

        except Exception as e:
            logger.error("[CR Sensor] Connection failed: %s", e)
            self._connected = False
            if self._serial and self._serial.is_open:
                self._serial.close()
            self._serial = None
            return False

    def disconnect(self) -> bool:
        """센서 연결 해제"""
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
            self._connected = False
            self._serial = None
            logger.info("[CR Sensor] Disconnected from %s", self.port)
            return True
        except Exception as e:
            logger.error("[CR Sensor] Disconnect error: %s", e)
            return False

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return (self._connected
                and self._serial is not None
                and self._serial.is_open)

    # ------------------------------------------------------------------
    # Low-level Serial Communication
    # ------------------------------------------------------------------

    def _send_command(self, command: str, timeout: float = 1.0) -> str:
        """
        센서에 명령을 전송하고 응답을 수신합니다.

        Args:
            command: 전송할 명령 (예: "RC Firmware", "M")
            timeout: 응답 대기 타임아웃(초)
        Returns:
            str: 수신한 원본 응답 문자열
        """
        if not self.is_connected():
            raise ConnectionError("Sensor not connected")

        with self._lock:
            cmd_bytes = (command + CR_NEW_LINE).encode('ascii')
            self._serial.write(cmd_bytes)
            self._serial.flush()
            time.sleep(CR_COMMAND_DELAY)

            logger.debug("[CR Sensor] TX: %s", command)

            old_timeout = self._serial.timeout
            self._serial.timeout = timeout
            response = self._read_response(command)
            self._serial.timeout = old_timeout

            logger.debug("[CR Sensor] RX: %s", response.strip())
            return response

    def _read_response(self, sent_command: str = "") -> str:
        """
        시리얼 포트에서 응답 라인을 수신합니다.
        에코 프롬프트('>') 및 명령 에코는 무시합니다.
        """
        response_lines = []
        while True:
            line = self._serial.readline().decode('ascii', errors='replace').strip()
            if not line:
                break

            # 에코 프롬프트 무시
            if line.startswith('>'):
                logger.debug("[CR Sensor] Echo discarded: %s", line)
                continue

            # 전송 에코 무시
            if line == sent_command:
                logger.debug("[CR Sensor] Command echo discarded: %s", line)
                continue

            response_lines.append(line)

            # OK 또는 ER 로 시작하면 메인 응답 라인
            if line.startswith(CR_RESPONSE_OK) or line.startswith(CR_RESPONSE_ERROR):
                break

        return "\n".join(response_lines) if response_lines else ""

    def _read_multi_line_response(self, num_lines: int,
                                  timeout: float = 2.0) -> List[str]:
        """
        다중 라인 응답을 읽습니다 (리스트 응답용).

        Args:
            num_lines: 읽어야 할 추가 라인 수
            timeout: 라인 당 타임아웃
        """
        lines = []
        old_timeout = self._serial.timeout
        self._serial.timeout = timeout

        for _ in range(num_lines):
            line = self._serial.readline().decode('ascii', errors='replace').strip()
            if line:
                lines.append(line)

        self._serial.timeout = old_timeout
        return lines

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        응답 문자열을 파싱합니다.

        형식: "OK:<code>:<command>:<result>"
              "ER:<code>:<command>:<description>"
        """
        parts = response.split(CR_RESPONSE_SEPARATOR, 3)
        return {
            'type':    parts[0] if len(parts) > 0 else '',
            'code':    parts[1] if len(parts) > 1 else '',
            'command': parts[2] if len(parts) > 2 else '',
            'result':  parts[3] if len(parts) > 3 else '',
        }

    def _send_and_parse(self, command: str,
                        timeout: float = 5.0) -> Dict[str, str]:
        """명령을 전송하고 파싱된 응답을 반환합니다."""
        response = self._send_command(command, timeout=timeout)
        parsed = self._parse_response(response)

        if parsed['type'] == CR_RESPONSE_ERROR:
            raise RuntimeError(
                "CR Sensor Error [{}]: {} - {}".format(
                    parsed['code'], parsed['command'], parsed['result']))
        return parsed

    def _send_and_get_result(self, command: str,
                             timeout: float = 5.0) -> str:
        """명령을 전송하고 result 값만 반환합니다."""
        parsed = self._send_and_parse(command, timeout=timeout)
        return parsed.get('result', '')

    def _send_and_get_list(self, command: str,
                           timeout: float = 5.0) -> Tuple[int, List[str]]:
        """
        리스트 형태 응답을 반환하는 명령을 전송합니다.
        result → 항목 수, 이후 각 라인이 항목입니다.
        """
        parsed = self._send_and_parse(command, timeout=timeout)
        count = int(parsed['result'])
        items = self._read_multi_line_response(count, timeout=timeout) if count > 0 else []
        return count, items

    # ------------------------------------------------------------------
    # Configuration Download  (RC 명령 세트)
    # ------------------------------------------------------------------

    def _download_version(self):
        """펌웨어 버전 다운로드"""
        result = self._send_and_get_result("RC Firmware")
        self.configuration.firmware = result
        logger.info("[CR Sensor] Firmware: %s", result)

    def _firmware_version_number(self) -> float:
        """펌웨어 버전을 숫자로 반환"""
        try:
            return float(self.configuration.firmware)
        except ValueError:
            return 0.0

    def _download_configuration(self):
        """센서 구성 정보 전체 다운로드 (RC 명령 세트)"""

        self.configuration.id = self._send_and_get_result("RC ID")
        self.configuration.model = self._send_and_get_result("RC Model")

        version = self._firmware_version_number()
        if version >= 1.17:
            try:
                self.configuration.instrument_type = int(
                    self._send_and_get_result("RC InstrumentType"))
            except Exception:
                pass

        # --- 리스트 항목들 ---

        # Accessories  (항목 형식: "ID),Name),Type")
        count, items = self._send_and_get_list("RC Accessory")
        self.configuration.accessories = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 3:
                self.configuration.accessories.append(
                    CRAccessoryItem(id=int(parts[0]), name=parts[1], type=parts[2]))

        # Filters  ("ID),Name),Type")
        count, items = self._send_and_get_list("RC Filter")
        self.configuration.filters = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 3:
                self.configuration.filters.append(
                    CRFilterItem(id=int(parts[0]), name=parts[1], type=parts[2]))

        # Apertures  ("ID),Name")
        count, items = self._send_and_get_list("RC Aperture")
        self.configuration.apertures = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.apertures.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Modes
        count, items = self._send_and_get_list("RC Mode")
        self.configuration.modes = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.modes.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Exposure Modes
        count, items = self._send_and_get_list("RC ExposureMode")
        self.configuration.exposure_modes = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.exposure_modes.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Range Modes
        count, items = self._send_and_get_list("RC RangeMode")
        self.configuration.range_modes = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.range_modes.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Ranges
        count, items = self._send_and_get_list("RC Range")
        self.configuration.ranges = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.ranges.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Speeds
        count, items = self._send_and_get_list("RC Speed")
        self.configuration.speeds = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.speeds.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Sync Modes
        count, items = self._send_and_get_list("RC SyncMode")
        self.configuration.sync_modes = []
        for item in items:
            parts = item.split("),")
            if len(parts) >= 2:
                self.configuration.sync_modes.append(
                    CRModeItem(id=int(parts[0]), name=parts[1]))

        # Matrix Modes (deprecated but still supported)
        try:
            count, items = self._send_and_get_list("RC MatrixMode")
            self.configuration.matrix_modes = []
            for item in items:
                parts = item.split("),")
                if len(parts) >= 2:
                    self.configuration.matrix_modes.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception:
            pass

        # User Calibration Modes
        try:
            count, items = self._send_and_get_list("RC UserCalibMode")
            self.configuration.user_calib_modes = []
            for item in items:
                parts = item.split("),")
                if len(parts) >= 2:
                    self.configuration.user_calib_modes.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception:
            pass

        # Match Set
        try:
            count, items = self._send_and_get_list("RC Match")
            self.configuration.match_set = []
            for item in items:
                parts = item.split("),")
                if len(parts) >= 2:
                    self.configuration.match_set.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception:
            pass

        # --- 스칼라 값 ---

        # Exposure limits  ("value msec")
        for attr, cmd in [
            ('min_exposure', 'RC MinExposure'),
            ('max_exposure', 'RC MaxExposure'),
        ]:
            try:
                result = self._send_and_get_result(cmd)
                parts = result.split()
                setattr(self.configuration, attr,
                        float(parts[0]) if parts else 0.0)
            except Exception:
                pass

        # SyncFreq limits  ("value Hz")
        for attr, cmd in [
            ('min_sync_freq', 'RC MinSyncFreq'),
            ('max_sync_freq', 'RC MaxSyncFreq'),
        ]:
            try:
                result = self._send_and_get_result(cmd)
                parts = result.split()
                setattr(self.configuration, attr,
                        float(parts[0]) if parts else 0.0)
            except Exception:
                pass

        # ExposureX limits  (int)
        for attr, cmd in [
            ('min_exposure_x', 'RC MinExposureX'),
            ('max_exposure_x', 'RC MaxExposureX'),
        ]:
            try:
                setattr(self.configuration, attr,
                        int(self._send_and_get_result(cmd)))
            except Exception:
                pass

        # Sampling Rate limits  (float)
        for attr, cmd in [
            ('min_sampling_rate', 'RC MinSamplingRate'),
            ('max_sampling_rate', 'RC MaxSamplingRate'),
        ]:
            try:
                setattr(self.configuration, attr,
                        float(self._send_and_get_result(cmd)))
            except Exception:
                pass

        # 각 액세서리에 대한 Matrix 다운로드
        for acc in self.configuration.accessories:
            try:
                parsed = self._send_and_parse("RC Matrix {}".format(acc.id))
                result = parsed.get('result', '')
                result_parts = result.split("),")
                if len(result_parts) >= 2 and result_parts[0] != "None":
                    mat_count = int(result_parts[0])
                    mat_items = self._read_multi_line_response(mat_count)
                    acc.matrices = []
                    for mat_item in mat_items:
                        mp = mat_item.split("),")
                        if len(mp) >= 2:
                            acc.matrices.append(
                                CRMatrixItem(id=int(mp[0]), name=mp[1]))
            except Exception:
                pass

        logger.info(
            "[CR Sensor] Configuration downloaded: %d modes, %d accessories",
            len(self.configuration.modes),
            len(self.configuration.accessories),
        )

    # ------------------------------------------------------------------
    # Setup Download / Upload  (RS / SM 명령 세트)
    # ------------------------------------------------------------------

    def _download_setup(self):
        """현재 센서 설정 다운로드 (RS 명령 세트)"""
        try:
            # Accessory
            result = self._send_and_get_result("RS Accessory")
            self.setup.accessory_id = self._find_id_by_name(
                self.configuration.accessories, result)
            self.setup_modified.accessory_id = self.setup.accessory_id

            # Filter  ("Name1),Name2),Name3")
            result = self._send_and_get_result("RS Filter")
            names = result.split("),")
            ids = [self._find_filter_id_by_name(n) for n in names]
            self.setup.filter1_id = ids[0] if len(ids) > 0 else -1
            self.setup.filter2_id = ids[1] if len(ids) > 1 else -1
            self.setup.filter3_id = ids[2] if len(ids) > 2 else -1
            self.setup_modified.filter1_id = self.setup.filter1_id
            self.setup_modified.filter2_id = self.setup.filter2_id
            self.setup_modified.filter3_id = self.setup.filter3_id

            # 단순 name→id 매핑 항목들
            _simple_map = [
                ('aperture_id',       'RS Aperture',      self.configuration.apertures),
                ('mode_id',           'RS Mode',          self.configuration.modes),
                ('range_mode_id',     'RS RangeMode',     self.configuration.range_modes),
                ('range_id',          'RS Range',         self.configuration.ranges),
                ('speed_id',          'RS Speed',         self.configuration.speeds),
                ('exposure_mode_id',  'RS ExposureMode',  self.configuration.exposure_modes),
                ('sync_mode_id',      'RS SyncMode',      self.configuration.sync_modes),
            ]
            for attr, cmd, collection in _simple_map:
                try:
                    result = self._send_and_get_result(cmd)
                    val = self._find_id_by_name(collection, result)
                    setattr(self.setup, attr, val)
                    setattr(self.setup_modified, attr, val)
                except Exception:
                    pass

            # Exposure  "1.000 msec"
            result = self._send_and_get_result("RS Exposure")
            parts = result.split()
            if parts:
                self.setup.exposure = float(parts[0])
                self.setup_modified.exposure = self.setup.exposure

            # MaxAutoExposure
            result = self._send_and_get_result("RS MaxAutoExposure")
            parts = result.split()
            if parts:
                self.setup.max_auto_exposure = float(parts[0])
                self.setup_modified.max_auto_exposure = self.setup.max_auto_exposure

            # SyncFreq  "60.00 Hz"
            result = self._send_and_get_result("RS SyncFreq")
            parts = result.split()
            if parts:
                self.setup.sync_freq = float(parts[0])
                self.setup_modified.sync_freq = self.setup.sync_freq

            # ExposureX (int)
            result = self._send_and_get_result("RS ExposureX")
            self.setup.exposure_x = int(result)
            self.setup_modified.exposure_x = self.setup.exposure_x

            # MatrixMode (deprecated)
            try:
                result = self._send_and_get_result("RS MatrixMode")
                self.setup.matrix_mode_id = self._find_id_by_name(
                    self.configuration.matrix_modes, result)
                self.setup_modified.matrix_mode_id = self.setup.matrix_mode_id
            except Exception:
                pass

            # UserCalibMode
            try:
                result = self._send_and_get_result("RS UserCalibMode")
                self.setup.user_calib_mode_id = self._find_id_by_name(
                    self.configuration.user_calib_modes, result)
                self.setup_modified.user_calib_mode_id = self.setup.user_calib_mode_id
            except Exception:
                pass

            # Matrix (int or "None")
            try:
                result = self._send_and_get_result("RS Matrix")
                self.setup.matrix_id = -1 if result == "None" else int(result)
                self.setup_modified.matrix_id = self.setup.matrix_id
            except Exception:
                pass

            # Match (int or "None")
            try:
                result = self._send_and_get_result("RS Match")
                self.setup.match_id = -1 if result == "None" else int(result)
                self.setup_modified.match_id = self.setup.match_id
            except Exception:
                pass

            # SamplingRate (float)
            try:
                self.setup.sampling_rate = float(
                    self._send_and_get_result("RS SamplingRate"))
                self.setup_modified.sampling_rate = self.setup.sampling_rate
            except Exception:
                pass

            # CMF (int)
            try:
                self.setup.cmf = int(self._send_and_get_result("RS CMF"))
                self.setup_modified.cmf = self.setup.cmf
            except Exception:
                pass

            logger.info("[CR Sensor] Setup downloaded successfully")

        except Exception as e:
            logger.error("[CR Sensor] Setup download error: %s", e)

    def upload_setup(self) -> bool:
        """
        변경된 설정을 센서에 업로드합니다 (SM 명령 세트).
        setup_modified ↔ setup 을 비교하여 변경분만 전송합니다.
        """
        try:
            _int_fields = [
                ('accessory_id',       'SM Accessory {}'),
                ('filter1_id',         'SM Filter1 {}'),
                ('filter2_id',         'SM Filter2 {}'),
                ('filter3_id',         'SM Filter3 {}'),
                ('aperture_id',        'SM Aperture {}'),
                ('mode_id',            'SM Mode {}'),
                ('exposure_mode_id',   'SM ExposureMode {}'),
                ('range_mode_id',      'SM RangeMode {}'),
                ('range_id',           'SM Range {}'),
                ('speed_id',           'SM Speed {}'),
                ('sync_mode_id',       'SM SyncMode {}'),
                ('exposure_x',         'SM ExposureX {}'),
                ('user_calib_mode_id', 'SM UserCalibMode {}'),
                ('matrix_id',          'SM Matrix {}'),
                ('match_id',           'SM Match {}'),
                ('cmf',                'SM CMF {}'),
            ]
            _float_fields = [
                ('exposure',           'SM Exposure {}'),
                ('max_auto_exposure',  'SM MaxAutoExposure {}'),
                ('sync_freq',          'SM SyncFreq {}'),
                ('sampling_rate',      'SM SamplingRate {}'),
            ]

            for attr, fmt in _int_fields:
                old_val = getattr(self.setup, attr)
                new_val = getattr(self.setup_modified, attr)
                if new_val != old_val:
                    self._send_and_parse(fmt.format(new_val))
                    setattr(self.setup, attr, new_val)

            for attr, fmt in _float_fields:
                old_val = getattr(self.setup, attr)
                new_val = getattr(self.setup_modified, attr)
                if new_val != old_val:
                    self._send_and_parse(fmt.format(new_val))
                    setattr(self.setup, attr, new_val)

            logger.info("[CR Sensor] Setup uploaded successfully")
            return True

        except Exception as e:
            logger.error("[CR Sensor] Setup upload error: %s", e)
            return False

    # ------------------------------------------------------------------
    # Measurement  (M + RM 명령 세트)
    # ------------------------------------------------------------------

    def capture(self) -> bool:
        """
        측정 명령(M)을 전송합니다.
        upload_setup() 로 변경 설정 반영 후 측정을 시작합니다.
        """
        try:
            self.upload_setup()
            # M 명령은 센서가 측정을 마칠 때까지 시간이 걸릴 수 있음
            self._send_and_parse("M", timeout=30.0)
            return True
        except Exception as e:
            logger.error("[CR Sensor] Capture error: %s", e)
            return False

    def download_reading(self) -> CRReading:
        """
        최근 측정 결과를 센서에서 다운로드합니다 (RM 명령 세트).
        """
        reading = CRReading()
        try:
            # 기본 정보
            reading.id               = self._send_and_get_result("RM ID")
            reading.model            = self._send_and_get_result("RM Model")
            reading.time             = self._send_and_get_result("RM Time")
            reading.accessory        = self._send_and_get_result("RM Accessory")
            reading.filter           = self._send_and_get_result("RM Filter")
            reading.aperture         = self._send_and_get_result("RM Aperture")
            reading.mode             = self._send_and_get_result("RM Mode")
            reading.exposure_mode    = self._send_and_get_result("RM ExposureMode")
            reading.exposure         = self._send_and_get_result("RM Exposure")
            reading.max_auto_exposure = self._send_and_get_result("RM MaxAutoExposure")
            reading.range_mode       = self._send_and_get_result("RM RangeMode")
            reading.range            = self._send_and_get_result("RM Range")
            reading.speed            = self._send_and_get_result("RM Speed")
            reading.sync_mode        = self._send_and_get_result("RM SyncMode")
            reading.sync_freq        = self._send_and_get_result("RM SyncFreq")
            reading.exposure_x       = self._send_and_get_result("RM ExposureX")
            reading.user_calib_mode  = self._send_and_get_result("RM UserCalibMode")

            # Matrix / Match  ("None" or int)
            r = self._send_and_get_result("RM Matrix")
            reading.matrix_id = "-1" if r == "None" else r
            r = self._send_and_get_result("RM Match")
            reading.match_id = "-1" if r == "None" else r

            reading.cmf = self._send_and_get_result("RM CMF")

            # CIE 2° Observer
            cie2 = reading.cie[CRObserver.DEGREE_2]
            cie2.X    = self._send_and_get_result("RM X")
            cie2.Y    = self._send_and_get_result("RM Y")
            cie2.Z    = self._send_and_get_result("RM Z")
            cie2.XYZ  = self._send_and_get_result("RM XYZ")
            cie2.xy   = self._send_and_get_result("RM xy")
            cie2.uv   = self._send_and_get_result("RM uv")
            cie2.upvp = self._send_and_get_result("RM upvp")
            cie2.CCT  = self._send_and_get_result("RM CCT")

            # CIE 10° Observer
            cie10 = reading.cie[CRObserver.DEGREE_10]
            cie10.X   = self._send_and_get_result("RM X10")
            cie10.Y   = self._send_and_get_result("RM Y10")
            cie10.Z   = self._send_and_get_result("RM Z10")
            cie10.XYZ = self._send_and_get_result("RM XYZ10")
            cie10.xy  = self._send_and_get_result("RM xy10")

            # Warnings
            try:
                count, items = self._send_and_get_list("RM Warnings")
                for item in items:
                    parts = item.split("),")
                    if len(parts) >= 2:
                        reading.warnings.append(
                            CRWarning(code=int(parts[0]), description=parts[1]))
                        reading.all_warnings += item + "\r"
            except Exception:
                pass

            # Spectrum
            try:
                parsed = self._send_and_parse("RM Spectrum")
                result = parsed.get('result', '')
                sp = result.split("),")
                if len(sp) >= 4:
                    reading.spectrum.starting_wavelength = float(sp[0])
                    reading.spectrum.ending_wavelength   = float(sp[1])
                    reading.spectrum.delta               = float(sp[2])
                    n = int(sp[3])
                    data_lines = self._read_multi_line_response(n)
                    reading.spectrum.data = [float(x.strip()) for x in data_lines]
            except Exception as e:
                logger.debug("[CR Sensor] Spectrum read error: %s", e)

            time.sleep(0.15)  # SDK 호환 대기

            # Radiometric, Yv
            reading.radiometric = self._send_and_get_result("RM Radiometric")
            reading.yv          = self._send_and_get_result("RM Yv")

            time.sleep(0.15)

            # Temporal
            try:
                cmd = ("RM TemporalY"
                       if self._firmware_version_number() >= 1.19
                       else "RM Temporal")
                parsed = self._send_and_parse(cmd)
                result = parsed.get('result', '')
                tp = result.split("),")
                if len(tp) >= 2:
                    reading.temporal.sampling_rate = float(tp[0])
                    n = int(tp[1])
                    data_lines = self._read_multi_line_response(n)
                    reading.temporal.data = [float(x.strip()) for x in data_lines]
            except Exception as e:
                logger.debug("[CR Sensor] Temporal read error: %s", e)

        except Exception as e:
            logger.error("[CR Sensor] Download reading error: %s", e)

        self.last_reading = reading
        return reading

    def read(self) -> SensorReading:
        """
        측정 수행 후 결과를 SensorReading 으로 반환합니다.
        capture() → download_reading() → SensorReading 변환
        """
        if not self.is_connected():
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0), luminance=0.0,
                timestamp=time.time(),
                is_valid=False, error_message="Sensor not connected")

        try:
            logger.info("[CR Sensor] Measuring... (#%d)",
                        self.measurement_count + 1)

            if not self.capture():
                return SensorReading(
                    rgb=np.array([0.0, 0.0, 0.0]),
                    xyz=np.array([0.0, 0.0, 0.0]),
                    cie_xy=(0.0, 0.0), luminance=0.0,
                    timestamp=time.time(),
                    is_valid=False, error_message="Capture failed")

            cr_reading = self.download_reading()

            xyz       = self._parse_xyz(cr_reading)
            cie_xy    = self._parse_xy(cr_reading)
            luminance = self._parse_luminance(cr_reading)
            rgb       = self._xyz_to_rgb(xyz)

            self.measurement_count += 1

            logger.info("[CR Sensor] XYZ=[%.4f, %.4f, %.4f]  "
                        "xy=(%.4f, %.4f)  Y=%.2f cd/m²",
                        xyz[0], xyz[1], xyz[2],
                        cie_xy[0], cie_xy[1], luminance)

            return SensorReading(
                rgb=rgb, xyz=xyz, cie_xy=cie_xy,
                luminance=luminance,
                timestamp=time.time(),
                is_valid=True, error_message="")

        except Exception as e:
            logger.error("[CR Sensor] Read error: %s", e)
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0), luminance=0.0,
                timestamp=time.time(),
                is_valid=False,
                error_message="Read error: {}".format(str(e)))

    # ------------------------------------------------------------------
    # Setup Convenience Methods
    # ------------------------------------------------------------------

    def set_mode(self, index: int):
        """측정 모드 설정"""
        if 0 <= index < len(self.configuration.modes):
            self.setup_modified.mode_id = self.configuration.modes[index].id

    def set_exposure_mode(self, index: int):
        """노출 모드 설정"""
        if 0 <= index < len(self.configuration.exposure_modes):
            self.setup_modified.exposure_mode_id = \
                self.configuration.exposure_modes[index].id

    def set_exposure(self, value: float):
        """노출 시간 설정 (msec)"""
        if self.configuration.min_exposure <= value <= self.configuration.max_exposure:
            self.setup_modified.exposure = value

    def set_speed(self, index: int):
        """측정 속도 설정"""
        if 0 <= index < len(self.configuration.speeds):
            self.setup_modified.speed_id = self.configuration.speeds[index].id

    def set_aperture(self, index: int):
        """조리개 설정"""
        if 0 <= index < len(self.configuration.apertures):
            self.setup_modified.aperture_id = \
                self.configuration.apertures[index].id

    def set_sync_mode(self, index: int):
        """동기화 모드 설정"""
        if 0 <= index < len(self.configuration.sync_modes):
            self.setup_modified.sync_mode_id = \
                self.configuration.sync_modes[index].id

    def set_sync_freq(self, value: float):
        """동기화 주파수 설정 (Hz)"""
        if self.configuration.min_sync_freq <= value <= self.configuration.max_sync_freq:
            self.setup_modified.sync_freq = value

    def set_sampling_rate(self, value: float):
        """샘플링 레이트 설정 (Hz)"""
        if self.configuration.min_sampling_rate <= value <= self.configuration.max_sampling_rate:
            self.setup_modified.sampling_rate = value

    def set_range_mode(self, index: int):
        """범위 모드 설정"""
        if 0 <= index < len(self.configuration.range_modes):
            self.setup_modified.range_mode_id = \
                self.configuration.range_modes[index].id

    def set_accessory(self, index: int):
        """액세서리 설정"""
        if 0 <= index < len(self.configuration.accessories):
            self.setup_modified.accessory_id = \
                self.configuration.accessories[index].id

    def set_cmf(self, value: int):
        """CMF (Color Matching Function) 설정"""
        self.setup_modified.cmf = value

    # ------------------------------------------------------------------
    # Flicker Analysis Settings
    # ------------------------------------------------------------------

    def set_flicker_filter_type(self, ftype: FlickerFilterType):
        self.flicker_settings.filter_type = ftype

    def set_flicker_filter_order(self, order: int):
        if 1 <= order <= 50:
            self.flicker_settings.order = order

    def set_flicker_filter_frequency(self, freq: float):
        if 10.0 <= freq <= 800.0:
            self.flicker_settings.frequency = freq

    def set_flicker_filter_bandwidth(self, bw: float):
        if 10.0 <= bw <= 800.0:
            self.flicker_settings.bandwidth = bw

    def set_flicker_max_search_frequency(self, freq: float):
        if 10.0 <= freq <= 800.0:
            self.flicker_settings.max_search_frequency = freq

    # ------------------------------------------------------------------
    # Response Time Analysis Settings
    # ------------------------------------------------------------------

    def set_response_time_mode(self, mode: ResponseTimeMode):
        self.response_time_settings.mode = mode

    def set_response_time_filter_type(self, ftype: ResponseTimeFilterType):
        self.response_time_settings.filter_type = ftype

    def set_response_time_average(self, avg: int):
        if 1 <= avg <= 20:
            self.response_time_settings.average = avg

    def set_response_time_clipping(self, enabled: bool,
                                   lo: float = 0.1, hi: float = 0.9):
        self.response_time_settings.clipping_enabled = enabled
        if 0.0 <= lo <= 1.0:
            self.response_time_settings.clipping_lo = lo
        if 0.0 <= hi <= 1.0:
            self.response_time_settings.clipping_hi = hi

    def set_response_time_noise_level(self, level: float):
        if 0.0 <= level <= 0.9:
            self.response_time_settings.noise_level = level

    def set_response_time_step_zone(self, lo: float = 0.1, hi: float = 0.9):
        if 0.0 <= lo <= 1.0:
            self.response_time_settings.step_response_zone_lo = lo
        if 0.0 <= hi <= 1.0:
            self.response_time_settings.step_response_zone_hi = hi

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _find_id_by_name(items, name: str) -> int:
        """이름으로 ID 검색"""
        if name == "None":
            return -1
        for item in items:
            if getattr(item, 'name', '') == name:
                return getattr(item, 'id', -1)
        return -1

    def _find_filter_id_by_name(self, name: str) -> int:
        return self._find_id_by_name(self.configuration.filters, name)

    @staticmethod
    def _parse_xyz(reading: CRReading,
                   observer: int = CRObserver.DEGREE_2) -> np.ndarray:
        """CRReading → XYZ ndarray"""
        try:
            cie = reading.cie[observer]
            x = float(cie.X) if cie.X else 0.0
            y = float(cie.Y) if cie.Y else 0.0
            z = float(cie.Z) if cie.Z else 0.0
            return np.array([x, y, z])
        except (ValueError, IndexError):
            return np.array([0.0, 0.0, 0.0])

    @staticmethod
    def _parse_xy(reading: CRReading,
                  observer: int = CRObserver.DEGREE_2) -> Tuple[float, float]:
        """CRReading → CIE xy tuple"""
        try:
            xy_str = reading.cie[observer].xy
            if xy_str:
                parts = xy_str.replace("),", ",").split(",")
                if len(parts) >= 2:
                    return (float(parts[0].strip()), float(parts[1].strip()))
            # fallback: XYZ → xy
            xyz = CRColorimeterSensor._parse_xyz(reading, observer)
            return CRColorimeterSensor._xyz_to_xy_static(xyz)
        except (ValueError, IndexError):
            return (0.3127, 0.3290)

    @staticmethod
    def _parse_luminance(reading: CRReading,
                         observer: int = CRObserver.DEGREE_2) -> float:
        """CRReading → 휘도 (Y 값)"""
        try:
            return float(reading.cie[observer].Y) if reading.cie[observer].Y else 0.0
        except ValueError:
            return 0.0

    @staticmethod
    def _xyz_to_xy_static(xyz: np.ndarray) -> Tuple[float, float]:
        """XYZ → CIE xy 변환"""
        s = xyz[0] + xyz[1] + xyz[2]
        if s < 1e-10:
            return (0.3127, 0.3290)   # D65 white point
        return (xyz[0] / s, xyz[1] / s)

    @staticmethod
    def _xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
        """XYZ → sRGB 변환 (BT.709 역행렬 + sRGB gamma)"""
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ])
        rgb_linear = M_inv @ xyz
        rgb = np.where(
            rgb_linear <= 0.0031308,
            12.92 * rgb_linear,
            1.055 * np.power(np.maximum(rgb_linear, 0), 1.0 / 2.4) - 0.055,
        )
        return np.clip(rgb, 0, 1)

    def get_measurement_count(self) -> int:
        return self.measurement_count

    def reset_measurement_count(self):
        self.measurement_count = 0

    def get_device_info(self) -> Dict[str, str]:
        """센서 기기 정보 반환"""
        return {
            'id':              self.configuration.id,
            'model':           self.configuration.model,
            'firmware':        self.configuration.firmware,
            'instrument_type': str(self.configuration.instrument_type),
            'port':            self.port,
            'modes':           str(len(self.configuration.modes)),
            'accessories':     str(len(self.configuration.accessories)),
        }

    def get_reading_value(self, data_type: str) -> str:
        """
        특정 측정 데이터를 직접 읽습니다.

        Args:
            data_type: "X", "Y", "Z", "xy", "CCT", "Spectrum" 등
        """
        return self._send_and_get_result("RM {}".format(data_type))

    def get_spectrum(self) -> CRSpectrumData:
        return self.last_reading.spectrum

    def get_temporal(self) -> CRTemporalData:
        return self.last_reading.temporal

    @staticmethod
    def list_available_ports() -> List[str]:
        """사용 가능한 시리얼 포트 목록 (포트 이름만)"""
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except ImportError:
            logger.warning("pyserial 패키지 필요 — pip install pyserial")
            return []

    @staticmethod
    def scan_ports_detailed() -> List[Dict[str, str]]:
        """
        사용 가능한 시리얼 포트 상세 정보 목록

        Returns:
            List[Dict]: 각 포트의 상세 정보
                - device: 포트 이름 (예: 'COM3')
                - description: 포트 설명
                - hwid: 하드웨어 ID
                - manufacturer: 제조사
                - vid_pid: VID:PID (USB 장치)
        """
        try:
            from serial.tools import list_ports
            result = []
            for p in list_ports.comports():
                vid_pid = ""
                if p.vid is not None and p.pid is not None:
                    vid_pid = "{:04X}:{:04X}".format(p.vid, p.pid)
                result.append({
                    'device': p.device,
                    'description': p.description or p.device,
                    'hwid': p.hwid or '',
                    'manufacturer': p.manufacturer or '',
                    'vid_pid': vid_pid,
                })
            # COM 번호 기준 정렬
            result.sort(key=lambda x: x['device'])
            return result
        except ImportError:
            logger.warning("pyserial 패키지 필요 — pip install pyserial")
            return []


# ============================================================================
# Virtual Sensor (Simulation)
# ============================================================================

class VirtualSensor(SensorInterface):
    """
    가상 센서 (시뮬레이션용)
    랜덤한 RGB 값을 생성하여 실제 센서 동작을 시뮬레이션
    """

    def __init__(self, noise_level: float = 0.02):
        self.connected = False
        self.noise_level = noise_level
        self.measurement_count = 0
        print("[Virtual Sensor] Initialized (Noise Level: {:.1f}%)"
              .format(noise_level * 100))

    def connect(self) -> bool:
        print("[Virtual Sensor] Connecting...")
        time.sleep(0.1)
        self.connected = True
        print("[Virtual Sensor] Connected successfully!")
        return True

    def disconnect(self) -> bool:
        print("[Virtual Sensor] Disconnecting...")
        self.connected = False
        print("[Virtual Sensor] Disconnected.")
        return True

    def is_connected(self) -> bool:
        return self.connected

    # ---- internal helpers ----

    def _generate_random_rgb(self) -> np.ndarray:
        rgb = np.random.random(3)
        noise = np.random.normal(0, self.noise_level, 3)
        return np.clip(rgb + noise, 0, 1)

    @staticmethod
    def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        return M @ np.power(rgb, 2.2)

    @staticmethod
    def _xyz_to_xy(xyz: np.ndarray) -> Tuple[float, float]:
        s = xyz[0] + xyz[1] + xyz[2]
        if s < 1e-10:
            return (0.3127, 0.3290)
        return (xyz[0] / s, xyz[1] / s)

    # ---- SensorInterface ----

    def read(self) -> SensorReading:
        if not self.connected:
            return SensorReading(
                rgb=np.array([0.0, 0.0, 0.0]),
                xyz=np.array([0.0, 0.0, 0.0]),
                cie_xy=(0.0, 0.0), luminance=0.0,
                timestamp=time.time(),
                is_valid=False, error_message="Sensor not connected")

        print("[Virtual Sensor] Reading... (Measurement #{})"
              .format(self.measurement_count + 1))
        time.sleep(0.2)

        measured_rgb = self._generate_random_rgb()
        xyz = self._rgb_to_xyz(measured_rgb)
        cie_xy = self._xyz_to_xy(xyz)
        luminance = xyz[1] * np.random.uniform(80, 120)

        self.measurement_count += 1

        print("[Virtual Sensor] RGB: R={:.3f}, G={:.3f}, B={:.3f}"
              .format(*measured_rgb))
        print("[Virtual Sensor] CIE xy: x={:.4f}, y={:.4f}"
              .format(*cie_xy))
        print("[Virtual Sensor] Luminance: {:.2f} cd/m²"
              .format(luminance))

        return SensorReading(
            rgb=measured_rgb, xyz=xyz, cie_xy=cie_xy,
            luminance=luminance,
            timestamp=time.time(),
            is_valid=True, error_message="")

    def get_measurement_count(self) -> int:
        return self.measurement_count

    def reset_measurement_count(self):
        self.measurement_count = 0
        print("[Virtual Sensor] Measurement count reset")


# ============================================================================
# Sensor Factory
# ============================================================================

def create_sensor(sensor_type: str = 'virtual', **kwargs) -> SensorInterface:
    """
    센서 인스턴스를 생성하는 팩토리 함수

    Args:
        sensor_type:
            'virtual'  — 가상 센서 (시뮬레이션)
            'cr'       — CR 색채계 (시리얼 연결)
        **kwargs:
            virtual : noise_level (float)
            cr      : port (str), baudrate (int), timeout (float)

    Returns:
        SensorInterface

    Examples:
        sensor = create_sensor('virtual', noise_level=0.02)
        sensor = create_sensor('cr', port='COM3')
    """
    if sensor_type == 'virtual':
        return VirtualSensor(noise_level=kwargs.get('noise_level', 0.02))
    elif sensor_type in ('cr', 'colorimeter', 'cr100', 'cr250', 'cr300'):
        return CRColorimeterSensor(
            port=kwargs.get('port', 'COM3'),
            baudrate=kwargs.get('baudrate', CR_DEFAULT_BAUDRATE),
            timeout=kwargs.get('timeout', CR_DEFAULT_TIMEOUT),
        )
    else:
        raise ValueError(
            "Unknown sensor type: '{}'. Use 'virtual' or 'cr'.".format(sensor_type))


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    print("=" * 80)
    print("Sensor Module Test")
    print("=" * 80)

    # === 1. 가상 센서 테스트 ===
    print("\n--- Virtual Sensor Test ---")
    sensor = create_sensor('virtual', noise_level=0.02)
    sensor.connect()

    for i in range(3):
        print("\nMeasurement #{}:".format(i + 1))
        print("-" * 60)
        reading = sensor.read()
        if reading.is_valid:
            print("  RGB: [{:.4f}, {:.4f}, {:.4f}]".format(*reading.rgb))
            print("  XYZ: [{:.4f}, {:.4f}, {:.4f}]".format(*reading.xyz))
            print("  CIE xy: ({:.4f}, {:.4f})".format(*reading.cie_xy))
            print("  Luminance: {:.2f} cd/m²".format(reading.luminance))
        print("-" * 60)

    print("\nTotal measurements:", sensor.get_measurement_count())
    sensor.disconnect()

    # === 2. CR 센서 사용 가능 포트 확인 ===
    print("\n--- Available Serial Ports ---")
    ports = CRColorimeterSensor.list_available_ports()
    if ports:
        for p in ports:
            print("  Found: {}".format(p))
    else:
        print("  No serial ports found (pyserial required)")

    # === 3. CR 센서 사용 예시 (실제 센서 연결 시 주석 해제) ===
    # print("\n--- CR Colorimeter Sensor Test ---")
    # cr_sensor = create_sensor('cr', port='COM3')
    # if cr_sensor.connect():
    #     info = cr_sensor.get_device_info()
    #     print("Device Info:", info)
    #
    #     # 설정 변경 예시
    #     # cr_sensor.set_exposure(10.0)
    #     # cr_sensor.set_sync_freq(60.0)
    #
    #     reading = cr_sensor.read()
    #     if reading.is_valid:
    #         print("XYZ:", reading.xyz)
    #         print("CIE xy:", reading.cie_xy)
    #         print("Luminance:", reading.luminance, "cd/m²")
    #
    #     cr_sensor.disconnect()

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
