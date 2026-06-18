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
import re
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
    speed_id: int = 1  # 기본값: 1 = 2x Fast (가장 빠른 속도)
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

            # ── Liveness probe (잘못된 COM 포트 빠른 실패) ──────────────
            # 잘못된/유휴 COM 포트도 open 은 성공하지만 응답이 없다. 이 체크가
            # 없으면 connect() 가 RC/RS 설정 명령 ~15개를 각 5초 타임아웃으로
            # 줄줄이 보내며 수십 초 멈췄다가 그래도 True 를 반환한다. RC Firmware
            # 를 짧은 타임아웃으로 두어 번 질의해 무응답이면 즉시 실패 반환 →
            # UI 가 실패 팝업을 띄우고 사용자가 다른 포트를 고를 수 있게 한다.
            firmware = self._probe_firmware(attempts=2, timeout=2.0)
            if firmware is None:
                logger.error("[CR Sensor] %s 가 열렸지만 센서가 응답하지 않습니다 "
                             "(잘못된 COM 포트?). connect 중단.", self.port)
                self._connected = False
                try:
                    if self._serial and self._serial.is_open:
                        self._serial.close()
                finally:
                    self._serial = None
                return False
            self.configuration.firmware = firmware
            logger.info("[CR Sensor] Firmware: %s", firmware)

            try:
                self._download_configuration()
            except Exception as e:
                logger.warning("[CR Sensor] Configuration download failed: %s (continuing...)", e)
                # 최소 구성 정보 설정
                if not self.configuration.model:
                    self.configuration.model = "Unknown Model"
                if not self.configuration.id:
                    self.configuration.id = "Unknown ID"

            try:
                self._download_setup()
            except Exception as e:
                logger.warning("[CR Sensor] Setup download failed: %s (continuing...)", e)

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

            # 상세 송신 로그
            logger.info("[CR Sensor] >>> TX: %s (timeout=%.1fs)", command, timeout)

            old_timeout = self._serial.timeout
            self._serial.timeout = timeout
            response = self._read_response(command)
            self._serial.timeout = old_timeout

            # 상세 수신 로그
            if response:
                logger.info("[CR Sensor] <<< RX: %s", response.strip()[:100])
            else:
                logger.warning("[CR Sensor] <<< RX: (EMPTY - timeout?)")
            
            return response

    def _read_response(self, sent_command: str = "") -> str:
        """
        시리얼 포트에서 응답 라인을 수신합니다.
        에코 프롬프트('>') 및 명령 에코는 무시합니다.
        """
        response_lines = []
        blank_retries = 0
        while True:
            line = self._serial.readline().decode('ascii', errors='replace').strip()
            if not line:
                # OK/ER 헤더를 받기 전의 순간 빈 줄(시리얼 청킹/지연)은 곧바로
                # 끝으로 간주하지 않고 잠깐 더 기다린다 — 멀티라인 리스트 응답
                # (RC Speed/RC ExposureMode)의 헤더를 놓치지 않기 위함.
                if not response_lines and blank_retries < 3:
                    blank_retries += 1
                    continue
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
                                  timeout: float = 5.0) -> List[str]:
        """
        다중 라인 응답을 읽습니다 (리스트 응답용).

        SDK(ProcessResponses)는 헤더+N개 항목 라인이 모두 버퍼에 들어올 때까지
        기다렸다 파싱한다. 동기 포트에서도 이를 흉내 내어, num_lines 개의
        '비어있지 않은' 항목 라인을 모을 때까지(또는 전체 deadline 까지) 계속
        읽는다 — 한 줄이 타임아웃돼도 드롭/포기하지 않고 재시도(이전 구현은
        타임아웃 라인을 조용히 버려 목록이 비거나 짧아졌음).
        """
        lines: List[str] = []
        old_timeout = self._serial.timeout
        self._serial.timeout = 0.5  # 짧은 per-readline → deadline 기반 루프
        end = time.time() + max(float(timeout), 3.0)
        try:
            while len(lines) < num_lines and time.time() < end:
                line = self._serial.readline().decode('ascii', errors='replace').strip()
                if not line:
                    continue  # 타임아웃 — deadline 까지 항목 라인을 계속 기다림
                if line.startswith('>'):
                    continue  # 에코 프롬프트 무시
                lines.append(line)
        finally:
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
        """
        명령을 전송하고 result 값만 반환합니다.
        
        Note: RM 명령은 측정 데이터 읽기이므로 충분한 타임아웃 필요
        C++ SDK에서는 각 명령마다 타임아웃 설정 가능
        """
        parsed = self._send_and_parse(command, timeout=timeout)
        return parsed.get('result', '')

    def _send_and_get_list(self, command: str,
                           timeout: float = 5.0) -> Tuple[int, List[str]]:
        """
        리스트 형태 응답을 반환하는 명령을 전송합니다.
        result → 항목 수, 이후 각 라인이 항목입니다.
        """
        parsed = self._send_and_parse(command, timeout=timeout)
        raw = (parsed.get('result') or '').strip()
        # 선행 정수만 안전 파싱 — 비거나 비정수면 count=0 (int() 예외로 목록이
        # 통째로 []가 되던 문제 방지). 진단을 위해 raw 를 로깅.
        m = re.match(r'\s*(\d+)', raw)
        count = int(m.group(1)) if m else 0
        if count <= 0:
            logger.info("[CR Sensor] %s -> count=0 (raw result=%r)", command, raw)
            return count, []
        items = self._read_multi_line_response(count, timeout=timeout)
        logger.info("[CR Sensor] %s -> count=%d, received %d items: %r",
                    command, count, len(items), items)
        return count, items

    @staticmethod
    def _split_list_item(item: str) -> List[str]:
        """리스트 항목 라인을 토큰으로 분리.

        SDK 는 RESULT_SEPARATOR 를 CString::Tokenize 로 처리 — '(' ')' ',' 를
        각각 구분자로 보고 빈 토큰을 버린다. 따라서 "0),Slow" 와 "(0),(Slow)"
        둘 다 ["0","Slow"] 가 된다. 기존 split("),") 는 후자에서 깨졌으므로
        (["(0","(Slow)"] → int("(0") 실패) 동일하게 토큰화한다.
        """
        return [p for p in re.split(r'[(),]+', item.strip()) if p]

    def _probe_firmware(self, attempts: int = 2, timeout: float = 2.0):
        """연결 직후 센서 생존 확인 (잘못된 COM 포트 빠른 실패용).

        RC Firmware 를 짧은 타임아웃으로 최대 attempts 회 질의해 첫 비어있지 않은
        응답이 오면 펌웨어 문자열을, 끝까지 무응답이면 None 을 반환한다. (정상
        센서는 RC Firmware 에 1초 내 응답 — 측정이 아니라 캐시된 문자열 읽기.)
        """
        for i in range(1, attempts + 1):
            try:
                fw = self._send_and_get_result("RC Firmware", timeout=timeout)
            except Exception as e:  # noqa: BLE001
                logger.warning("[CR Sensor] liveness probe %d/%d error: %s",
                               i, attempts, e)
                fw = ""
            if fw and fw.strip():
                logger.info("[CR Sensor] liveness probe %d/%d OK (fw=%s)",
                            i, attempts, fw.strip())
                return fw
            logger.warning("[CR Sensor] liveness probe %d/%d: 응답 없음 "
                           "(timeout=%.1fs)", i, attempts, timeout)
        return None

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
        """센서 구성 정보 전체 다운로드 (RC 명령 세트)
        
        선택적 다운로드: 지원되지 않는 항목은 skip하고 계속 진행
        (SOM10 같은 제한된 기능 센서 호환성을 위해)
        """

        # 필수 항목: ID, Model
        try:
            self.configuration.id = self._send_and_get_result("RC ID")
        except Exception as e:
            logger.warning("[CR Sensor] RC ID failed: %s (using default)", e)
            self.configuration.id = "N/A"

        try:
            self.configuration.model = self._send_and_get_result("RC Model")
        except Exception as e:
            logger.warning("[CR Sensor] RC Model failed: %s (using default)", e)
            self.configuration.model = "Unknown"

        # 선택적 항목: InstrumentType
        version = self._firmware_version_number()
        if version >= 1.17:
            try:
                self.configuration.instrument_type = int(
                    self._send_and_get_result("RC InstrumentType"))
            except Exception as e:
                logger.debug("[CR Sensor] RC InstrumentType not available: %s", e)

        # --- 리스트 항목들 (선택적) ---

        # Accessories  (항목 형식: "ID),Name),Type")
        try:
            count, items = self._send_and_get_list("RC Accessory")
            self.configuration.accessories = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 3:
                    self.configuration.accessories.append(
                        CRAccessoryItem(id=int(parts[0]), name=parts[1], type=parts[2]))
        except Exception as e:
            logger.debug("[CR Sensor] RC Accessory not available: %s", e)
            self.configuration.accessories = []

        # Filters  ("ID),Name),Type")
        try:
            count, items = self._send_and_get_list("RC Filter")
            self.configuration.filters = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 3:
                    self.configuration.filters.append(
                        CRFilterItem(id=int(parts[0]), name=parts[1], type=parts[2]))
        except Exception as e:
            logger.debug("[CR Sensor] RC Filter not available: %s", e)
            self.configuration.filters = []

        # Apertures  ("ID),Name")
        try:
            count, items = self._send_and_get_list("RC Aperture")
            self.configuration.apertures = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.apertures.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.debug("[CR Sensor] RC Aperture not available: %s", e)
            self.configuration.apertures = []

        # Modes — ★ SOM10에서 실패하는 명령
        try:
            count, items = self._send_and_get_list("RC Mode")
            self.configuration.modes = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.modes.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.debug("[CR Sensor] RC Mode not available: %s (sensor may have limited features)", e)
            self.configuration.modes = []

        # Exposure Modes
        try:
            count, items = self._send_and_get_list("RC ExposureMode")
            self.configuration.exposure_modes = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.exposure_modes.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.info("[CR Sensor] RC ExposureMode list unavailable/parse failed: %s", e)
            self.configuration.exposure_modes = []

        # Range Modes
        try:
            count, items = self._send_and_get_list("RC RangeMode")
            self.configuration.range_modes = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.range_modes.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.debug("[CR Sensor] RC RangeMode not available: %s", e)
            self.configuration.range_modes = []

        # Ranges
        try:
            count, items = self._send_and_get_list("RC Range")
            self.configuration.ranges = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.ranges.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.debug("[CR Sensor] RC Range not available: %s", e)
            self.configuration.ranges = []

        # Speeds
        try:
            count, items = self._send_and_get_list("RC Speed")
            self.configuration.speeds = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.speeds.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.info("[CR Sensor] RC Speed list unavailable/parse failed: %s", e)
            self.configuration.speeds = []

        # Sync Modes
        try:
            count, items = self._send_and_get_list("RC SyncMode")
            self.configuration.sync_modes = []
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    self.configuration.sync_modes.append(
                        CRModeItem(id=int(parts[0]), name=parts[1]))
        except Exception as e:
            logger.debug("[CR Sensor] RC SyncMode not available: %s", e)
            self.configuration.sync_modes = []

        # Matrix Modes (deprecated but still supported)
        try:
            count, items = self._send_and_get_list("RC MatrixMode")
            self.configuration.matrix_modes = []
            for item in items:
                parts = self._split_list_item(item)
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
                parts = self._split_list_item(item)
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
                parts = self._split_list_item(item)
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
            # ★ SOM10 대응: configuration 리스트가 비어있으면 RC에서 ID를 직접 가져옴
            _simple_map = [
                ('aperture_id',       'RS Aperture',      self.configuration.apertures,      'RC Aperture'),
                ('mode_id',           'RS Mode',          self.configuration.modes,          'RC Mode'),
                ('range_mode_id',     'RS RangeMode',     self.configuration.range_modes,    'RC RangeMode'),
                ('range_id',          'RS Range',         self.configuration.ranges,         'RC Range'),
                ('speed_id',          'RS Speed',         self.configuration.speeds,         'RC Speed'),
                ('exposure_mode_id',  'RS ExposureMode',  self.configuration.exposure_modes, 'RC ExposureMode'),
                ('sync_mode_id',      'RS SyncMode',      self.configuration.sync_modes,     'RC SyncMode'),
            ]
            for attr, rs_cmd, collection, rc_cmd in _simple_map:
                try:
                    result = self._send_and_get_result(rs_cmd)
                    val = self._find_id_by_name(collection, result)
                    
                    # ★ configuration 리스트가 비어서 ID를 찾지 못했으면 RC에서 직접 가져오기
                    if val == -1 and len(collection) == 0:
                        try:
                            rc_result = self._send_and_get_result(rc_cmd)
                            # RC 응답이 숫자면 직접 사용
                            if rc_result.isdigit():
                                val = int(rc_result)
                                logger.info("[CR Sensor] Using RC %s=%d (configuration list empty)", 
                                          attr, val)
                        except Exception as e:
                            logger.debug("[CR Sensor] Could not get %s from RC: %s", rc_cmd, e)
                    
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
        
        Note: C++ SDK에서는 M 명령에 30초 타임아웃 사용
        실제 측정은 노출 시간, 속도 설정 등에 따라 시간이 오래 걸릴 수 있음
        """
        try:
            # Helper: ID에서 이름 가져오기
            def get_name(id_val, mode_list):
                if id_val < 0:
                    return "NOT_SET"
                for item in mode_list:
                    if item.id == id_val:
                        return item.name
                return f"ID={id_val}"
            
            # 현재 설정 표시
            logger.info("="*60)
            logger.info("[CR Sensor] CAPTURE START")
            logger.info("  Current Settings:")
            logger.info("    Exposure Mode: %s (ID=%d)", 
                       get_name(self.setup.exposure_mode_id, self.configuration.exposure_modes),
                       self.setup.exposure_mode_id)
            logger.info("    Exposure: %s ms", self.setup.exposure)
            logger.info("    Max Auto Exposure: %s ms", self.setup.max_auto_exposure)
            logger.info("    Speed: %s (ID=%d)", 
                       get_name(self.setup.speed_id, self.configuration.speeds),
                       self.setup.speed_id)
            logger.info("    Range Mode: %s (ID=%d)", 
                       get_name(self.setup.range_mode_id, self.configuration.range_modes),
                       self.setup.range_mode_id)
            logger.info("    Aperture: %s (ID=%d)", 
                       get_name(self.setup.aperture_id, self.configuration.apertures),
                       self.setup.aperture_id)
            logger.info("="*60)
            
            # 설정 업로드
            self.upload_setup()
            
            # M 명령 타임아웃: C++ SDK 참조 (최대 30초)
            # 자동 노출 모드에서는 최대 30초까지 소요 가능
            logger.info("[CR Sensor] Sending M command (measuring, may take up to 30s)...")
            start_time = time.time()
            
            self._send_and_parse("M", timeout=30.0)
            
            elapsed = time.time() - start_time
            logger.info("[CR Sensor] M command completed in %.1f seconds", elapsed)
            logger.info("="*60)
            
            return True
        except Exception as e:
            logger.error("[CR Sensor] Capture error: %s", e)
            return False

    def download_reading(self) -> CRReading:
        """
        최근 측정 결과를 센서에서 다운로드합니다 (RM 명령 세트).
        선택적 읽기: 일부 명령이 실패해도 필수 데이터(XYZ)는 읽기 시도
        
        Note: C++ SDK 참조 - 각 RM 명령마다 타임아웃 설정
        측정 데이터 읽기는 일반 명령보다 시간이 걸릴 수 있음
        """
        reading = CRReading()
        
        # RM 명령 타임아웃: 데이터 읽기이므로 여유있게 설정
        rm_timeout = 10.0  # C++ SDK에서는 기본 타임아웃 사용
        
        logger.info("="*60)
        logger.info("[CR Sensor] DOWNLOAD READING START")
        logger.info("="*60)
        
        # 필수 CIE 데이터를 먼저 시도 (XYZ가 가장 중요)
        try:
            cie2 = reading.cie[CRObserver.DEGREE_2]
            
            logger.info("[CR Sensor] Reading XYZ data...")
            x_str = self._send_and_get_result("RM X", timeout=rm_timeout)
            y_str = self._send_and_get_result("RM Y", timeout=rm_timeout)
            z_str = self._send_and_get_result("RM Z", timeout=rm_timeout)
            
            logger.info("[CR Sensor] Raw XYZ strings: X='%s', Y='%s', Z='%s'", 
                       x_str, y_str, z_str)
            
            # 빈 문자열 체크
            if not x_str or not y_str or not z_str:
                logger.error("[CR Sensor] Critical: XYZ data is EMPTY!")
                logger.error("  Possible causes:")
                logger.error("    1. M command didn't complete measurement")
                logger.error("    2. Sensor not detecting light")
                logger.error("    3. Communication timeout")
                logger.error("  Check: Sensor pointing at bright surface?")
                self.last_reading = reading
                return reading
            
            # float 변환
            try:
                cie2.X = x_str
                cie2.Y = y_str
                cie2.Z = z_str
                logger.info("[CR Sensor] XYZ data retrieved: X=%.4f Y=%.4f Z=%.4f",
                           float(cie2.X), float(cie2.Y), float(cie2.Z))
            except ValueError as ve:
                logger.error("[CR Sensor] Critical: Cannot convert XYZ to float: %s", ve)
                logger.error("  X='%s', Y='%s', Z='%s'", x_str, y_str, z_str)
                self.last_reading = reading
                return reading
                
        except Exception as e:
            logger.error("[CR Sensor] Critical: Failed to read XYZ data: %s", e)
            self.last_reading = reading
            return reading
        
        # 선택적 필드들 - 개별 try-except로 보호
        try:
            reading.id = self._send_and_get_result("RM ID")
        except Exception as e:
            logger.debug("[CR Sensor] RM ID not available: %s", e)
            reading.id = "N/A"
        
        try:
            reading.model = self._send_and_get_result("RM Model")
        except Exception as e:
            logger.debug("[CR Sensor] RM Model not available: %s", e)
            reading.model = "Unknown"
        
        try:
            reading.time = self._send_and_get_result("RM Time")
        except Exception as e:
            logger.debug("[CR Sensor] RM Time not available: %s", e)
            reading.time = str(time.time())
        
        try:
            reading.accessory = self._send_and_get_result("RM Accessory")
        except Exception as e:
            logger.debug("[CR Sensor] RM Accessory not available: %s", e)
            reading.accessory = "N/A"
        
        try:
            reading.filter = self._send_and_get_result("RM Filter")
        except Exception as e:
            logger.debug("[CR Sensor] RM Filter not available: %s", e)
            reading.filter = "N/A"
        
        try:
            reading.aperture = self._send_and_get_result("RM Aperture")
        except Exception as e:
            logger.debug("[CR Sensor] RM Aperture not available: %s", e)
            reading.aperture = "N/A"
        
        try:
            reading.mode = self._send_and_get_result("RM Mode")
        except Exception as e:
            logger.debug("[CR Sensor] RM Mode not available: %s (limited sensor)", e)
            reading.mode = "Colorimeter"
        try:
            reading.mode = self._send_and_get_result("RM Mode")
        except Exception as e:
            logger.debug("[CR Sensor] RM Mode not available: %s (limited sensor)", e)
            reading.mode = "Colorimeter"
        
        try:
            reading.exposure_mode = self._send_and_get_result("RM ExposureMode")
        except Exception as e:
            logger.debug("[CR Sensor] RM ExposureMode not available: %s", e)
            reading.exposure_mode = "Auto"
        
        try:
            reading.exposure = self._send_and_get_result("RM Exposure")
        except Exception as e:
            logger.debug("[CR Sensor] RM Exposure not available: %s", e)
            reading.exposure = "N/A"
        
        try:
            reading.max_auto_exposure = self._send_and_get_result("RM MaxAutoExposure")
        except Exception as e:
            logger.debug("[CR Sensor] RM MaxAutoExposure not available: %s", e)
            reading.max_auto_exposure = "N/A"
        
        try:
            reading.range_mode = self._send_and_get_result("RM RangeMode")
        except Exception as e:
            logger.debug("[CR Sensor] RM RangeMode not available: %s", e)
            reading.range_mode = "Auto"
        
        try:
            reading.range = self._send_and_get_result("RM Range")
        except Exception as e:
            logger.debug("[CR Sensor] RM Range not available: %s", e)
            reading.range = "N/A"
        
        try:
            reading.speed = self._send_and_get_result("RM Speed")
        except Exception as e:
            logger.debug("[CR Sensor] RM Speed not available: %s", e)
            reading.speed = "Normal"
        
        try:
            reading.sync_mode = self._send_and_get_result("RM SyncMode")
        except Exception as e:
            logger.debug("[CR Sensor] RM SyncMode not available: %s", e)
            reading.sync_mode = "None"
        
        try:
            reading.sync_freq = self._send_and_get_result("RM SyncFreq")
        except Exception as e:
            logger.debug("[CR Sensor] RM SyncFreq not available: %s", e)
            reading.sync_freq = "N/A"
        
        try:
            reading.exposure_x = self._send_and_get_result("RM ExposureX")
        except Exception as e:
            logger.debug("[CR Sensor] RM ExposureX not available: %s", e)
            reading.exposure_x = "N/A"
        
        try:
            reading.user_calib_mode = self._send_and_get_result("RM UserCalibMode")
        except Exception as e:
            logger.debug("[CR Sensor] RM UserCalibMode not available: %s", e)
            reading.user_calib_mode = "N/A"
        
        # Matrix / Match
        try:
            r = self._send_and_get_result("RM Matrix")
            reading.matrix_id = "-1" if r == "None" else r
        except Exception as e:
            logger.debug("[CR Sensor] RM Matrix not available: %s", e)
            reading.matrix_id = "-1"
        
        try:
            r = self._send_and_get_result("RM Match")
            reading.match_id = "-1" if r == "None" else r
        except Exception as e:
            logger.debug("[CR Sensor] RM Match not available: %s", e)
            reading.match_id = "-1"
        
        try:
            reading.cmf = self._send_and_get_result("RM CMF")
        except Exception as e:
            logger.debug("[CR Sensor] RM CMF not available: %s", e)
            reading.cmf = "N/A"
        
        # CIE 2° Observer 추가 데이터
        try:
            cie2 = reading.cie[CRObserver.DEGREE_2]
            cie2.XYZ = self._send_and_get_result("RM XYZ")
            cie2.xy = self._send_and_get_result("RM xy")
            cie2.uv = self._send_and_get_result("RM uv")
            cie2.upvp = self._send_and_get_result("RM upvp")
            cie2.CCT = self._send_and_get_result("RM CCT")
        except Exception as e:
            logger.debug("[CR Sensor] Extended CIE data not available: %s", e)
        
        # CIE 10° Observer (선택적)
        try:
            cie10 = reading.cie[CRObserver.DEGREE_10]
            cie10.X = self._send_and_get_result("RM X10")
            cie10.Y = self._send_and_get_result("RM Y10")
            cie10.Z = self._send_and_get_result("RM Z10")
            cie10.XYZ = self._send_and_get_result("RM XYZ10")
            cie10.xy = self._send_and_get_result("RM xy10")
        except Exception as e:
            logger.debug("[CR Sensor] CIE 10° data not available: %s", e)
        
        # Warnings (선택적)
        try:
            count, items = self._send_and_get_list("RM Warnings")
            for item in items:
                parts = self._split_list_item(item)
                if len(parts) >= 2:
                    reading.warnings.append(
                        CRWarning(code=int(parts[0]), description=parts[1]))
                    reading.all_warnings += item + "\r"
        except Exception as e:
            logger.debug("[CR Sensor] Warnings not available: %s", e)
        
        # Spectrum (선택적)
        try:
            parsed = self._send_and_parse("RM Spectrum")
            result = parsed.get('result', '')
            sp = result.split("),")
            if len(sp) >= 4:
                reading.spectrum.starting_wavelength = float(sp[0])
                reading.spectrum.ending_wavelength = float(sp[1])
                reading.spectrum.delta = float(sp[2])
                n = int(sp[3])
                data_lines = self._read_multi_line_response(n)
                reading.spectrum.data = [float(x.strip()) for x in data_lines]
        except Exception as e:
            logger.debug("[CR Sensor] Spectrum not available: %s", e)
        
        time.sleep(0.15)  # SDK 호환 대기
        
        # Radiometric, Yv (선택적)
        try:
            reading.radiometric = self._send_and_get_result("RM Radiometric")
        except Exception as e:
            logger.debug("[CR Sensor] RM Radiometric not available: %s", e)
            reading.radiometric = "N/A"
        
        try:
            reading.yv = self._send_and_get_result("RM Yv")
        except Exception as e:
            logger.debug("[CR Sensor] RM Yv not available: %s", e)
            reading.yv = "N/A"
        
        time.sleep(0.15)
        
        # Temporal (선택적)
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
            logger.debug("[CR Sensor] Temporal data not available: %s", e)

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
        """측정 XYZ → 표시용 sRGB 스와치 색 (BT.709 역행렬 + sRGB gamma).

        CR 센서의 XYZ 는 절대값(Y 가 cd/m², 예: 57.65)이라 그대로 역변환하면
        선형 RGB 가 1 을 훌쩍 넘겨 전부 흰색으로 클립된다(주황을 재도 흰색).
        스와치는 '무슨 색을 쟀나'를 보여주는 용도이므로, 선형 RGB 의 최댓값을
        1 로 정규화해 절대 밝기는 버리고 색도(hue/sat)를 보존한 풀-밝기 색으로
        렌더한다. (calibration 은 이 rgb 가 아니라 xyz/cie_xy/luminance 를 사용.)
        """
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ])
        rgb_linear = np.maximum(M_inv @ xyz, 0.0)   # 음수(색역 밖) 클램프
        peak = float(np.max(rgb_linear))
        if peak > 1.0:                               # 밝기 정규화 → 색도 보존
            rgb_linear = rgb_linear / peak
        rgb = np.where(
            rgb_linear <= 0.0031308,
            12.92 * rgb_linear,
            1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
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
    실제 센서 동작을 시뮬레이션하되, Grayscale 정확도 개선
    
    개선 사항:
      - 마지막 측정 패턴 기억 (pattern_hint)
      - Grayscale 패턴 시 R≈G≈B (편차 < 0.05)
      - Color 패턴 시 랜덤 노이즈 추가
    
    INFERENCE: VirtualSensor는 넓은 색역 디스플레이(BT.2020)를 시뮬레이션합니다.
    이는 모든 타겟 색공간(BT.709, DCI-P3, BT.2020)을 포함할 수 있어
    캘리브레이션 로직 테스트에 적합합니다.
    약간의 오차(노이즈)를 추가하여 실제 디스플레이의 불완전성을 반영합니다.
    """

    def __init__(self, noise_level: float = 0.02, display_colorspace: str = 'BT.2020',
                 max_luminance: float = 100.0, black_level: float = 0.05,
                 native_gamma: float = 2.2):
        self.connected = False
        self.noise_level = noise_level
        self.measurement_count = 0
        self.last_pattern_rgb = None  # 마지막 측정 요청 패턴
        self.display_colorspace = display_colorspace

        # 디스플레이 고정 특성 (매 측정마다 달라지지 않음)
        # VirtualSensor가 일관된 응답을 주어야 iterative calibration이 수렴함
        self.max_luminance = max_luminance   # 최대 휘도 (cd/m²) — 고정
        self.black_level = black_level       # 블랙 레벨 (cd/m²) — 고정
        self.native_gamma = native_gamma     # 디스플레이 네이티브 EOTF 감마

        # 디스플레이 원색 행렬 설정 (RGB → XYZ)
        if display_colorspace == 'BT.2020':
            # ITU-R BT.2020 primaries (wide gamut)
            self.rgb_to_xyz_matrix = np.array([
                [0.6370, 0.1446, 0.1689],
                [0.2627, 0.6780, 0.0593],
                [0.0000, 0.0281, 1.0610],
            ])
        elif display_colorspace == 'DCI-P3':
            # DCI-P3 primaries
            self.rgb_to_xyz_matrix = np.array([
                [0.4865, 0.2657, 0.1982],
                [0.2290, 0.6917, 0.0793],
                [0.0000, 0.0451, 1.0439],
            ])
        else:  # 'sRGB' or 'BT.709'
            # sRGB / BT.709 primaries
            self.rgb_to_xyz_matrix = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ])

        print("[Virtual Sensor] Initialized "
              "(Noise: {:.1f}%, CS: {}, Lw: {:.1f} cd/m², "
              "Lb: {:.3f} cd/m², gamma: {:.2f})"
              .format(noise_level * 100, display_colorspace,
                      self.max_luminance, self.black_level, self.native_gamma))

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
    
    def set_pattern_hint(self, rgb: Tuple[float, float, float]):
        """다음 측정할 패턴 힌트 설정 (정확도 개선용)"""
        self.last_pattern_rgb = rgb

    # ---- internal helpers ----

    def _generate_realistic_rgb(self, pattern_rgb: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        현실적인 RGB 값 생성
        
        Grayscale (R=G=B) 패턴의 경우:
          - 실제 측정값도 R≈G≈B (편차 < 0.05)
          - 약간의 노이즈만 추가
        
        Color 패턴의 경우:
          - 패턴 중심으로 노이즈 추가
        """
        if pattern_rgb is not None:
            r, g, b = pattern_rgb
            # Grayscale 패턴 감지 (R=G=B)
            if abs(r - g) < 0.01 and abs(g - b) < 0.01:
                # Grayscale: R≈G≈B 유지하되 약간의 노이즈
                gray_level = (r + g + b) / 3.0
                noise = np.random.normal(0, self.noise_level * 0.5, 3)
                rgb = np.array([gray_level, gray_level, gray_level]) + noise
                return np.clip(rgb, 0, 1)
            else:
                # Color: 패턴 중심으로 노이즈 추가
                noise = np.random.normal(0, self.noise_level, 3)
                rgb = np.array([r, g, b]) + noise
                return np.clip(rgb, 0, 1)
        else:
            # 패턴 힌트 없으면 랜덤 생성
            rgb = np.random.random(3)
            noise = np.random.normal(0, self.noise_level, 3)
            return np.clip(rgb + noise, 0, 1)

    def _rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """
        RGB → XYZ 변환 (디스플레이 색공간 기준)

        네이티브 감마(power-law EOTF) 적용 후 원색 행렬 적용.
        실제 디스플레이의 EOTF 시뮬레이션:
          L_linear = code ^ native_gamma
          XYZ = M_display @ L_linear
        """
        linear = np.power(np.clip(rgb, 0.0, 1.0), self.native_gamma)
        return self.rgb_to_xyz_matrix @ linear

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
        time.sleep(0.05)  # 실제 센서 응답 시간 시뮬

        # 패턴 힌트 사용하여 현실적인 RGB 생성
        measured_rgb = self._generate_realistic_rgb(self.last_pattern_rgb)
        xyz_normalized = self._rgb_to_xyz(measured_rgb)
        cie_xy = self._xyz_to_xy(xyz_normalized)

        # 절대 휘도 스케일링 — 고정된 디스플레이 특성 사용
        # VirtualSensor 예: Lw=100 cd/m², Lb=0.05 cd/m²
        # relative_Y=0 → Lb, relative_Y=1 → Lw
        relative_luminance = float(np.clip(xyz_normalized[1], 0.0, 1.0))
        luminance = (self.black_level
                     + (self.max_luminance - self.black_level) * relative_luminance)

        # XYZ를 절대 휘도 기반으로 스케일링
        if xyz_normalized[1] > 1e-10:
            scale_factor = luminance / xyz_normalized[1]
        else:
            # 완전한 블랙 (R=G=B=0)
            scale_factor = self.black_level
        xyz = xyz_normalized * scale_factor

        self.measurement_count += 1

        print("[Virtual Sensor] hint_rgb={} → "
              "measured_rgb=({:.3f},{:.3f},{:.3f}) "
              "Y={:.2f} cd/m²"
              .format(
                  tuple(round(v, 3) for v in self.last_pattern_rgb)
                  if self.last_pattern_rgb is not None else None,
                  *measured_rgb, luminance))

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
            print("  Luminance: {:.2f} cd/m^2".format(reading.luminance))
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
