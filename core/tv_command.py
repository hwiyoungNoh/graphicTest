"""
core/tv_command.py — TV Command 모듈.

Calibration 모듈은 이 모듈의 고수준 API만 호출한다(보정 데이터를 '그냥' 넘김).
GET→SET 명령 변환, 페이로드 인코딩(base64 uint16 / float32), 전송(SSH/WSS),
키 처리는 전부 여기서 처리하고, 실제 전송은 연결된 TVControlAPI(_send 라우팅,
core.tv_control 로 연결)에 위임한다.

명령 어휘 (GET 기준; SET 은 prefix 만 GET_→SET_):
    3D_LUT_DATA            Color 3D LUT        uint16
    1D_DPG_DATA            Gamma 1D LUT(DPG)   uint16
    3BY3_GAMUT_DATA        3x3 gamut (SDR)     float x9
    HDR_3BY3_GAMUT_DATA    3x3 gamut (HDR)     float x9
    1D_2_2_EN / 1D_0_45_EN 1D 감마 곡선 enable 플래그
    CHARACTERISTICS        패널 특성 (read-only)

전송 경로 (TVControlAPI 가 모드별로 알아서):
    SET_3D_LUT_DATA  → api.send_lut() 재사용 (대용량 → BT709/ARG_MAX 회피, SSH/WS)
    그 외 setExternalPqData → SSH=set_external_pq(generic luna-send) / WS=fire-and-forget

인코딩 관례 (기존 코드 근거):
    float x9  = base64(little-endian float32 x9)   ← CAL_START payload 와 동일
    uint16    = base64(little-endian uint16),  3D 는 0..4095(12-bit) 정규화

⚠️ 실기 확인 필요: 1D_DPG 의 비트심도/dataCount, 1D_*_EN 플래그 인코딩은
   read_*()(GET)로 현재 값을 받아보면 dataType/dataCount 가 그대로 드러난다 —
   적용 전 한 번 읽어 스키마를 맞추는 것을 권장.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

GET_URI = "palm://externalpq/getExternalPqData"


def get_to_set(command: str) -> str:
    """GET_* → SET_* (prefix 만 교체). 이미 SET_/기타면 그대로."""
    return command.replace("GET_", "SET_", 1) if command.startswith("GET_") else command


class TVCommand:
    """Calibration 이 호출하는 TV 보정-데이터 API.

        cmd = TVCommand(conn.api)          # conn = core.tv_control.connect_tv_control(...)
        cmd.cal_start()
        cmd.apply_1d_lut(gamma_lut)        # Gamma → SET_1D_DPG_DATA
        cmd.apply_3d_lut(color_lut)        # Color → 3D LUT
        cmd.apply_3x3_gamut(matrix3x3)     # 또는 3x3 gamut matrix
        cmd.cal_end()                      # 보정 종료 → TV 에 잔류
    """

    def __init__(self, api: Any, *, pic_mode: str = "cinema", profile: int = 0):
        self._api = api          # 연결된 TVControlAPI
        self.pic_mode = pic_mode
        self.profile = profile

    # ── 인코더 ────────────────────────────────────────────────
    @staticmethod
    def _b64_uint16(arr, normalize_12bit: bool = True) -> tuple[str, int]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        if normalize_12bit and a.size and float(a.max()) <= 1.0:
            a = a * 4095.0
        a = np.clip(np.round(a), 0, 65535).astype("<u2")
        return base64.b64encode(a.tobytes()).decode("ascii"), int(a.size)

    @staticmethod
    def _b64_float32(arr) -> tuple[str, int]:
        a = np.asarray(arr, dtype="<f4").ravel()
        return base64.b64encode(a.tobytes()).decode("ascii"), int(a.size)

    @staticmethod
    def _b64_flag(on: bool) -> str:
        a = np.array([1 if on else 0], dtype="<u2")
        return base64.b64encode(a.tobytes()).decode("ascii")

    # ── 저수준 setExternalPqData (TVControlAPI 라우팅 위임) ────
    def _set_pq(self, command: str, *, data_type: str, data_count: int, data: str,
                program_id: int = 1, data_opt: int = 1) -> bool:
        payload = {
            "command": command,
            "programID": program_id,
            "picMode": self.pic_mode,
            "profileNo": self.profile,
            "dataOpt": data_opt,
            "dataType": data_type,
            "dataCount": data_count,
            "data": data,
        }
        request = self._api._build_external_pq_request(payload)
        return self._api._send(request)

    # ── 캘리브레이션 브래킷 ───────────────────────────────────
    def cal_start(self) -> bool:
        return self._api.send_cal_start(self.pic_mode, self.profile)

    def cal_end(self) -> bool:
        return self._api.send_cal_end(self.pic_mode, self.profile)

    # ── 적용 (SET) : Calibration 은 이것만 호출 ───────────────
    def apply_3d_lut(self, lut) -> bool:
        """Color 3D LUT. 대용량이라 검증된 send_lut(BT709/ARG_MAX 회피) 재사용."""
        return self._api.send_lut(lut, self.pic_mode, self.profile)

    def apply_1d_lut(self, lut) -> bool:
        """Gamma 1D LUT → SET_1D_DPG_DATA (uint16). float[0,1] 입력 시 12-bit 정규화."""
        data, n = self._b64_uint16(lut)
        return self._set_pq("SET_1D_DPG_DATA", data_type="unsigned integer16",
                            data_count=n, data=data)

    def apply_3x3_gamut(self, matrix, hdr: bool = False) -> bool:
        """3x3 gamut matrix → SET_(HDR_)3BY3_GAMUT_DATA (float x9, row-major)."""
        data, n = self._b64_float32(matrix)
        if n != 9:
            logger.warning("[tv_command] 3x3 gamut expects 9 floats, got %d", n)
        cmd = "SET_HDR_3BY3_GAMUT_DATA" if hdr else "SET_3BY3_GAMUT_DATA"
        return self._set_pq(cmd, data_type="float", data_count=n, data=data)

    def set_gamma_2_2_enable(self, on: bool = True) -> bool:
        """1D 2.2 감마 곡선 enable → SET_1D_2_2_EN."""
        return self._set_pq("SET_1D_2_2_EN", data_type="unsigned integer16",
                            data_count=1, data=self._b64_flag(on))

    def set_gamma_0_45_enable(self, on: bool = True) -> bool:
        """1D 0.45(역 2.2) 감마 곡선 enable → SET_1D_0_45_EN."""
        return self._set_pq("SET_1D_0_45_EN", data_type="unsigned integer16",
                            data_count=1, data=self._b64_flag(on))

    # ── 읽기 (GET) : 검증 / 특성 / 스키마 발견 ────────────────
    def read_3d_lut(self, callback: Optional[Callable] = None) -> bool:
        """현재 3D LUT 읽기 (WS 콜백 경로 재사용)."""
        return self._api.get_3d_lut_data(callback, self.pic_mode, self.profile)

    def read_raw(self, get_command: str) -> Optional[dict]:
        """임의 GET_* 를 동기 읽기 (SSH 모드). 응답 dict(dataType/dataCount/data 포함)
        반환 — 적용 전 스키마 확인용. WS 모드는 read_3d_lut 처럼 콜백 경로가 필요해
        여기선 None 을 반환한다(추후 확장)."""
        ssh = getattr(self._api, "_ssh", None)
        if ssh is not None and bool(getattr(ssh, "is_connected", False)):
            payload = {
                "command": get_command,
                "programID": 1,
                "picMode": self.pic_mode,
                "profileNo": self.profile,
                "dataOpt": 1,
            }
            try:
                return ssh.luna_raw(GET_URI, payload)
            except Exception as e:  # noqa: BLE001
                logger.error("[tv_command] read_raw(%s) failed: %s", get_command, e)
                return None
        logger.info("[tv_command] read_raw(%s): WS 동기 읽기 미지원 — read_3d_lut "
                    "또는 콜백 경로 확장 필요", get_command)
        return None

    def read_characteristics(self) -> Optional[dict]:
        return self.read_raw("GET_CHARACTERISTICS")
