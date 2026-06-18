# export_pkg — 포팅용 서브시스템 묶음

다른 프로젝트로 가져갈 3개 기능을 **원본 패키지 구조 그대로** 추출한 것.
모든 절대 import(`from lut import ...`, `from device import ...`, `from color.color_math import ...`)가
**무수정으로 동작**하도록 디렉토리 이름을 보존했다. `export_pkg/`를 sys.path 루트에 두면 됨.

> ✅ 동봉된 11개 핵심 모듈 import 스모크 테스트 통과 확인.

## 설치
```bash
pip install -r requirements.txt
# Mac SSH 쓸 경우:
cp config/_mac_ssh.local.json.template _mac_ssh.local.json   # 루트에 두고 값 채우기
```

## 사용법 (sys.path 루트 = export_pkg)
```python
import sys; sys.path.insert(0, "/path/to/export_pkg")
```

---

## 1. 패턴 생성  →  `color/`
| 파일 | 역할 |
|---|---|
| `color/color_pattern_generator.py` | **전부.** 10여 종 차트 생성기 1개 클래스. 순수 numpy+cv2, headless |

```python
from color.color_pattern_generator import ColorChartGenerator
gen = ColorChartGenerator(resolution="UHD", orientation="horizontal")

# (A) generate()/create_and_save() 디스패치 대상은 5종뿐: gradient, cube, wheel, mixed, gamma
img = gen.generate("gradient")
gen.create_and_save("gamma", "gamma.png", bit_depth=16)

# (B) 그 외 차트는 create_*() 직접 호출 후 save_image()
img = gen.create_grid_ramp_chart(num_angles=6, num_sats=4, bit_depth=16)
gen.save_image(img, "grid_ramp.png")
```
- 의존성: `numpy`, `opencv-python` 뿐. 내부 import 0.
- `generate()` 디스패치: `gradient / cube / wheel / mixed / gamma`
- 직접 호출 메서드: `create_grid_ramp_chart / create_hue_sat_plane_chart / create_hue_sat_zoom_chart / create_sat_val_plane_chart / create_hsv_cube_tile_chart / create_full_hsv_cube_chart`
- 주의: `bit_depth=16` → uint16 PNG, 10-bit 코드값 `round(code*65535/1023)` 스케일(DaVinci full-range용). HSV는 OpenCV 관례(H 0–179).

---

## 2. Mac 제어 + DaVinci 제어 + 패턴 전송  →  `davinci/`
| 파일 | 역할 |
|---|---|
| `davinci/remote_control.py` | **클라이언트**(어디서든 → Mac). `upload()` 호출 |
| `davinci/remote_player.py`   | **Mac DaVinci 내부 HTTP 서버.** Resolve API를 JSON 엔드포인트로 노출 |
| `davinci/README.md`          | 배포·실행 가이드 |

**전송 흐름**: `upload(png)` → HTTP POST `/upload` → `remote_player`가 `~/Pictures/pattern_new`에 저장 → ImportMedia+AppendToTimeline+SetCurrentTimecode → **TV(HDMI)에 표시**.
```python
from davinci import remote_control
remote_control.upload("pattern.png")            # replace=True 면 ReplaceClip
remote_control.check()
```
- 의존성: 클라이언트는 stdlib(http)만. 서버는 DaVinci Fusion Scripting API + osascript(transport).
- env: `RESOLVE_REMOTE_HOST`(기본 192.168.0.3) / `RESOLVE_REMOTE_PORT`(7777) / `RESOLVE_PATTERN_DIR`.
- ⚠️ `remote_player.py`는 **Mac의 `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/`에 복사 후 Resolve 메뉴 Scripts>Utility에서 수동 실행**해야 로드됨(수정 후 재실행 필요). Preferences>System>External scripting = Local 이상.
- ⚠️ transport 키(play/JKL)는 osascript GUI 자동화 → macOS 접근성 권한 필요, **non-macOS 불가**.
- (선택) Mac 디스플레이 진단 `_mac_probe.py / _mac_probe_extra.py / _mac_pull_icc.py`는 원본 repo에 있고 미포함. `_mac_pull_icc`는 `color/color_math.py`(동봉)에 의존.

---

## 3. TV SSH + Calibration + 3D LUT + 설정 제어  →  `device/`, `lut/`, `wss/`, `utils/`

### Transport / 제어 (`device/`)
| 파일 | 역할 |
|---|---|
| `device/device_tv_control_panel.py` | **`TVControlAPI` — 통합 진입점.** SSH/WS 라우팅을 숨기고 picture/calibration/3D-LUT 단일 API |
| `device/device_tv_ssh_client.py`    | **`TVSSHClient`** — webOS dev SSH(paramiko + luna-send). Qt 워커는 try/except로 선택적 |
| `device/device_connect.py`          | **`WebSocketClient`** — SSAP/WSS, PIN 페어링, 명령 디스패치 |
| `device/device_connect_info.py`     | WS 매니페스트(`SECURE_MANIFEST`=`WRITE_SETTINGS`) + luna 커맨드 템플릿 |

```python
from device.device_tv_ssh_client import TVSSHClient
from device.device_tv_control_panel import TVControlAPI

ssh = TVSSHClient("192.168.0.8"); ssh.connect()
api = TVControlAPI()
api.attach_ssh(ssh); api.set_mode("ssh")     # 또는 set_command_sender(ws.send_request)

api.set_brightness(50); api.set_contrast(85); api.set_color_temperature(-50)  # OSD 설정
api.send_cal_start(); api.send_lut(lut_array); api.send_cal_end()             # 3D LUT
```
**라우팅**: `_send()`가 `_ssh_live()`(mode=='ssh' AND ssh.is_connected)면 SSH, 아니면 WS(fire-and-forget). 페이지/호출부는 transport를 모름.

### 3D LUT 빌드 엔진 (`lut/`)
| 파일 | 역할 |
|---|---|
| `lut/lut_algorithm.py`            | 3D LUT 엔진: 색공간 변환, 보간(trilinear/gaussian/IDW), CP 그리드 |
| `lut/lut_compensation.py`         | send-time 사전보상(패널 under-gain f(V) + OSD remap w). 순수 numpy |
| `lut/lut_slots.py`                | 세션 저장/로드(NPZ+JSON, `%APPDATA%/PictureCalibration/slots`) |
| `lut/recon/lutrec_loader.py`      | `.cube` 로드/저장(`load_cube/save_cube/build_identity`) |
| `lut/recon/lutrec_reconstruct.py` | sparse CP 편집 → 33³ LUT 재구성 |
| `lut/recon/lutrec_oklab.py`       | OKLab 보간/정확도 메트릭 |

### 페어링 + 유틸 (`wss/`, `utils/`)
| 파일 | 역할 |
|---|---|
| `wss/pair_tv.py`              | 1회성 PIN 페어링 → client-key 저장 (`from device.device_connect import WebSocketClient`) |
| `wss/wss_guide.md`           | SSAP 프로토콜/매니페스트/인증 문서 |
| `utils/reset_tv_lut.py`      | 3D LUT identity 리셋(CAL 사이클) |
| `utils/probe_tv_settings.py` | OSD 값 어휘/현재상태 read |
| `utils/capture_rgb.py`       | SSH raw 8-bit RGB 캡처 → (H,W,3) uint8 |

**LUT → TV 전송 경로**
- **SSH(주)**: uint16 바이너리를 `/tmp/ssg/volatile/externalpq/_lut3d.bin`에 업로드 → **Node.js palmbus**가 `palm://com.webos.service.pqcontroller/setExternalPqData`를 path 참조로 호출(**luna-send ARG_MAX ~128KB 회피**).
- **WS**: base64 인코딩 → `BT709_3D_LUT_DATA` 커맨드.
- 양쪽 float[0,1] → uint16[0,4095] 정규화.
- **TV측 요구**: webOS 5+ (`pqcontroller` 서비스), 2021+ LG OLED 검증. SSH 폴백 시 `node` 바이너리 PATH 필요.

---

## ⚠️ 포팅 시 주의

1. **tkinter 의존(GUI baggage)**: `device_tv_control_panel.py`와 `device_connect.py`는 **모듈 레벨에서 `import tkinter`**. tkinter는 stdlib라 보통 import 성공하지만, headless 서버 빌드엔 없을 수 있음. `TVControlAPI`/`WebSocketClient` 로직 자체는 tkinter 불필요 → 깔끔히 쓰려면 파일에서 tkinter import와 `TVControlPanel`(tkinter GUI 클래스)을 제거하는 가벼운 정리 권장.
2. **하드코딩 값 파라미터화**: TV IP `192.168.0.7/0.8`, WS 포트 `3001`, Mac `192.168.0.3:7777`, TV측 경로 `/tmp/ssg/volatile/externalpq/_lut3d.bin`. config/env로 빼기.
3. **CAL_START/END로 LUT 전송 감싸기**. CAL_START가 colorTemperature를 교란 → 이후 복원(`utils/reset_tv_lut.py` 참조).
4. **페어링 키**: `%APPDATA%/PictureCalibration/client_keys.json`(IP별). `device/_embedded_key.py`는 원본에도 없음 → 없으면 PIN 페어링으로 폴백(안전).
5. **screenshot은 post-framebuffer**(패널 WB 뒤) → webOS 스크린샷으로 색온도 측정 불가.
6. **picture mode 변경은 out-of-band**(colorGamut/gamma 의존 체인 때문에 TVControlAPI 외부).

## 미포함(필요 시 원본 repo에서 추가)
- **Auto-Match OSD 타겟 계산**: `utils/apply_tv_osd_target.py` / `check_tv_osd_target.py` + `display/display_match_engine.py` + `display_analysis/effective_state.py`(+ model_db, panel_priors, ddc_ci…). source 디스플레이로부터 OSD 타겟을 *계산*하는 별도 대형 서브시스템 — 직접 설정 제어엔 불필요해 제외.
- `lut/lut_vector_engine.py`(Qt 시각화/dead), `pages/*`, `core/core_ui_common.py`(PySide6 UI glue).
