# DaVinci Resolve Remote Control

클라이언트 PC에서 **다른 Mac의 DaVinci Resolve를 원격 제어**하는 도구.
프로젝트/미디어/타임라인 조회, 플레이헤드 이동, **패턴 이미지 업로드→타임라인 추가→실시간 교체**, transport 재생까지.

DaVinci Resolve **무료 버전 호환** — 외부 스크립팅 API(Studio 전용) 대신, Resolve 내부에서 도는 작은 HTTP 서버 방식.

```
[클라이언트 PC] remote_control.py  --HTTP :7777-->  [Mac] Resolve 내부 remote_player.py
```

## 구성

| 파일 | 위치 | 역할 |
|------|------|------|
| `remote_player.py` | **Mac**, Resolve 내부에서 실행 | HTTP 서버(:7777). 받은 요청을 Resolve API로 수행 |
| `remote_control.py` | **클라이언트 PC** (어디든) | CLI + 라이브러리. 서버에 요청 전송 |

> 클라이언트는 **stdlib만** 사용 → Python 3.7+ 있으면 다른 프로젝트에 `remote_control.py` 하나만 복사해도 동작.

## 빠른 시작

### 1. 서버 — Mac에서
1. `remote_player.py`를 Resolve 스크립트 폴더로 복사:
   ```
   ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/
   ```
2. Resolve 실행 → 제어할 **프로젝트 + 타임라인 열기**.
3. (스크립팅 활성화) `Preferences → System → General → External scripting using = Local 이상`.
   - 클라이언트는 Resolve 스크립팅 포트가 아니라 **이 서버의 7777 포트**로 붙으므로 `Network`까지는 불필요.
4. `Workspace → Scripts → Utility → remote_player` 실행.
5. 콘솔에 `remote_player 0.5-portable listening on http://0.0.0.0:7777` 뜨면 OK.
6. macOS 방화벽이 Python 인바운드(:7777)를 막으면 한 번 **허용**.

### 2. 클라이언트 — 어디서든
```bash
# 대상 Mac 주소 (기본 192.168.0.3:7777, 다르면 지정)
set RESOLVE_REMOTE_HOST=192.168.0.3       # Windows
export RESOLVE_REMOTE_HOST=192.168.0.3    # macOS/Linux

python remote_control.py check            # 연결 + 서버 version 확인
```

## 설정 (환경변수)

| 변수 | 위치 | 기본값 | 용도 |
|------|------|--------|------|
| `RESOLVE_REMOTE_HOST` | 클라이언트 | `192.168.0.3` | 대상 Mac IP |
| `RESOLVE_REMOTE_PORT` | 클라이언트 | `7777` | 포트 |
| `RESOLVE_PATTERN_DIR` | 서버(Mac) | `~/Pictures/pattern_new` | 업로드 파일 저장 폴더(없으면 자동 생성) |

## 사용법

### CLI
```bash
python remote_control.py help                 # 전체 사용법
python remote_control.py check                # 연결 확인 (+ version)
python remote_control.py info                 # 프로젝트/타임라인 요약
python remote_control.py timeline             # 현재 타임라인 상세 (트랙/아이템)
python remote_control.py media                # 미디어풀 클립 목록
python remote_control.py jump <index> [track] # N번째 클립 위치로 점프 (시작+1초)
python remote_control.py add <mac_path>       # Mac 로컬 파일 임포트→타임라인 끝→점프
python remote_control.py upload <file> [--replace] [--as <name>]
python remote_control.py <file> [--replace]   # 단축: = upload <file>
python remote_control.py goto <timecode>      # 플레이헤드 이동
python remote_control.py play|pause|reverse|toggle   # transport
```

### 라이브러리 (생성기에서 import)
```python
from remote_control import upload

status, res = upload(r"C:\patterns\ramp.png")     # 전송→임포트→타임라인 끝→콘텐츠 위치
status, res = upload(path, replace=True)          # 같은 클립 제자리 교체 (실시간)
# res = {"ok": True, "msg": "added ramp.png @ 01:13:38:21 (N bytes)"}
```

### 실시간 패턴 갱신 루프
```python
from remote_control import upload
upload(path)                          # 최초 1회: 타임라인에 추가 + 점프
while updating:
    path = generate_pattern(...)      # 생성기가 새 패턴 파일 생성
    upload(path, replace=True)        # 같은 클립 제자리 교체 → 뷰어 즉시 갱신
```
> 파일이 **이미 Mac에 있으면**(생성기가 Mac/공유드라이브에 출력) 복사 생략: `add <mac_path>`.

## 패턴 이미지 — 8-bit / 16-bit PNG

**둘 다 됩니다.** 파이프라인은 파일 바이트를 그대로 전송 후 `ImportMedia`만 하므로 비트뎁스에 무관하고, Resolve가 8/16-bit PNG를 모두 임포트합니다.
- **HDR/그라디언트/그레이램프**처럼 부드러운 계조가 중요하면 **16-bit 권장**(밴딩 회피). 8-bit는 256단계.
- 색공간은 Resolve의 입력 색관리(프로젝트 설정)에 따라 해석됨(보통 sRGB/Rec.709). HDR 패턴이면 클립 **Input Color Space**를 맞춰주면 됨 — 이건 Resolve 설정이지 본 도구 영역 아님.
- TIFF/EXR/DPX 등도 동일 경로로 임포트 가능.

## 동작 메커니즘 / 한계

- **재생(transport)**: Resolve 스크립팅 API엔 `Play()`/`Stop()`이 **없음**. `play`/`pause` 등은 macOS `osascript`로 Resolve에 키(스페이스/JKL)를 보내는 GUI 자동화 →
  `System Settings → Privacy & Security → Accessibility`에서 **DaVinci Resolve 허용** 필요, Resolve가 포커스여야 함.
- **플레이헤드 = 프레임 표시**: `goto`/`jump`는 해당 프레임을 뷰어에 띄움(정지). 실제 영상이 흐르려면 transport. 단 이미 재생/루프 중이면 점프만으로 그 지점부터 계속 재생.
- **+1초 오프셋**: `jump`/`add`/`upload`는 클립 **시작+1초** 지점에 착지(짧으면 끝 직전 클램프) → 시작 경계에서 검정이 뜨는 문제 회피.
- **실시간 교체**: `--replace`는 `MediaPoolItem.ReplaceClip()`으로 같은 클립의 에셋만 교체(타임라인 위치 유지).

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `Errno 48 Address already in use` | 이전 서버 인스턴스가 포트 점유. 신버전은 재실행 시 자동 교체(`/shutdown`). 안 되면 Mac에서 `lsof -ti tcp:7777 \| xargs kill -9` |
| 업로드가 `UnicodeDecodeError`(500) | **구버전 서버**가 `/upload`를 JSON으로 파싱. → 최신 파일 복사 후 **스크립트 재실행** |
| 파일 바꿨는데 그대로 | **파일 복사 ≠ 리로드.** Resolve에서 스크립트를 **재실행**해야 새 코드 로드. `check`의 `version`으로 확인 |
| 점프했는데 검정 | `+1초` 오프셋이 적용된 0.4+ 인지 확인. 더 안쪽으로: `jump <i>`의 서버 `offset`(초) 조정 |
| 연결 안 됨 | 서버 실행 여부, `RESOLVE_REMOTE_HOST` 일치, 방화벽(:7777), 같은 네트워크 도달 확인 |

## 다른 프로젝트로 이식

- `davinci/` 폴더를 통째로 복사. 클라이언트 쪽에선 **`remote_control.py` 한 파일이면 충분**(stdlib only).
- 대상이 바뀌면 `RESOLVE_REMOTE_HOST`/`PORT`만 지정.
- 서버(`remote_player.py`)는 Mac의 Scripts/Utility에 두고 재실행. `RESOLVE_PATTERN_DIR`는 기본이 현재 사용자 홈(`~/Pictures/pattern_new`)이라 사용자별로 자동.

## API 레퍼런스

전체 HTTP 엔드포인트는 `remote_player.py` 상단 docstring, 클라이언트 사용법은 `python remote_control.py help` 참고.
