# SSG (Second Screen Gateway) 외부 연결 및 권한 획득 가이드

## 1. 개요

webOS TV의 SSG 서비스는 포트 3001(WSS)을 통해 외부 클라이언트의 연결을 허용합니다.
서명된 manifest를 사용하면 SECURE 수준의 권한(WRITE_SETTINGS 등)을 획득할 수 있습니다.

---

## 2. 접속 구조

```
외부 클라이언트 → WSS (port 3001) → Second Screen Gateway → API Adapter → Luna Bus
```

- **프로토콜**: WebSocket over TLS (`wss://`)
- **포트**: 3001
- **TLS**: Self-signed CA (클라이언트에서 인증서 검증 무시 필요)
- **클라이언트 인증서**: 불필요 (mTLS 아님)

---

## 3. 인증 흐름

### 3.1 최초 접속 (client-key 없음)

```
1. WSS 연결
2. register 메시지 전송 (서명된 manifest 포함)
3. TV에 PIN 팝업 표시 (8자리 랜덤)
4. 사용자가 TV 화면의 PIN 확인 → 클라이언트에서 setPin 전송
5. 인증 성공 → client-key 반환
6. client-key 저장 (재접속용)
```

### 3.2 재접속 (client-key 있음)

```
1. WSS 연결
2. register 메시지 전송 (client-key만 포함)
3. 즉시 인증 완료 (PIN 불필요)
```

---

## 4. 서명 생성

### 4.1 필요한 파일

| 파일 | 용도 |
|------|------|
| `tools/manifest/test-signing-key.pem` | 개발 TV용 서명 키 |
| `tools/manifest/svl-signing-key.pem` | 양산 TV용 서명 키 |

### 4.2 서명 알고리즘

```javascript
const crypto = require('crypto');

// 1. canonicalStringify: 키를 알파벳순 정렬 후 JSON 직렬화
function canonicalStringify(obj) {
    if (Array.isArray(obj)) {
        return "[" + obj.map(canonicalStringify).join(",") + "]";
    } else if (obj !== null && typeof obj === "object") {
        var keys = Object.keys(obj).sort();
        return "{" + keys.map(function(key) {
            return JSON.stringify(key) + ":" + canonicalStringify(obj[key]);
        }).join(",") + "}";
    } else {
        return JSON.stringify(obj);
    }
}

// 2. 서명 생성
function signManifest(manifest, privateKeyPem) {
    var header = {
        signatureVersion: 1,
        algorithm: "RSA-SHA256",
        keyId: "test-signing-cert"  // 양산: "svl-signing-cert"
    };

    // 서명 대상 데이터
    var signingInput = canonicalStringify({
        header: header,
        payload: manifest.signed
    });

    // RSA-SHA256 서명
    var signer = crypto.createSign('RSA-SHA256');
    signer.update(signingInput, 'binary');
    var signature = signer.sign(privateKeyPem, 'base64');

    // 최종 조합: base64(header) + "." + signature
    var encodedHeader = Buffer.from(canonicalStringify(header)).toString('base64');
    var combined = encodedHeader + "." + signature;

    // manifest에 signatures 추가
    manifest.signatures = [{
        signatureVersion: 1,
        signature: combined
    }];

    return manifest;
}
```

### 4.3 CLI 사용법 (기존 도구)

```bash
cd tools/manifest
node sign-manifest.js your-unsigned-manifest.json
```

---

## 5. Manifest 구조

```json
{
    "manifestVersion": 1,
    "appVersion": "1.0",
    "signed": {
        "created": "20260609",
        "appId": "com.your.app",
        "vendorId": "com.your",
        "localizedAppNames": {
            "": "Your App Name"
        },
        "localizedVendorNames": {
            "": "Your Company"
        },
        "permissions": [
            "WRITE_SETTINGS",
            "CONTROL_POWER",
            "CONTROL_DISPLAY",
            "READ_RUNNING_APPS",
            "READ_NOTIFICATIONS"
        ],
        "serial": "3c6f8f1519ac979d1ce1064329e0f2e7"
    },
    "permissions": [
        "LAUNCH",
        "LAUNCH_WEBAPP",
        "CONTROL_AUDIO",
        "CONTROL_INPUT_TV",
        "CONTROL_INPUT_MEDIA_PLAYBACK",
        "CONTROL_INPUT_TEXT",
        "CONTROL_MOUSE_AND_KEYBOARD",
        "READ_APP_STATUS",
        "READ_INSTALLED_APPS",
        "READ_CURRENT_CHANNEL",
        "WRITE_NOTIFICATION_TOAST"
    ],
    "signatures": [
        {
            "signatureVersion": 1,
            "signature": "<base64_header>.<base64_rsa_signature>"
        }
    ]
}
```

- **`signed` 블록**: 서명으로 보호됨. 변경 시 서명 불일치.
- **최상위 `permissions`**: unsigned로 처리됨. SECURE 타입 제외.
- **`serial`**: 임의 32자 hex 문자열.

---

## 6. WebSocket 프로토콜

### 6.1 Register (서명된 manifest)

```json
{
    "type": "register",
    "id": 1,
    "payload": {
        "pairingType": "PIN",
        "manifest": { /* 서명된 manifest 전체 */ }
    }
}
```

### 6.2 Register (기존 client-key)

```json
{
    "type": "register",
    "id": 1,
    "payload": {
        "client-key": "abc123def456..."
    }
}
```

### 6.3 PIN 응답

```json
{
    "type": "request",
    "id": 2,
    "uri": "ssap://pairing/setPin",
    "payload": {
        "pin": "12345678"
    }
}
```

### 6.4 API 호출 (Request)

```json
{
    "type": "request",
    "id": 3,
    "uri": "ssap://system/getSystemInfo"
}
```

### 6.5 구독 (Subscribe)

```json
{
    "type": "subscribe",
    "id": 4,
    "uri": "ssap://audio/getVolume"
}
```

### 6.6 구독 해제

```json
{
    "type": "unsubscribe",
    "id": 4
}
```

---

## 7. 서버 응답 형식

### 등록 성공

```json
{
    "type": "registered",
    "id": 1,
    "payload": {
        "client-key": "a1b2c3d4e5f6..."
    }
}
```

### 요청 성공

```json
{
    "type": "response",
    "id": 3,
    "payload": {
        "returnValue": true,
        ...
    }
}
```

### 에러

```json
{
    "type": "error",
    "id": 3,
    "error": "...",
    "payload": {
        "returnValue": false,
        "errorCode": -1000,
        "errorText": "..."
    }
}
```

---

## 8. 권한 수준별 차이

### Signed 권한 (서명 필수)

| 권한 | 기능 |
|------|------|
| WRITE_SETTINGS | 시스템 설정 변경 |
| READ_LGE_SDX | LG SDX 데이터 읽기 |
| READ_LGE_TV_INPUT_EVENTS | TV 입력 이벤트 수신 |
| READ_NOTIFICATIONS | 알림 읽기 |
| UPDATE_FROM_REMOTE_APP | 원격 업데이트 |
| TEST_SECURE | 테스트 보안 API |

### Unsigned 권한 (서명 불필요, manifest에 나열만)

| 권한 | 기능 |
|------|------|
| LAUNCH / LAUNCH_WEBAPP | 앱 실행 |
| CONTROL_AUDIO | 볼륨 제어 |
| CONTROL_POWER | 전원 제어 |
| CONTROL_DISPLAY | 디스플레이 제어 |
| CONTROL_INPUT_* | 입력 제어 |
| CONTROL_MOUSE_AND_KEYBOARD | 마우스/키보드 |
| READ_APP_STATUS | 앱 상태 조회 |
| READ_INSTALLED_APPS | 설치 앱 목록 |
| READ_CURRENT_CHANNEL | 현재 채널 |
| WRITE_NOTIFICATION_TOAST | 토스트 알림 |

---

## 9. 보안 관련 참고

### 9.1 TV 서버 인증서 체인

```
RootCA_Cert_webOS.pem (Root CA)
  └── ica.sub.ssl.cert.pem (Intermediate CA)
       └── ssg.server.cert.pem (Server Cert)
Private Key: luna://com.webos.service.sm/crypto/getSSGSSL 로 런타임 획득
```

### 9.2 PIN 보안

- 8자리 숫자 (`crypto.randomBytes`로 생성)
- PIN 실패 20회 초과 시 PIN 방식 차단 → PROMPT 강제 전환
- 타임아웃: 120초

### 9.3 Config 보안 플래그

```javascript
// config.js (기본값)
"disable-interface-security": false,  // 인터페이스 보안 적용
"disable-origin-security": false,     // 오리진 제한 적용
"bypass-pairing": false,              // 페어링 필수
"force-start": false                  // allowMobileDeviceAccess 체크
```

---

## 10. 구현 체크리스트

```
[ ] WSS 클라이언트 (TLS 인증서 검증 skip)
[ ] canonicalStringify 함수 구현
[ ] RSA-SHA256 서명 생성 (private key 사용)
[ ] Manifest 조립 (signed + unsigned permissions)
[ ] Register 메시지 전송
[ ] PIN 입력 처리 (TV 화면 확인 필요)
[ ] client-key 영구 저장
[ ] Request/Subscribe/Unsubscribe 메시지 처리
[ ] 에러 핸들링 및 재접속 로직
```
