import json
import websocket
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
import logging
import ssl  # Import the ssl module
import http.client
import base64
import os
from device import device_connect_info as connectInfo
from tkinter import filedialog

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logDisable = False
server_ip = "192.168.0.7"

# ── client-key 영속 저장 (TV IP별로 저장) ─────────────────────────────────
_KEY_FILE = os.path.join(
    os.environ.get("APPDATA") or os.path.expanduser("~"),
    "PictureCalibration", "client_keys.json",
)


def _load_client_key(ip: str):
    """client-key 조회. 저장키(페어링 발급) 우선, 없으면 내장 타겟키 폴백.

    저장키(_KEY_FILE)는 그 TV가 페어링 때 발급한 유효 키라 항상 우선한다.
    없으면 내장 키(device/_embedded_key.py, 타겟 .7용 XOR 암호화)로 폴백 →
    배포본이 타겟 TV엔 페어링 없이 바로 연결. 둘 다 없으면 None → TV가 PIN
    프롬프트를 띄우고 앱이 PIN 페어링을 진행(set_pin)한다.

    저장키를 우선하는 이유: 내장키는 특정 물리 TV(.7) 전용이라 .7에 다른 TV가
    오면 무효다. 그 경우 PIN 재페어링으로 새 키가 저장되는데, 내장키 우선이면
    매번 PIN을 다시 묻게 된다. 저장키 우선이면 재페어링 후 자동 연결된다.
    (무효 키를 보내도 TV는 pairingType:PIN으로 폴백함 — 실측 확인.)
    """
    try:
        with open(_KEY_FILE) as f:
            k = json.load(f).get(ip)
        if k:
            return k
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        from device._embedded_key import get_embedded_key
        return get_embedded_key(ip)
    except Exception:
        return None


def _save_client_key(ip: str, key: str):
    """client-key를 파일에 저장 (IP 키로 관리)."""
    try:
        os.makedirs(os.path.dirname(_KEY_FILE), exist_ok=True)
        try:
            with open(_KEY_FILE) as f:
                keys = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            keys = {}
        keys[ip] = key
        with open(_KEY_FILE, "w") as f:
            json.dump(keys, f, indent=2)
        logging.info(f"[WS] client-key saved for {ip}")
    except Exception as exc:
        logging.error(f"[WS] Failed to save client-key: {exc}")

class WebSocketClient:
    def __init__(self, server, message_callback, port: int = 3001):
        self.server = server
        self.port = port
        self.ws = None
        self.id_counter = 1
        self.current_client_key = None
        self.auto_register = True  # Set to True or False based on your requirement
        self.message_callback = message_callback  # Callback to update GUI with messages
        # 추가 응답 핸들러 목록 (TVControlAPI.handle_response 등 등록)
        self._response_handlers: list = []

    def add_response_handler(self, handler):
        """응답 메시지를 받을 핸들러 추가.

        handler 시그니처: handler(message: str)
        TVControlAPI.handle_response 를 등록하면 자동으로 설정 응답이 라우팅됩니다.
        """
        if handler not in self._response_handlers:
            self._response_handlers.append(handler)

    def remove_response_handler(self, handler):
        """응답 핸들러 제거."""
        try:
            self._response_handlers.remove(handler)
        except ValueError:
            pass

    def on_message(self, ws, message):
        # Call the message callback to update the GUI
        self.message_callback(message)

        # 추가 응답 핸들러에게 메시지 전달
        for handler in self._response_handlers:
            try:
                handler(message)
            except Exception as e:
                logging.error(f"Response handler error: {e}")

        # Parse the message
        try:
            parsed_message = json.loads(message)
            response_type = parsed_message.get('type')

            if response_type == 'registered':
                # TV 페어링 완료 → client-key 저장
                client_key = parsed_message.get('payload', {}).get('client-key')
                if client_key:
                    self.current_client_key = client_key
                    _save_client_key(self.server, client_key)
                    self.message_callback(
                        f"[WS] Registered! client-key={client_key[:8]}... saved")
                    logging.info(f"[WS] Registered with TV ({self.server})")
                else:
                    self.message_callback(
                        "[WS] 'registered' received but no client-key in payload")

            elif response_type == 'error':
                logging.error(f"Error received: {parsed_message.get('message')}")

        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from message.")

    def on_error(self, ws, error):
        # Close 프레임이 error 콜백으로 날아올 때 close code 추출
        code_hint = ""
        if isinstance(error, Exception) and hasattr(error, 'args'):
            raw = getattr(error, 'status_code', None)
            if raw is None:
                s = str(error)
                # b'\x03\xf0...' 형태에서 0x03f0 = 1008 추출
                import re as _re
                m = _re.search(r"data=b'(\\x[0-9a-f]{2})(\\x[0-9a-f]{2})", s)
                if m:
                    b1 = int(m.group(1).replace('\\x', ''), 16)
                    b2 = int(m.group(2).replace('\\x', ''), 16)
                    code_hint = f" (WS close code={b1 << 8 | b2})"
        msg = f"[WS ERROR]{code_hint} {error}"
        logging.error(msg)
        self.message_callback(msg)

    def on_close(self, ws, close_status_code, close_msg):
        code = close_status_code or "(none)"
        msg_text = close_msg or ""
        msg = f"[WS CLOSED] code={code}  {msg_text}"
        logging.info(msg)
        self.message_callback(msg)

    def on_open(self, ws):
        logging.info(f"[WS] Connection established to {self.server}")
        self.message_callback(f"[WS] Connected to {self.server}")

        # 저장된 client-key 로드 (없으면 None → TV 화면에 페어링 프롬프트 표시됨)
        client_key = _load_client_key(self.server)
        if client_key:
            self.current_client_key = client_key
            self.message_callback(f"[WS] Using stored client-key={client_key[:8]}...")
        else:
            self.message_callback("[WS] No stored client-key — TV 화면의 페어링 프롬프트를 승인해 주세요")

        register_id = self.id_counter
        self.id_counter += 1
        # SECURE_MANIFEST(서명 포함)로 register해야 WRITE_SETTINGS(SECURE) 권한이
        # 부여되어 setExternalPqData(LUT/Cal/Pattern)가 동작함. PROTECTED_MANIFEST는
        # 서명이 없어 WRITE_SETTINGS를 못 받아 권한 거부됨.
        # client-key가 있으면 TV가 그것으로 즉시 인증(PIN 생략), 없으면 pairingType
        # "PIN"으로 TV 화면에 8자리 PIN을 표시 → set_pin()으로 응답해야 함.
        request = {
            "type": "register",
            "id": register_id,
            "payload": {
                "client-key": client_key,
                "pairingType": "PIN",
                "manifest": connectInfo.SECURE_MANIFEST,
            }
        }
        ws.send(json.dumps(request))
        logging.info(f"[WS] Sent register request (id={register_id})")

    def connect(self):
        logging.info(f"[WS] Starting connection to {self.server}:{self.port}")
        context = ssl._create_unverified_context()
        try:
            # HTTP pre-check (self.server 로 수정 — 전역 server_ip 버그 수정)
            connection = http.client.HTTPSConnection(self.server, self.port, timeout=10, context=context)
            connection.request('GET', '/')
            response = connection.getresponse()
            logging.info(f"[WS] HTTP pre-check: {response.status} {response.reason}")
            response.read()  # flush
        except Exception as e:
            # HTTP pre-check 실패는 경고만 — WebSocket 연결은 계속 시도
            logging.warning(f"[WS] HTTP pre-check failed (ignored): {e}")
            self.message_callback(f"[WS] HTTP pre-check skipped: {e}")

        websocket.enableTrace(False)
        try:
            self.ws = websocket.WebSocketApp(
                f"wss://{self.server}:{self.port}/",
                on_message=self.on_message,
                on_open=self.on_open,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            # suppress_origin=True: LG WebOS TV는 Origin 헤더가 있으면 1008(유효하지 않은 원본)로 거부함
            self.ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                suppress_origin=True,
            )
        except Exception as e:
            logging.error(f"[WS] Connection error: {e}")
            self.message_callback(f"[WS ERROR] {e}")

    def send_request(self, request):
        if self.ws is None:
            logging.warning("[WS] send_request called but ws is None")
            return
        # 호출자가 id를 미리 지정했으면(예: TVControlAPI가 100+ 부여) 보존한다.
        # 이래야 응답의 id가 TVControlAPI.handle_response의 _pending_responses
        # 키와 일치해 load 완료 콜백이 발화한다. 과거엔 무조건 덮어써서
        # 응답 라우팅이 깨졌음. id가 없을 때만 자체 카운터로 부여.
        if request.get('id') is None:
            request['id'] = self.id_counter
            self.id_counter += 1
        self.ws.send(json.dumps(request))

    def set_pin(self, pin: str):
        """TV 화면에 표시된 8자리 PIN으로 페어링 완료 (저장된 client-key가 없을 때).

        register(pairingType="PIN") 후 유효한 client-key가 없으면 TV가 PIN을
        화면에 표시한다. 그 PIN을 setPin으로 보내면 'registered' 응답과 함께
        client-key가 발급되어 on_message에서 저장된다.
        타겟 TV는 내장 client-key로 자동 인증되므로 이 경로는 보통 불필요.
        """
        if self.ws is None:
            logging.warning("[WS] set_pin called but ws is None")
            return
        request = {
            "type": "request",
            "id": self.id_counter,
            "uri": "palm://pairing/setPin",
            "payload": {"pin": str(pin)},
        }
        self.id_counter += 1
        self.ws.send(json.dumps(request))
        logging.info("[WS] Sent setPin")

class App:
    def __init__(self, master, websocket_client):
        self.master = master
        self.websocket_client = websocket_client
        self.master.title("WebSocket Command Sender")
        self.master.geometry("600x600")  # Set the window size

        self.label = tk.Label(master, text="Select a command to send:")
        self.label.pack(pady=10)

        self.command_listbox = tk.Listbox(master, width=80, height=10)  # Adjusted size
        for message in connectInfo.canned_messages:
            self.command_listbox.insert(tk.END, message['name'])
        self.command_listbox.pack(pady=10)

        self.json_text = scrolledtext.ScrolledText(master, width=80, height=10)  # Text area for JSON
        self.json_text.pack(pady=10)

        self.response_text = scrolledtext.ScrolledText(master, width=80, height=10)  # Text area for responses
        self.response_text.pack(pady=10)

        self.command_listbox.bind('<<ListboxSelect>>', self.on_command_select)  # Bind selection event

        self.send_button = tk.Button(master, text="Send Command", command=self.send_command)
        self.send_button.pack(pady=5)

        self.connect_button = tk.Button(master, text="Connect", command=self.start_connection)
        self.connect_button.pack(pady=5)

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.pack(pady=5)

        self.auto_register_var = tk.BooleanVar(value=True)  # Checkbox for auto-register
        self.auto_register_checkbox = tk.Checkbutton(master, text="Auto Register", variable=self.auto_register_var)
        self.auto_register_checkbox.pack(pady=5)

        # Server IP Entry
        self.server_label = tk.Label(master, text="Server IP:")
        self.server_label.pack(pady=5)

        self.server_entry = tk.Entry(master, width=30)
        self.server_entry.pack(pady=5)
        self.server_entry.insert(0, server_ip)  # Default value

    def on_command_select(self, event):
        selected_index = self.command_listbox.curselection()
        if selected_index:
            command_data = connectInfo.canned_messages[selected_index[0]]['data']
            self.json_text.delete(1.0, tk.END)  # Clear the text area
            self.json_text.insert(tk.END, json.dumps(command_data, indent=4))  # Show JSON data
            logging.info(f"Selected command: {connectInfo.canned_messages[selected_index[0]]['name']}")

    def send_command(self):
        selected_index = self.command_listbox.curselection()
        if selected_index:
            command_name = connectInfo.canned_messages[selected_index[0]]['name']
            command = json.loads(self.json_text.get(1.0, tk.END).strip())  # Get JSON from text area
            
            # "BT709_3D_LUT_DATA" 명령이 선택된 경우에만 파일 선택
            if command_name == "BT709_3D_LUT_DATA":
                # 파일 선택 대화상자 열기
                #file_path = filedialog.askopenfilename(title="Select a file to encode")
                file_path = "./lutArrayBin"
                if file_path:
                    with open(file_path, "rb") as file:
                        file_data = file.read()
                        file_size = len(file_data)
                        # Base64로 인코딩
                        encoded_data = base64.b64encode(file_data).decode('utf-8')
                        # 인코딩된 데이터를 JSON에 추가
                        command['payload']['data'] = encoded_data
                        command['payload']['dataCount'] = file_size/2
                else:
                    messagebox.showwarning("No File Selected", "Please select a file to encode.")
                    return  # 파일이 선택되지 않으면 종료

            try:
                self.websocket_client.send_request(command)
                if command_name != "BT709_3D_LUT_DATA":
                    messagebox.showinfo("Command Sent", f"Sent command: {command_name}")
                else :
                    messagebox.showinfo("Command Sent", f"Sent command: 3DLUT")
            except json.JSONDecodeError:
                messagebox.showerror("Invalid JSON", "Please enter valid JSON data.")
                logging.error("Invalid JSON data entered.")
        else:
            messagebox.showwarning("No Selection", "Please select a command to send.")

    def start_connection(self):
        server = self.server_entry.get()  # Get server IP from the entry widget
        self.websocket_client.server = server
        self.websocket_client.auto_register = self.auto_register_var.get()  # Get auto-register checkbox value

        logging.info("Starting WebSocket connection...")
        # Start the WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self.websocket_client.connect)
        ws_thread.start()

    def update_response(self, message):
        # Insert the received message into the response text area
        self.response_text.insert(tk.END, f"Received: {message}\n")
        
        # Optionally, you can parse the message here as well
        try:
            parsed_message = json.loads(message)
            if 'data' in parsed_message:
                self.response_text.insert(tk.END, f"Data: {parsed_message['data']}\n")
            if 'message' in parsed_message:
                self.response_text.insert(tk.END, f"Message: {parsed_message['message']}\n")
        except json.JSONDecodeError:
            self.response_text.insert(tk.END, "Failed to parse message.\n")
        
        self.response_text.yview(tk.END)  # Scroll to the end

if __name__ == "__main__":
    # Create WebSocket client
    websocket_client = WebSocketClient(server_ip, App.update_response)  # Pass the update_response method

    # Create the main application window
    root = tk.Tk()
    app = App(root, websocket_client)

    # Start the Tkinter event loop
    root.mainloop()