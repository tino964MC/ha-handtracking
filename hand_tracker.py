import cv2
import mediapipe as mp
import requests
import time
import math
import os
import sys
import json
import logging
import threading
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("HandControl")

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["GLOG_minloglevel"]        = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]   = "1"
os.environ["QT_QPA_PLATFORM"]         = "offscreen"
os.environ["DISPLAY"]                 = ""
logging.getLogger("absl").setLevel(logging.ERROR)

def log_sep():
    log.info("─" * 50)

# ──────────────────────────────────────────────────────────────
# MJPEG STREAM SERVER (Multi-threaded)
# ──────────────────────────────────────────────────────────────
class PreviewState:
    def __init__(self):
        self.frame     = None
        self.lock      = threading.Lock()
        self.enabled   = False

preview = PreviewState()

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with preview.lock: frame = preview.frame
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except: pass
        elif self.path in ["/", "/index.html"]:
            html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Hand Control Live</title><style>
            body { background:#111; color:#eee; display:flex; flex-direction:column; align-items:center; font-family:sans-serif; padding:20px; }
            img { max-width:100%; border-radius:8px; background:#222; }
            h1 { font-weight:400; color:#aaa; margin-bottom:20px; }
            </style></head><body><h1>Hand Control - Live Preview</h1><img src="/stream"></body></html>""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_response(404)
            self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def start_preview_server(port=8765):
    server = ThreadedHTTPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f"Live Preview: http://<HA-IP>:{port}")
    return server

# ──────────────────────────────────────────────────────────────
# THREADED CAMERA READER (Prevents 10s latency)
# ──────────────────────────────────────────────────────────────
class ThreadedCamera:
    """
    Background thread that keeps reading frames as fast as possible.
    Only the LATEST frame is kept, effectively clearing the OpenCV buffer.
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap      = None
        self.frame    = None
        self.status   = False
        self.stopped  = False
        self.lock     = threading.Lock()
        
        self.open()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def open(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        attempt = 0
        backoff = [3, 5, 10, 20, 30]
        while not self.stopped:
            attempt += 1
            wait = backoff[min(attempt-1, len(backoff)-1)]
            log.info(f"Connecting to camera (Attempt {attempt}) → {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Still good to have
            if self.cap.isOpened():
                log.info("Camera connected ✓")
                return
            self.cap.release()
            log.warning(f"Not reachable — retrying in {wait}s")
            time.sleep(wait)

    def update(self):
        while not self.stopped:
            if self.cap is not None and self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self.frame  = frame
                        self.status = True
                else:
                    log.warning("Frame read fail — Reconnecting...")
                    self.cap.release()
                    self.status = False
                    self.open()
            else:
                time.sleep(1)

    def read(self):
        with self.lock:
            return self.status, self.frame

    def stop(self):
        self.stopped = True
        if self.cap: self.cap.release()

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
HA_OPTIONS_PATH = "/data/options.json"
IS_ADDON        = os.path.exists(HA_OPTIONS_PATH)
RTSP_URL  = os.getenv("RTSP_URL")
HA_URL    = os.getenv("HA_URL")
HA_TOKEN  = os.getenv("HA_TOKEN")
COOLDOWN_SEK  = 3.0
BEWEGUNG_MIN  = 0.008
ANALYSE_WIDTH = 240   # Reduced from 320 for 1-core CPU efficiency
COMBO_WINDOW  = 2.0

mp_hands = mp.solutions.hands

def load_config():
    config = {
        "settings": {
            "global_cooldown": COOLDOWN_SEK,
            "combo_window":    COMBO_WINDOW,
            "ha_url":          HA_URL,
            "ha_token":        HA_TOKEN,
            "rtsp_url":        RTSP_URL,
            "debug_logging":   False,
            "min_detection":   0.6,
            "min_tracking":    0.6,
            "motion_thr":      0.008
        },
        "gestures": {}, "combos": {}
    }

    if IS_ADDON:
        try:
            with open(HA_OPTIONS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            s = config["settings"]
            s["global_cooldown"] = float(data.get("global_cooldown", COOLDOWN_SEK))
            s["combo_window"]    = float(data.get("combo_window",    COMBO_WINDOW))
            s["ha_url"]          = data.get("ha_url",    "http://supervisor/core")
            s["ha_token"]        = data.get("ha_token",  "")
            s["rtsp_url"]        = data.get("rtsp_url",  "")
            s["debug_logging"]   = bool(data.get("debug_logging", False))
            s["live_preview"]    = bool(data.get("live_preview",   False))
            s["min_detection"]   = float(data.get("min_detection_confidence", 0.6))
            s["min_tracking"]    = float(data.get("min_tracking_confidence", 0.6))
            s["motion_thr"]      = float(data.get("motion_threshold", 0.008))

            if s["debug_logging"]: log.setLevel(logging.DEBUG)

            # Einzelne Gesten
            m = {"peace_sign_action":"PEACE_SIGN", "index_pointing_action":"INDEX_POINTING", 
                 "thumbs_up_action":"THUMBS_UP", "open_hand_action":"OPEN_HAND", 
                 "fist_action":"FIST", "rock_on_action":"ROCK_ON"}
            for k, n in m.items():
                v = data.get(k, "")
                if v and "," in v:
                    svc, eid = [x.strip() for x in v.split(",", 1)]
                    config["gestures"][n] = {"service":svc, "entity_id":eid, "data":{}}

            # Combos
            for i in range(1, 8):
                v = data.get(f"combo_{i}_action", "")
                if not v or v.count(",") < 2: continue
                parts = v.split(",")
                eid, svc = parts[-1].strip(), parts[-2].strip()
                ck = ",".join(parts[:-2]).strip().upper()
                if "+" in ck: config["combos"][ck] = {"service":svc, "entity_id":eid, "data":{}}

            log.info(f"Loaded {len(config['gestures'])} gestures and {len(config['combos'])} combos")
        except Exception as e: log.error(f"Config Error: {e}")
    return config

# ──────────────────────────────────────────────────────────────
# GESTURE DETECTION
# ──────────────────────────────────────────────────────────────
TIPS = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
PIPS = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
        mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]

def detect_gesture(lm):
    lms = lm.landmark
    o = [lms[t].y < lms[p].y for t, p in zip(TIPS, PIPS)]
    i_o, m_o, r_o, k_o = o
    wrist, t_tip, t_mcp = lms[0], lms[4], lms[2]
    t_o = math.hypot(t_tip.x-wrist.x, t_tip.y-wrist.y) > math.hypot(t_mcp.x-wrist.x, t_mcp.y-wrist.y)

    if i_o and m_o and r_o and k_o: return "OPEN_HAND"
    if not i_o and not m_o and not r_o and not k_o: return "FIST"
    if i_o and m_o and not r_o and not k_o: return "PEACE_SIGN"
    if i_o and not m_o and not r_o and not k_o: return "INDEX_POINTING"
    if t_o and not i_o and not m_o and not r_o and not k_o: return "THUMBS_UP"
    if i_o and k_o and not m_o and not r_o: return "ROCK_ON"
    return "UNKNOWN"

class ComboDetector:
    def __init__(self, window):
        self.window = window
        self.history = deque(maxlen=3)
        self.last_stable = None
    def update(self, g):
        now = time.time()
        if g == "UNKNOWN":
            self.last_stable = None
            return None
        if g == self.last_stable: return None
        self.last_stable = g
        self.history.append((g, now))
        if len(self.history) >= 3:
            (g1,t1),(g2,t2),(g3,t3) = list(self.history)[-3:]
            if t2-t1 <= self.window and t3-t2 <= self.window: return f"{g1}+{g2}+{g3}"
        if len(self.history) >= 2:
            (g1,t1),(g2,t2) = list(self.history)[-2:]
            if t2-t1 <= self.window: return f"{g1}+{g2}"
        return None
    def reset(self):
        self.history.clear()
        self.last_stable = None

def call_ha(service, entity_id, config, data=None):
    s = config["settings"]
    url = f"{s.get('ha_url', HA_URL)}/api/services/{service.replace('.','/',1)}"
    token = s.get('ha_token', HA_TOKEN) or os.getenv("SUPERVISOR_TOKEN", "")
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, 
                          json={"entity_id":entity_id, **(data or {})}, timeout=5)
        if r.status_code < 300:
            log.info(f"HA OK: {service} → {entity_id}")
            return True
        log.warning(f"HA Fail {r.status_code}")
    except Exception as e: log.error(f"HA Error: {e}")
    return False

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    config = load_config()
    s      = config["settings"]
    rtsp_url = s.get("rtsp_url", RTSP_URL)
    headless = IS_ADDON or not os.environ.get("DISPLAY")

    if s.get("live_preview"):
        preview.enabled = True
        start_preview_server()

    # Latency Fix: Start threaded reader
    cam = ThreadedCamera(rtsp_url)

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=s.get("min_detection", 0.6),
        min_tracking_confidence=s.get("min_tracking", 0.6)
    )
    
    detector = ComboDetector(s.get("combo_window", 2.0))
    last_trigger = {}
    frame_count  = 0
    prev_gray    = None
    last_hand_t  = time.time()
    debug_last   = time.time()
    
    log_sep()
    log.info("Hand Control - Threaded (v1.4.8)")
    log.info(f"  Headless: {headless}  |  RTSP: {rtsp_url}")
    log_sep()

    try:
        while True:
            loop_start = time.time()
            success, frame = cam.read()

            if not success:
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()
            
            if now - debug_last >= 30:
                log.info(f"[Heartbeat] Hand last seen {round(now-last_hand_t)}s ago")
                debug_last = now

            # Dynamic Frame Skipping (CPU-Friendly)
            idle_time = now - last_hand_t
            # Idle: Skip 6 frames (~3 FPS analysis) to save CPU
            # Active: Skip 2 frames (~10 FPS analysis) for responsiveness
            skip_rate = 6 if idle_time > 10 else 2
            if frame_count % skip_rate != 0: continue

            h, w = frame.shape[:2]
            scale = ANALYSE_WIDTH / w
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21,21), 0)

            # Motion Detection
            if (now - last_hand_t) > 3.0 and prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                moved = cv2.countNonZero(cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]) / gray.size
                if moved < s.get("motion_thr", 0.008):
                    prev_gray = gray
                    time.sleep(0.05)
                    continue
            prev_gray = gray

            results = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            gesture = "UNKNOWN"

            if results.multi_hand_landmarks:
                last_hand_t = now
                gesture = detect_gesture(results.multi_hand_landmarks[0])
                log.debug(f"Gesture: {gesture}")

            # Preview
            if preview.enabled:
                viz = small.copy()
                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(viz, hl, mp_hands.HAND_CONNECTIONS)
                cv2.putText(viz, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                _, buf = cv2.imencode(".jpg", viz)
                with preview.lock: preview.frame = buf.tobytes()

            # Actions
            ck = detector.update(gesture)
            if ck and ck in config["combos"]:
                if now - last_trigger.get(ck, 0) >= s.get("global_cooldown", 3.0):
                    a = config["combos"][ck]
                    log.info(f"COMBO: {ck}")
                    call_ha(a["service"], a["entity_id"], config)
                    last_trigger[ck] = now
                    detector.reset()
            elif gesture != "UNKNOWN" and gesture in config["gestures"]:
                if not any(gesture in k.split("+") for k in config["combos"]):
                    if now - last_trigger.get(gesture, 0) >= s.get("global_cooldown", 3.0):
                        a = config["gestures"][gesture]
                        log.info(f"GESTURE: {gesture}")
                        call_ha(a["service"], a["entity_id"], config)
                        last_trigger[gesture] = now

            time.sleep(max(0.1 - (time.time() - loop_start), 0.01))
    except KeyboardInterrupt: pass
    finally:
        cam.stop()
        hands.close()

if __name__ == "__main__":
    main()