import cv2
import mediapipe as mp
import requests
import time
import math
import os
import sys
import json
import logging
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# LOGGING — Direct output to HA log panel
# ──────────────────────────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("HandControl")

# Suppress MediaPipe / TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["GLOG_minloglevel"]             = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]        = "1"
logging.getLogger("absl").setLevel(logging.ERROR)

def log_separator():
    log.info("─" * 50)

# --- CONFIGURATION ---
HA_OPTIONS_PATH = "/data/options.json"
IS_ADDON        = os.path.exists(HA_OPTIONS_PATH)

RTSP_URL  = os.getenv("RTSP_URL")
HA_URL    = os.getenv("HA_URL")
HA_TOKEN  = os.getenv("HA_TOKEN")

# --- PARAMETERS ---
COOLDOWN_SEK    = 3.0   # Pause after triggering an action
BEWEGUNG_MIN    = 0.008 # Minimum motion threshold
ANALYSE_WIDTH   = 320   # Resolution for analysis

# --- MEDIAPIPE ---
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ──────────────────────────────────────────────────────────────
# CONFIGURATION LOADING
# ──────────────────────────────────────────────────────────────
def load_config():
    config = {
        "settings": {
            "global_cooldown": COOLDOWN_SEK,
            "ha_url":    HA_URL,
            "ha_token":  HA_TOKEN,
            "rtsp_url":  RTSP_URL,
            "debug_logging": False
        },
        "gestures": {}
    }

    if IS_ADDON:
        try:
            with open(HA_OPTIONS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config["settings"]["global_cooldown"] = float(data.get("global_cooldown", COOLDOWN_SEK))
            config["settings"]["ha_url"]         = data.get("ha_url",    "http://supervisor/core")
            config["settings"]["ha_token"]       = data.get("ha_token",  "")
            config["settings"]["rtsp_url"]       = data.get("rtsp_url",  "")
            config["settings"]["debug_logging"]  = bool(data.get("debug_logging", False))

            if config["settings"]["debug_logging"]:
                log.setLevel(logging.DEBUG)
                log.info("Debug logging ENABLED")

            gesture_mapping = {
                "peace_sign_action":    "PEACE_SIGN",
                "index_pointing_action":"INDEX_POINTING",
                "thumbs_up_action":     "THUMBS_UP",
                "open_hand_action":     "OPEN_HAND",
                "fist_action":          "FIST",
                "rock_on_action":       "ROCK_ON"
            }
            for key, name in gesture_mapping.items():
                val = data.get(key, "")
                if val and "," in val:
                    svc, eid = [x.strip() for x in val.split(",", 1)]
                    config["gestures"][name] = {"service": svc, "entity_id": eid, "data": {}}

            log.info(f"Loaded {len(config['gestures'])} gesture actions")
        except Exception as e:
            log.error(f"Config error: {e}", exc_info=True)

    return config


# ──────────────────────────────────────────────────────────────
# GESTURE DETECTION
# ──────────────────────────────────────────────────────────────
TIPS = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]
PIPS = [
    mp_hands.HandLandmark.INDEX_FINGER_PIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    mp_hands.HandLandmark.RING_FINGER_PIP,
    mp_hands.HandLandmark.PINKY_PIP,
]

def detect_gesture(lm):
    lms = lm.landmark
    fingers = [lms[t].y < lms[p].y for t, p in zip(TIPS, PIPS)]
    i_o, m_o, r_o, k_o = fingers

    wrist = lms[mp_hands.HandLandmark.WRIST]
    t_tip = lms[mp_hands.HandLandmark.THUMB_TIP]
    t_mcp = lms[mp_hands.HandLandmark.THUMB_MCP]
    t_o   = math.hypot(t_tip.x - wrist.x, t_tip.y - wrist.y) > \
            math.hypot(t_mcp.x - wrist.x, t_mcp.y - wrist.y)

    if     i_o and  m_o and  r_o and  k_o:              return "OPEN_HAND"
    if not i_o and not m_o and not r_o and not k_o:      return "FIST"
    if     i_o and  m_o and not r_o and not k_o:         return "PEACE_SIGN"
    if     i_o and not m_o and not r_o and not k_o:      return "INDEX_POINTING"
    if     t_o and not i_o and not m_o and not r_o and not k_o: return "THUMBS_UP"
    if     i_o and  k_o and not m_o and not r_o:         return "ROCK_ON"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────
# HOME ASSISTANT API
# ──────────────────────────────────────────────────────────────
def call_ha(service, entity_id, config, data=None):
    settings  = config.get("settings", {})
    url_base  = settings.get("ha_url",   HA_URL)
    token     = settings.get("ha_token", HA_TOKEN)

    if not token and IS_ADDON:
        token = os.getenv("SUPERVISOR_TOKEN", "")
    if not service or not entity_id:
        return False

    domain, sname = service.split(".", 1)
    url     = f"{url_base}/api/services/{domain}/{sname}"
    headers = {"Authorization": f"Bearer {token}", "content-type": "application/json"}
    payload = {"entity_id": entity_id, **(data or {})}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=5)
        if r.status_code < 300:
            log.info(f"HA OK [{r.status_code}] {service} → {entity_id}")
            return True
        else:
            log.warning(f"HA HTTP {r.status_code} at {service} → {entity_id}")
            return False
    except Exception as e:
        log.error(f"HA Error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# CAMERA WITH AUTO-RECONNECT
# ──────────────────────────────────────────────────────────────
def open_camera(rtsp_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    attempt  = 0
    backoff  = [3, 5, 10, 20, 30, 60]

    while True:
        attempt += 1
        wait = backoff[min(attempt - 1, len(backoff) - 1)]
        log.info(f"Connecting to camera (Attempt {attempt}) → {rtsp_url}")
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                log.info("Camera connected ✓")
                return cap
            cap.release()
        except Exception as e:
            log.error(f"Camera error: {e}")
        log.warning(f"Camera not reachable — Retrying in {wait}s")
        time.sleep(wait)


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────
def main():
    config        = load_config()
    settings      = config["settings"]
    gestures_cfg  = config["gestures"]
    cooldown_val  = settings.get("global_cooldown", COOLDOWN_SEK)
    rtsp_url      = settings.get("rtsp_url", RTSP_URL)
    headless      = IS_ADDON or not os.environ.get("DISPLAY")

    # Simple cooldown map for instant triggering
    last_trigger_times = {}

    frame_count   = 0
    prev_gray     = None
    last_hand_t   = time.time()
    consecutive_fails = 0
    
    debug_last_report     = time.time()
    DEBUG_INTERVAL        = 30
    
    log_separator()
    log.info("Hand Control started (Instant Reaction v1.3.2)")
    log.info(f"  Add-on mode  : {IS_ADDON}")
    log.info(f"  Headless     : {headless}")
    log.info(f"  RTSP URL     : {rtsp_url}")
    log.info(f"  Cooldown     : {cooldown_val}s")
    log.info(f"  Reaction     : INSTANT")
    log_separator()

    cap = open_camera(rtsp_url)

    while True:
        loop_start = time.time()
        success, frame = cap.read()

        if not success:
            consecutive_fails += 1
            if consecutive_fails >= 5:
                log.warning("Connection lost — Reconnecting...")
                cap.release()
                prev_gray = None
                cap = open_camera(rtsp_url)
                consecutive_fails = 0
            time.sleep(0.5)
            continue

        consecutive_fails = 0
        frame_count      += 1

        # Periodic status heartbeat
        now = time.time()
        if now - debug_last_report >= DEBUG_INTERVAL:
            log.info("[Heartbeat] System running | Hand detected recently: %s" % (now - last_hand_t < 10))
            debug_last_report = now

        # Dynamic Frame Skipping for high responsiveness
        idle_time = now - last_hand_t
        skip_rate = 4 if idle_time > 5 else 2
        
        if frame_count % skip_rate != 0:
            if not headless:
                cv2.imshow("Hand Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        h, w   = frame.shape[:2]
        scale  = ANALYSE_WIDTH / w
        small  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray   = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # Motion check
        if prev_gray is not None:
            delta  = cv2.absdiff(prev_gray, gray)
            thr    = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            moved  = cv2.countNonZero(thr) / gray.size
            if moved < BEWEGUNG_MIN:
                prev_gray = gray
                if not headless:
                    cv2.imshow("Hand Control", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                time.sleep(max(0.05 - (time.time() - loop_start), 0.01))
                continue

        prev_gray = gray
        rgb       = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb)

        gesture = "UNKNOWN"
        if results.multi_hand_landmarks:
            last_hand_t = time.time()
            gesture     = detect_gesture(results.multi_hand_landmarks[0])
            log.debug(f"Hand detected → Gesture: {gesture}")

            if gesture in gestures_cfg:
                now = time.time()
                last_fire = last_trigger_times.get(gesture, 0)
                
                # Instant trigger if cooldown expired
                if now - last_fire >= cooldown_val:
                    action = gestures_cfg[gesture]
                    log.info(f"INSTANT ACTION: {gesture} → {action['service']}")
                    call_ha(action["service"], action["entity_id"], config, action.get("data"))
                    last_trigger_times[gesture] = now
                else:
                    log.debug(f"Gesture {gesture} detected but cooldown active (wait {round(cooldown_val - (now - last_fire), 1)}s)")

        # Display
        if not headless:
            color  = (0, 255, 0) if gesture != "UNKNOWN" else (100, 100, 100)
            cv2.putText(frame, gesture, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Hand Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Maintain 10 FPS analysis loop
        time.sleep(max(0.1 - (time.time() - loop_start), 0.01))

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()