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
# Python buffers stdout if not a TTY (e.g., Docker/HA Add-on).
# Line-buffering ensures each line appears immediately.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO, # INFO level for production
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)   # Output to HA Log-Panel
    ]
)
log = logging.getLogger("HandControl")

# Suppress MediaPipe / TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"   # Silent TensorFlow
os.environ["GLOG_minloglevel"]             = "3"   # Silent MediaPipe C++
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

# --- STABILITY PARAMETERS ---
STABIL_FRAMES   = 6     # Frames required for a gesture to be considered stable
HISTORY_LEN     = 10    # Buffer length for gesture history
HALTE_DAUER     = 0.8   # Minimum duration (seconds) to hold a gesture
COOLDOWN_SEK    = 3.0   # Pause after triggering an action (overridden by config)
BEWEGUNG_MIN    = 0.008 # Minimum motion threshold (0.8% pixel change)
ANALYSE_WIDTH   = 320   # Resolution for analysis

# --- MEDIAPIPE ---
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,          # Lite model for performance
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
            "rtsp_url":  RTSP_URL
        },
        "gestures": {}
    }

    if IS_ADDON:
        try:
            with open(HA_OPTIONS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config["settings"]["global_cooldown"] = float(data.get("global_cooldown", COOLDOWN_SEK))
            config["settings"]["ha_url"]    = data.get("ha_url",    "http://supervisor/core")
            config["settings"]["ha_token"]  = data.get("ha_token",  "")
            config["settings"]["rtsp_url"]  = data.get("rtsp_url",  "")

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
            for name, action in config["gestures"].items():
                log.info(f"  {name} → {action['service']} | {action['entity_id']}")
        except Exception as e:
            log.error(f"Config error: {e}", exc_info=True)

    return config


# ──────────────────────────────────────────────────────────────
# GESTURE DETECTION (Coordinate-based)
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
    """
    Finger tip above PIP joint (smaller y) = extended.
    Thumb: tip further from wrist than MCP.
    """
    lms = lm.landmark

    # Check if 4 fingers are extended
    fingers = [lms[t].y < lms[p].y for t, p in zip(TIPS, PIPS)]
    i_o, m_o, r_o, k_o = fingers

    # Check if thumb is extended
    wrist = lms[mp_hands.HandLandmark.WRIST]
    t_tip = lms[mp_hands.HandLandmark.THUMB_TIP]
    t_mcp = lms[mp_hands.HandLandmark.THUMB_MCP]
    t_o   = math.hypot(t_tip.x - wrist.x, t_tip.y - wrist.y) > \
            math.hypot(t_mcp.x - wrist.x, t_mcp.y - wrist.y)

    # Gesture Logic
    if     i_o and  m_o and  r_o and  k_o:              return "OPEN_HAND"
    if not i_o and not m_o and not r_o and not k_o:      return "FIST"
    if     i_o and  m_o and not r_o and not k_o:         return "PEACE_SIGN"
    if     i_o and not m_o and not r_o and not k_o:      return "INDEX_POINTING"
    if     t_o and not i_o and not m_o and not r_o and not k_o: return "THUMBS_UP"
    if     i_o and  k_o and not m_o and not r_o:         return "ROCK_ON"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────
# STABILITY FILTER (Prevents accidental triggers)
# ──────────────────────────────────────────────────────────────
class GestureFilter:
    """
    Timer starts at the FIRST occurrence of a gesture.
    Triggered when: gesture appears frequently enough AND duration >= HALTE_DAUER.
    """
    def __init__(self):
        self.history      = deque(maxlen=HISTORY_LEN)
        self.timer        = {}   # gesture → timestamp of first occurrence
        self.last_fire    = {}   # gesture → timestamp of last trigger

    def check(self, gesture, cooldown):
        now = time.time()
        self.history.append(gesture)

        if gesture == "UNKNOWN":
            self.timer.clear()
            return False

        # Start timer at the first appearance of this gesture
        if gesture not in self.timer:
            self.timer = {gesture: now}   # discard others
            log.debug(f"Timer started for {gesture}")
            return False

        # Count occurrences in history
        count      = sum(1 for g in self.history if g == gesture)
        duration   = now - self.timer[gesture]
        since_last = now - self.last_fire.get(gesture, 0)

        stable = count >= STABIL_FRAMES

        if stable and duration >= HALTE_DAUER and since_last >= cooldown:
            self.last_fire[gesture] = now
            self.timer = {}   # Reset
            return True

        return False

    @property
    def active_gesture(self):
        return next(iter(self.timer), None)

    @property
    def gesture_start_time(self):
        return next(iter(self.timer.values()), None)

    def reset(self):
        self.history.clear()
        self.timer.clear()


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
            log.warning(f"HA HTTP {r.status_code} at {service} → {entity_id} | Response: {r.text[:120]}")
            return False
    except requests.exceptions.ConnectionError as e:
        log.error(f"HA not reachable: {e}")
        return False
    except Exception as e:
        log.error(f"HA Error: {e}", exc_info=True)
        return False


# ──────────────────────────────────────────────────────────────
# CAMERA WITH AUTO-RECONNECT
# ──────────────────────────────────────────────────────────────
def open_camera(rtsp_url):
    """
    Continuously attempts to open the camera stream.
    Implements increasing backoff to avoid CPU spinning.
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    attempt  = 0
    # Backoff steps in seconds
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

    gfilter       = GestureFilter()
    frame_count   = 0
    prev_gray     = None
    last_hand_t   = time.time()
    consecutive_fails = 0
    
    # Debug counters for status reports
    debug_frames_total    = 0
    debug_frames_skipped  = 0
    debug_frames_no_move  = 0
    debug_frames_mediapipe= 0
    debug_hand_found      = 0
    debug_last_report     = time.time()
    DEBUG_INTERVAL        = 15   # Report every 15s
    
    log_separator()
    log.info("Hand Control started")
    log.info(f"  Add-on mode  : {IS_ADDON}")
    log.info(f"  Headless     : {headless}")
    log.info(f"  RTSP URL     : {rtsp_url}")
    log.info(f"  Cooldown     : {cooldown_val}s")
    log.info(f"  Stability    : {STABIL_FRAMES}/{HISTORY_LEN} frames")
    log.info(f"  Hold duration: {HALTE_DAUER}s")
    log.info(f"  Motion min   : {BEWEGUNG_MIN}")
    if gestures_cfg:
        log.info(f"  Gestures ({len(gestures_cfg)}):")
        for name, action in gestures_cfg.items():
            log.info(f"    {name:20s} → {action['service']} | {action['entity_id']}")
    else:
        log.warning("  !! NO GESTURES CONFIGURED — Actions will not be triggered !!")
    log_separator()

    cap = open_camera(rtsp_url)
    log.info("Main loop running — Press Ctrl+C to stop")

    while True:
        loop_start = time.time()
        success, frame = cap.read()

        # ── RECONNECT on connection loss ──────────────────────
        if not success:
            consecutive_fails += 1
            log.warning(f"Frame error #{consecutive_fails}/5")
            if consecutive_fails >= 5:
                log.warning("Connection lost — Reconnecting...")
                cap.release()
                gfilter.reset()
                prev_gray = None
                cap = open_camera(rtsp_url)
                consecutive_fails = 0
            time.sleep(0.2)
            continue

        consecutive_fails = 0
        frame_count      += 1
        debug_frames_total += 1

        # ── Periodic Status Report ────────────────────────────
        now = time.time()
        if now - debug_last_report >= DEBUG_INTERVAL:
            log.info(
                f"[Status] total_frames={debug_frames_total} | "
                f"skipped={debug_frames_skipped} | "
                f"no_motion={debug_frames_no_move} | "
                f"mediapipe={debug_frames_mediapipe} | "
                f"hand_detected={debug_hand_found}"
            )
            # Log filter state for debugging
            history_slice = list(gfilter.history)[-5:]
            log.debug(f"[Filter] history={history_slice} | active={gfilter.active_gesture}")
            
            debug_last_report     = now
            debug_frames_skipped  = 0
            debug_frames_no_move  = 0
            debug_frames_mediapipe= 0
            debug_hand_found      = 0

        # ── Dynamic Frame Skipping ────────────────────────────
        idle_time = time.time() - last_hand_t
        skip_rate = 20 if idle_time > 10 else 6
        if frame_count % skip_rate != 0:
            debug_frames_skipped += 1
            if not headless:
                cv2.imshow("Hand Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ── Resolution scaling for analysis ───────────────────
        h, w   = frame.shape[:2]
        scale  = ANALYSE_WIDTH / w
        small  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray   = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # ── Motion Detection ──────────────────────────────────
        if prev_gray is not None:
            delta  = cv2.absdiff(prev_gray, gray)
            thr    = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            moved  = cv2.countNonZero(thr) / gray.size
            if moved < BEWEGUNG_MIN:
                debug_frames_no_move += 1
                prev_gray = gray
                if not headless:
                    cv2.imshow("Hand Control", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(max(0.2 - (time.time() - loop_start), 0.01))
                continue
            log.debug(f"Motion detected: {moved:.3f}")

        prev_gray = gray

        # ── MediaPipe Processing ─────────────────────────────
        debug_frames_mediapipe += 1
        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "UNKNOWN"
        if results.multi_hand_landmarks:
            debug_hand_found += 1
            last_hand_t = time.time()
            gesture     = detect_gesture(results.multi_hand_landmarks[0])
            log.debug(f"Hand detected → Gesture: {gesture}")

            if gfilter.check(gesture, cooldown_val):
                if gesture in gestures_cfg:
                    action = gestures_cfg[gesture]
                    log.info(f"ACTION TRIGGERED: {gesture} → {action['service']} | {action['entity_id']}")
                    ok = call_ha(action["service"], action["entity_id"],
                                 config, action.get("data"))
                    if not ok:
                        log.warning(f"HA call failed for {gesture}")
                else:
                    log.warning(f"Gesture '{gesture}' stable but NOT configured!")
            else:
                count    = sum(1 for g in gfilter.history if g == gesture)
                duration = round(time.time() - gfilter.gesture_start_time, 1) if gfilter.gesture_start_time else 0
                log.debug(f"Filter: {gesture} {count}/{STABIL_FRAMES} | held={duration}s")
        else:
            gfilter.check("UNKNOWN", cooldown_val)  # Reset timer

        # ── UI Overlay & Display ──────────────────────────────
        if not headless:
            color  = (0, 255, 0) if gesture != "UNKNOWN" else (100, 100, 100)
            text   = gesture if gesture != "UNKNOWN" else "Searching..."
            cv2.putText(frame, text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Hand Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ── FPS Cap ───────────────────────────────────────────
        time.sleep(max(0.2 - (time.time() - loop_start), 0.01))

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    log.info("Terminated.")


if __name__ == "__main__":
    main()