import cv2
import mediapipe as mp
import requests
import time
import math
import os
import json
import numpy as np
import yaml # PyYAML wird für die Gesten-Konfiguration benötigt
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# Pfade für Home Assistant Add-on
HA_OPTIONS_PATH = "/data/options.json"
HA_CONFIG_GESTURES = "/config/hand_control_pro_gestures.yaml"
IS_ADDON = os.path.exists(HA_OPTIONS_PATH)

# Lokale Fallbacks
RTSP_URL = os.getenv("RTSP_URL")
HA_URL = os.getenv("HA_URL")
HA_TOKEN = os.getenv("HA_TOKEN")
LOCAL_GESTURES = "gestures.json"

# Globale Variablen
last_action_time = 0
last_executed_gesture = None 

# --- MEDIAPIPE INITIALISIERUNG ---
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
# ---
# Extreme CPU-Optimierung: model_complexity=0 (Lite)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_config():
    """Lädt Basis-Einstellungen und Gesten direkt vom Home Assistant Add-on Dashboard."""
    config = {
        "settings": {
            "global_cooldown": 5.0,
            "ha_url": HA_URL,
            "ha_token": HA_TOKEN,
            "rtsp_url": RTSP_URL
        },
        "gestures": {}
    }

    # 1. Einstellungen vom Supervisor laden
    if IS_ADDON:
        try:
            with open(HA_OPTIONS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                config["settings"]["global_cooldown"] = float(data.get("global_cooldown", 5.0))
                config["settings"]["ha_url"] = data.get("ha_url", "http://supervisor/core")
                config["settings"]["ha_token"] = data.get("ha_token", "")
                config["settings"]["rtsp_url"] = data.get("rtsp_url", "")

                # Gesten aus dem Dashboard parsen (Format: "service,entity_id")
                gesture_mapping = {
                    "peace_sign_action": "PEACE_SIGN",
                    "index_pointing_action": "INDEX_POINTING",
                    "thumbs_up_action": "THUMBS_UP",
                    "open_hand_action": "OPEN_HAND",
                    "fist_action": "FIST",
                    "rock_on_action": "ROCK_ON"
                }

                for key, gesture_name in gesture_mapping.items():
                    action_str = data.get(key, "")
                    if action_str and "," in action_str:
                        service, entity_id = [x.strip() for x in action_str.split(",", 1)]
                        config["gestures"][gesture_name] = {
                            "service": service,
                            "entity_id": entity_id,
                            "data": {}
                        }
                
                if config["gestures"]:
                    print(f"🟢 {len(config['gestures'])} Gesten aus dem Dashboard geladen.")
        except Exception as e:
            print(f"🔴 Add-on Options Fehler: {e}")

    # Fallback für lokale Tests (Env-Variablen)
    if not config["gestures"] and not IS_ADDON:
        # Hier könnten wir noch eine lokale config laden, falls nötig
        pass

    return config

def calculate_distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def detect_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    def is_open(tip_idx, pip_idx):
        return calculate_distance(wrist, hand_landmarks.landmark[tip_idx]) > \
               calculate_distance(wrist, hand_landmarks.landmark[pip_idx])
    
    i_o = is_open(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    m_o = is_open(mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    r_o = is_open(mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    p_o = is_open(mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    
    t_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    t_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    t_o = calculate_distance(t_tip, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]) > \
          calculate_distance(t_mcp, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])

    if i_o and m_o and r_o and p_o: return "OPEN_HAND"
    if not i_o and not m_o and not r_o and not p_o: return "FIST"
    if i_o and m_o and not r_o and not p_o: return "PEACE_SIGN"
    if i_o and not m_o and not r_o and not p_o: return "INDEX_POINTING"
    if t_o and not i_o and not m_o and not r_o and not p_o: return "THUMBS_UP"
    if i_o and p_o and not m_o and not r_o: return "ROCK_ON"
    return "UNKNOWN"

def call_ha_service(service, entity_id, config, data=None):
    settings = config.get("settings", {})
    url_base = settings.get("ha_url", HA_URL)
    token = settings.get("ha_token", HA_TOKEN)
    if token == "" and IS_ADDON:
        token = os.getenv("SUPERVISOR_TOKEN", "")
    if not service or not entity_id: return False
    domain, s_name = service.split(".")
    url = f"{url_base}/api/services/{domain}/{s_name}"
    headers = {"Authorization": f"Bearer {token}", "content-type": "application/json"}
    payload = {"entity_id": entity_id, **(data or {})}
    try:
        requests.post(url, headers=headers, json=payload, timeout=5)
        return True
    except: return False

def main():
    global last_executed_gesture, last_action_time
    config = load_config()
    settings = config.get("settings", {})
    gestures_config = config.get("gestures", {})
    cooldown_val = settings.get("global_cooldown", 5.0)
    rtsp_url = settings.get("rtsp_url", RTSP_URL)
    headless = IS_ADDON or os.getenv("DISPLAY") is None

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"🔴 Fehler: Kamera nicht erreichbar.")
        return

    print("🟢 Ultra CPU-Optimierung aktiv (Motion-Aware, 5 FPS Limit, 320px Res)")

    frame_count = 0
    prev_gray = None
    last_hand_time = 0
    while cap.isOpened():
        loop_start = time.time()
        success, frame = cap.read()
        if not success: continue
        
        frame_count += 1
        
        # 1. Dynamischer Frame-Skip
        # Wenn wir seit 10s keine Hand gesehen haben, verarbeiten wir nur jeden 20. Frame
        # Sonst jeden 10. Frame (reicht für Gesten völlig aus)
        idle_time = time.time() - last_hand_time
        skip_rate = 20 if idle_time > 10 else 10
        
        if frame_count % skip_rate != 0:
            if not headless:
                cv2.imshow('Smart Home Hand Controller Pro', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 2. Resolution Scaling (320px)
        h, w = frame.shape[:2]
        analysis_scale = 320 / w
        analysis_frame = cv2.resize(frame, (0, 0), fx=analysis_scale, fy=analysis_scale)
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 3. Simple Motion Detection Check
        # Wenn sich im Bild gar nichts bewegt, sparen wir uns MediaPipe komplett
        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            if cv2.countNonZero(thresh) < (analysis_frame.size * 0.005): # Weniger als 0.5% Änderung
                prev_gray = gray
                continue # Skip MediaPipe
        
        prev_gray = gray
        results = hands.process(cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB))
        
        curr_t = time.time()
        time_diff = curr_t - last_action_time
        in_cd = time_diff < cooldown_val
        gesture = "UNKNOWN"

        if results.multi_hand_landmarks:
            last_hand_time = curr_t # Hand gesehen, Idle-Timer zurücksetzen
            for hl in results.multi_hand_landmarks:
                gesture = detect_gesture(hl)
                if gesture != "UNKNOWN" and gesture in gestures_config:
                    if not in_cd and gesture != last_executed_gesture:
                        action = gestures_config[gesture]
                        if call_ha_service(action["service"], action["entity_id"], config, action.get("data")):
                            print(f"🎬 Geste: {gesture}")
                            last_executed_gesture = gesture
                            last_action_time = curr_t
        
        if gesture == "UNKNOWN":
            last_executed_gesture = None

        if not headless:
            ui_status = "BEREIT" if gesture == "UNKNOWN" else f"GESTE: {gesture}"
            ui_color = (0, 255, 0) if not in_cd and gesture != "UNKNOWN" else (0, 165, 255)
            cv2.putText(frame, "HAND CONTROL PRO", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, ui_status, (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_color, 1)
            cv2.imshow('Smart Home Hand Controller Pro', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        # 4. FPS Cap: Max 5 Mal pro Sekunde drosseln (reicht für Smart Home locker)
        elapsed = time.time() - loop_start
        wait = max(0.2 - elapsed, 0.02) 
        time.sleep(wait)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()