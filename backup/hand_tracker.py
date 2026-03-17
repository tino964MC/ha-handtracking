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
# LIVE PREVIEW — MJPEG Stream Server
# Aufrufbar unter http://<ha-ip>:8765
# In HA als Generic Camera: stream_source: http://localhost:8765/stream
# ──────────────────────────────────────────────────────────────
class PreviewState:
    """Shared state zwischen Main-Loop und HTTP-Server."""
    def __init__(self):
        self.frame     = None   # aktuelles JPEG-Bytes
        self.lock      = threading.Lock()
        self.enabled   = False

preview = PreviewState()

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # HTTP-Requests nicht ins Log

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with preview.lock:
                        frame = preview.frame
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)   # ~10fps im Preview
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/" or self.path == "/index.html":
            html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Hand Control Live</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:#111; display:flex; flex-direction:column;
         align-items:center; min-height:100vh;
         font-family:sans-serif; color:#eee; padding: 16px; }
  h1 { margin: 16px 0 12px; font-size:1.1rem; font-weight:400;
       color:#aaa; letter-spacing:.05em; }
  .stream-wrap { width:100%; max-width:800px; }
  img { width:100%; height:auto; border-radius:8px;
        display:block; background:#222; }
  p  { color:#555; font-size:.8rem; margin-top:10px; text-align:center; }
</style></head>
<body>
<h1>Hand Control - Live Preview</h1>
<div class="stream-wrap">
  <img src="/stream" alt="Live stream">
</div>
<p>Landmarks and detected gesture shown live</p>
</body></html>""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_response(404)
            self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread to prevent blocking."""
    daemon_threads = True

def start_preview_server(port=8765):
    server = ThreadedHTTPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f"Live Preview: http://<HA-IP>:{port}  |  Stream: http://<HA-IP>:{port}/stream")
    return server

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
ANALYSE_WIDTH = 320
COMBO_WINDOW  = 2.0   # Sekunden zwischen zwei Gesten für eine Combo

mp_hands = mp.solutions.hands
# hands will be initialized in main() with configurable thresholds


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
        "gestures": {},   # einzelne Gesten
        "combos":   {}    # Combo-Gesten: "GESTE1+GESTE2" → action
    }

    if IS_ADDON:
        try:
            with open(HA_OPTIONS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            config["settings"]["global_cooldown"] = float(data.get("global_cooldown", COOLDOWN_SEK))
            config["settings"]["combo_window"]    = float(data.get("combo_window",    COMBO_WINDOW))
            config["settings"]["ha_url"]          = data.get("ha_url",    "http://supervisor/core")
            config["settings"]["ha_token"]        = data.get("ha_token",  "")
            config["settings"]["rtsp_url"]        = data.get("rtsp_url",  "")
            config["settings"]["debug_logging"]   = bool(data.get("debug_logging", False))
            config["settings"]["live_preview"]    = bool(data.get("live_preview",   False))
            config["settings"]["min_detection"]   = float(data.get("min_detection_confidence", 0.6))
            config["settings"]["min_tracking"]    = float(data.get("min_tracking_confidence", 0.6))
            config["settings"]["motion_thr"]      = float(data.get("motion_threshold", 0.008))

            if config["settings"]["debug_logging"]:
                log.setLevel(logging.DEBUG)
                log.info("Debug logging ENABLED")

            # ── Einzelne Gesten ───────────────────────────────
            single_map = {
                "peace_sign_action":     "PEACE_SIGN",
                "index_pointing_action": "INDEX_POINTING",
                "thumbs_up_action":      "THUMBS_UP",
                "open_hand_action":      "OPEN_HAND",
                "fist_action":           "FIST",
                "rock_on_action":        "ROCK_ON"
            }
            for key, name in single_map.items():
                val = data.get(key, "")
                if val and "," in val:
                    svc, eid = [x.strip() for x in val.split(",", 1)]
                    config["gestures"][name] = {"service": svc, "entity_id": eid, "data": {}}

            # ── Combos ────────────────────────────────────────
            # Format: "GESTE1+GESTE2,service,entity_id"       (2er)
            # Format: "GESTE1+GESTE2+GESTE3,service,entity_id" (3er)
            for i in range(1, 8):   # bis zu 7 Combos
                val = data.get(f"combo_{i}_action", "")
                if not val or val.count(",") < 1:
                    continue
                # Letztes Komma trennt entity_id, vorletztes den service
                # Alles davor ist der Combo-Key (kann + enthalten)
                parts = val.split(",")
                if len(parts) < 3:
                    log.warning(f"Combo {i}: braucht Format 'GESTE1+GESTE2,service,entity' — übersprungen")
                    continue
                # entity_id = letzter Teil, service = vorletzter, combo_key = Rest
                eid       = parts[-1].strip()
                svc       = parts[-2].strip()
                combo_key = ",".join(parts[:-2]).strip().upper()

                gesten = combo_key.split("+")
                if len(gesten) < 2:
                    log.warning(f"Combo {i}: kein '+' in '{combo_key}' — übersprungen")
                    continue
                if len(gesten) > 3:
                    log.warning(f"Combo {i}: maximal 3 Gesten erlaubt — übersprungen")
                    continue

                config["combos"][combo_key] = {"service": svc, "entity_id": eid, "data": {}}

            log.info(f"Einzelgesten geladen : {len(config['gestures'])}")
            for n, a in config["gestures"].items():
                log.info(f"  {n:20s} → {a['service']} | {a['entity_id']}")

            log.info(f"Combos geladen       : {len(config['combos'])}")
            for n, a in config["combos"].items():
                log.info(f"  {n:30s} → {a['service']} | {a['entity_id']}")

        except Exception as e:
            log.error(f"Config-Fehler: {e}", exc_info=True)

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
    lms     = lm.landmark
    fingers = [lms[t].y < lms[p].y for t, p in zip(TIPS, PIPS)]
    i_o, m_o, r_o, k_o = fingers

    wrist = lms[mp_hands.HandLandmark.WRIST]
    t_tip = lms[mp_hands.HandLandmark.THUMB_TIP]
    t_mcp = lms[mp_hands.HandLandmark.THUMB_MCP]
    t_o   = math.hypot(t_tip.x - wrist.x, t_tip.y - wrist.y) > \
            math.hypot(t_mcp.x - wrist.x, t_mcp.y - wrist.y)

    if     i_o and  m_o and  r_o and  k_o:                          return "OPEN_HAND"
    if not i_o and not m_o and not r_o and not k_o:                  return "FIST"
    if     i_o and  m_o and not r_o and not k_o:                     return "PEACE_SIGN"
    if     i_o and not m_o and not r_o and not k_o:                  return "INDEX_POINTING"
    if     t_o and not i_o and not m_o and not r_o and not k_o:      return "THUMBS_UP"
    if     i_o and  k_o and not m_o and not r_o:                     return "ROCK_ON"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────
# COMBO DETECTOR
# Merkt sich die letzte N erkannte Geste mit Timestamp
# Prüft ob eine Combo-Sequenz erfüllt wurde
# ──────────────────────────────────────────────────────────────
class ComboDetector:
    def __init__(self, combo_window):
        self.window      = combo_window
        # Ring-Buffer: (geste, timestamp) der letzten 3 stabilen Gesten
        self.history     = deque(maxlen=3)
        self.last_stable = None   # letzte stabile Geste (kein UNKNOWN)

    def update(self, gesture):
        """
        Aufgerufen bei jeder erkannten Geste.
        Gibt die Combo-Key zurück wenn eine Combo erkannt wurde, sonst None.
        """
        now = time.time()

        if gesture == "UNKNOWN":
            # UNKNOWN nach einer Geste = Geste abgeschlossen → bereit für nächste
            if self.last_stable is not None:
                self.last_stable = None
            return None

        # Geste stabil halten — erst in History wenn sie sich von letzter unterscheidet
        if gesture == self.last_stable:
            return None   # gleiche Geste wird gehalten — nicht als neue zählen

        # Neue Geste erkannt
        self.last_stable = gesture
        self.history.append((gesture, now))
        log.debug(f"Combo-History: {[(g, round(now-t,1)) for g,t in self.history]}")

        # ── 3er-Combo zuerst prüfen (spezifischer) ────────────
        if len(self.history) >= 3:
            g1, t1 = self.history[-3]
            g2, t2 = self.history[-2]
            g3, t3 = self.history[-1]
            # Alle 3 müssen innerhalb combo_window voneinander liegen
            if (t2 - t1) <= self.window and (t3 - t2) <= self.window:
                combo_key = f"{g1}+{g2}+{g3}"
                log.debug(f"3er-Combo-Kandidat: {combo_key}")
                return combo_key

        # ── 2er-Combo prüfen ──────────────────────────────────
        if len(self.history) >= 2:
            g1, t1 = self.history[-2]
            g2, t2 = self.history[-1]
            if (t2 - t1) <= self.window:
                combo_key = f"{g1}+{g2}"
                log.debug(f"2er-Combo-Kandidat: {combo_key}")
                return combo_key

        return None

    def reset(self):
        self.history.clear()
        self.last_stable = None


# ──────────────────────────────────────────────────────────────
# HOME ASSISTANT
# ──────────────────────────────────────────────────────────────
def call_ha(service, entity_id, config, data=None):
    settings = config.get("settings", {})
    url_base = settings.get("ha_url",   HA_URL)
    token    = settings.get("ha_token", HA_TOKEN)

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
        log.warning(f"HA HTTP {r.status_code}: {r.text[:100]}")
        return False
    except Exception as e:
        log.error(f"HA Fehler: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────
def open_camera(rtsp_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    attempt = 0
    backoff = [3, 5, 10, 20, 30, 60]
    while True:
        attempt += 1
        wait = backoff[min(attempt - 1, len(backoff) - 1)]
        log.info(f"Verbinde Kamera (Versuch {attempt}) ...")
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            log.info("Kamera verbunden ✓")
            return cap
        cap.release()
        log.warning(f"Nicht erreichbar — warte {wait}s")
        time.sleep(wait)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    config       = load_config()
    settings     = config["settings"]
    gestures_cfg = config["gestures"]
    combos_cfg   = config["combos"]
    cooldown_val = settings.get("global_cooldown", COOLDOWN_SEK)
    combo_window = settings.get("combo_window",    COMBO_WINDOW)
    rtsp_url     = settings.get("rtsp_url", RTSP_URL)
    live_preview = settings.get("live_preview", False)
    headless     = IS_ADDON or not os.environ.get("DISPLAY")

    # Re-initialize hands with configured thresholds
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=settings.get("min_detection", 0.6),
        min_tracking_confidence=settings.get("min_tracking", 0.6)
    )

    motion_threshold = settings.get("motion_thr", BEWEGUNG_MIN)

    if live_preview:
        preview.enabled = True
        start_preview_server(port=8765)

    last_trigger_times = {}   # geste/combo → letzter fire timestamp
    combo_detector     = ComboDetector(combo_window)

    frame_count       = 0
    prev_gray         = None
    last_hand_t       = time.time()
    consecutive_fails = 0
    debug_last_report = time.time()
    DEBUG_INTERVAL    = 30

    log_sep()
    log.info("Hand Control gestartet")
    log.info(f"  Headless     : {headless}")
    log.info(f"  RTSP         : {rtsp_url}")
    log.info(f"  Cooldown     : {cooldown_val}s")
    log.info(f"  Combo-Fenster: {combo_window}s")
    log_sep()

    cap = open_camera(rtsp_url)

    while True:
        loop_start = time.time()
        success, frame = cap.read()

        # ── Reconnect ─────────────────────────────────────────
        if not success:
            consecutive_fails += 1
            if consecutive_fails >= 5:
                log.warning("Verbindung verloren — Reconnect ...")
                cap.release()
                prev_gray = None
                cap = open_camera(rtsp_url)
                consecutive_fails = 0
            time.sleep(0.5)
            continue

        consecutive_fails = 0
        frame_count      += 1

        # ── Heartbeat ─────────────────────────────────────────
        now = time.time()
        if now - debug_last_report >= DEBUG_INTERVAL:
            log.info(f"[Heartbeat] läuft | Hand zuletzt vor {round(now - last_hand_t)}s")
            debug_last_report = now

        # ── Frame-Skip ────────────────────────────────────────
        idle_time = now - last_hand_t
        skip_rate = 4 if idle_time > 5 else 2
        if frame_count % skip_rate != 0:
            continue

        # ── Analyse-Frame ─────────────────────────────────────
        h, w  = frame.shape[:2]
        scale = ANALYSE_WIDTH / w
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # ── Bewegungsdetektion ────────────────────────────────
        hand_kuerzelich = (time.time() - last_hand_t) < 3.0
        if not hand_kuerzelich and prev_gray is not None:
            delta  = cv2.absdiff(prev_gray, gray)
            moved  = cv2.countNonZero(cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]) / gray.size
            if moved < motion_threshold:
                prev_gray = gray
                time.sleep(max(0.05 - (time.time() - loop_start), 0.01))
                continue

        prev_gray = gray

        # ── MediaPipe ─────────────────────────────────────────
        results = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        gesture = "UNKNOWN"

        if results.multi_hand_landmarks:
            last_hand_t = time.time()
            gesture     = detect_gesture(results.multi_hand_landmarks[0])
            log.debug(f"Geste: {gesture}")

        # ── Live Preview Frame aufbauen ────────────────────────
        if preview.enabled:
            viz = small.copy()

            # Landmarks zeichnen
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        viz, hl, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 200, 100), thickness=2, circle_radius=3),
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=1)
                    )

            # Geste + Combo-History als Text
            combo_hist = [g for g, _ in combo_detector.history]
            farbe = (0, 220, 80) if gesture != "UNKNOWN" else (160, 160, 160)
            cv2.putText(viz, gesture, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, farbe, 2)
            if combo_hist:
                cv2.putText(viz, " + ".join(combo_hist), (8, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 80), 1)

            # Als JPEG kodieren und in shared state schreiben
            # Hochskalieren für bessere Darstellung im Browser
            viz_large = cv2.resize(viz, (640, int(640 * viz.shape[0] / viz.shape[1])))
            ok, buf = cv2.imencode(".jpg", viz_large, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with preview.lock:
                    preview.frame = buf.tobytes()

        # ── Combo-Check ───────────────────────────────────────
        combo_key = combo_detector.update(gesture)

        if combo_key and combo_key in combos_cfg:
            now        = time.time()
            last_fire  = last_trigger_times.get(combo_key, 0)
            if now - last_fire >= cooldown_val:
                action = combos_cfg[combo_key]
                log.info(f"COMBO: {combo_key} → {action['service']} | {action['entity_id']}")
                call_ha(action["service"], action["entity_id"], config, action.get("data"))
                last_trigger_times[combo_key] = now
                combo_detector.reset()   # History leeren nach Combo-Fire
            else:
                log.debug(f"Combo {combo_key} erkannt aber Cooldown aktiv")

        # ── Einzelgesten-Check (nur wenn keine Combo gefeuert) ─
        elif gesture != "UNKNOWN" and gesture in gestures_cfg:
            # Prüfe ob diese Geste Teil einer konfigurierten Combo sein könnte
            # Wenn ja → warte kurz bevor wir einzeln auslösen
            geste_in_combo = any(
                gesture in combo_key.split("+")
                for combo_key in combos_cfg
            )

            if geste_in_combo:
                # Geste könnte noch eine Combo einleiten → kleines Delay
                # Erst nach combo_window ohne Folge-Geste als einzeln auslösen
                pass   # combo_detector entscheidet — wir feuern nicht sofort
            else:
                now       = time.time()
                last_fire = last_trigger_times.get(gesture, 0)
                if now - last_fire >= cooldown_val:
                    action = gestures_cfg[gesture]
                    log.info(f"GESTE: {gesture} → {action['service']} | {action['entity_id']}")
                    call_ha(action["service"], action["entity_id"], config, action.get("data"))
                    last_trigger_times[gesture] = now

        # ── FPS-Cap ───────────────────────────────────────────
        time.sleep(max(0.1 - (time.time() - loop_start), 0.01))

    cap.release()
    if 'hands' in locals():
        hands.close()


if __name__ == "__main__":
    main()