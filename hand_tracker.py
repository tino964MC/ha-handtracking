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
# LOGGING — alle Ausgaben sofort sichtbar im HA Log-Panel
# ──────────────────────────────────────────────────────────────
# Python puffert stdout wenn es keine TTY ist (z.B. Docker/HA Addon).
# flush=True und line-buffered sorgen dafür dass jede Zeile sofort erscheint.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)   # → HA Log-Panel
    ]
)
log = logging.getLogger("HandControl")

# MediaPipe / TF Warnungen unterdrücken (die WARNING-Spam-Zeilen oben)
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"   # TensorFlow still
os.environ["GLOG_minloglevel"]             = "3"   # MediaPipe C++ still
os.environ["MEDIAPIPE_DISABLE_GPU"]        = "1"
logging.getLogger("absl").setLevel(logging.ERROR)

def log_trennlinie():
    log.info("─" * 50)

# --- KONFIGURATION ---
HA_OPTIONS_PATH = "/data/options.json"
IS_ADDON        = os.path.exists(HA_OPTIONS_PATH)

RTSP_URL  = os.getenv("RTSP_URL")
HA_URL    = os.getenv("HA_URL")
HA_TOKEN  = os.getenv("HA_TOKEN")

# --- STABILITÄTS-PARAMETER (hier anpassen) ---
STABIL_FRAMES   = 6     # FIX: war 12 — 6 von 10 reicht, weniger Wartezeit
HISTORY_LEN     = 10    # FIX: war 15 — kleinerer Puffer füllt sich schneller
HALTE_DAUER     = 0.8   # FIX: war 1.2s — 0.8s reicht für bewusste Geste
COOLDOWN_SEK    = 5.0   # Pause nach Auslösung (überschrieben durch config)
BEWEGUNG_MIN    = 0.008 # Mindest-Bewegung für MediaPipe (0.8% Pixel-Änderung)
ANALYSE_BREITE  = 320   # Auflösung für Analyse

# --- MEDIAPIPE ---
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,          # Lite-Modell
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ──────────────────────────────────────────────────────────────
# KONFIGURATION LADEN
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

            log.info(f"Gesten geladen: {len(config['gestures'])} Stück")
            for name, action in config["gestures"].items():
                log.info(f"  {name} → {action['service']} | {action['entity_id']}")
        except Exception as e:
            log.error(f"Config-Fehler: {e}", exc_info=True)

    return config


# ──────────────────────────────────────────────────────────────
# GESTEN-ERKENNUNG  (y-basiert, stabiler als Distanz-Methode)
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
    Fingerspitze über PIP-Gelenk (y kleiner) = gestreckt.
    Daumen: Spitze weiter vom Handgelenk als MCP.
    """
    lms = lm.landmark

    # 4 Finger gestreckt?
    fingers = [lms[t].y < lms[p].y for t, p in zip(TIPS, PIPS)]
    i_o, m_o, r_o, k_o = fingers

    # Daumen gestreckt? (x-Achse, Abstand von Handgelenk)
    wrist = lms[mp_hands.HandLandmark.WRIST]
    t_tip = lms[mp_hands.HandLandmark.THUMB_TIP]
    t_mcp = lms[mp_hands.HandLandmark.THUMB_MCP]
    t_o   = math.hypot(t_tip.x - wrist.x, t_tip.y - wrist.y) > \
            math.hypot(t_mcp.x - wrist.x, t_mcp.y - wrist.y)

    # Gesten-Regeln
    if     i_o and  m_o and  r_o and  k_o:              return "OPEN_HAND"
    if not i_o and not m_o and not r_o and not k_o:      return "FIST"
    if     i_o and  m_o and not r_o and not k_o:         return "PEACE_SIGN"
    if     i_o and not m_o and not r_o and not k_o:      return "INDEX_POINTING"
    if     t_o and not i_o and not m_o and not r_o and not k_o: return "THUMBS_UP"
    if     i_o and  k_o and not m_o and not r_o:         return "ROCK_ON"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────
# STABILITÄTS-FILTER  (verhindert Fehlauslösungen beim Vorbeigehen)
# ──────────────────────────────────────────────────────────────
class GestureFilter:
    """
    Timer startet beim ERSTEN Auftreten der Geste.
    Auslösung wenn: Geste kommt oft genug vor UND Timer >= HALTE_DAUER.
    So ist die Haltezeit nicht nach dem Stabilitätspuffer, sondern parallel.
    """
    def __init__(self):
        self.history      = deque(maxlen=HISTORY_LEN)
        self.timer        = {}   # geste → timestamp erstes Auftreten
        self.letzter_fire = {}   # geste → timestamp letztes Auslösen

    def check(self, geste, cooldown):
        now = time.time()
        self.history.append(geste)

        if geste == "UNKNOWN":
            self.timer.clear()
            return False

        # Timer starten beim ersten Auftreten dieser Geste
        if geste not in self.timer:
            self.timer = {geste: now}   # alles andere verwerfen
            log.debug(f"Timer gestartet für {geste}")
            return False

        # Wie oft kommt die Geste im Puffer vor?
        count     = sum(1 for g in self.history if g == geste)
        gehalten  = now - self.timer[geste]
        seit_letzt = now - self.letzter_fire.get(geste, 0)

        stabil = count >= STABIL_FRAMES

        if stabil and gehalten >= HALTE_DAUER and seit_letzt >= cooldown:
            self.letzter_fire[geste] = now
            self.timer = {}   # Reset
            return True

        return False

    @property
    def aktive(self):
        return next(iter(self.timer), None)

    @property
    def geste_start(self):
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
            log.warning(f"HA HTTP {r.status_code} bei {service} → {entity_id} | Antwort: {r.text[:120]}")
            return False
    except requests.exceptions.ConnectionError as e:
        log.error(f"HA nicht erreichbar: {e}")
        return False
    except Exception as e:
        log.error(f"HA Fehler: {e}", exc_info=True)
        return False


# ──────────────────────────────────────────────────────────────
# KAMERA MIT ENDLOSEM AUTO-RECONNECT
# Versucht für immer — Kamera kann beliebig lange aus sein
# ──────────────────────────────────────────────────────────────
def open_camera(rtsp_url):
    """
    Versucht endlos die Kamera zu öffnen.
    Wartet zwischen Versuchen immer länger (max 60s),
    damit kein CPU-Spin bei dauerhaft offline Kamera.
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    attempt  = 0
    # Backoff-Stufen in Sekunden: 3, 5, 10, 20, 30, 60, 60, 60 ...
    backoff  = [3, 5, 10, 20, 30, 60]

    while True:
        attempt += 1
        wait = backoff[min(attempt - 1, len(backoff) - 1)]
        log.info(f"Kamera-Verbindung Versuch {attempt} → {rtsp_url}")
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                log.info("Kamera verbunden ✓")
                return cap
            cap.release()
        except Exception as e:
            log.error(f"Kamera-Fehler: {e}")
        log.warning(f"Kamera nicht erreichbar — nächster Versuch in {wait}s")
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
    last_hand_t   = time.time()   # FIX: nicht 0 — sonst skip_rate sofort 20
    consecutive_fails = 0
    
    # Debug-Zähler für regelmäßige Status-Ausgabe
    debug_frames_total    = 0
    debug_frames_skipped  = 0
    debug_frames_no_move  = 0
    debug_frames_mediapipe= 0
    debug_hand_found      = 0
    debug_last_report     = time.time()
    DEBUG_INTERVAL        = 10   # alle 10s Status ausgeben

    log_trennlinie()
    log.info("Hand Control startet")
    log.info(f"  Addon-Modus  : {IS_ADDON}")
    log.info(f"  Headless     : {headless}")
    log.info(f"  RTSP-URL     : {rtsp_url}")
    log.info(f"  Cooldown     : {cooldown_val}s")
    log.info(f"  Stabil-Frames: {STABIL_FRAMES}/{HISTORY_LEN}")
    log.info(f"  Halte-Dauer  : {HALTE_DAUER}s")
    log.info(f"  Bewegung-Min : {BEWEGUNG_MIN}")
    if gestures_cfg:
        log.info(f"  Gesten ({len(gestures_cfg)}):")
        for name, action in gestures_cfg.items():
            log.info(f"    {name:20s} → {action['service']} | {action['entity_id']}")
    else:
        log.warning("  !! KEINE GESTEN KONFIGURIERT — nichts wird ausgelöst !!")
    log_trennlinie()

    # open_camera läuft endlos bis Verbindung steht
    cap = open_camera(rtsp_url)
    log.info("Hauptschleife läuft — Strg+C zum Beenden")

    while True:
        loop_start = time.time()
        success, frame = cap.read()

        # ── RECONNECT bei Verbindungsabbruch ──────────────────
        if not success:
            consecutive_fails += 1
            log.warning(f"Frame-Fehler #{consecutive_fails}/5")
            if consecutive_fails >= 5:
                log.warning("Verbindung verloren — starte Reconnect ...")
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

        # ── Regelmäßiger Status-Report ────────────────────────
        now = time.time()
        if now - debug_last_report >= DEBUG_INTERVAL:
            log.info(
                f"[Status] Frames total={debug_frames_total} | "
                f"skipped={debug_frames_skipped} | "
                f"kein-move={debug_frames_no_move} | "
                f"mediapipe={debug_frames_mediapipe} | "
                f"hand={debug_hand_found}"
            )
            log.info(
                f"[Filter] history={list(gfilter.history)[-5:]} | "
                f"aktiv={gfilter.aktive} | "
                f"start={round(now - gfilter.geste_start, 1) if gfilter.geste_start else 'None'}s"
            )
            debug_last_report     = now
            debug_frames_skipped  = 0
            debug_frames_no_move  = 0
            debug_frames_mediapipe= 0
            debug_hand_found      = 0

        # ── Dynamischer Frame-Skip ────────────────────────────
        idle_t    = time.time() - last_hand_t
        skip_rate = 20 if idle_t > 10 else 6   # FIX: war 10, jetzt 6 — schneller reagieren
        if frame_count % skip_rate != 0:
            debug_frames_skipped += 1
            if not headless:
                cv2.imshow("Hand Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ── Analyse-Frame verkleinern ─────────────────────────
        h, w   = frame.shape[:2]
        scale  = ANALYSE_BREITE / w
        small  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray   = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # ── Bewegungsdetektion ────────────────────────────────
        if prev_gray is not None:
            delta  = cv2.absdiff(prev_gray, gray)
            thr    = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            moved  = cv2.countNonZero(thr) / gray.size
            if moved < BEWEGUNG_MIN:
                debug_frames_no_move += 1
                prev_gray = gray
                # KEIN gfilter.update("UNKNOWN") hier!
                # Wenn Hand still gehalten wird → Bewegung = 0 → aber Geste soll trotzdem zählen
                # Nur überspringen, Filter in Ruhe lassen
                if not headless:
                    cv2.imshow("Hand Control", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(max(0.2 - (time.time() - loop_start), 0.01))
                continue
            log.debug(f"Bewegung erkannt: {moved:.3f} (min={BEWEGUNG_MIN})")

        prev_gray = gray

        # ── MediaPipe ─────────────────────────────────────────
        debug_frames_mediapipe += 1
        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "UNKNOWN"
        if results.multi_hand_landmarks:
            debug_hand_found += 1
            last_hand_t = time.time()
            gesture     = detect_gesture(results.multi_hand_landmarks[0])
            log.debug(f"Hand erkannt → Geste: {gesture}")

            if gfilter.check(gesture, cooldown_val):
                if gesture in gestures_cfg:
                    action = gestures_cfg[gesture]
                    log.info(f"AUSLÖSUNG: {gesture} → {action['service']} | {action['entity_id']}")
                    ok = call_ha(action["service"], action["entity_id"],
                                 config, action.get("data"))
                    if not ok:
                        log.warning(f"HA-Aufruf fehlgeschlagen für {gesture}")
                else:
                    log.warning(
                        f"Geste '{gesture}' stabil aber NICHT konfiguriert! "
                        f"Konfiguriert: {list(gestures_cfg.keys())}"
                    )
            else:
                count    = sum(1 for g in gfilter.history if g == gesture)
                gehalten = round(time.time() - gfilter.geste_start, 1) if gfilter.geste_start else 0
                log.debug(
                    f"Filter: {gesture} {count}/{STABIL_FRAMES} frames | "
                    f"gehalten={gehalten}s/{HALTE_DAUER}s | aktiv={gfilter.aktive}"
                )
        else:
            log.debug("Keine Hand im Bild")
            gfilter.check("UNKNOWN", cooldown_val)  # Timer reset

            cv2.imshow("Hand Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ── FPS-Cap ───────────────────────────────────────────
        time.sleep(max(0.2 - (time.time() - loop_start), 0.01))

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    log.info("Beendet.")


if __name__ == "__main__":
    main()