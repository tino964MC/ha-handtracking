# Hand Tracking Home Assistant Controller

Dieses Skript verbindet sich mit dem RTSP-Stream deiner Kamera, erkennt Handgesten mittels MediaPipe und sendet HTTP-Befehle an dein Home Assistant, um beispielsweise ein Licht aus- oder einzuschalten.

## Voraussetzungen installieren
Die benötigten Python-Bibliotheken wurden bereits installiert. Falls sie auf einem anderen Rechner benötigt werden:
```bash
pip install opencv-python mediapipe requests python-dotenv
```

## Konfiguration (Fortgeschritten)

Du kannst nun jede Geste einer eigenen Aktion in Home Assistant zuweisen. Dafür gibt es die Datei `config.json`.

### 1. Die .env Datei
Diese enthält weiterhin deine Zugangsdaten:
```env
RTSP_URL=rtsp://...
HA_URL=http://...
HA_TOKEN=...
```

### 2. Die config.json Datei
Hier legst du fest, welche Geste was tun soll. Du kannst Lichter, Schalter oder andere Services steuern und sogar Helligkeit oder Farben übergeben:

```json
{
    "gestures": {
        "OPEN_HAND": {
            "service": "light.turn_on",
            "entity_id": "light.wohnzimmer",
            "data": { "brightness_pct": 100, "rgb_color": [255, 255, 255] }
        },
        "FIST": {
            "service": "light.turn_off",
            "entity_id": "light.wohnzimmer"
        },
        "INDEX_POINTING": {
            "service": "light.turn_on",
            "entity_id": "light.schreibtisch",
            "data": { "color_name": "blue" }
        },
        "PEACE_SIGN": {
            "service": "light.turn_on",
            "entity_id": "light.schreibtisch",
            "data": { "color_name": "red" }
        }
    }
}
```

## Unterstützte Gesten
- `OPEN_HAND`: Alle Finger ausgestreckt.
- `FIST`: Alle Finger zur Faust geballt.
- `INDEX_POINTING`: Nur der Zeigefinger zeigt nach oben.
- `PEACE_SIGN`: Zeige- und Mittelfinger sind ausgestreckt.
- `THUMBS_UP`: Daumen nach oben (alle anderen zu).
- `ROCK_ON`: Zeigefinger und kleiner Finger sind ausgestreckt.

## Starten
```bash
python hand_tracker.py
```
