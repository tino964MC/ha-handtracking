# Hand Tracking Home Assistant Controller

Dieses Skript verbindet sich mit dem RTSP-Stream deiner Kamera, erkennt Handgesten mittels MediaPipe und sendet HTTP-Befehle an dein Home Assistant, um beispielsweise ein Licht aus- oder einzuschalten.

## Voraussetzungen installieren
Die benötigten Python-Bibliotheken wurden bereits installiert. Falls sie auf einem anderen Rechner benötigt werden:
```bash
pip install opencv-python mediapipe requests python-dotenv
```

## Konfiguration (Empfohlen: YAML)

Für das Home Assistant Add-on wird die Konfiguration über eine YAML-Datei empfohlen. Diese ist übersichtlicher und einfacher zu bearbeiten.

Die Datei findest du unter: `/config/hand_control_pro_gestures.yaml`

Eine vollständige Anleitung mit Beispielen findest du hier:
👉 **[GESTURES_YAML_GUIDE.md](GESTURES_YAML_GUIDE.md)**

### Beispiel (YAML):
```yaml
PEACE_SIGN:
  service: "light.toggle"
  entity_id: "light.schreibtisch"
```

---

## Konfiguration (Alternativ: JSON)

Falls du JSON bevorzugst, kannst du die Datei `gestures.json` verwenden.

```json
{
    "PEACE_SIGN": {
        "service": "light.toggle",
        "entity_id": "light.schreibtisch"
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
