# Hand Control Pro - Gesten-Konfiguration (YAML)

Diese Anleitung zeigt dir, wie du deine eigenen Gesten und Aktionen in der Datei `/config/hand_control_pro_gestures.yaml` definierst.

## 📋 Grundstruktur

Jeder Eintrag besteht aus dem Namen der Geste und der Aktion, die in Home Assistant ausgeführt werden soll:

```yaml
NAME_DER_GESTE:
  service: "domain.service"
  entity_id: "domain.entity_id"
  data:
    key: "value" # Optional
```

## 🖐️ Verfügbare Gesten

Du kannst folgende Gesten-Namen verwenden:

| Geste | Beschreibung |
| :--- | :--- |
| `PEACE_SIGN` | Zeigefinger und Mittelfinger sind oben (V-Zeichen). |
| `INDEX_POINTING` | Nur der Zeigefinger zeigt nach oben. |
| `THUMBS_UP` | Daumen nach oben, andere Finger zur Faust. |
| `OPEN_HAND` | Alle Finger sind gestreckt. |
| `FIST` | Alle Finger sind zur Faust geballt. |
| `ROCK_ON` | Zeigefinger und kleiner Finger sind oben. |

## 💡 Beispiele

### Einfaches Schalten (Toggle)
Schaltet das Licht um, wenn du das Peace-Zeichen zeigst:
```yaml
PEACE_SIGN:
  service: "light.toggle"
  entity_id: "light.mein_zimmer_licht"
```

### Mehrere Gesten kombinieren
```yaml
THUMBS_UP:
  service: "light.turn_on"
  entity_id: "light.wohnzimmer"
  data:
    brightness_pct: 100
    rgb_color: [255, 255, 255]

FIST:
  service: "light.turn_off"
  entity_id: "light.wohnzimmer"
```

### Scenen aktivieren
```yaml
INDEX_POINTING:
  service: "scene.turn_on"
  entity_id: "scene.kino_modus"
```

## ⚠️ Wichtige Hinweise
1. **Einrücken:** Achte auf die Leerzeichen am Zeilenanfang. YAML ist hier sehr streng.
2. **Neustart:** Nach jeder Änderung an der Datei musst du das Add-on einmal **neu starten**.
3. **Priorität:** Falls beide Dateien existieren, bevorzugt das Add-on die `gestures.json`. Lösche diese am besten, wenn du YAML nutzt.
