# Hand Control: Home Assistant Gesture Controller

Control your Home Assistant smart home using simple hand gestures captured via any RTSP camera stream. This project uses MediaPipe for hand tracking and gesture recognition.

## Features

- **Real-time Hand Tracking**: Powered by Google's MediaPipe.
- **RTSP Support**: Works with most IP cameras and NVRs.
- **Home Assistant Integration**: Trigger any service or script directly.
- **Customizable Gestures**: Map specific gestures to Home Assistant actions.
- **Resource Efficient**: Optimized for low CPU usage with dynamic frame skipping.
- **Home Assistant Add-on**: Easy installation as an official-style add-on.

## Supported Gestures

| Gesture          | Description                       |
| :--------------- | :-------------------------------- |
| `OPEN_HAND`      | All fingers extended              |
| `FIST`           | All fingers curled                |
| `INDEX_POINTING` | Only index finger extended        |
| `PEACE_SIGN`     | Index and middle fingers extended |
| `THUMBS_UP`      | Thumb pointed upwards             |
| `ROCK_ON`        | Index and pinky fingers extended  |

## Installation

### As a Home Assistant Add-on

1. Go to **Settings** > **Add-ons** > **Add-on Store**.
2. Click the three dots (top right) > **Repositories**.
3. Add: `https://github.com/tino964MC/ha-handtracking`
4. Find **Hand Control** and click **Install**.

## Configuration

The add-on can be configured directly through the Home Assistant UI or via `config.yaml`.

### Example Gesture Mapping (Add-on Options)

```yaml
peace_sign_action: "light.toggle,light.living_room"
index_pointing_action: "media_player.media_play_pause,media_player.tv"
thumbs_up_action: "scene.turn_on,scene.movie_night"
open_hand_action: "light.turn_off,light.all_lights"
fist_action: "lock.lock,lock.front_door"
rock_on_action: "script.party_mode"
```

## Contributing

Contributions are welcome! If you have ideas for new features or gestures, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
