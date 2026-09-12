#
# tts / sound_manager.py
# Saves and manages sounds in data / sounds /
#

import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

# from atlas.core.config import cfg
from atlas.core.events import EventManager
from atlas.utils.config import CONFIG_DIR, DATA_DIR, Config

log = logging.getLogger(__name__)

CONFIG_EXAMPLE = Path(__file__).parent / "tts_exapmle.toml"
cfg = Config.load_config("tts.toml", CONFIG_EXAMPLE.read_text())["tts"]


class SoundManager:
    def __init__(self, events: EventManager | None = None) -> None:
        self.events = events or EventManager()

        self.commands = self.load_commands() or {}
        self.silence_duration = cfg["silence_duration"]
        self._healthy = False

    @staticmethod
    def load_commands() -> dict[str, dict[str, str | list[str] | None]]:
        path = CONFIG_DIR / "commands.json"
        if not path.exists():
            Config.write_from_eample(
                path, Path(__file__).parent / "commands_example.json"
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return {}

        commands: dict[str, dict[str, str | list[str] | None]] = {}
        for intent, values in payload.items():
            if not isinstance(values, dict):
                continue

            commands[str(intent)] = {
                "sounds": values.get("sounds", []),
                "triggers": values.get("triggers", []),
            }

        return commands

    def load(self) -> None:
        self.commands = self.load_commands()
        self._generate_basic_sounds()
        self._healthy = True

    def play_audio(self, path: Path) -> None:
        """Plays audio from path"""
        if not self._healthy:
            log.warning("Ignoring sound request because TTS audio is unavailable")
            return
        try:
            log.debug("Playing audio: %s", path.name)
            audio, samplerate = sf.read(path)

            target_sr = 48000
            if samplerate != target_sr:
                gcd = math.gcd(target_sr, samplerate)
                up = target_sr // gcd
                down = samplerate // gcd

                axis = 0 if audio.ndim > 1 else -1
                audio = scipy.signal.resample_poly(audio, up, down, axis=axis)
                samplerate = target_sr

            sd.check_output_settings(
                samplerate=samplerate,
                channels=audio.shape[1] if audio.ndim > 1 else 1,
            )

            if audio.ndim > 1:  # if stereo file
                silence = np.zeros(
                    (int(samplerate * self.silence_duration), audio.shape[1]),
                    dtype=audio.dtype,
                )
            else:
                silence = np.zeros(
                    int(samplerate * self.silence_duration), dtype=audio.dtype
                )

            padded_audio = np.concatenate((silence, audio))  # audio with silence before

            sd.play(padded_audio, samplerate)
            sd.wait()

            self.events.emit("tts.free", {})
        except Exception:
            self._healthy = False
            log.exception("Error playing audio %s", path.name)

    def play_sound(self, payload: Path | dict[str, str | Path | None]) -> None:
        """Plays sound from payload."""
        if not self._healthy:
            log.warning("Ignoring sound request because TTS audio is unavailable")
            return
        if isinstance(payload, dict):
            path = payload.get("path") or payload.get("sound")
            text = payload.get("text")

            formatted_text = str(text).format(
                username=cfg["username"], name=cfg["name"]
            )
            if formatted_text:
                self.events.emit("ui.say", {"text": formatted_text})
            if not path:
                return
            payload = Path(path)
        elif isinstance(payload, str):
            payload = Path(payload)

        self.events.emit("tts.busy", {})
        self.play_audio(payload)

    def interrupt(self) -> None:
        """Stops sound playback."""
        sd.stop()

    def _get_current_state(self) -> dict:
        """Returns structured dict of current TTS settings and formatted sounds."""
        state = {
            "settings": {
                "name": cfg["name"],
                "username": cfg["username"],
                "model_path": cfg["model_path"],
                "use_cuda": cfg["use_cuda"],
                "volume": cfg["volume"],
                "length_scale": cfg["length_scale"],
                "noise_scale": cfg["noise_scale"],
                "noise_w_scale": cfg["noise_w_scale"],
                "normalize_audio": cfg["normalize_audio"],
            },
            "sounds": {},
        }

        for intent, data in self.commands.items():
            formatted_sounds = []
            sounds_val: list[dict[str, str]] = data.get("sounds", [])  # type: ignore

            for sound_obj in sounds_val:
                path_str = sound_obj.get("path", "")
                text_template = sound_obj.get("text", "")

                if not path_str or not text_template:
                    continue

                try:
                    # KeyError occurs if template has {unknown_key}
                    formatted_text = text_template.format(
                        name=cfg["name"], username=cfg["username"]
                    )
                    formatted_sounds.append({"path": path_str, "text": formatted_text})
                except KeyError:
                    log.exception("Missing config key for string '%s'", text_template)
                    continue

            if formatted_sounds:
                state["sounds"][intent] = formatted_sounds

        return state

    def _generate_basic_sounds(self):
        log.info("Checking sounds")

        sounds_dir = DATA_DIR / "sounds"
        sounds_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = sounds_dir / "manifest.json"

        current_state = self._get_current_state()
        old_state = {"settings": {}, "sounds": {}}
        if manifest_file.exists():
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    old_state = json.load(f)
            except json.JSONDecodeError:
                log.warning("Manifest file is corrupted; regenerating all")

        re_generate_all = old_state.get("settings") != current_state["settings"]

        if re_generate_all:
            log.info("TTS settings changed; all sounds will be regenerated")

        for intent, sounds_list in current_state["sounds"].items():
            old_intent_sounds = old_state.get("sounds", {}).get(intent, [])

            for sound_obj in sounds_list:
                path_str = sound_obj["path"]
                formatted_text = sound_obj["text"]
                full_path = sounds_dir / path_str

                if (
                    not re_generate_all
                    and full_path.exists()
                    and sound_obj in old_intent_sounds
                ):
                    continue

                log.info("Generating sound: %s", path_str)
                self.events.emit(
                    "tts.sounds.generate_sound",
                    {"text": formatted_text.strip(), "path": full_path},
                )

        # Updating manifest
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(current_state, f, indent=4)

        log.info("Sounds check complete")

    def play_category(self, category: str):
        """Plays random sound from category."""
        if not self._healthy:
            log.warning(
                "Ignoring sound category '%s' because TTS is unavailable", category
            )
            return None
        conf = self.commands.get(category, {})
        sounds = conf.get("sounds", [])

        log.debug("Fetching sound for '%s'", category)

        if isinstance(sounds, list) and sounds:
            sound = random.choice(sounds)

            path_str = ""
            text_str = ""

            if isinstance(sound, dict):
                path_str = sound.get("path", "")
                text_str = sound.get("text", "")
            elif isinstance(sound, str):
                path_str = sound
            else:
                log.warning(
                    "Invalid sound type in config for '%s': %s", category, type(sound)
                )

            if text_str:
                try:
                    text_str = text_str.format(
                        username=cfg["username"], name=cfg["name"]
                    )
                except KeyError:
                    log.exception("Formatting text failed for '%s'", text_str)

            if path_str:
                path = Path(path_str)
                if not path.is_absolute():
                    path = DATA_DIR / "sounds" / path

                log.debug("Playing sound payload: %s | text: %s", path_str, text_str)
                payload = {"path": str(path), "text": text_str if text_str else None}
                self.play_sound(payload)
                return payload

        log.warning("No sounds available for category: %s", category)
        return None
