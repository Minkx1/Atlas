#
# sound_manager.py
#


import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

_MAIN = __name__ == "__main__"
if not _MAIN:
    from ..core.config import DATA_DIR, cfg
    from ..core.events import EventType, emit_event, log
else:
    # changing execution dir to src/ for proper importing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import DATA_DIR, cfg
    from core.events import EventType, emit_event, log


class SoundManager:
    def __init__(self) -> None:
        self.commands = cfg.op.load_commands() or {}
        self.silence_duration = cfg.tts.silence_duration

    def load(self) -> None:
        self.commands = cfg.op.load_commands()
        self._generate_basic_sounds()

    def play_audio(self, path: Path) -> None:
        """Plays audio from path"""
        try:
            log(f"Playing audio: {path.name}", "TTS", "DEBUG")
            audio, samplerate = sf.read(path)

            target_sr = 48000
            if samplerate != target_sr:
                gcd = math.gcd(target_sr, samplerate)
                up = target_sr // gcd
                down = samplerate // gcd

                axis = 0 if audio.ndim > 1 else -1
                audio = scipy.signal.resample_poly(audio, up, down, axis=axis)
                samplerate = target_sr

            log(
                f"Audio: {path.name}, shape={audio.shape}, dtype={audio.dtype}, sr={samplerate}",
                "TTS",
                "INFO",
            )

            log(
                f"Output device: {sd.default.device}",
                "TTS",
                "INFO",
            )

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

            emit_event(EventType.TTS_FREE, {})
        except Exception as e:
            log(
                f"Error playing audio {path.name}: {type(e).__name__}: {e}",
                "TTS",
                "ERROR",
            )

    def play_sound(self, payload: Path | dict[str, str | Path | None]) -> None:
        """Plays `payload`: if Path - play_audio(payload), else - prints text to UI and play_audio(payload['path'])."""
        if isinstance(payload, dict):
            path = payload.get("path") or payload.get("sound")
            text = payload.get("text")

            formatted_text = str(text).format(username=cfg.username, name=cfg.name)
            if formatted_text:
                emit_event(EventType.UI_ASSISTANT_SAY, {"text": formatted_text})
            if not path:
                return
            payload = Path(path)
        elif isinstance(payload, str):
            payload = Path(payload)

        emit_event(EventType.TTS_BUSY, {})
        self.play_audio(payload)

    def interrupt(self) -> None:
        """Stops sound playback."""
        sd.stop()

    def _get_current_state(self) -> dict:
        """Returns structured dict of current TTS settings and formatted sounds."""
        c = cfg.tts
        state = {
            "settings": {
                "name": cfg.name,
                "username": cfg.username,
                "model_path": cfg.tts.model_path,
                "use_cuda": c.use_cuda,
                "volume": c.volume,
                "length_scale": c.length_scale,
                "noise_scale": c.noise_scale,
                "noise_w_scale": c.noise_w_scale,
                "normalize_audio": c.normalize_audio,
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
                        name=cfg.name, username=cfg.username
                    )
                    formatted_sounds.append({"path": path_str, "text": formatted_text})
                except KeyError as e:
                    log(
                        f"Missing config key {e} for string '{text_template}'",
                        "TTS",
                        "ERROR",
                    )
                    continue

            if formatted_sounds:
                state["sounds"][intent] = formatted_sounds

        return state

    def _generate_basic_sounds(self):
        log("Checking sounds...", "TTS", "INFO")

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
                log("Manifest file is corrupted. Regenerating all.", "TTS", "WARN")

        re_generate_all = old_state.get("settings") != current_state["settings"]

        if re_generate_all:
            log("TTS settings changed. All sounds will be regenerated.", "TTS", "INFO")

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

                log(f"Generating sound: {path_str}", "TTS", "INFO")
                emit_event(
                    EventType.SOUNDS_GENERATE_SOUND,
                    {"text": formatted_text.strip(), "path": full_path},
                )

        # Updating manifest
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(current_state, f, indent=4)

        log("Sounds check complete.", "TTS", "INFO")

    def play_category(self, category: str):
        """Plays random sound from category."""
        conf = self.commands.get(category, {})
        sounds = conf.get("sounds", [])

        log(
            f"Fetching sound for '{category}'.",
            "OP",
            "DEBUG",
        )

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
                log(
                    f"Invalid sound type in config for '{category}': {type(sound)}",
                    "OP",
                    "WARN",
                )

            if text_str:
                try:
                    text_str = text_str.format(username=cfg.username, name=cfg.name)
                except KeyError as e:
                    log(
                        f"Formatting text failed for sound '{text_str}': Missing key {e}",
                        "OP",
                        "DEBUG",
                    )

            if path_str:
                path = Path(path_str)
                if not path.is_absolute():
                    path = DATA_DIR / "sounds" / path

                log(
                    f"Playing sound payload: {path_str} | text: {text_str}",
                    "OP",
                    "DEBUG",
                )
                payload = {"path": str(path), "text": text_str if text_str else None}
                self.play_sound(payload)
                return payload

        log(f"No sounds available for category: {category}", "OP", "WARN")
        return None
