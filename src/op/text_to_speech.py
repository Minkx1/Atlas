# text_to_speech.py

import json
import queue
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice, SynthesisConfig

from ..core.config import DATA_DIR, cfg
from ..core.events import EventType, emit_event, log

VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


class TextToSpeech:
    def __init__(self) -> None:
        s = cfg.tts
        self.path = DATA_DIR / s.model_path
        if not self.path.exists():
            self._download_model()

        self.syn_config = SynthesisConfig(
            volume=s.volume,  # half as loud
            length_scale=s.length_scale,  # twice as slow
            noise_scale=s.noise_scale,  # more audio variation
            noise_w_scale=s.noise_w_scale,  # more speaking variation
            normalize_audio=s.normalize_audio,  # use raw audio from voice
        )

        self.queue: queue.Queue[str | Path | None] = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._tts_worker, name="TTS_THREAD", daemon=True
        )
        self._busy = False
        self._busy_lock = threading.Lock()

    def load(self):
        _start = time.perf_counter()
        self.voice = PiperVoice.load(
            self.path, use_cuda=cfg.tts.use_cuda, download_dir=self.path.parent
        )
        self._generate_basic_sounds()
        log(
            f"TTS model loaded in {(time.perf_counter() - _start) * 1000:.0f}ms",
            "TTS",
            "SUCCESS",
        )
        emit_event(EventType.TTS_LOADED, f"{(time.perf_counter() - _start) * 1000}ms")

    def _download_model(self):
        model_path = DATA_DIR / cfg.tts.model_path

        model_key = model_path.stem
        model_dir = model_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)

        log(
            f"Downloading PiperTTS model '{model_key}' via HuggingFace index...",
            "TTS",
            "INFO",
        )

        try:
            req = urllib.request.Request(
                VOICES_JSON_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp:
                voices_data = json.loads(resp.read().decode("utf-8"))

            if model_key not in voices_data:
                raise ValueError(
                    f"Model '{model_key}' not found in Piper voices index."
                )

            # 2. Знаходимо всі файли для цього голосу (.onnx та .onnx.json)
            files = voices_data[model_key].get("files", {})

            for rel_path in files:
                file_name = Path(rel_path).name
                target_path = model_dir / file_name
                download_url = HF_BASE_URL + rel_path

                if not target_path.exists():
                    log(f"Downloading {file_name}...", "TTS", "INFO")
                    file_req = urllib.request.Request(
                        download_url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with (
                        urllib.request.urlopen(file_req) as response,
                        open(target_path, "wb") as out_file,
                    ):
                        out_file.write(response.read())

            log(
                f"PiperTTS model '{model_key}' downloaded successfully.",
                "TTS",
                "SUCCESS",
            )

        except Exception as e:
            log(
                f"Failed to download PiperTTS model '{model_key}': {type(e).__name__}: {e}",
                "TTS",
                "ERROR",
            )
            raise

    def _tts_worker(self):
        """Background thread that gathers sentences(text chunks) from queue and voices them."""
        while True:
            value = self.queue.get()
            if value is None:
                self.queue.task_done()
                return

            if isinstance(value, Path):
                self._set_busy(True)
                try:
                    self.play_audio(value)
                finally:
                    self._set_busy(False)
            elif isinstance(value, str):
                self._set_busy(True)
                try:
                    self.tts(value)
                finally:
                    self._set_busy(False)

            self.queue.task_done()

    def _set_busy(self, value: bool) -> None:
        emit_event(EventType.TTS_BUSY) if value == True else emit_event(
            EventType.TTS_FREE
        )
        with self._busy_lock:
            self._busy = value

    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    def tts(self, text: str) -> None:
        if text.strip():
            try:
                log(f"Synthesizing TTS: {text}...", "TTS", "DEBUG")
                audio_chunks = list(self.voice.synthesize(text, self.syn_config))
                audio_array = np.concatenate(
                    [chunk.audio_float_array for chunk in audio_chunks]
                )

                samplerate = audio_chunks[0].sample_rate
                silence = np.zeros(
                    int(samplerate * cfg.tts.silence_duration), dtype=audio_array.dtype
                )
                padded_audio = np.concatenate((silence, audio_array))

                sd.play(padded_audio, samplerate=samplerate)
                sd.wait()
                log("TTS playback completed.", "TTS", "DEBUG")
            except Exception as e:  # noqa: BLE001
                log(
                    f"Error during TTS synthesis: {type(e).__name__}: {e}",
                    "TTS",
                    "ERROR",
                )

    def _text_to_file(self, text: str, output_path: Path) -> None:
        if not text.strip():
            return

        try:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            wav_file: Path = output_path.with_suffix(".wav")

            with wave.open(str(wav_file), "wb") as f:
                self.voice.synthesize_wav(text, f, self.syn_config)

            if output_path == wav_file:
                return  # .wav file was already generated

            if output_path.suffix.lower() in {".flac", ".ogg"}:
                log(
                    f"Compressing to {output_path.suffix.lower()}: {output_path.name}",
                    "TTS",
                    "DEBUG",
                )

                sf.write(output_path, *sf.read(wav_file))

                wav_file.unlink()
            else:
                log(f"Unsupported output format: {output_path.suffix}", "TTS", "ERROR")

        except Exception as e:  # noqa: BLE001
            log(
                f"Error generating audio {output_path.name}: {type(e).__name__}: {e}",
                "TTS",
                "ERROR",
            )

    @staticmethod
    def play_audio(path: Path) -> None:
        try:
            log(f"Playing audio: {path.name}", "TTS", "DEBUG")
            audio, samplerate = sf.read(path)

            if audio.ndim > 1:  # if stereo file
                silence = np.zeros(
                    (int(samplerate * cfg.tts.silence_duration), audio.shape[1]),
                    dtype=audio.dtype,
                )
            else:
                silence = np.zeros(
                    int(samplerate * cfg.tts.silence_duration), dtype=audio.dtype
                )

            padded_audio = np.concatenate((silence, audio))  # audio with silence before

            sd.play(padded_audio, samplerate)
            sd.wait()
        except Exception as e:  # noqa: BLE001
            log(
                f"Error playing audio {path.name}: {type(e).__name__}: {e}",
                "TTS",
                "ERROR",
            )

    def start(self):
        if not hasattr(self, "voice"):
            raise RuntimeError("TTS.start() was called before TTS.load()")

        self.worker_thread.start()

    def speak(self, text: str) -> None:
        self.queue.put(text)

    def play_sound(self, payload: str | Path | dict[str, str | Path | None]) -> None:
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

        self.queue.put(payload)

    def close(self):
        self.queue.put(None)
        self.worker_thread.join()

    def _get_current_state(self) -> dict:
        """Returns structured dict of current TTS settings and formatted sounds."""
        c = cfg.tts
        state = {
            "settings": {
                "name": cfg.name,
                "username": cfg.username,
                "model_path": c.model_path,
                "use_cuda": c.use_cuda,
                "volume": c.volume,
                "length_scale": c.length_scale,
                "noise_scale": c.noise_scale,
                "noise_w_scale": c.noise_w_scale,
                "normalize_audio": c.normalize_audio,
            },
            "sounds": {},
        }

        commands = cfg.op.load_commands() or {}

        for intent, data in commands.items():
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
                with open(manifest_file, "r", encoding="utf-8") as f:
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
                self._text_to_file(formatted_text.strip(), full_path)

        # Updating manifest
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(current_state, f, indent=4)

        log("Sounds check complete.", "TTS", "INFO")


if __name__ == "__main__":
    tts = TextToSpeech()
    tts._download_model()
    # tts.load()
