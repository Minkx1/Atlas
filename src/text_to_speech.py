# text_to_speech.py

import queue
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from playsound3 import playsound

if __name__ == "__main__":
    from config import DATA_DIR, cfg
    from events import EventType, emit_event, log
else:
    from .config import DATA_DIR, cfg
    from .events import EventType, emit_event, log


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
        emit_event(EventType.TTS_LOADED, f"{(time.perf_counter() - _start) * 1000}ms")

    def _download_model(self):
        model_path = DATA_DIR / cfg.tts.model_path
        name = model_path.stem
        log(f"Downloading PiperTTS model: {name}", "TTS", "INFO")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log(f"Starting download subprocess for {name}...", "TTS", "DEBUG")
            subprocess.run(
                f"{sys.executable} -m piper.download_voices {name}",
                cwd=model_path.parent,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
            log(f"PiperTTS model {name} downloaded successfully.", "TTS", "INFO")
        except Exception as e:  # noqa: BLE001
            log(
                f"Error downloading PiperTTS model {name}: {type(e).__name__}: {e}",
                "TTS",
                "ERROR",
            )

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
                log(f"Synthesizing TTS: {text[:50]}...", "TTS", "DEBUG")
                audio_chunks = list(self.voice.synthesize(text, self.syn_config))
                audio_array = np.concatenate(
                    [chunk.audio_float_array for chunk in audio_chunks]
                )

                sd.play(audio_array, samplerate=audio_chunks[0].sample_rate)
                sd.wait()
                log("TTS playback completed.", "TTS", "DEBUG")
            except Exception as e:  # noqa: BLE001
                log(
                    f"Error during TTS synthesis: {type(e).__name__}: {e}",
                    "TTS",
                    "ERROR",
                )

    def _text_to_wav(self, text: str, wav_file_path: Path) -> None:
        if text.strip():
            try:
                wav_file_path.parent.mkdir(exist_ok=True, parents=True)
                log(f"Writing WAV: {wav_file_path.name}", "TTS", "DEBUG")
                with wave.open(f"{wav_file_path}", "wb") as f:
                    self.voice.synthesize_wav(text, f, self.syn_config)
                log(f"WAV written: {wav_file_path.name}", "TTS", "DEBUG")
            except Exception as e:  # noqa: BLE001
                log(
                    f"Error writing WAV {wav_file_path.name}: {type(e).__name__}: {e}",
                    "TTS",
                    "ERROR",
                )

    @staticmethod
    def play_audio(path: Path) -> None:
        try:
            log(f"Playing audio: {path.name}", "TTS", "DEBUG")
            playsound(path)
            log(f"Audio playback done: {path.name}", "TTS", "DEBUG")
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

    def _generate_basic_sounds(self):
        log("Checking builtin sounds...", "TTS", "INFO")
        sounds_dir = DATA_DIR / "sounds"

        builtin_cmds = cfg.op.load_builtin_commands() or {}

        for data in builtin_cmds.values():
            sounds_list: list[dict[str, str]] = data.get("sounds", [])  # type: ignore

            for sound_obj in sounds_list:
                path_str = sound_obj.get("path")
                text_template = sound_obj.get("text")

                if not path_str or not text_template:
                    continue

                full_path = sounds_dir / path_str

                # if full_path.exists():
                #     continue

                try:
                    formatted_text = text_template.format(
                        username=cfg.username, name=cfg.name
                    )
                except KeyError as e:
                    log(
                        f"Missing config key {e} for string '{text_template}'",
                        "TTS",
                        "ERROR",
                    )
                    continue

                log(f"Generating missing sound: {path_str}", "TTS", "INFO")
                self._text_to_wav(formatted_text.strip(), full_path)


if __name__ == "__main__":
    tts = TextToSpeech()
    tts.load()
    # tts = TextToSpeech.play_audio(Path("data/sounds/greet/greet1.wav"))
    # playsound("data/sounds/greet/greet1.wav")
