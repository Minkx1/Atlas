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
        emit_event(EventType.TTS_LOADED, f"{(time.perf_counter() - _start) * 1000}ms")

    def _download_model(self):
        model_path = DATA_DIR / cfg.tts.model_path
        name = model_path.stem
        log(f"[I] Downloading PiperTTS model: {name}", "TTS", "INFO")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                f"{sys.executable} -m piper.download_voices {name}",
                cwd=model_path.parent,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
        except Exception as e:  # noqa: BLE001
            log(
                f"[!] Error during downloading PiperTTS model({name}): {e}",
                "TTS",
                "ERROR",
            )
        else:
            log(
                f"[$] PiperTTS model({name}) was downloaded succesfully.",
                "TTS",
                "SUCCES",
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
                audio_chunks = list(self.voice.synthesize(text, self.syn_config))
                audio_array = np.concatenate(
                    [chunk.audio_float_array for chunk in audio_chunks]
                )

                sd.play(audio_array, samplerate=audio_chunks[0].sample_rate)
                sd.wait()
            except Exception as e:  # noqa: BLE001
                print(f"[!] Error occured during TTS synthesis: {e}")

    def _text_to_wav(self, text: str, wav_file_path: Path) -> None:
        if text.strip():
            wav_file_path.parent.mkdir(exist_ok=True, parents=True)

            with wave.open(f"{wav_file_path}", "wb") as f:
                self.voice.synthesize_wav(text, f, self.syn_config)

    def play_audio(self, path: Path) -> None:
        playsound(path)

    def start(self):
        if not hasattr(self, "voice"):
            raise RuntimeError("TTS.start() was called before TTS.load()")

        self.worker_thread.start()

    def speak(self, text: str) -> None:
        self.queue.put(text)

    def play_sound(self, path: str | Path) -> None:
        self.queue.put(Path(path))

    def close(self):
        self.queue.put(None)
        self.worker_thread.join()


def _main():
    # if file is run generates basic sound files
    tts = TextToSpeech()

    sounds_dir = DATA_DIR / "sounds"
    basic_sounds = {
        ("greet/greet1.wav", "Good Evening, sir."),
        ("greet/greet2.wav", "Welcome back, sir."),
        ("greet/greet3.wav", "Greetings, sir."),
        ("farewell/bye1.wav", "Goodbye, sir."),
        ("farewell/bye2.wav", "Farewell, sir."),
        ("farewell/bye3.wav", "Have a good day, sir."),
        ("thanks/thanks1.wav", "Thank you, sir."),
        ("thanks/thanks2.wav", "I truly apreciate this, sir."),
        ("sorry/sorry1.wav", "I am really sorry, sir."),
        ("sorry/sorry2.wav", "Please accept my apologies."),
        ("sorry/sorry3.wav", "My apologies, sir."),
        # ("welcome/welcome1.wav", "You're welcome, sir."),
    }

    for path, text in basic_sounds:
        path = sounds_dir / path
        tts._text_to_wav(text, path)


if __name__ == "__main__":
    _main()
