# text_to_speech.py

import queue
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from playsound3 import playsound

if __name__ == "__main__":
    from config import DATA_DIR, cfg
else:
    from .config import DATA_DIR, cfg


def print(msg: str, end="\n", force=False):
    if cfg.profiler or force:
        sys.stdout.write(msg + end)


class TextToSpeech:
    def __init__(self) -> None:
        s = cfg.tts
        path = DATA_DIR / s.model_path
        if not path.exists():
            self._download_model()

        self.voice = PiperVoice.load(
            path, use_cuda=s.use_cuda, download_dir=path.parent
        )
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

    def _download_model(self):
        model_path = DATA_DIR / cfg.tts.model_path
        name = model_path.stem
        print(f"[I] Downloading PiperTTS model: {name}")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [sys.executable, "-m", "piper.download_voices", name],
                cwd=model_path.parent,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[!] Error during downloading PiperTTS model({name}): {e}")
        else:
            print(f"[$] PiperTTS model({name}) was downloaded succesfully.")

    def _tts_worker(self):
        """Background thread that gathers sentences(text chunks) from queue and voices them."""
        while True:
            value = self.queue.get()
            if value is None:
                self.queue.task_done()
                return

            if isinstance(value, Path):
                self.play_audio(value)
            elif isinstance(value, str):
                self.tts(value)

            self.queue.task_done()

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
        ("thanks/thanks2.wav", "I trully apreciate this, sir."),
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
