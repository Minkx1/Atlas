# text_to_speech.py

import queue
import subprocess
import sys
import threading

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

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

        self.chunk_queue: queue.Queue[str | None] = queue.Queue()
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
            text = self.chunk_queue.get()
            if text is None:
                self.chunk_queue.task_done()
                break

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

            self.chunk_queue.task_done()

    def start(self):
        self.worker_thread.start()

    def speak(self, text: str) -> None:
        self.chunk_queue.put(text)

    def close(self):
        self.chunk_queue.put(None)
        self.worker_thread.join()
