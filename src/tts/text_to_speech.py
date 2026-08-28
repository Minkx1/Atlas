#
# text_to_speech.py
#

import math
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice, SynthesisConfig

from ..core.config import DATA_DIR, cfg
from ..core.events import EventType, emit_event, log

VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


class TextToSpeech:
    def __init__(
        self,
        model_path=cfg.tts.model_path,
        volume=cfg.tts.volume,
        length_scale=cfg.tts.length_scale,
        noise_scale=cfg.tts.noise_scale,
        noise_w_scale=cfg.tts.noise_w_scale,
        normalize_audio=cfg.tts.normalize_audio,
    ) -> None:
        self.path = DATA_DIR / model_path
        if not self.path.exists():
            self._download_model()

        self.syn_config = SynthesisConfig(
            volume=volume,  # half as loud
            length_scale=length_scale,  # twice as slow
            noise_scale=noise_scale,  # more audio variation
            noise_w_scale=noise_w_scale,  # more speaking variation
            normalize_audio=normalize_audio,  # use raw audio from voice
        )

        self.queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._tts_worker, name="TTS_THREAD", daemon=True
        )
        self._busy = False
        self._busy_lock = threading.Lock()

        self.silence_duration = cfg.tts.silence_duration
        self.use_cuda = cfg.tts.use_cuda

    def load(self):
        _start = time.perf_counter()
        self.voice = PiperVoice.load(
            self.path, use_cuda=self.use_cuda, download_dir=self.path.parent
        )
        # self._generate_basic_sounds()
        log(
            f"TTS model loaded in {(time.perf_counter() - _start) * 1000:.0f}ms",
            "TTS",
            "SUCCESS",
        )
        emit_event(EventType.TTS_LOADED, f"{(time.perf_counter() - _start) * 1000}ms")

    def start(self):
        if not hasattr(self, "voice"):
            raise RuntimeError("TTS.start() was called before TTS.load()")

        self.worker_thread.start()

    def _download_model(self):
        import json
        import shutil
        import ssl
        import urllib.request

        # SSL Certificate fix
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        model_path = self.path

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
            with urllib.request.urlopen(req, context=ctx) as resp:
                voices_data = json.loads(resp.read().decode("utf-8"))

            if model_key not in voices_data:
                raise ValueError(
                    f"Model '{model_key}' not found in Piper voices index."
                )

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
                        urllib.request.urlopen(file_req, context=ctx) as response,
                        open(target_path, "wb") as out_file,
                    ):
                        shutil.copyfileobj(response, out_file)

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

            self._set_busy(True)
            try:
                self._text_to_speech(value)
            finally:
                self._set_busy(False)

            self.queue.task_done()

    def _set_busy(self, value: bool) -> None:
        emit_event(EventType.TTS_BUSY) if value == True else emit_event(
            EventType.TTS_FREE
        )
        with self._busy_lock:
            self._busy = value

    def _text_to_speech(self, text: str) -> None:
        """Generates and plays audio from text(str)."""
        if text.strip():
            try:
                log(f"Synthesizing TTS: {text}...", "TTS", "DEBUG")
                audio_chunks = list(self.voice.synthesize(text, self.syn_config))
                audio_array = np.concatenate(
                    [chunk.audio_float_array for chunk in audio_chunks]
                )

                samplerate = audio_chunks[0].sample_rate

                target_sr = 48000
                if samplerate != target_sr:
                    gcd = math.gcd(target_sr, samplerate)
                    audio_array = scipy.signal.resample_poly(
                        audio_array, target_sr // gcd, samplerate // gcd
                    )
                    samplerate = target_sr

                silence = np.zeros(
                    int(samplerate * self.silence_duration), dtype=audio_array.dtype
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

    def _text_to_file(self, text: str, path: Path) -> None:
        """Generates audio from `text` and writes it to the `output_path`."""
        if not text.strip():
            return

        try:
            path.parent.mkdir(exist_ok=True, parents=True)
            wav_file: Path = path.with_suffix(".wav")

            with wave.open(str(wav_file), "wb") as f:
                self.voice.synthesize_wav(text, f, self.syn_config)

            if path == wav_file:
                return  # .wav file was already generated

            if path.suffix.lower() in {".flac", ".ogg"}:
                log(
                    f"Compressing to {path.suffix.lower()}: {path.name}",
                    "TTS",
                    "DEBUG",
                )

                sf.write(path, *sf.read(wav_file))

                wav_file.unlink()
            else:
                log(f"Unsupported output format: {path.suffix}", "TTS", "ERROR")

        except Exception as e:  # noqa: BLE001
            log(
                f"Error generating audio {path.name}: {type(e).__name__}: {e}",
                "TTS",
                "ERROR",
            )

    def speak(self, text: str) -> None:
        self.queue.put(text)

    def close(self):
        self.queue.put(None)
        self.worker_thread.join()


if __name__ == "__main__":
    tts = TextToSpeech()
    tts._download_model()
    # tts.load()
