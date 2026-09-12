#
# tts / text_to_speech.py
# Uses PiperTTS model to dynamically speak text
#

import logging
import math
import queue
import threading
import wave
from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice, SynthesisConfig

from atlas.core.events import EventManager
from atlas.utils.config import DATA_DIR, Config

log = logging.getLogger(__name__)

CONFIG_EXAMPLE = Path(__file__).parent / "tts_exapmle.toml"
cfg = Config.load_config("tts.toml", CONFIG_EXAMPLE.read_text())["tts"]

VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


class TextToSpeech:
    def __init__(
        self,
        events: EventManager | None = None,
        model_path=cfg["model_path"],
        volume=cfg["volume"],
        length_scale=cfg["length_scale"],
        noise_scale=cfg["noise_scale"],
        noise_w_scale=cfg["noise_w_scale"],
        normalize_audio=cfg["normalize_audio"],
    ) -> None:
        self.events = events or EventManager()

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
        self._healthy = False

        self.silence_duration = cfg["silence_duration"]
        self.use_cuda = cfg["use_cuda"]

    def load(self):
        self.voice = PiperVoice.load(
            self.path, use_cuda=self.use_cuda, download_dir=self.path.parent
        )
        # self._generate_basic_sounds()
        self._healthy = True
        log.info("TTS model loaded")

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

        log.info("Downloading PiperTTS model '%s' via HuggingFace index", model_key)

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
                    log.info("Downloading %s...", file_name)
                    file_req = urllib.request.Request(
                        download_url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with (
                        urllib.request.urlopen(file_req, context=ctx) as response,
                        open(target_path, "wb") as out_file,
                    ):
                        shutil.copyfileobj(response, out_file)

            log.info("PiperTTS model '%s' downloaded successfully", model_key)

        except Exception:
            log.exception("Failed to download Piper '%s'", model_key)
            raise

    def _tts_worker(self):
        """Background thread that gathers text chunks from queue and voices them."""
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

    def interrupt(self) -> None:
        """Stops current playback and clears the TTS queue."""
        log.info("TTS interrupted")
        with self.queue.mutex:
            self.queue.queue.clear()

        sd.stop()
        self._set_busy(False)

    def _set_busy(self, value: bool) -> None:
        self.events.emit("tts.busy" if value else "tts.free", {})
        with self._busy_lock:
            self._busy = value

    def _text_to_speech(self, text: str) -> None:
        """Generates and plays audio from text(str)."""
        if not self._healthy:
            log.warning("Ignoring TTS request because the subsystem is unavailable")
            return
        if text.strip():
            try:
                log.debug("Synthesizing TTS text")
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
                log.debug("TTS playback completed")
            except Exception:
                self._healthy = False
                log.exception("TTS synthesis or playback failed; subsystem disabled")

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
                log.debug("Compressing to %s: %s", path.suffix.lower(), path.name)

                sf.write(path, *sf.read(wav_file))

                wav_file.unlink()
            else:
                log.error("Unsupported output format: %s", path.suffix)

        except Exception:
            log.exception("Error generating audio %s", path.name)

    def speak(self, text: str) -> None:
        self.queue.put(text)

    def close(self):
        self.queue.put(None)
        self.worker_thread.join(timeout=2.0)
