#
# stt / speech_recognition.py
# Unites Silero VAD with Faster-Whisper recognition models to process audio chunks and
# recognize spoken text as fast as possible
#

import os
import queue
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Literal

import numpy as np

from ..core.config import DATA_DIR, cfg
from ..core.events import EventType, emit_event, log

# disables HF symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class VAD:
    def __init__(self) -> None:
        self.is_speaking = False
        self.model_path: Path = DATA_DIR / cfg.vad.model_path

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def _download_model(self):
        import shutil
        import ssl
        import urllib.request
        from urllib.error import URLError

        # SSL Certificate fix
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

        try:
            print(f"[I] Downloading Silero VAD ONNX model from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with (
                urllib.request.urlopen(req, context=ctx) as response,
                open(self.model_path, "wb") as out_file,
            ):
                shutil.copyfileobj(response, out_file)
            print(f"[I] Model successfully installed to {self.model_path}")
        except (URLError, OSError) as e:
            if self.model_path.exists():
                self.model_path.unlink()
            raise RuntimeError(f"[!] Failed to download Silero VAD model: {e}") from e

    def load(self):
        import onnxruntime as ort

        try:
            log("Loading Silero VAD ONNX model...", "VAD", "INFO")

            if not self.model_path.exists():
                log("No VAD model found. Downloading...", "VAD", "WARN")
                self._download_model()

            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )

            self.sample_rate = cfg.audio.sample_rate
            self.threshold = cfg.vad.threshold
            self.min_silence_samples = (
                self.sample_rate * cfg.vad.min_silence_duration_ms
            ) / 1000

            self.reset_state()

            log("VAD model loaded.", "VAD", "SUCCESS")
            emit_event(EventType.VAD_LOADED, {})
        except Exception as e:
            log(
                f"Error loading VAD model: {type(e).__name__}: {e}",
                "VAD",
                "ERROR",
            )
            raise

    def reset_state(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @staticmethod
    def _normalize_chunk(audio_chunk: np.ndarray) -> np.ndarray:
        chunk = audio_chunk.squeeze()
        if chunk.ndim == 1:
            chunk = np.expand_dims(chunk, axis=0)

        if chunk.dtype != np.float32:
            if chunk.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            else:
                chunk = chunk.astype(np.float32)
        return chunk

    def process(
        self, audio_chunk: np.ndarray
    ) -> Literal["silence", "start", "speaking", "end"]:
        """Returns state of speech: 'silence', 'start', 'speaking', 'end'."""
        if not hasattr(self, "session"):
            raise RuntimeError("VAD was used before vad.load()")

        chunk = self._normalize_chunk(audio_chunk)
        chunk_length = chunk.shape[1]
        chunk_with_context = np.concatenate((self.context, chunk), axis=1)
        self.context = chunk[:, -64:]  # updating context

        ort_inputs = {
            "input": chunk_with_context,
            "state": self.state,
            "sr": np.array([self.sample_rate], dtype=np.int64),
        }

        ort_outs = self.session.run(None, ort_inputs)
        prob = ort_outs[0].item()  # type: ignore
        self.state = ort_outs[1]

        self.current_sample += chunk_length

        if prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if prob >= self.threshold and not self.triggered:
            self.triggered = True
            self.is_speaking = True
            return "start"

        if prob < (self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample

            if self.current_sample - self.temp_end >= self.min_silence_samples:
                self.triggered = False
                self.is_speaking = False
                self.temp_end = 0
                return "end"

        return "speaking" if self.triggered else "silence"


class Whisper:
    def __init__(self):
        # should make downloading Whisper models from HF faster
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

        w = cfg.stt
        self.model_dir: Path = DATA_DIR / w.download_root

    def load(self):
        from faster_whisper import WhisperModel

        try:
            w = cfg.stt
            if not self.model_dir.exists():
                log(
                    f"Faster-Whisper model not found: {self.model_dir}. Downloading...",
                    "STT",
                    "INFO",
                )
            else:
                log(f"Using Whisper model from {self.model_dir}", "STT", "DEBUG")

            log(f"Loading Whisper model: {w.model_size}...", "STT", "INFO")
            self.model = WhisperModel(
                w.model_size,
                device=w.device,
                compute_type="int8",
                cpu_threads=w.cpu_threads,
                num_workers=1,
                download_root=str(self.model_dir),
            )
            log("Whisper model loaded.", "STT", "SUCCESS")
            emit_event(EventType.WHISPER_LOADED, {})
        except Exception as e:
            log(
                f"Error loading Whisper model: {type(e).__name__}: {e}",
                "STT",
                "ERROR",
            )
            raise

    def transcribe(self, audio_array: np.ndarray) -> str:
        """Turn a speech audio array into text."""
        if not hasattr(self, "model"):
            raise RuntimeError("Whisper was used before whisper.load()")

        w = cfg.stt
        segments, _ = self.model.transcribe(
            audio=audio_array,
            beam_size=w.beam_size,
            language=w.language,
            initial_prompt=w.initial_prompt,
            condition_on_previous_text=False,
        )
        text = " ".join([segment.text for segment in segments]).strip()
        return text


class SpeechRecognizer:
    def __init__(self):
        self.vad = VAD()
        self.whisper = Whisper()

        self.preroll = deque(maxlen=cfg.vad.preroll_blocks)
        self.buffer: list[np.ndarray] = []
        self.audio_queue = queue.Queue()  # Queue containg (audio_array, listen_ms)

        self._recording = False
        self.sample_rate = cfg.audio.sample_rate
        self.min_command_ms = cfg.stt.min_command_ms

        self.stt_worker_thread = Thread(
            target=self._stt_worker, name="STT_WORKER_THREAD", daemon=True
        )

    def load(self):
        self.vad.load()
        self.whisper.load()

    def start(self) -> None:
        self.stt_worker_thread.start()

    def close(self) -> None:
        self.audio_queue.put(None)  # breaks loop
        if self.stt_worker_thread is not None and self.stt_worker_thread.is_alive():
            self.stt_worker_thread.join(timeout=2.0)

        if hasattr(self, "whisper") and hasattr(self.whisper, "model"):
            del self.whisper.model  # finishing low-level C process

    def _stt_worker(self):
        """Processes audio from audio_queue to text and emits event."""
        while True:
            item = self.audio_queue.get()
            if item is None:
                break

            audio_array, _ = item
            text = self.whisper.transcribe(audio_array)

            text = text.strip()

            if text:
                emit_event(
                    EventType.UI_TRANSCRIPTION,
                    {
                        "text": text,
                    },
                )

                emit_event(EventType.STT_TRANSCRIBED, {"text": text})

            self.audio_queue.task_done()

    def process(self, raw_chunk: np.ndarray, allow_recording: bool):
        chunk = raw_chunk.squeeze(1) if raw_chunk.ndim > 1 else raw_chunk

        self.preroll.append(chunk.copy())
        vad_state = self.vad.process(chunk)

        if not allow_recording:
            if self._recording:
                self._recording = False
                self.buffer.clear()
            return vad_state

        if vad_state == "start":
            self._recording = True
            self.buffer = list(self.preroll)
            emit_event(EventType.VAD_START, {})

        elif vad_state == "speaking" and self._recording:
            self.buffer.append(chunk)

        elif vad_state == "end" and self._recording:
            self._recording = False
            self.buffer.append(chunk)
            if self.buffer:
                full_audio = np.concatenate(self.buffer)
                listen_ms = (len(full_audio) / self.sample_rate) * 1000.0

                if listen_ms > self.min_command_ms:
                    self.audio_queue.put((full_audio, listen_ms))

            self.buffer.clear()
            emit_event(EventType.VAD_END, {})

        return vad_state
