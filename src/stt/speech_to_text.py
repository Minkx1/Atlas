#
# speech_to_text.py
# SoundDevice.InputStream[microphone] -> KeyWordSpotter[Sherpa ONNX KWS] -> Voice Activity Detector[Silero VAD] + STT[faster-whisper] -> "recognized text"
#

import os
import queue
import time
from collections import deque
from enum import StrEnum
from pathlib import Path
from threading import Thread
from typing import Literal

import numpy as np

from ..core.config import DATA_DIR, cfg
from ..core.events import EventManager, EventType, emit_event, log


class VAD:
    def __init__(self) -> None:
        self.is_speaking = False
        self.model_path: Path = DATA_DIR / cfg.vad.model_path

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def _download_model(self):
        import urllib.request
        from urllib.error import URLError

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

        try:
            print(f"[I] Downloading Silero VAD ONNX model from {url}...")
            urllib.request.urlretrieve(url, self.model_path)
            print(f"[I] Model successfully installed to {self.model_path}")
        except (URLError, OSError) as e:
            if self.model_path.exists():
                self.model_path.unlink()
            raise RuntimeError(f"[!] Failed to download Silero VAD model: {e}") from e

    def load(self):
        import onnxruntime as ort

        try:
            _start = time.perf_counter()
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

            elapsed = (time.perf_counter() - _start) * 1000
            log(f"VAD model loaded in {elapsed:.0f}ms", "VAD", "SUCCESS")
            emit_event(EventType.VAD_LOADED, f"{elapsed}ms")
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
                    f"No Faster-Whisper model found in {self.model_dir}. Downloading...",
                    "STT",
                    "INFO",
                )
            else:
                log(f"Using Whisper model from {self.model_dir}", "STT", "DEBUG")

            _start = time.perf_counter()
            log(f"Loading Whisper model: {w.model_size}...", "STT", "INFO")
            self.model = WhisperModel(
                w.model_size,
                device=w.device,
                compute_type="int8",
                cpu_threads=w.cpu_threads,
                num_workers=1,
                download_root=str(self.model_dir),
            )
            elapsed = (time.perf_counter() - _start) * 1000
            log(f"Whisper model loaded in {elapsed:.0f}ms", "STT", "SUCCESS")
            emit_event(EventType.WHISPER_LOADED, f"{elapsed}ms")
        except Exception as e:
            log(
                f"Error loading Whisper model: {type(e).__name__}: {e}",
                "STT",
                "ERROR",
            )
            raise

    def transcribe(self, audio_array: np.ndarray) -> tuple[str, int]:
        """Turns Spech(audio array) into a text. Returns (text, time_to_process)."""
        if not hasattr(self, "model"):
            raise RuntimeError("Whisper was used before whisper.load()")

        w = cfg.stt
        start_time = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio=audio_array,
            beam_size=w.beam_size,
            language=w.language,
            initial_prompt=w.initial_prompt,
            condition_on_previous_text=False,
        )
        text = " ".join([segment.text for segment in segments]).strip()
        return text, int((time.perf_counter() - start_time) * 1000)


class SRState(StrEnum):
    SLEEPING = "SLEEPING"
    AWAKE = "AWAKE"
    RECORDING = "RECORDING"
    WAITING = "WAITING"


class SpeechRecognizer:
    def __init__(self):
        self.vad = VAD()
        self.whisper = Whisper()

        self.preroll = deque(maxlen=cfg.vad.preroll_blocks)
        self.buffer: list[np.ndarray] = []
        self.audio_queue = queue.Queue()  # Queue containg (audio_array, listen_ms)

        self.stt_worker_thread = Thread(
            target=self._stt_worker, name="STT_WORKER_THREAD", daemon=True
        )

        if cfg.stt.start_state == "AWAKE":
            self.state = SRState.AWAKE
        else:
            self.state = SRState.SLEEPING

        self.awake_deadline = 0.0

    def update_deadline(self) -> None:
        """Updates deadline when needed, so it is not reached during talking or processing."""
        self.awake_deadline = time.monotonic() + cfg.stt.awake_timeout

    def is_deadline_expired(self) -> bool:
        return time.monotonic() > self.awake_deadline

    def set_state(self, new_state: SRState, detail: str | None = None) -> None:
        if self.state != new_state:
            self.state = new_state
            if new_state == SRState.AWAKE:
                self.update_deadline()

            emit_event(EventType.STT_CHANGED_STATE, new_state.value)

            payload = {"state": new_state.value}
            if detail:
                payload["detail"] = detail
            emit_event(EventType.UI_STATE_CHANGE, payload)

    def load(self):
        self.vad.load()
        self.whisper.load()

        em = EventManager()
        em.subscribe(
            EventType.KWS_KEYWORD_DETECTED,
            lambda e: self.set_state(SRState.AWAKE, f"Keyword: '{e.content}'"),
        )

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

            audio_array, listen_ms = item
            text, recog_ms = self.whisper.transcribe(audio_array)
            rtf = recog_ms / listen_ms

            text = text.strip()

            if text:
                emit_event(
                    EventType.UI_TRANSCRIPTION,
                    {
                        "text": text,
                        "listen_ms": listen_ms,
                        "recog_ms": recog_ms,
                        "rtf": rtf,
                    },
                )

                emit_event(EventType.STT_TRANSCRIBED, text)
                self.set_state(SRState.WAITING)

            self.audio_queue.task_done()

    def process(self, raw_chunk: np.ndarray) -> None:
        chunk = raw_chunk.squeeze(1) if raw_chunk.ndim > 1 else raw_chunk

        self.preroll.append(chunk.copy()) if self.state != SRState.WAITING else ...

        match self.state.value:
            case "WAITING":
                self.update_deadline()
                return
            case "SLEEPING":
                return
            case "RECORDING":
                self.buffer.append(chunk)
                vad_state = self.vad.process(chunk)
                if vad_state == "end":
                    if self.buffer:
                        full_audio = np.concatenate(self.buffer)
                        listen_ms = (len(full_audio) / cfg.audio.sample_rate) * 1000.0

                        if listen_ms > cfg.stt.min_command_ms:
                            self.audio_queue.put((full_audio, listen_ms))

                    self.buffer.clear()
                    self.update_deadline()
                    self.set_state(SRState.AWAKE)

            case "AWAKE":
                if self.is_deadline_expired():
                    self.set_state(
                        SRState.SLEEPING,
                        detail=f"Timeout ({int(cfg.stt.awake_timeout)}s)",
                    )
                    return

                vad_state = self.vad.process(chunk)
                if vad_state == "start":
                    self.buffer = list(self.preroll)
                    self.set_state(SRState.RECORDING)
