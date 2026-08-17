#
# speech_to_text.py
# SoundDevice.InputStream[microphone] -> KeyWordSpotter[Sherpa ONNX KWS] -> Voice Activity Detector[Silero VAD] -> STT[faster-whisper] -> "recognized text"
#

from __future__ import annotations  # some type annotations shit

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


import os
import queue
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from threading import Thread

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from .config import DATA_DIR, cfg
from .events import EventType, emit_event, log

# makes downloading Whisper models from HF faster
load_dotenv()  # loads HF_TOKEN from .env file.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


class VAD:
    def __init__(self) -> None:
        self.is_speaking = False

    def load(self):
        from silero_vad import VADIterator, load_silero_vad

        try:
            _start = time.perf_counter()
            log("Loading Silero VAD model...", "VAD", "INFO")
            self.model = load_silero_vad()
            self.iterator = VADIterator(
                self.model,
                threshold=cfg.vad.threshold,
                min_silence_duration_ms=cfg.vad.min_silence_duration_ms,
                sampling_rate=cfg.audio.sample_rate,
                speech_pad_ms=cfg.vad.speech_pad_ms,
            )
            elapsed = (time.perf_counter() - _start) * 1000
            log(f"VAD model loaded in {elapsed:.0f}ms", "VAD", "SUCCESS")
            emit_event(EventType.VAD_LOADED, f"{elapsed}ms")
        except Exception as e:  # noqa: BLE001
            log(
                f"Error loading VAD model: {type(e).__name__}: {e}",
                "VAD",
                "ERROR",
            )

    @staticmethod
    def _normalize_chunk(audio_chunk: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio_chunk, np.ndarray):
            import torch

            return torch.from_numpy(
                audio_chunk.squeeze(1) if audio_chunk.ndim > 1 else audio_chunk
            )
        return audio_chunk

    def process(self, audio_chunk: np.ndarray | torch.Tensor) -> str:
        """Returns state of speech: 'silence', 'start', 'speaking', 'end'."""
        if not hasattr(self, "iterator"):
            raise RuntimeError("VAD was used before vad.load()")

        chunk = self._normalize_chunk(audio_chunk)

        voice_dict = self.iterator(chunk)
        if voice_dict:
            if "start" in voice_dict:
                self.is_speaking = True
                return "start"
            elif "end" in voice_dict:
                self.is_speaking = False
                return "end"

        return "speaking" if self.is_speaking else "silence"


class KeyWordSpotter:
    """Sherpa-ONNX Keyword Spotter model.

    Usage:
    ```python
    kws = KeyWordSpotter()

    kw = kws.process_chunk(audio_chunk)
    if kw:
        print("Keyword was detected: " + kw)
    ```
    """

    def __init__(self):
        path: Path = DATA_DIR / cfg.kws.model_dir
        self.tokens = str(path / "tokens.txt")
        self.encoder = str(path / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        self.decoder = str(path / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        self.joiner = str(path / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")

        if not os.path.exists(self.tokens):
            log(
                f"No Sherpa model in: {path}. Donwloading...",
                source="KWS",
                level="WARN",
            )
            self._download_sherpa_onnx_model(path)
            # raise FileNotFoundError(f"No Sherpa model in: {cfg.kws.model_dir}")

    def load(self):
        import sherpa_onnx

        try:
            _start = time.perf_counter()
            log("Loading Sherpa-ONNX KWS model...", "KWS", "INFO")
            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=self.tokens,
                encoder=self.encoder,
                decoder=self.decoder,
                joiner=self.joiner,
                keywords_file=f"{DATA_DIR / cfg.kws.keywords_file}",
                num_threads=cfg.kws.num_threads,
                keywords_threshold=cfg.kws.score_threshold,
                feature_dim=80,
            )

            self.stream = self.kws.create_stream()
            elapsed = (time.perf_counter() - _start) * 1000
            log(f"KWS model loaded in {elapsed:.0f}ms", "KWS", "SUCCESS")
            emit_event(EventType.KWS_LOADED, f"{elapsed}ms")
        except Exception as e:  # noqa: BLE001
            log(
                f"Error loading KWS model: {type(e).__name__}: {e}",
                "KWS",
                "ERROR",
            )

    @staticmethod
    def _download_sherpa_onnx_model(model_path: Path):
        import shutil
        import tarfile
        import urllib.request
        from urllib.error import URLError

        url = (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
        )
        archive_name = "sherpa_kws_temp.tar.bz2"
        archive_path = model_path.parent / archive_name
        extracted_folder_name = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
        extracted_path = model_path.parent / extracted_folder_name

        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[I] Downloading Sherpa-ONNX KWS model from {url}...")
            urllib.request.urlretrieve(url, archive_path)

            print("[I] Extracting model archive...")
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(path=model_path.parent)

            if model_path.exists():
                shutil.rmtree(model_path)
            extracted_path.rename(model_path)

            print(f"[I] Model successfully installed to {model_path}")

        except (URLError, tarfile.TarError, OSError) as e:
            if extracted_path.exists():
                shutil.rmtree(extracted_path, ignore_errors=True)
            raise RuntimeError(
                f"[!] Failed to download or extract Sherpa-ONNX model: {e}"
            ) from e
        finally:
            if archive_path.exists():
                archive_path.unlink()

    def process_chunk(self, chunk_np: np.ndarray) -> str | None:
        """Processes audio chunk. If keyword was spotted: returns it. Else: returns None."""
        if not hasattr(self, "kws"):
            raise RuntimeError("KWS was used before kws.load()")

        self.stream.accept_waveform(cfg.audio.sample_rate, chunk_np)
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            result = self.kws.get_result(self.stream)
            if result:
                keyword = result.strip()
                self.reset()
                emit_event(EventType.STT_KEYWORD_DETECTED, keyword)
                return keyword
        return None

    def reset(self):
        if not hasattr(self, "kws"):
            raise RuntimeError("KWS was used before kws.load()")
        self.stream = self.kws.create_stream()


class Whisper:
    def __init__(self):
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
                compute_type=w.compute_type,
                cpu_threads=w.cpu_threads,
                num_workers=1,
                download_root=str(self.model_dir),
            )
            elapsed = (time.perf_counter() - _start) * 1000
            log(f"Whisper model loaded in {elapsed:.0f}ms", "STT", "SUCCESS")
            emit_event(EventType.WHISPER_LOADED, f"{elapsed}ms")
        except Exception as e:  # noqa: BLE001
            log(
                f"Error loading Whisper model: {type(e).__name__}: {e}",
                "STT",
                "ERROR",
            )

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


class LState(StrEnum):
    SLEEPING = "SLEEPING"
    AWAKE = "AWAKE"
    RECORDING = "RECORDING"
    WAITING = "WAITING"


class AudioPipeline:
    """Operates states & processes raw audio."""

    def __init__(
        self,
        vad: VAD,
        kws: KeyWordSpotter,
        on_audio_ready: Callable[[np.ndarray, float], None] | None = None,
    ) -> None:
        self.vad = vad
        self.kws = kws
        self.on_audio_ready = on_audio_ready

        if cfg.stt.start_state == "AWAKE":
            self.state = LState.AWAKE
        else:
            self.state = LState.SLEEPING

        self.awake_deadline = 0.0

        self.preroll_buffer = deque(maxlen=cfg.vad.preroll_blocks)
        self.speech_buffer = []

        self.update_deadline()

    def set_state(self, new_state: LState, detail: str | None = None) -> None:
        if self.state != new_state:
            self.state = new_state
            if new_state == LState.AWAKE:
                self.update_deadline()

            emit_event(EventType.STT_CHANGED_STATE, new_state.value)

            payload = {"state": new_state.value}
            if detail:
                payload["detail"] = detail
            emit_event(EventType.UI_STATE_CHANGE, payload)

    def get_state(self) -> LState:
        return self.state

    def update_deadline(self) -> None:
        self.awake_deadline = time.monotonic() + cfg.stt.awake_timeout

    def is_deadline_expired(self) -> bool:
        return time.monotonic() > self.awake_deadline

    def process(self, raw_chunk: np.ndarray) -> None:
        chunk = raw_chunk.squeeze(1) if raw_chunk.ndim > 1 else raw_chunk

        if self.state == LState.WAITING:
            self.update_deadline()
            return

        self.preroll_buffer.append(chunk.copy())

        if self.state == LState.SLEEPING:
            self._handle_sleeping(chunk)
        elif self.state == LState.AWAKE:
            self._handle_awake(chunk)
        elif self.state == LState.RECORDING:
            self._handle_recording(chunk)

    def _handle_sleeping(self, chunk: np.ndarray) -> None:
        detected_keyword = self.kws.process_chunk(chunk)
        if detected_keyword:
            self.update_deadline()
            self.kws.reset()
            self.set_state(LState.AWAKE, detail=f"Keyword: '{detected_keyword}'")

    def _handle_awake(self, chunk: np.ndarray) -> None:
        if self.is_deadline_expired():
            self.kws.reset()
            self.set_state(
                LState.SLEEPING, detail=f"Timeout ({int(cfg.stt.awake_timeout)}s)"
            )
            return

        speech_state = self.vad.process(chunk)
        if speech_state == "start":
            self.speech_buffer = list(self.preroll_buffer)
            self.set_state(LState.RECORDING)

    def _handle_recording(self, chunk: np.ndarray) -> None:
        self.speech_buffer.append(chunk.copy())
        speech_state = self.vad.process(chunk)

        if speech_state == "end":
            if self.speech_buffer:
                full_audio = np.concatenate(self.speech_buffer)
                listen_ms = (len(full_audio) / cfg.audio.sample_rate) * 1000.0

                if listen_ms > cfg.stt.min_command_ms:
                    if not self.on_audio_ready:
                        log(
                            "[!] AudioPipeline was registered, but `on_audio_ready` callback was not.",
                            "AudioPipeline",
                            "ERROR",
                        )
                        raise RuntimeError(
                            "[!] AudioPipeline was registered, but `on_audio_ready` callback was not."
                        )
                    self.on_audio_ready(full_audio, listen_ms)

            self.speech_buffer.clear()
            self.update_deadline()
            self.set_state(LState.AWAKE)

    def reset_buffers(self) -> None:
        self.preroll_buffer.clear()
        self.speech_buffer.clear()


class Listener:
    def __init__(self, vad: VAD, whisper: Whisper, kws: KeyWordSpotter) -> None:
        self.whisper = whisper

        self.vad = vad
        self.kws = kws
        self.pipeline = AudioPipeline(self.vad, self.kws, self._on_audio_recorded)

        self.audio_input_thread = Thread(
            target=self._audio_input, name="LISTENER_INPUT_THREAD", daemon=True
        )
        self.stt_worker_thread = Thread(
            target=self._stt_worker, name="LISTENER_WORKER_THREAD", daemon=True
        )
        self.audio_queue = queue.Queue()  # Queue containg (audio_array, listen_ms)

        self._running = False
        self._is_muted = False

        self._last_wave_emit = 0.0
        self._wave_fps_interval = 0.04

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        if not self._is_muted:
            self.pipeline.process(indata)

            now = time.monotonic()
            if now - self._last_wave_emit >= self._wave_fps_interval:
                self._last_wave_emit = now
                audio_mono = indata[:, 0] if indata.ndim > 1 else indata

                rms = float(np.sqrt(np.mean(audio_mono**2)))
                emit_event(EventType.STT_AUDIOWAVE, rms)

    def _on_audio_recorded(self, full_audio: np.ndarray, listen_ms: float):
        self.audio_queue.put((full_audio.copy(), listen_ms))

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
                self.pipeline.set_state(LState.WAITING)

                # self.pipeline.set_state(
                #     LState.AWAKE
                #     if self.pipeline.get_state() != LState.SLEEPING
                #     else self.pipeline.get_state()
                # )
                # self.pipeline.update_deadline()

            self.audio_queue.task_done()

    def _audio_input(self):
        try:
            with sd.InputStream(
                samplerate=cfg.audio.sample_rate,
                channels=cfg.audio.channels,
                blocksize=cfg.audio.blocksize,
                dtype=cfg.audio.dtype,
                callback=self._audio_callback,
            ):
                while self._running:
                    sd.sleep(100)
        except Exception as e:  # noqa: BLE001
            log(f"Microphone input error: {e}", "LISTENER", "ERROR")

    def start(self):
        self._running = True
        emit_event(EventType.STT_START)

        self.stt_worker_thread.start()
        self.audio_input_thread.start()

    def mute(self):
        self._is_muted = True

    def unmute(self):
        self._is_muted = False

    def close(self):
        self._running = False
        self.audio_queue.put(None)

        if (
            hasattr(self, "stt_worker_thread")
            and self.stt_worker_thread is not None
            and self.stt_worker_thread.is_alive()
        ):
            self.stt_worker_thread.join(timeout=2.0)
        if (
            hasattr(self, "audio_input_thread")
            and self.audio_input_thread is not None
            and self.audio_input_thread.is_alive()
        ):
            self.audio_input_thread.join(timeout=2.0)

        emit_event(EventType.STT_FINISH)
