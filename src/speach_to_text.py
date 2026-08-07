#
# speech_to_text.py
# SoundDevice.InputStream[microphone] -> KeyWordSpotter[Sherpa ONNX KWS] -> Voice Activity Detector[Silero VAD] -> STT[faster-whisper] -> "recognized text"
#

import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import rich
import sherpa_onnx
import sounddevice as sd
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

from .config import DATA_DIR, cfg
from .events import EventManager, emit_event

# makes downloading Whisper models from HF faster
load_dotenv()  # loads HF_TOKEN from .env file.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# # limiting ONNX Runtime CPU Usage in Sleaping Mode # Note: Makes faster-whisper slow as hell
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


def print(msg: str, end="\n", force=False):
    if cfg.profiler or force:
        rich.print(msg, end=end)


class KeyWordSpotter:
    def __init__(self):
        path: Path = DATA_DIR / cfg.kws.model_dir
        tokens = str(path / "tokens.txt")
        encoder = str(path / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        decoder = str(path / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        joiner = str(path / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")

        if not os.path.exists(tokens):
            print(f"[WARN] No Sherpa model in: {path}. Donwloading...")
            self._download_sherpa_onnx_model(path)
            # raise FileNotFoundError(f"No Sherpa model in: {cfg.kws.model_dir}")

        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=f"{DATA_DIR / cfg.kws.keywords_file}",
            num_threads=cfg.kws.num_threads,
            keywords_threshold=cfg.kws.score_threshold,
            feature_dim=80,
        )

        # self.stream = self.kws.create_stream()
        self.reset()

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
        self.stream.accept_waveform(cfg.audio.sample_rate, chunk_np)
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            result = self.kws.get_result(self.stream)
            if result:
                keyword = result.strip()
                self.reset()
                return keyword
        return None

    def reset(self):
        self.stream = self.kws.create_stream()


class Whisper:
    def __init__(self):
        w = cfg.stt
        model_dir: Path = DATA_DIR / w.download_root

        if not model_dir.exists():
            print(f"[I] No Faster-Whisper model found in {model_dir}. Downloading...")

        self.model = WhisperModel(
            w.model_size,
            device=w.device,
            compute_type=w.compute_type,
            cpu_threads=w.cpu_threads,
            num_workers=1,
            download_root=str(model_dir),
        )

    def transcribe(self, audio_array: np.ndarray) -> tuple[str, int]:
        """Turns Spech(audio array) into a text. Returns (text, time_to_process)."""
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


class Listener:
    def __init__(self, stt_model: Whisper, kws_model: KeyWordSpotter | None = None):
        self.MODE: Literal["KWS", "DIRECT"] = cfg.stt.pipeline_mode

        self.stt = stt_model
        self.kws = kws_model

        if self.kws:
            self.MODE = "KWS"

        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=cfg.vad.threshold,
            min_silence_duration_ms=cfg.vad.min_silence_duration_ms,
            sampling_rate=cfg.audio.sample_rate,
            speech_pad_ms=cfg.vad.speech_pad_ms,
        )

        self.speech_buffer = []  # Buffer for accumulating audio chunks while the user is speaking
        self.preroll_buffer = deque(maxlen=cfg.vad.preroll_blocks)
        self.state: Literal["SLEEPING", "AWAKE", "RECORDING"] = (
            "SLEEPING" if self.MODE == "KWS" else "AWAKE"
        )
        emit_event("PROFILER_SET_STATE", self.state)

        self.awake_deadline = 0.0
        self.audio_queue = queue.Queue()

        self.is_busy = False

        self.stt_thread = threading.Thread(target=self._stt_worker, daemon=True)

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        if self.is_busy:  # if assistant is generating response or speaking
            self.awake_deadline = time.time() + cfg.stt.awake_timeout
            return

        # ID array
        chunk_np = indata.squeeze(1)
        chunk_torch = torch.from_numpy(chunk_np)

        self.preroll_buffer.append(chunk_np.copy())  # updating preroll
        speech_dict = self.vad_iterator(chunk_torch)  # checking VAD

        if self.state == "SLEEPING":
            # kws exists
            if not isinstance(self.kws, KeyWordSpotter):
                raise SystemError(
                    "[!] ERROR: Listener.state is 'SLEEPING' but KWS was not initialized."
                )
            detected_keyword = self.kws.process_chunk(chunk_np)
            if detected_keyword:
                self.state = "AWAKE"
                emit_event("PROFILER_SET_STATE", "AWAKE")

                self.awake_deadline = time.time() + cfg.stt.awake_timeout
                self.kws.reset()

                emit_event(
                    "UI_STATE_CHANGE",
                    {"state": self.state, "detail": f"Keyword: '{detected_keyword}'"},
                )

        elif self.state == "AWAKE":
            if (self.kws) and time.time() > self.awake_deadline:
                self.state = "SLEEPING"
                emit_event("PROFILER_SET_STATE", self.state)

                self.kws.reset()
                emit_event(
                    "UI_STATE_CHANGE",
                    {
                        "state": "SLEEPING",
                        "detail": f"Timeout ({int(cfg.stt.awake_timeout)}s)",
                    },
                )

                return

            if speech_dict and "start" in speech_dict:
                self.state = "RECORDING"
                emit_event("PROFILER_SET_STATE", self.state)

                emit_event("UI_STATE_CHANGE", {"state": self.state})
                self.speech_buffer = list(self.preroll_buffer)

        elif self.state == "RECORDING":
            self.speech_buffer.append(chunk_np.copy())

            if speech_dict and "end" in speech_dict:  # VAD detected end of speach
                if self.speech_buffer:
                    full_audio = np.concatenate(self.speech_buffer)
                    listen_ms: float = (
                        len(full_audio) / cfg.audio.sample_rate
                    ) * 1000.0

                    # random sound protection
                    if listen_ms > cfg.stt.min_command_ms:
                        self.audio_queue.put((full_audio.copy(), listen_ms))

                self.speech_buffer = []
                self.state = "AWAKE"
                emit_event("PROFILER_SET_STATE", self.state)

                self.awake_deadline = time.time() + cfg.stt.awake_timeout

    def _stt_worker(self):
        """Different thread awaits audio array in queue and then processes it"""
        while True:
            item = self.audio_queue.get()
            if item is None:
                break

            audio_array, listen_ms = item
            text, recog_ms = self.stt.transcribe(audio_array)
            rtf = recog_ms / listen_ms

            if text:
                emit_event(
                    "UI_TRANSCRIPTION",
                    {
                        "text": text,
                        "listen_ms": listen_ms,
                        "recog_ms": recog_ms,
                        "rtf": rtf,
                    },
                )

                emit_event(
                    "STT_TRANSCRIBE", text
                )  # This events shows that now transcribed text must be operated

                em = EventManager.get_instace()
                em.set_flag("stt_runtime", False)

                self.is_busy = True
                # stt thread is blocked until the "stt_runtime" flag is not set
                _ = em.wait_for("stt_runtime")

                self.awake_deadline = time.time() + cfg.stt.awake_timeout
                emit_event("PROFILER_SET_STATE", "AWAKE")

                self.is_busy = False

            self.audio_queue.task_done()

    def close(self):
        self.audio_queue.put(None)  # Ending worker thread
        self.stt_thread.join()

        emit_event("STT_FINISH")

    def start(self):
        emit_event("PROFILER_START")

        # Starting background Whisper thread
        self.stt_thread.start()
        emit_event("UI_BANNER")

        try:
            with sd.InputStream(
                samplerate=cfg.audio.sample_rate,
                channels=cfg.audio.channels,
                blocksize=cfg.audio.blocksize,
                dtype=cfg.audio.dtype,
                callback=self._audio_callback,
            ):
                while True:
                    sd.sleep(100)
        finally:
            self.close()
