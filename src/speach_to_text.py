#
# speech_to_text.py
# SoundDevice.InputStream[microphone] -> Voice Activity Detector[Silero VAD] -> KeyWordSpotter[Sherpa ONNX KWS] -> STT[faster-whisper] -> "result text"
#

import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import sherpa_onnx
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

from .config import cfg
from .ui import AssistantUI

# makes downloading Whisper models from HF faster
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


class KeyWordSpotter:
    def __init__(self):
        path = Path(cfg.kws.model_dir)
        tokens = str(path / "tokens.txt")
        encoder = str(path / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        decoder = str(path / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        joiner = str(path / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")

        if not os.path.exists(tokens):
            raise FileNotFoundError(f"No Sherpa model in: {cfg.kws.model_dir}")

        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=cfg.kws.keywords_file,
            num_threads=cfg.kws.num_threads,
            keywords_threshold=cfg.kws.score_threshold,
            feature_dim=80,
        )
        self.stream = self.kws.create_stream()

    def process_chunk(self, chunk_np: np.ndarray) -> str | None:
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


class SpeechToText:
    def __init__(self):
        w = cfg.whisper
        self.model = WhisperModel(
            w.model_size,
            device=w.device,
            compute_type=w.compute_type,
            cpu_threads=w.cpu_threads,
            num_workers=1,
            download_root=w.download_root,
        )

    def transcribe(self, audio_array: np.ndarray) -> tuple[str, int]:
        w = cfg.whisper
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
    def __init__(
        self,
        stt_model: SpeechToText,
        kws_model_dir: str = "./data/sherpa_onnx_kws",
        keywords_file: str = "keywords.txt",
    ):
        self.stt = stt_model
        self.kws = KeyWordSpotter()

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
        self.state: Literal["SLEEPING", "AWAKE", "RECORDING"] = "SLEEPING"

        self.awake_deadline = 0.0
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        # ID array
        chunk_np = indata.squeeze(1)
        chunk_torch = torch.from_numpy(chunk_np)

        self.preroll_buffer.append(chunk_np.copy())  # updating preroll
        speech_dict = self.vad_iterator(chunk_torch)  # checking VAD

        if self.state == "SLEEPING":
            detected_keyword = self.kws.process_chunk(chunk_np)
            if detected_keyword:
                self.state = "AWAKE"
                self.awake_deadline = time.time() + cfg.awake_timeout
                self.kws.reset()

                AssistantUI.print_state_change(
                    self.state, f"Wake word: '{detected_keyword}'"
                )

        elif self.state == "AWAKE":
            if time.time() > self.awake_deadline:
                self.state = "SLEEPING"
                self.kws.reset()
                AssistantUI.print_state_change("SLEEPING", "Timeout (10s)")
                return

            if speech_dict and "start" in speech_dict:
                self.state = "RECORDING"
                AssistantUI.print_state_change("RECORDING")
                self.speech_buffer = list(self.preroll_buffer)
                # self.speech_buffer = [c.copy() for c in self.preroll_buffer]

        elif self.state == "RECORDING":
            self.speech_buffer.append(chunk_np.copy())

            if speech_dict and "end" in speech_dict:  # VAD detected end of speach
                if self.speech_buffer:
                    full_audio = np.concatenate(self.speech_buffer)
                    listen_ms: float = (
                        len(full_audio) / cfg.audio.sample_rate
                    ) * 1000.0

                    # random sound protection
                    if listen_ms > cfg.min_command_ms:
                        self.audio_queue.put((full_audio.copy(), listen_ms))

                self.speech_buffer = []
                self.state = "AWAKE"
                self.awake_deadline = time.time() + cfg.awake_timeout

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
                AssistantUI.print_transcription(text, listen_ms, recog_ms, rtf)

            self.audio_queue.task_done()

    def start(self):
        # Starting background Whisper thread
        stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
        stt_thread.start()
        AssistantUI.print_banner()

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
        except KeyboardInterrupt:
            from .ui import console

            console.print("\n[dim]Stopping assistant...[/dim]")
        finally:
            self.audio_queue.put(None)  # Ending worker thread
            stt_thread.join()
