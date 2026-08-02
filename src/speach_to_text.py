#
# speech_to_text.py
# SoundDevice.InputStream[microphone] -> Voice Activity Detector[Silero VAD] -> KeyWordSpotter[Sherpa ONNX KWS] -> STT[faster-whisper] -> "result text"
#

import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import sherpa_onnx
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

# makes downloading Whisper models from HF faster
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


class KeyWordSpotter:
    def __init__(
        self,
        model_dir: str = "./data/sherpa_onnx_kws",
        keywords_file: str = "keywords.txt",
        num_threads: int = 2,
        score_threshold: float = 0.25,
    ):
        path = Path(model_dir)
        tokens = str(path / "tokens.txt")
        encoder = str(path / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        decoder = str(path / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        joiner = str(path / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")

        if not os.path.exists(tokens):
            raise FileNotFoundError(
                f"No Sherpa-ONNX model file found in directory: {model_dir}"
            )

        print("[KWS] Loading Sherpa-ONNX KeyWordSpotter model...", flush=True)
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            num_threads=num_threads,
            # score_threshold=score_threshold,
            feature_dim=80,
        )
        self.stream = self.kws.create_stream()
        print("[KWS] WakeWord ready!", flush=True)

    def process_chunk(
        self, chunk_np: np.ndarray, sample_rate: int = 16000
    ) -> str | None:
        """Takes audio chunk and compares with keywords.txt, returning spotted word if so."""
        self.stream.accept_waveform(sample_rate, chunk_np)
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            result = self.kws.get_result(self.stream)
            if result:
                keyword = result.strip()
                self.reset()
                return keyword
        return None

    def reset(self):
        """Resets stream for a new recognition."""
        self.stream = self.kws.create_stream()


class SpeechToText:
    def __init__(
        self,
        model_size: Literal["small", "small.en", "medium", "large-v3"] = "small",
        device: Literal["cpu", "cuda"] = "cpu",
        compute_type="int8",
        transcribe_beam_size: int = 5,
        language: str = "en",
        initial_prompt: str = "English language, speach to an assistant. Termins: Newt, Python, Linux, C++, code, programming.",
    ):
        self.trans_beam_size = transcribe_beam_size
        self.lang = language
        self.init_prompt = initial_prompt

        print(f"[{device.upper()}] Loading Whisper ({model_size})...", flush=True)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=6,
            num_workers=1,
            download_root="./data/whisper_models_cache",
        )
        print("Loading complete!", flush=True)

    def transcribe(self, audio_array: np.ndarray) -> tuple[str, int]:
        # beam_size=1 gives max speed, but less accurate results
        # language="uk" boosts speed
        start_time = time.perf_counter()

        segments, _ = self.model.transcribe(
            audio=audio_array,
            beam_size=self.trans_beam_size,
            language=self.lang,
            initial_prompt=self.init_prompt,
            condition_on_previous_text=False,  # Обов'язково для коротких команд!
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
        self.kws = KeyWordSpotter(
            model_dir=kws_model_dir,
            keywords_file=keywords_file,
            score_threshold=0.25,
        )

        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=0.5,
            min_silence_duration_ms=600,
            sampling_rate=16000,
            speech_pad_ms=60,
        )

        # Buffer for accumulating audio chunks while the user is speaking
        self.speech_buffer = []
        self.state: Literal["SLEEPING", "AWAKE", "RECORDING"] = "SLEEPING"

        self.awake_timeout = 10.0  # time in s, when assistant waits for commands
        self.awake_deadline = 0.0  # timestamp, when assistant goes to sleep

        self.is_speaking = False

        # queue for passing audio chunks to the STT worker thread
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        # ID array
        chunk_np = indata.squeeze(1)
        chunk_torch = torch.from_numpy(chunk_np)

        # checking VAD
        speech_dict = self.vad_iterator(chunk_torch)

        if speech_dict:
            if "start" in speech_dict:
                self.is_speaking = True
            elif "end" in speech_dict:
                self.is_speaking = False

        if (
            self.state == "SLEEPING" and self.is_speaking
        ):  # VAD detectes voice -> KWS recognizes keyword
            detected_keyword = self.kws.process_chunk(chunk_np)
            if detected_keyword:
                print(f"\n[!] Wake word detected: '{detected_keyword}'")
                print("[*] Assistant AWAKE! Listening for commands...")

                self.state = "AWAKE"
                self.awake_deadline = time.time() + self.awake_timeout
                self.speech_buffer = [chunk_np.copy()]

        elif self.state == "AWAKE":
            if time.time() > self.awake_deadline:
                print("\n[zZz] 10s timeout. Going back to SLEEPING...     ")
                self.state = "SLEEPING"
                self.kws.reset()
                return

            if speech_dict and "start" in speech_dict:
                sys.stdout.write("[@] Recording command...                \r")
                sys.stdout.flush()
                self.state = "RECORDING"
                self.speech_buffer = [chunk_np.copy()]

        elif self.state == "RECORDING":
            self.speech_buffer.append(chunk_np.copy())

            if speech_dict and "end" in speech_dict:  # VAD detected end of speach
                if self.speech_buffer:
                    full_audio = np.concatenate(self.speech_buffer)
                    listening_time_ms: float = (len(full_audio) / 16000) * 1000.0

                    # random sound protection
                    if listening_time_ms > 600:
                        self.audio_queue.put((full_audio.copy(), listening_time_ms))

                self.speech_buffer = []
                self.state = "AWAKE"
                self.awake_deadline = time.time() + self.awake_timeout
                print("\n[*] Command sent! Awake for next 10s...")

    def _stt_worker(self):
        """Different thread awaits audio array in queue and then processes it"""
        while True:
            item = self.audio_queue.get()
            if item is None:
                break

            audio_array, listening_time_ms = item

            sys.stdout.write("\r[#] Processing...               \r")
            sys.stdout.flush()

            text, recognition_time_ms = self.stt.transcribe(audio_array)
            rtf = recognition_time_ms / listening_time_ms
            if text:
                print(f"\n[:] You said: >> {text} <<\n")
                print(
                    f"⏱️  Listening time: {listening_time_ms:.0f} ms | "
                    f"Recognition time: {recognition_time_ms:.0f} ms | "
                    f"RTF: {rtf:.2f}x"
                )

            self.audio_queue.task_done()

    def start(self):
        # Starting background Whisper thread
        stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
        stt_thread.start()

        print("\n" + "=" * 50)
        print(" Ready to speak?")
        print(" Press Ctrl+C to quit.")
        print("=" * 50 + "\n")

        try:
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                blocksize=512,
                dtype="float32",
                callback=self._audio_callback,
            ):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.audio_queue.put(None)  # Ending worker thread
            stt_thread.join()
