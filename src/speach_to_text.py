#
# speech_to_text.py
# SoundDevice -> VAD -> WakeWord[not-for-tests] -> STT -> "text"
#

import os
import queue
import sys
import threading
import time
from typing import Literal

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

os.environ["HF_XET_HIGH_PERFORMANCE"] = (
    "1"  # makes downloading Whisper models from HF faster
)


class SpeechToText:
    def __init__(
        self,
        model_size: Literal["small", "small.en", "medium", "large-v3"] = "small",
        device: Literal["cpu", "cuda"] = "cpu",
        compute_type="int8",
        transcribe_beam_size: int = 5,
        language: str = "uk",
        initial_prompt: str = "Українська мова, правильна пунктуація. Терміни: Newt, Python, Linux, C++, код.",
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
            download_root="./data/models_cache",
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
    def __init__(self, stt_model: SpeechToText):
        self.stt = stt_model
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
                self.speech_buffer = []  # Clearing buffer at the start of speech
                sys.stdout.write("\r@  [Listening...]                  \r")
                sys.stdout.flush()

            elif "end" in speech_dict and self.is_speaking:
                self.is_speaking = False
                # Constructing complete audio array
                if self.speech_buffer:
                    full_audio = np.concatenate(self.speech_buffer)
                    listening_time_ms = (len(full_audio) / 16000.0) * 1000.0

                    # putting in stt queue
                    self.audio_queue.put((full_audio.copy(), listening_time_ms))
                self.speech_buffer = []

        if self.is_speaking:
            self.speech_buffer.append(chunk_np.copy())

    def _stt_worker(self):
        """Different thread awaits audio array in queue and then processes it"""
        while True:
            item = self.audio_queue.get()
            if item is None:
                break

            audio_array, listening_time_ms = item

            sys.stdout.write("\r# [Processing...]               \r")
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
