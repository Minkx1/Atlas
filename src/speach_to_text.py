#
# speech_to_text.py
# STT + VAD
#

import sys
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad


class SpeachToText:
    def __init__(self):
        pass

    def transcribe(self, audio_file):
        pass


class VAD:
    def __init__(self, treshold: float = 0.5, min_silence_ms: int = 200) -> None:
        print("Loading Silero VAD...", flush=True)

        self.treshold = treshold
        self.min_silence_ms = (
            min_silence_ms  # 100 - great for tests, >200 greater for spech
        )

        self.is_speaking = False

        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.model,
            threshold=self.treshold,
            min_silence_duration_ms=self.min_silence_ms,
            sampling_rate=16000,
        )

        print("Loading complete.", flush=True)

    def run(
        self, callback: Callable[[np.ndarray, int, dict, sd.CallbackFlags], None]
    ) -> None:
        # Another debug print
        print("\n" + "=" * 50)
        print(" Silero VAD is running!")
        print(f" Configuration: {self.treshold=}, {self.min_silence_ms=}ms")
        print(" Press Ctrl+C to exit.")
        print("=" * 50 + "\n")

        try:
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                blocksize=512,
                dtype="float32",
                callback=callback,
            ):
                while True:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            print("\n\nVAD was finished. ")


def test1():
    vad = VAD(treshold=0.5, min_silence_ms=150)

    def audio_callback(indata: np.ndarray, frames, time_info, status):
        # nonlocal is_speaking

        audio_chunk = torch.from_numpy(indata.squeeze(1))

        speech_dict = vad.vad_iterator(audio_chunk)

        # Оновлюємо стан мовлення при виявленні подій start / end
        if speech_dict:
            if "start" in speech_dict:
                vad.is_speaking = True
            elif "end" in speech_dict:
                vad.is_speaking = False

        # Rewrting console
        if vad.is_speaking:
            sys.stdout.write("\r🗣️  [ * SPEACH * ] ")
        else:
            sys.stdout.write("\r🤫  [ . silens . ] ")
        sys.stdout.flush()

    vad.run(audio_callback)


def test2(): ...


if __name__ == "__main__":
    test1()
