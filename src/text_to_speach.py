#
# This module contains logic of Natural Language Understanding and speech recognition
#
# Made by Minkx1 (Optimized for non-blocking asyncio & sounddevice)
#

import asyncio
import subprocess
import sys

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

if __name__ == "__main__":
    from config import DATA_DIR, cfg
else:
    from .config import DATA_DIR, cfg


class TextToSpeech:
    def __init__(self) -> None:
        s = cfg.tts
        path = DATA_DIR / s.model_path
        if not path.exists():
            self._download_model()

        self.voice = PiperVoice.load(
            path, use_cuda=s.use_cuda, download_dir=path.parent
        )
        self.syn_config = SynthesisConfig(
            volume=s.volume,  # half as loud
            length_scale=s.length_scale,  # twice as slow
            noise_scale=s.noise_scale,  # more audio variation
            noise_w_scale=s.noise_w_scale,  # more speaking variation
            normalize_audio=s.normalize_audio,  # use raw audio from voice
        )

    def _download_model(self):
        model_path = DATA_DIR / cfg.tts.model_path
        name = model_path.stem
        print(f"[I] Downloading PiperTTS model: {name}")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [sys.executable, "-m", "piper.download_voices", name],
                cwd=model_path.parent,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[!] Error during downloading PiperTTS model({name}): {e}")
        else:
            print(f"[V] PiperTTS model({name}) was downloaded succesfully.")

    def _synthesize_and_play_blocking(self, text: str) -> None:
        """Synchronicaly creates and plays sound in bg"""
        if not text or not text.strip():
            return

        try:
            audio_chunks = list(self.voice.synthesize(text, self.syn_config))
            audio_array = np.concatenate(
                [chunk.audio_float_array for chunk in audio_chunks]
            )

            sd.play(audio_array, samplerate=audio_chunks[0].sample_rate)
            sd.wait()
        except Exception as e:  # noqa: BLE001
            print(f"[!] Error occured during TTS synthesis: {e}")

    async def speak(self, text: str) -> None:
        """Async Text-To-Speech (non-blocking for asyncio loop)"""
        await asyncio.to_thread(self._synthesize_and_play_blocking, text)


if __name__ == "__main__":
    tts = TextToSpeech()
    asyncio.run(tts.speak("Hello World!"))
