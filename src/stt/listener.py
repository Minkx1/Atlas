#
#  listener.py
#

import time
from collections.abc import Callable
from threading import Thread

import numpy as np
import sounddevice as sd

from ..core.config import cfg
from ..core.events import EventType, emit_event, log


class Listener:
    def __init__(self, chunk_processor: Callable[[np.ndarray], None]) -> None:
        self.processor = chunk_processor
        self.audio_input_thread = Thread(
            target=self._audio_input, name="LISTENER_INPUT_THREAD", daemon=True
        )

        self._running = False
        self._is_muted = False

        self._last_wave_emit = 0.0
        self._wave_fps_interval = 0.04  # passing audio wave data to UI

    def _audio_input(self):
        try:
            with sd.InputStream(
                samplerate=cfg.audio.sample_rate,
                channels=cfg.audio.channels,
                blocksize=cfg.audio.blocksize,
                dtype=cfg.audio.dtype,
            ) as stream:
                while self._running:
                    indata, _ = stream.read(cfg.audio.blocksize)

                    if not self._is_muted:
                        try:
                            self.processor(indata)

                            now = time.monotonic()
                            if now - self._last_wave_emit >= self._wave_fps_interval:
                                self._last_wave_emit = now
                                audio_mono = indata[:, 0] if indata.ndim > 1 else indata
                                rms = float(np.sqrt(np.mean(audio_mono**2)))
                                emit_event(EventType.STT_AUDIOWAVE, rms)
                        except Exception as e:
                            log(
                                f"Error processing audio chunk: {e}",
                                "LISTENER",
                                "ERROR",
                            )
                            raise

        except Exception as e:
            log(f"Microphone input error: {e}", "LISTENER", "ERROR")
            raise

    def start(self):
        self._running = True
        emit_event(EventType.STT_START)

        self.audio_input_thread.start()

    def mute(self):
        self._is_muted = True

    def unmute(self):
        self._is_muted = False

    def close(self):
        self._running = False

        if self.audio_input_thread is not None and self.audio_input_thread.is_alive():
            self.audio_input_thread.join(timeout=2.0)

        emit_event(EventType.STT_FINISH)
