#
# stt / listener.py
# Listens InputStream and processes audio data
#

import logging
import time
from collections.abc import Callable
from threading import Thread

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from atlas.core.config import cfg
from atlas.core.events import EventManager, EventType

log = logging.getLogger(__name__)


class Listener:
    def __init__(
        self,
        chunk_processor: Callable[[np.ndarray], None],
        events: EventManager | None = None,
    ) -> None:
        self.events = events or EventManager()
        self.processor = chunk_processor

        self.audio_input_thread = Thread(
            target=self._audio_input,
            name="LISTENER_INPUT_THREAD",
            daemon=True,
        )

        self._running = False

        self._last_wave_emit = 0.0
        self._wave_fps_interval = 0.04

    def _get_input_samplerate(self) -> int:
        target_sr = cfg.audio.sample_rate

        device = sd.query_devices(kind="input")
        native_sr = int(device["default_samplerate"])

        try:
            sd.check_input_settings(
                samplerate=target_sr,
                channels=cfg.audio.channels,
                dtype=cfg.audio.dtype,
            )

            log.info("Input device supports requested sample rate: %s Hz", target_sr)

            return target_sr

        except sd.PortAudioError:
            log.warning(
                "Input device does not support %s Hz; using native rate %s Hz",
                target_sr,
                native_sr,
            )

            return native_sr

    def _audio_input(self):
        try:
            input_sr = self._get_input_samplerate()
            target_sr = cfg.audio.sample_rate

            with sd.InputStream(
                samplerate=input_sr,
                channels=cfg.audio.channels,
                blocksize=cfg.audio.blocksize,
                dtype=cfg.audio.dtype,
            ) as stream:
                log.info("Audio stream opened: %s Hz -> %s Hz", input_sr, target_sr)

                while self._running:
                    indata, _ = stream.read(cfg.audio.blocksize)

                    try:
                        if input_sr != target_sr:
                            audio = resample_poly(
                                indata,
                                target_sr,
                                input_sr,
                                axis=0,
                            ).astype(np.float32, copy=False)
                        else:
                            audio = indata

                        self.processor(audio)

                        # checks whether to send wave event
                        now = time.monotonic()
                        if now - self._last_wave_emit >= self._wave_fps_interval:
                            self._last_wave_emit = now
                            audio_mono = audio[:, 0] if audio.ndim > 1 else audio

                            rms = float(np.sqrt(np.mean(audio_mono**2)))
                            self.events.emit(EventType.STT_AUDIOWAVE, {"rms": rms})

                    except Exception:
                        log.exception("Error processing audio chunk")
                        raise

        except Exception:
            log.exception("Microphone input error; STT input stopped")
            raise

    def start(self):
        self._running = True
        self.events.emit(EventType.STT_START, {})
        self.audio_input_thread.start()

    def close(self):
        self._running = False

        if self.audio_input_thread is not None and self.audio_input_thread.is_alive():
            self.audio_input_thread.join(timeout=2.0)

        self.events.emit(EventType.STT_FINISH, {})
