import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Thread

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

_MAIN = __name__ == "__main__"
if not _MAIN:
    from ..core.config import cfg
    from ..core.events import EventType, emit_event, log
else:
    # changing execution dir to src/ for proper importing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import cfg
    from core.events import EventType, emit_event, log


class Listener:
    def __init__(self, chunk_processor: Callable[[np.ndarray], None]) -> None:
        self.processor = chunk_processor

        self.audio_input_thread = Thread(
            target=self._audio_input,
            name="LISTENER_INPUT_THREAD",
            daemon=True,
        )

        self._running = False
        self._is_muted = False

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

            log(
                f"Input device supports requested sample rate: {target_sr} Hz",
                "LISTENER",
                "INFO",
            )

            return target_sr

        except sd.PortAudioError:
            log(
                f"Input device does not support {target_sr} Hz. "
                f"Using native rate {native_sr} Hz and resampling to {target_sr} Hz.",
                "LISTENER",
                "WARN",
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
                log(
                    f"Audio stream opened: {input_sr} Hz → {target_sr} Hz",
                    "LISTENER",
                    "SUCCESS",
                )

                while self._running:
                    indata, _ = stream.read(cfg.audio.blocksize)

                    if self._is_muted:
                        continue

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

                        now = time.monotonic()
                        if now - self._last_wave_emit >= self._wave_fps_interval:
                            self._last_wave_emit = now
                            audio_mono = audio[:, 0] if audio.ndim > 1 else audio

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


if _MAIN:
    ...
