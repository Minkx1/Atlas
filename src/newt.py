# newt.py


import signal
import sys
import threading
import time

print("[DEBUG] Loading libs...")
_start = time.perf_counter()

from .cmd_operator import LLM, CommandOperator, Operator
from .config import cfg
from .events import (
    Event,
    EventLogger,
    EventManager,
    EventType,
    emit_event,
    log,
)
from .profiler import profiler
from .speech_to_text import VAD, KeyWordSpotter, Listener, LState, Whisper
from .text_to_speech import TextToSpeech
from .ui import AssistantUI, console

print(f"[DEBUG] Loading complete. Took: {(time.perf_counter()-_start)*1000}ms")

class Newt:
    def __init__(self) -> None:
        # light components
        self.events = EventManager()
        self.alive = True
        self.logger = None

        if cfg.log:
            self.logger = EventLogger()
            self.logger._write_file(
                f" ===== New Session: [{time.strftime('%H:%M:%S', time.localtime(time.time()))}] | SUCCESS ===== \n",
                time.time(),
            )

        # main components

        self.tts = TextToSpeech()

        self.cmd = CommandOperator()
        self.llm = LLM()

        self.operator = Operator(self.cmd, self.llm)

        # STT pipeline

        self.kws = KeyWordSpotter()
        self.vad = VAD()
        self.whisper = Whisper()
        self.listener = Listener(self.vad, self.whisper, self.kws)

        self._setup_subscriptions()

    def load_models(self):
        try:
            log("Starting model loading...", "NEWT", "INFO")
            self.tts.load()

            self.llm.load()

            self.vad.load()
            self.kws.load()
            self.whisper.load()

            log("All models loaded successfully.", "NEWT", "SUCCESS")
        except Exception as e:
            log(
                f"Error loading models: {type(e).__name__}: {e}",
                "NEWT",
                "ERROR",
            )
            raise

    def _setup_subscriptions(self):
        """Subscribe all nececessary callbacks for events."""
        em = EventManager()
        app = self

        # TTS
        em.subscribe(EventType.TTS_SPEAK, lambda e: app.tts.speak(e.content))
        em.subscribe(EventType.TTS_PLAY_SOUND, lambda e: app.tts.play_sound(e.content))

        em.subscribe(EventType.TTS_BUSY, lambda e: app.listener.mute())
        em.subscribe(EventType.TTS_FREE, lambda e: app.listener.unmute())
        em.subscribe(EventType.TTS_FREE, lambda e: emit_event(EventType.STT_CONTINUE))

        em.subscribe(
            EventType.STT_CHANGED_STATE,
            lambda e: emit_event(EventType.PROFILER_SET_STATE, e.content),
        )
        em.subscribe(
            EventType.STT_SET_STATE,
            lambda e: app.listener.pipeline.set_state(LState(e.content)),
        )
        em.subscribe(
            EventType.STT_TRANSCRIBED,
            lambda e: emit_event(EventType.OP_RECEIVE_CMD, e.content),
        )
        em.subscribe(
            EventType.STT_KEYWORD_DETECTED,
            lambda e: emit_event(EventType.OP_RECEIVE_CMD, "!EVENT_KEYWORD_DETECTED"),
        )
        em.subscribe(
            EventType.STT_START, lambda e: emit_event(EventType.PROFILER_START)
        )

        em.subscribe(EventType.OP_ASK_FINISH, lambda e: app._shutdown())
        em.subscribe(EventType.OP_RECEIVE_CMD, lambda e: app.operator.submit(e.content))
        em.subscribe(EventType.OP_READY, lambda e: emit_event(EventType.STT_CONTINUE))

        em.subscribe(EventType.UI_BANNER, lambda e: AssistantUI.print_banner())
        em.subscribe(
            EventType.UI_STATE_CHANGE,
            lambda e: AssistantUI.print_state_change(**e.content),
        )
        em.subscribe(
            EventType.UI_TRANSCRIPTION,
            lambda e: AssistantUI.print_transcription(**e.content),
        )
        em.subscribe(
            EventType.UI_LLM_CHUNK,
            lambda e: AssistantUI.print_llm_chunk(**e.content),
        )
        em.subscribe(
            EventType.UI_LLM_RESPONSE,
            lambda e: AssistantUI.print_llm_response(**e.content),
        )
        em.subscribe(
            EventType.UI_ASSISTANT_SAY,
            lambda e: AssistantUI.print_assistant_say(**e.content),
        )

        def _prof_start(event: Event | None = None):
            if cfg.profiler:
                profiler.start()

        em.subscribe(EventType.PROFILER_START, lambda e: _prof_start(e))
        em.subscribe(
            EventType.PROFILER_SET_STATE,
            lambda e: profiler.set_state(e.content),
        )
        em.subscribe(EventType.PROFILER_FINISH, app._prof_finish)

    def _shutdown(self):
        self.alive = False

    def _prof_finish(self, *args):
        if cfg.profiler:
            profiler.stop()
            AssistantUI.print_benchmark_report(profiler.get_summary())

    def close(self):
        try:
            log("Shutting down assistant...", "NEWT", "INFO")
            emit_event(EventType.PROFILER_FINISH)

            if getattr(self, "operator", None):
                self.operator.close()
                log("Operator closed.", "NEWT", "DEBUG")
            if getattr(self, "tts", None):
                self.tts.close()
                log("TTS closed.", "NEWT", "DEBUG")
            if getattr(self, "listener", None):
                self.listener.close()
                log("Listener closed.", "NEWT", "DEBUG")

            self._shutdown()

            self.events.flush_and_stop()
            log("Shutdown complete.", "NEWT", "INFO")
        except Exception as e:  # noqa: BLE001
            log(
                f"Error during shutdown: {type(e).__name__}: {e}",
                "NEWT",
                "ERROR",
            )
        if getattr(self, "listener", None):
            self.listener.close()

        self._shutdown()

        self.events.flush_and_stop()

    def main(self):
        self.load_models()

        self.tts.start()
        self.operator.start()
        self.listener.start()

        emit_event(EventType.UI_BANNER)

        while self.alive:
            threading.Event().wait(1.0)

    def start(self):
        def _handle_sigint(signum, frame):
            self._shutdown()
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGINT, _handle_sigint)
        except ValueError:
            pass

        try:
            self.main()
        except KeyboardInterrupt:
            console.print("[dim]Stopping assistant...[/]")
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            console.print("[bold red]][!] ERROR[/]: " + str(e))
            sys.exit(1)
        finally:
            self.close()
