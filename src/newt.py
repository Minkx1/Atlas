# newt.py

import sys
import time

from .cmd_operator import LLM, CommandOperator, Operator
from .config import cfg
from .events import (
    EventLogger,
    EventManager,
    EventType,
    emit_event,
    log,
)

# from .ui import AssistantUI, console
from .new_ui import UI
from .speech_to_text import VAD, KeyWordSpotter, Listener, LState, Whisper
from .text_to_speech import TextToSpeech


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

        # STT Pipeline
        self.kws = KeyWordSpotter()
        self.vad = VAD()
        self.whisper = Whisper()
        self.listener = Listener(self.vad, self.whisper, self.kws)

        # TTS
        self.tts = TextToSpeech()

        # Operator
        self.cmd = CommandOperator()
        self.llm = LLM()
        self.operator = Operator(self.cmd, self.llm)

        self._setup_subscriptions()

    def _shutdown(self):
        self.alive = False
        if hasattr(self, "ui") and getattr(self.ui, "is_running", False):
            self.ui.call_from_thread(self.ui.exit)

    def load_models(self):
        try:
            log("Starting model loading...", "NEWT", "INFO")
            self.vad.load()
            self.kws.load()
            self.whisper.load()

            self.tts.load()

            self.cmd.load()
            self.llm.load()

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

        em.subscribe(EventType.OP_ASK_FINISH, lambda e: app._shutdown())
        em.subscribe(EventType.OP_RECEIVE_CMD, lambda e: app.operator.submit(e.content))
        em.subscribe(EventType.OP_READY, lambda e: emit_event(EventType.STT_CONTINUE))

        # em.subscribe(EventType.UI_BANNER, lambda e: AssistantUI.print_banner())
        # em.subscribe(
        #     EventType.UI_STATE_CHANGE,
        #     lambda e: AssistantUI.print_state_change(**e.content),
        # )
        # em.subscribe(
        #     EventType.UI_TRANSCRIPTION,
        #     lambda e: AssistantUI.print_transcription(**e.content),
        # )
        # em.subscribe(
        #     EventType.UI_LLM_CHUNK,
        #     lambda e: AssistantUI.print_llm_chunk(**e.content),
        # )
        # em.subscribe(
        #     EventType.UI_LLM_RESPONSE,
        #     lambda e: AssistantUI.print_llm_response(**e.content),
        # )
        # em.subscribe(
        #     EventType.UI_ASSISTANT_SAY,
        #     lambda e: AssistantUI.print_assistant_say(**e.content),
        # )

    def close(self):
        try:
            log("Shutting down assistant...", "NEWT", "INFO")

            if getattr(self, "listener", None):
                self.listener.close()
                log("Listener closed.", "NEWT", "DEBUG")

            if getattr(self, "operator", None):
                self.operator.close()
                log("Operator closed.", "NEWT", "DEBUG")
            if getattr(self, "tts", None):
                self.tts.close()
                log("TTS closed.", "NEWT", "DEBUG")

            if hasattr(self, "kws"):
                if hasattr(self.kws, "stream"):
                    del self.kws.stream
                if hasattr(self.kws, "kws"):
                    del self.kws.kws
            if hasattr(self, "vad") and hasattr(self.vad, "iterator"):
                del self.vad.iterator
            if hasattr(self, "whisper") and hasattr(self.whisper, "model"):
                del self.whisper.model

            self._shutdown()
            self.events.flush_and_stop()
            log("Shutdown complete.", "NEWT", "INFO")
        except Exception as e:  # noqa: BLE001
            log(f"Error during shutdown: {type(e).__name__}: {e}", "NEWT", "ERROR")

        finally:
            import sys

            sys.stdout.write(
                "\x1b[?1000l"
                "\x1b[?1003l"
                "\x1b[?1015l\x1b[?1006l"
                "\x1b[?25h"
                "\x1b[=0u"
                "\x1b[<u"
                "\x1b[>4m"
                "\x1b[?2004l"
            )
            sys.stdout.flush()

    def main(self):
        self.load_models()

        self.tts.start()
        self.operator.start()
        self.listener.start()

        emit_event(EventType.UI_BANNER)

        self.ui = UI(newt_app=self)
        self.ui.run()  # this blocks main thread
        # from threading import Event

        # while self.alive:
        #     Event().wait(1.0)

    def start(self):
        try:
            self.main()
        except Exception as e:  # noqa: BLE001
            # console.print("[bold red]][!] ERROR[/]: " + str(e))
            print(f"[!] FATAL ERROR: {e}")
            sys.exit(1)
        finally:
            self.close()
