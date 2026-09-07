#
# core / atlas.py
# Main Atlas entry point and orchestrator for the application
#

import sys

# from ..op import CommandOperator, Llama, Operator
from ..op import OpModule

# from ..stt import KeyWordSpotter, Listener, SpeechRecognizer, State, StateMachine
from ..stt import SttModule

# from ..tts import SoundManager, TextToSpeech
from ..tts import TtsModule
from ..utils import UI, KeyBindManager
from .config import cfg
from .events import (
    CommandType,
    EventLogger,
    EventManager,
    EventType,
    command,
    emit_event,
    log,
)


class Atlas:
    def __init__(self) -> None:
        # Events and Logger
        self.events = EventManager()
        self.alive = True
        self.logger = None

        if cfg.log:
            self.logger = EventLogger()
            import time

            timestamp = time.strftime("%H:%M:%S", time.localtime(time.time()))
            self.logger._write_file(
                f" ===== New Atlas Session: [{timestamp}] | SUCCESS ===== \n",
                time.time(),
            )

        # Utils

        self.keybinds = KeyBindManager()
        self.keybinds.register_keybind(
            cfg.kws.awake_keybind,
            lambda: emit_event(EventType.KWS_KEYWORD_DETECTED, {"keyword": "{HotKey}"}),
        )

        self.ui = UI(app=self, events=self.events)

        # Modules

        self.stt_module = SttModule(self.events)
        self.tts_module = TtsModule(self.events)
        self.op_module = OpModule(self.events)

        self._setup_subscriptions()

    def shutdown(self):
        self.alive = False
        if hasattr(self, "ui") and getattr(self.ui, "is_running", False):
            self.ui.call_from_thread(self.ui.exit)

    def load_models(self):
        try:
            log("Starting model loading...", "ATLAS", "INFO")
            # self.kws.load()
            # self.sr.load()
            self.stt_module.load()

            self.tts_module.load()
            # self.tts.load()
            # self.sound_manager.load()

            self.op_module.load()
            # self.cmd.load()
            # self.llama.load()

            log("All models loaded successfully.", "ATLAS", "SUCCESS")
        except Exception as e:
            log(
                f"Error loading models: {type(e).__name__}: {e}",
                "ATLAS",
                "ERROR",
            )
            raise

    def _setup_subscriptions(self):
        """Subscribe all nececessary callbacks for events."""

        def handle_intent(event):
            intent: str = event.payload["intent"]
            self.tts_module.play_category(intent)

            if intent == "farewell":
                self.shutdown()
            if intent == "sleep":
                command(CommandType.SET_STATE, {"state": "SLEEPING"})

        self.events.subscribe(EventType.OP_INTENT, handle_intent)

    def _close(self):
        try:
            log("Shutting down assistant...", "ATLAS", "INFO")

            self.keybinds.close()

            if getattr(self, "stt_module", None):
                self.stt_module.close()
                log("STT closed.", "ATLAS", "DEBUG")
            if getattr(self, "op_module", None):
                self.op_module.close()
                log("Operator closed.", "ATLAS", "DEBUG")
            if getattr(self, "tts_module", None):
                self.tts_module.close()
                log("TTS closed.", "ATLAS", "DEBUG")

            self.shutdown()
            self.events.flush_and_stop()
            log("Shutdown complete.", "ATLAS", "INFO")
        except Exception as e:
            log(f"Error during shutdown: {type(e).__name__}: {e}", "ATLAS", "ERROR")
        finally:
            import sys

            sys.stdout.write(  # Textual ui fix
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

    def _main(self):
        self.load_models()

        self.keybinds.start()
        self.stt_module.start()
        self.tts_module.start()
        self.op_module.start()

        emit_event(EventType.UI_BANNER, {})

        self.ui.run()  # this blocks main thread

        # from threading import Event
        # while self.alive:
        #     Event().wait(1.0)

    def start(self):
        """Starts Atlas Assistant."""
        try:
            self._main()
        except Exception as e:
            print(f"[!] FATAL ERROR: {e}")
            sys.exit(1)
        finally:
            self._close()
