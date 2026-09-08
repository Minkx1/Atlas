#
# core / atlas.py
# Main Atlas entry point and orchestrator for the application
#

import logging
import sys

from atlas.op import OpModule
from atlas.stt import SttModule
from atlas.tts import TtsModule
from atlas.utils import UI, KeyBindManager

from .config import DATA_DIR, cfg
from .events import CommandType, EventManager, EventType
from .logging_config import configure_logging

log = logging.getLogger(__name__)


class Atlas:
    def __init__(self) -> None:
        configure_logging(DATA_DIR / "logs", enabled=cfg.log, level=cfg.log_level)
        # Events and Logger
        self.events = EventManager()
        self.alive = True
        # Utils

        self.keybinds = KeyBindManager()
        self.keybinds.register_keybind(
            cfg.kws.awake_keybind,
            lambda: self.events.emit(
                EventType.KWS_KEYWORD_DETECTED, {"keyword": "{HotKey}"}
            ),
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
            log.info("Starting model loading")
            # self.kws.load()
            # self.sr.load()
            self.stt_module.load()

            self.tts_module.load()
            # self.tts.load()
            # self.sound_manager.load()

            self.op_module.load()
            # self.cmd.load()
            # self.llama.load()

            log.info("All models loaded successfully")
        except Exception:
            log.exception("Error loading models")
            raise

    def _setup_subscriptions(self):
        """Subscribe all nececessary callbacks for events."""

        def handle_intent(event):
            intent: str = event.payload["intent"]
            self.tts_module.play_category(intent)

            if intent == "farewell":
                self.shutdown()
            if intent == "sleep":
                self.events.emit_command(CommandType.SET_STATE, {"state": "SLEEPING"})

        self.events.subscribe(EventType.OP_INTENT, handle_intent)

    def _close(self):
        try:
            log.info("Shutting down assistant")

            self.keybinds.close()

            if getattr(self, "stt_module", None):
                self.stt_module.close()
                log.debug("STT closed")
            if getattr(self, "op_module", None):
                self.op_module.close()
                log.debug("Operator closed")
            if getattr(self, "tts_module", None):
                self.tts_module.close()
                log.debug("TTS closed")

            self.shutdown()
            self.events.flush_and_stop()
            log.info("Shutdown complete")
        except Exception:
            log.exception("Error during shutdown")
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

        self.events.emit(EventType.UI_BANNER, {})

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
