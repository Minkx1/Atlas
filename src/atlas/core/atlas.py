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
from .events import EventManager
from .logging_config import configure_logging
from .module import Module, on_event

log = logging.getLogger(__name__)


class Atlas(Module):
    def __init__(self) -> None:
        # logs and events
        configure_logging(DATA_DIR / "logs", enabled=cfg.log, level=cfg.log_level)
        self.events = EventManager()
        self._register_events(self.events)

        # utils

        self.alive = True

        self.keybinds = KeyBindManager()
        self.keybinds.register_keybind(
            cfg.kws.awake_keybind,
            lambda: self.events.emit(
                "stt.kws.keyword_detected", {"keyword": "{HotKey}"}
            ),
        )

        self.ui = UI(app=self, events=self.events)

        # Modules

        self.stt_module = SttModule(self.events)
        self.tts_module = TtsModule(self.events)
        self.op_module = OpModule(self.events)

    def _shutdown(self):
        self.alive = False
        if hasattr(self, "ui") and getattr(self.ui, "is_running", False):
            self.ui.call_from_thread(self.ui.exit)

    def load_models(self):
        try:
            log.info("Starting model loading")
            self.stt_module.load()
            self.tts_module.load()
            self.op_module.load()

            log.info("All models loaded successfully")
        except Exception:
            log.exception("Error loading models")
            raise

    @on_event("core.terminate")
    def close(self, **kwargs):
        try:
            log.info("Shutting down assistant")
            self.events.close()

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

            self._shutdown()

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

        self.ui.run()  # this blocks main thread

        # from threading import Event
        # while self.alive:
        #     Event().wait(1.0)

    def start(self):
        """Starts Atlas."""
        try:
            self._main()
        except Exception as e:
            print(f"[!] FATAL ERROR: {e}")
            sys.exit(1)
        finally:
            self.close()
