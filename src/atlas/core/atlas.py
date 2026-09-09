#
# core / atlas.py
# Main Atlas entry point and orchestrator for the application
#

import logging
import sys
import threading

import atlas.modules
from atlas.core.module import Module, discover_modules, on_event
from atlas.utils import UI

from .config import DATA_DIR, cfg
from .events import EventManager
from .logging_config import configure_logging

log = logging.getLogger(__name__)


class Atlas(Module):
    name = "core"

    def __init__(self, log: bool = True, level: str = "INFO") -> None:
        self.alive: bool = True

        # logs and events
        configure_logging(DATA_DIR / "logs", enabled=cfg.log, level=cfg.log_level)
        self.events = EventManager()
        self._register_events(self.events)

        self.ui = UI(app=self, events=self.events)

        # Modules
        self.modules: dict[str, Module] = {}
        for name, cls in discover_modules(atlas.modules).items():
            self.modules[name] = cls(self.events)

    def _shutdown(self):
        self.alive = False
        if hasattr(self, "ui") and getattr(self.ui, "is_running", False):
            self.ui.call_from_thread(self.ui.exit)

    def load_models(self):
        try:
            log.info("Starting model loading")

            for module in self.modules.values():
                module.load()

            log.info("All models loaded successfully")
        except Exception:
            log.exception("Error loading models")
            raise

    @on_event("core.terminate", mode="immediate")
    def _handle_terminate_event(self, **kwargs):
        log.info("Received terminate event, starting shutdown thread")
        threading.Thread(target=self.close, name="ATLAS_SHUTDOWN", daemon=True).start()

    def close(self, **kwargs):
        try:
            log.info("Shutting down assistant")
            for module in self.modules.values():
                module.close()

            self.events.close()
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
        self.events.start()
        for module in self.modules.values():
            module.start()
        self.ui.run()

    def start(self):
        """Starts Atlas."""
        try:
            self._main()
        except Exception as e:
            log.error("[!] FATAL ERROR: %s", e, exc_info=True)
            sys.exit(1)
        finally:
            self.close()
