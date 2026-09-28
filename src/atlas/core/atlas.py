#
# core / atlas.py
# Main Atlas entry point and orchestrator for the application
#

import logging
import sys
import threading

from atlas.core.events import EventManager
from atlas.core.module import Module

# utils
from atlas.utils.config import DATA_DIR

# global logger for each python-module
log = logging.getLogger(__name__)


class Atlas:
    """Module manager and entrypoint for Atlas Assistant"""

    def __init__(self, *, ignored_modules: list[str] | None = None) -> None:
        self.alive: bool = True
        self._closed: bool = False
        self.ignored_modules = ignored_modules or []

        self.events = EventManager()
        self.events.subscribe(
            "core.terminate", lambda e: self._handle_terminate_event(**e.content)
        )
        self._configure_logging()

        # setting up modules
        self.modules: dict[str, Module] = {}
        self.init_modules()

    def _configure_logging(self):
        from atlas.utils.logging_config import configure_logging

        configure_logging(DATA_DIR / "logs", enabled=True, level="DEBUG")

    def init_modules(self):
        """Initializes all Atlas' modules"""
        import atlas.modules
        from atlas.core.module import discover_modules

        for name, cls in discover_modules(atlas.modules).items():
            if name not in self.ignored_modules:
                self.modules[name] = cls(self.events)

    def load_modules(self):
        log.info("Starting module loading")

        for name, module in self.modules.items():
            try:
                module.load()
            except Exception as e:
                log.exception("Module " + name + " loaded with error: %s", e)
                raise
            log.info("All modules loaded.")

    def close(self, **kwargs):
        """Closes all atlas' modules.

        Note that this doesn't automatically ends execution.
        See: Atlas.shutdown()

        Idempotent: safe to call multiple times, e.g. once from the
        `core.terminate` handler's shutdown thread and once from `run()`'s
        `finally` block.
        """
        if self._closed:
            return
        self._closed = True

        log.info("Shutting down assistant")

        for name, module in self.modules.items():
            try:
                module.close()
            except Exception as e:
                log.exception("Module " + name + " closed with error: %s", e)
                raise

        self.events.close()

        log.info("Shutdown complete")

    def start(self) -> None:
        """Loads and starts all mod"""
        log.info("Starting all modules")

        self.load_modules()
        self.events.start()
        for module in self.modules.values():
            try:
                module.start()
            except Exception as e:
                log.exception("Error starting modules: %s", e)
                raise

            log.info("All modules started successfully.")

    def _handle_terminate_event(self, **kwargs):
        log.info("Received terminate event, starting shutdown thread")
        threading.Thread(
            target=self.shutdown, name="ATLAS_SHUTDOWN", daemon=True
        ).start()

    def shutdown(self):
        """Shuts Atlas down."""
        self.alive = False

    def _main(self):
        self.start()
        while self.alive:
            threading.Event().wait(0.1)

    def run(self):
        """Runs Atlas"""
        try:
            self._main()
        except KeyboardInterrupt:
            self.events.emit("core.terminate")
        except Exception as e:
            log.critical("[!] ERROR: %s", e, exc_info=True)
            sys.exit(1)
        finally:
            self.close()
