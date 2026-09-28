#
# keybinds.py
#

import logging
from collections.abc import Callable

from atlas.core.events import EventManager
from atlas.core.module import Module
from atlas.utils.config import Config

log = logging.getLogger(__name__)


KEYBINDS_ORIGIN_CFG = """<ctrl>+<alt>+w"""

cfg = Config.load_raw("keybinds.cfg", KEYBINDS_ORIGIN_CFG)


class KeyBindManager:
    def __init__(self) -> None:
        from pynput import keyboard

        self._keyboard = keyboard

        self.keybinds: dict[str, list[Callable[[], None]]] = {}
        self._listener: keyboard.GlobalHotKeys | None = None

    def start(self) -> None:
        if self._listener is not None:
            self.close()

        hotkeys_map = {kb: (lambda k=kb: self._dispatch(k)) for kb in self.keybinds}

        self._listener = self._keyboard.GlobalHotKeys(hotkeys_map)
        self._listener.start()

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _dispatch(self, keybind: str) -> None:
        for cb in self.keybinds.get(keybind, []):
            try:
                cb()
            except Exception as e:
                log.exception(f"Error handling '{keybind}'" + ": %s", e)

    def register_keybind(self, keybind: str, callback: Callable) -> None:
        """Registers callback for the keybind."""
        if keybind not in self.keybinds:
            self.keybinds[keybind] = []
        self.keybinds[keybind].append(callback)


class KeybindsModule(Module):
    name = "keybinds"

    def __init__(
        self,
        events: EventManager | None,
        keybind: str = cfg["content"],
        **kwargs,
    ) -> None:
        super().__init__(events, **kwargs)

        self.keybinds = KeyBindManager()
        self.keybind = keybind

    def start(self) -> None:
        self.keybinds.start()

    def load(self) -> None:
        self.keybinds.register_keybind(
            self.keybind,
            lambda: self.events.emit(
                "stt.kws.keyword_detected", {"keyword": "{HotKey}"}
            ),
        )

    def close(self) -> None:
        self.keybinds.close()
