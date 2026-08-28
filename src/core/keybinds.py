#
# keybins.py
#

from collections.abc import Callable

from pynput import keyboard


class KeyBindManager:
    def __init__(self) -> None:
        self.keybinds: dict[str, list[Callable[[], None]]] = {}
        self._listener: keyboard.GlobalHotKeys | None = None

    def start(self) -> None:
        if self._listener is not None:
            self.close()

        hotkeys_map = {kb: (lambda k=kb: self._dispatch(k)) for kb in self.keybinds}

        self._listener = keyboard.GlobalHotKeys(hotkeys_map)
        self._listener.start()

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _dispatch(self, keybind: str) -> None:
        for cb in self.keybinds.get(keybind, []):
            try:
                cb()
            except Exception as e:  # noqa: BLE001
                print(f"Error handling '{keybind}': {e}", "KeyBind", "ERROR")

    def register_keybind(self, keybind: str, callback: Callable) -> None:
        """Registers callback for the keybind."""
        if keybind not in self.keybinds:
            self.keybinds[keybind] = []
        self.keybinds[keybind].append(callback)
