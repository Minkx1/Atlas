# events.py

import queue
import threading
from typing import Any


class Event:
    def __init__(self, type: str = "", content: Any = None) -> None:
        self.type = type
        self.content = content


class EventManager:
    instance: "EventManager|None" = None

    def __init__(self) -> None:
        if EventManager.instance:
            raise RuntimeError(
                "[WARN] Singleton class EventManager was initialized twice."
            )
            # return EventManager.instance  # type: ignore

        self.queue = queue.Queue()
        self.flags: dict[str, threading.Event] = {}

        EventManager.instance = self

    def emit(self, type: str, content: Any = None):
        self.queue.put(Event(type, content))

    def set_flag(self, id: str, set: bool = True):
        """Function that initializes flags(threading.Event) and sets them if needed."""
        if id not in self.flags:
            self.flags[id] = threading.Event()

        if set:
            self.flags[id].set()
        else:
            self.flags[id].clear()

    def wait_for(self, id: str, timeout: float | None = None) -> bool:
        if id not in self.flags:
            raise RuntimeError(f"Unknown or not initialized event-flag: {id}.")

        return self.flags[id].wait(timeout)

    @classmethod
    def get_instace(cls) -> "EventManager":
        return cls.instance or EventManager()

    def get_next_event(self) -> Event:
        return self.queue.get()


def emit_event(type: str, content: Any = None):
    EventManager.get_instace().emit(type, content)


# def wait_for(flag_id: str, timeout: float | None = None) -> bool:
#     return EventManager.get_instace().wait_for(flag_id, timeout)
