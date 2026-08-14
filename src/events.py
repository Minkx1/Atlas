# events.py

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.console import Console


@dataclass
class Event:
    name: str
    content: Any = None
    timestamp: float = 0


class EventType(StrEnum):
    TTS_LOADED = "TTS_LOADED"
    TTS_SPEAK = "TTS_SPEAK"
    TTS_PLAY_SOUND = "TTS_PLAY_SOUND"
    TTS_BUSY = "TTS_BUSY"
    TTS_FREE = "TTS_FREE"

    KWS_LOADED = "KWS_LOADED"
    WHISPER_LOADED = "WHISPER_LOADED"
    VAD_LOADED = "VAD_LOADED"

    STT_CHANGED_STATE = "STT_CHANGED_STATE"
    STT_TRANSCRIBED = "STT_TRANSCRIBED"
    STT_KEYWORD_DETECTED = "STT_KEYWORD_DETECTED"
    STT_SET_STATE = "STT_SET_STATE"
    STT_CONTINUE = "STT_CONTINUE"
    STT_START = "STT_START"
    STT_FINISH = "STT_FINISH"

    OP_ASK_FINISH = "OP_ASK_FINISH"
    OP_RECEIVE_CMD = "OP_RECEIVE_CMD"
    OP_READY = "OP_READY"

    OP_CMD_LEVEL = "OP_CMD_LEVEL"
    UI_BANNER = "UI_BANNER"
    UI_STATE_CHANGE = "UI_STATE_CHANGE"
    UI_TRANSCRIPTION = "UI_TRANSCRIPTION"
    UI_LLM_CHUNK = "UI_LLM_CHUNK"
    UI_LLM_RESPONSE = "UI_LLM_RESPONSE"
    UI_LLM_RESPONSE_DONE = "UI_LLM_RESPONSE_DONE"
    UI_ASSISTANT_SAY = "UI_ASSISTANT_SAY"

    PROFILER_START = "PROFILER_START"
    PROFILER_SET_STATE = "PROFILER_SET_STATE"
    PROFILER_FINISH = "PROFILER_FINISH"
    DEBUG_LOG = "DEBUG_LOG"

    LLM_RESPONSE = "LLM_RESPONSE"
    LLM_LOADED = "LLM_LOADED"

    WILDCARD = "*"


class EventManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.queue = queue.Queue()
        self.callbacks: dict[str, list[Callable]] = {}

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="EVENT_DISPATCHER", daemon=True
        )
        self._dispatcher.start()

    def emit(self, event: EventType | None, content: Any = None):
        self.queue.put(Event(event.value, content, time.time()) if event else None)

    def subscribe(self, event: EventType, callback: Callable[[Event], Any]):
        if event.value not in self.callbacks:
            self.callbacks[event.value] = []
        self.callbacks[event.value].append(callback)

    def unsubscribe(self, event: EventType, callback: Callable):
        name = event.value
        if name in self.callbacks and callback in self.callbacks[name]:
            self.callbacks[name].remove(callback)

    def _dispatch_loop(self):
        while True:
            event = self.queue.get()

            if event is None:
                self.queue.task_done()
                break

            callbacks_to_call = self.callbacks.get(event.name, []) + self.callbacks.get(
                EventType.WILDCARD.value, []
            )

            for callback in callbacks_to_call:
                try:
                    callback(event)
                except Exception as exc:  # noqa: BLE001
                    log(
                        f"Error in callback for {event.name}: {exc}",
                        source="EVENTS",
                        level="ERROR",
                    )

            self.queue.task_done()

    def wait_for(self, event: EventType, timeout: float | None = None) -> Event | None:
        """Blocks thread until event is emitted."""
        wait_event = threading.Event()
        received_event = None

        def _unblock(e: Event):
            nonlocal received_event
            received_event = e
            wait_event.set()

        self.subscribe(event, _unblock)
        wait_event.wait(timeout)
        self.unsubscribe(event, _unblock)

        return received_event

    def flush_and_stop(self, timeout: float = 2.0):
        def _wait():
            self.queue.join()
            self.stop()

        wait_thread = threading.Thread(target=_wait, daemon=True)
        wait_thread.start()
        wait_thread.join(timeout=timeout)

        if wait_thread.is_alive():
            self.stop()

    def stop(self):
        self.queue.put(None)


def emit_event(event: EventType | None, content: Any = None):
    EventManager().emit(event, content)


def wait_for(event: EventType, timeout: float | None = None) -> Event | None:
    return EventManager().wait_for(event, timeout)


def log(message: str, source: str = "SYSTEM", level: str = "INFO"):
    emit_event(
        EventType.DEBUG_LOG,
        {"message": str(message), "source": source, "level": level.upper()},
    )


class EventLogger:
    def __init__(self):
        from .config import DATA_DIR

        self.logs_dir = DATA_DIR / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

        # sub to all ('*') events
        EventManager().subscribe(EventType.WILDCARD, self._log_event)

    def _get_log_filepath(self, timestamp: float) -> Path:
        date_str = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        return self.logs_dir / f"{date_str}.log"

    def _log_event(self, event: Event):
        message_text = self._format_message(event)
        # self.console.print(
        #     message_text
        # )  # this prints into console, which right now is not needed.

        log_file = self._get_log_filepath(event.timestamp)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message_text + "\n")

    def _format_message(self, event: Event) -> str:
        timestamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))

        if event.name == EventType.DEBUG_LOG.value and isinstance(event.content, dict):
            level = str(event.content.get("level", "INFO")).upper()
            source = str(event.content.get("source", "SYSTEM"))
            message = str(event.content.get("message", ""))
            return f"[{timestamp}] [{source}] {level}: {message}"

        content_str = (
            str(event.content)[:100] + "..."
            if len(str(event.content)) > 100
            else str(event.content)
        )
        return f"[{timestamp}] {event.name}: {content_str}"
