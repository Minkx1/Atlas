#
# core / events.py
# Contains core event system
#

import queue
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

from rich.console import Console

from .schema import (
    AudioWavePayload,
    CommandType,
    EmptyPayload,
    EventType,
    IntentPayload,
    KeywordPayload,
    LLMChunkPayload,
    LogPayload,
    SetStatePayload,
    SoundGenerationPayload,
    SoundPlaybackPayload,
    TextPayload,
)


@dataclass
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    kind: str = "event"

    @property
    def content(self) -> dict[str, Any]:
        """Compatibility alias for subscribers migrating to ``payload``."""
        return self.payload


Payload = Mapping[str, Any]
Callback = Callable[[Event], Any]


class EventManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.queue = queue.Queue()
        self.callbacks: dict[str, list[tuple[Callback, bool]]] = {}
        self._callbacks_lock = threading.RLock()
        self._async_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="EVENT_CALLBACK"
        )
        self._futures: set[Future[Any]] = set()
        self._futures_lock = threading.Lock()
        self._stopping = False

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="EVENT_DISPATCHER", daemon=True
        )
        self._dispatcher.start()

    def subscribe(
        self,
        event: EventType | CommandType,
        callback: Callback,
        *,
        asynchronous: bool = False,
    ):
        with self._callbacks_lock:
            self.callbacks.setdefault(event.value, []).append((callback, asynchronous))

    def unsubscribe(self, event: EventType | CommandType, callback: Callback):
        name = event.value
        with self._callbacks_lock:
            callbacks = self.callbacks.get(name, [])
            self.callbacks[name] = [
                item for item in callbacks if item[0] is not callback
            ]

    def _track_future(self, future: Future[Any]):
        with self._futures_lock:
            self._futures.add(future)
        future.add_done_callback(self._discard_future)

    def _discard_future(self, future: Future[Any]):
        with self._futures_lock:
            self._futures.discard(future)

    def _run_callback(self, callback: Callback, event: Event):
        try:
            callback(event)
        except Exception as exc:
            log(
                f"Error in callback for {event.name}: {exc}",
                source="EVENTS",
                level="ERROR",
            )

    def _dispatch_loop(self):
        while True:
            event = self.queue.get()

            if event is None:
                self.queue.task_done()
                break

            with self._callbacks_lock:
                callbacks_to_call = self.callbacks.get(
                    event.name, []
                ) + self.callbacks.get(EventType.WILDCARD.value, [])

            for callback, asynchronous in callbacks_to_call:
                if asynchronous:
                    self._track_future(
                        self._async_executor.submit(self._run_callback, callback, event)
                    )
                else:
                    self._run_callback(callback, event)

            self.queue.task_done()

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
        if self._stopping:
            return
        self._stopping = True
        EventManager._instance = None
        self.queue.put(None)
        self._dispatcher.join(timeout=2.0)
        self._async_executor.shutdown(wait=True, cancel_futures=False)

    # Emit overloads
    @overload
    def emit(
        self, event: Literal[EventType.OP_INTENT], payload: IntentPayload
    ) -> None: ...
    @overload
    def emit(
        self,
        event: Literal[
            EventType.OP_LLM_CHUNK,
            EventType.UI_LLM_RESPONSE_DONE,
            EventType.LLM_RESPONSE,
        ],
        payload: TextPayload,
    ) -> None: ...
    @overload
    def emit(
        self, event: Literal[EventType.UI_LLM_CHUNK], payload: LLMChunkPayload
    ) -> None: ...
    @overload
    def emit(
        self, event: Literal[EventType.KWS_KEYWORD_DETECTED], payload: KeywordPayload
    ) -> None: ...
    @overload
    def emit(
        self,
        event: Literal[
            EventType.UI_TRANSCRIPTION,
            EventType.STT_TRANSCRIBED,
            EventType.UI_LLM_RESPONSE,
            EventType.UI_ASSISTANT_SAY,
        ],
        payload: TextPayload,
    ) -> None: ...
    @overload
    def emit(
        self, event: Literal[EventType.STT_AUDIOWAVE], payload: AudioWavePayload
    ) -> None: ...
    @overload
    def emit(
        self,
        event: Literal[EventType.STT_CHANGED_STATE, EventType.UI_STATE_CHANGE],
        payload: SetStatePayload,
    ) -> None: ...
    @overload
    def emit(
        self,
        event: Literal[EventType.SOUNDS_GENERATE_SOUND],
        payload: SoundGenerationPayload,
    ) -> None: ...
    @overload
    def emit(
        self,
        event: Literal[
            EventType.TTS_LOADED,
            EventType.KWS_LOADED,
            EventType.VAD_LOADED,
            EventType.WHISPER_LOADED,
            EventType.STT_MUTE,
            EventType.STT_UNMUTE,
            EventType.STT_START,
            EventType.STT_FINISH,
            EventType.OP_INTERRUPT,
            EventType.LLM_LOADED,
            EventType.UI_BANNER,
        ],
        payload: EmptyPayload | None = None,
    ) -> None: ...
    @overload
    def emit(
        self,
        event: EventType | None,
        payload: Payload | None = None,
    ) -> None: ...
    @overload
    def emit(
        self,
        event: Literal[
            EventType.OP_START,
            EventType.OP_FINISH,
            EventType.VAD_START,
            EventType.VAD_END,
            EventType.TTS_BUSY,
            EventType.TTS_FREE,
        ],
        payload: EmptyPayload | None = None,
    ) -> None: ...
    @overload
    def emit(
        self, event: Literal[EventType.DEBUG_LOG], payload: LogPayload
    ) -> None: ...

    def emit(
        self, event: EventType | None, payload: Mapping[str, Any] | None = None
    ) -> None:
        if event is None:
            self.queue.put(None)
            return
        self.queue.put(Event(event.value, dict(payload or {})))

    # command overloads
    @overload
    def emit_command(
        self,
        cmd: Literal[CommandType.TTS_SPEAK, CommandType.OP_SUBMIT],
        payload: TextPayload,
    ) -> None: ...
    @overload
    def emit_command(
        self, cmd: Literal[CommandType.SET_STATE], payload: SetStatePayload
    ) -> None: ...
    @overload
    def emit_command(
        self, cmd: Literal[CommandType.TTS_PLAY_SOUND], payload: SoundPlaybackPayload
    ) -> None: ...
    @overload
    def emit_command(
        self, cmd: CommandType, payload: Payload | None = None
    ) -> None: ...

    def emit_command(
        self, cmd: CommandType, payload: Mapping[str, Any] | None = None
    ) -> None:
        self.queue.put(Event(cmd.value, dict(payload or {}), kind="command"))


def emit_event(event: EventType | None, payload: Payload | None = None):
    EventManager().emit(event, payload)


# def command(cmd: CommandType, payload: Payload | None = None) -> None:
#     """Dispatch an operation request while preserving its command identity."""
#     EventManager().emit_command(cmd, payload)


def log(message: str, source: str = "SYS", level: str = "DEBUG"):
    emit_event(
        EventType.DEBUG_LOG,
        LogPayload(message=str(message), source=source, level=level.upper()),  # type: ignore
    )


class EventLogger:
    def __init__(self):
        from .config import DATA_DIR

        self.logs_dir = DATA_DIR / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

        # Subscribe only to DEBUG_LOG events (explicit, not wildcard)
        EventManager().subscribe(EventType.DEBUG_LOG, self._log_event)

    def _get_log_filepath(self, timestamp: float) -> Path:
        date_str = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        return self.logs_dir / f"{date_str}.log"

    def _write_file(self, msg: str, timestamp: float):
        log_file = self._get_log_filepath(timestamp)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    def _log_event(self, event: Event):
        message_text = self._format_message(event)
        # self.console.print(message_text)

        self._write_file(message_text, event.timestamp)

    def _format_message(self, event: Event) -> str:
        timestamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))

        if event.name == EventType.DEBUG_LOG.value:
            level = str(event.payload.get("level", "INFO")).upper()
            source = str(event.payload.get("source", "SYSTEM"))
            message = str(event.payload.get("message", ""))
            return f"[{timestamp}] [{source}]  [{level}]: {message}"

        content_str = (
            str(event.payload)[:150] + "..."
            if len(str(event.payload)) > 150
            else str(event.payload)
        )
        return f"[{timestamp}] {event.name}: {content_str}"
