#
# core / events.py
# Contains core event system
#

from .event_api import Event, EventManager

# import logging
# import queue
# import threading
# import time
# from collections.abc import Callable, Mapping
# from concurrent.futures import Future, ThreadPoolExecutor
# from dataclasses import dataclass, field
# from typing import Any, Literal, overload

# from .schema import (
#     AudioWavePayload,
#     CommandType,
#     EmptyPayload,
#     EventType,
#     IntentPayload,
#     KeywordPayload,
#     LLMChunkPayload,
#     SetStatePayload,
#     SoundGenerationPayload,
#     SoundPlaybackPayload,
#     TextPayload,
# )

# log = logging.getLogger(__name__)


# @dataclass
# class Event:
#     name: str
#     payload: dict[str, Any] = field(default_factory=dict)
#     timestamp: float = field(default_factory=time.time)
#     kind: str = "event"

#     @property
#     def content(self) -> dict[str, Any]:
#         """Compatibility alias for subscribers migrating to ``payload``."""
#         return self.payload


# Payload = Mapping[str, Any]
# Callback = Callable[[Event], Any]


# class EventManager:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._init()
#         return cls._instance

#     def _init(self):
#         self.queue = queue.Queue()
#         self.callbacks: dict[str, list[tuple[Callback, bool]]] = {}
#         self._callbacks_lock = threading.RLock()
#         self._async_executor = ThreadPoolExecutor(
#             max_workers=4, thread_name_prefix="EVENT_CALLBACK"
#         )
#         self._futures: set[Future[Any]] = set()
#         self._futures_lock = threading.Lock()
#         self._stopping = False

#         self._dispatcher = threading.Thread(
#             target=self._dispatch_loop, name="EVENT_DISPATCHER", daemon=True
#         )
#         self._dispatcher.start()

#     def subscribe(
#         self,
#         event: EventType | CommandType,
#         callback: Callback,
#         *,
#         asynchronous: bool = False,
#     ):
#         with self._callbacks_lock:
#             self.callbacks.setdefault(event.value, []).append((callback, asynchronous))

#     def unsubscribe(self, event: EventType | CommandType, callback: Callback):
#         name = event.value
#         with self._callbacks_lock:
#             callbacks = self.callbacks.get(name, [])
#             self.callbacks[name] = [
#                 item for item in callbacks if item[0] is not callback
#             ]

#     def _track_future(self, future: Future[Any]):
#         with self._futures_lock:
#             self._futures.add(future)
#         future.add_done_callback(self._discard_future)

#     def _discard_future(self, future: Future[Any]):
#         with self._futures_lock:
#             self._futures.discard(future)

#     def _run_callback(self, callback: Callback, event: Event):
#         try:
#             callback(event)
#         except Exception:
#             log.exception("Error in callback for event %s", event.name)

#     def _dispatch_loop(self):
#         while True:
#             event = self.queue.get()

#             if event is None:
#                 self.queue.task_done()
#                 break

#             with self._callbacks_lock:
#                 callbacks_to_call = self.callbacks.get(
#                     event.name, []
#                 ) + self.callbacks.get(EventType.WILDCARD.value, [])

#             for callback, asynchronous in callbacks_to_call:
#                 if asynchronous:
#                     self._track_future(
#                         self._async_executor.submit(self._run_callback, callback, event)
#                     )
#                 else:
#                     self._run_callback(callback, event)

#             self.queue.task_done()

#     def flush_and_stop(self, timeout: float = 2.0):
#         def _wait():
#             self.queue.join()
#             self.stop()

#         wait_thread = threading.Thread(target=_wait, daemon=True)
#         wait_thread.start()
#         wait_thread.join(timeout=timeout)

#         if wait_thread.is_alive():
#             self.stop()

#     def stop(self):
#         if self._stopping:
#             return
#         self._stopping = True
#         EventManager._instance = None
#         self.queue.put(None)
#         self._dispatcher.join(timeout=2.0)
#         self._async_executor.shutdown(wait=True, cancel_futures=False)

#     # Emit overloads
#     @overload
#     def emit(
#         self, event: Literal["op.intent"], payload: IntentPayload
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: Literal[
#             "op.llm_chunk",
#             "ui.llm_response_done",
#             "op.llm_response",
#         ],
#         payload: TextPayload,
#     ) -> None: ...
#     @overload
#     def emit(
#         self, event: Literal["ui.llm_chunk"], payload: LLMChunkPayload
#     ) -> None: ...
#     @overload
#     def emit(
#         self, event: Literal["stt.kws.keyword_detected"], payload: KeywordPayload
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: Literal[
#             "ui.transcription",
#             "stt.transcribed",
#             EventType.UI_LLM_RESPONSE,
#             "ui.say",
#         ],
#         payload: TextPayload,
#     ) -> None: ...
#     @overload
#     def emit(
#         self, event: Literal["stt.audiowave"], payload: AudioWavePayload
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: Literal["stt.changed_state", "ui.state_change"],
#         payload: SetStatePayload,
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: Literal["tts.sounds.generate_sound"],
#         payload: SoundGenerationPayload,
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: Literal[
#             EventType.TTS_LOADED,
#             EventType.KWS_LOADED,
#             EventType.VAD_LOADED,
#             EventType.WHISPER_LOADED,
#             EventType.STT_MUTE,
#             EventType.STT_UNMUTE,
#             EventType.STT_START,
#             EventType.STT_FINISH,
#             "op.interrupt",
#             EventType.LLM_LOADED,
#             "ui.banner",
#         ],
#         payload: EmptyPayload | None = None,
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: EventType | None,
#         payload: Payload | None = None,
#     ) -> None: ...
#     @overload
#     def emit(
#         self,
#         event: Literal[
#             "op.start",
#             "op.finish",
#             "stt.vad.start",
#             "stt.vad.end",
#             "tts.busy",
#             "tts.free",
#         ],
#         payload: EmptyPayload | None = None,
#     ) -> None: ...
#     def emit(
#         self, event: EventType | None, payload: Mapping[str, Any] | None = None
#     ) -> None:
#         if event is None:
#             self.queue.put(None)
#             return
#         self.queue.put(Event(event.value, dict(payload or {})))

#     # command overloads
#     @overload
#     def emit_command(
#         self,
#         cmd: Literal["tts.speak", "op.submit"],
#         payload: TextPayload,
#     ) -> None: ...
#     @overload
#     def emit_command(
#         self, cmd: Literal[CommandType.SET_STATE], payload: SetStatePayload
#     ) -> None: ...
#     @overload
#     def emit_command(
#         self, cmd: Literal["tts.sounds.play"], payload: SoundPlaybackPayload
#     ) -> None: ...
#     @overload
#     def emit_command(
#         self, cmd: CommandType, payload: Payload | None = None
#     ) -> None: ...

#     def emit_command(
#         self, cmd: CommandType, payload: Mapping[str, Any] | None = None
#     ) -> None:
#         self.queue.put(Event(cmd.value, dict(payload or {}), kind="command"))
