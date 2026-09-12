#
# core / events.py
# Contains core event system
#

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

# global vars

log = logging.getLogger(__name__)
DispatchMode = Literal["immediate", "worker"]
WILDCARD = "*"


@dataclass(slots=True, frozen=True)
class Event:
    type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    @property
    def content(self) -> dict:
        # for backwards compatability purposes
        return self.payload


class _Subscription:
    __slots__ = ("callback", "mode")

    def __init__(self, callback: Callable[[Event], None], mode: DispatchMode) -> None:
        self.callback = callback
        self.mode = mode


class EventManager:
    def __init__(
        self,
        *,
        max_workers: int = 4,
        immediate_warn_threshold: float = 0.1,
    ) -> None:
        """
        - **max_workers**:
            ThreadPool size for mode="worker".

        - **immediate_warn_threshold**:
            If immediate-callback takes too long
            (sec), warning is logged: callback
            is blocking dispatcher and, probably,
            shoud've been mode="worker".
        """
        self._queue: queue.Queue[Event | None] = queue.Queue()
        self._subs: dict[str, list[_Subscription]] = {}
        self._subs_lock = threading.RLock()

        self._max_workers = max_workers
        self._immediate_warn_threshold = immediate_warn_threshold

        self._executor: ThreadPoolExecutor | None = None
        self._dispatcher: threading.Thread | None = None
        self._running = False
        self._lifecycle_lock = threading.Lock()

    # lifecycle

    def start(self) -> None:
        """Starts worker and queue"""
        with self._lifecycle_lock:
            if self._running:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="EVENTS_WORKER",
            )
            self._running = True
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="EVENTS_DISPATCHER",
                daemon=True,
            )
            self._dispatcher.start()

    def close(self, timeout: float = 2.0) -> None:
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            self._queue.put(None)

            if self._dispatcher is not None:
                if threading.current_thread() is not self._dispatcher:
                    self._dispatcher.join(timeout=timeout)
                self._dispatcher = None

            if self._executor is not None:
                current_thread = threading.current_thread()
                is_worker_thread = any(
                    t == current_thread for t in self._executor._threads
                )

                self._executor.shutdown(wait=not is_worker_thread, cancel_futures=True)
                self._executor = None

    # API

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        *,
        mode: DispatchMode = "worker",
    ) -> None:
        """
        - **event_type**=WILDCARD ("*"):
            Subscribes callback on EACH event.

        - **mode**="immediate":
            Fast, order-critical callback.

        - **mode**="worker":
            Heavy/long callback (default).
        """
        with self._subs_lock:
            self._subs.setdefault(event_type, []).append(_Subscription(callback, mode))

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._subs_lock:
            subs = self._subs.get(event_type)
            if not subs:
                return
            self._subs[event_type] = [s for s in subs if s.callback is not callback]

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if not self._running:
            log.warning(
                "emit(%r) after close()/before start() -- event ignored", event_type
            )
            return
        self._queue.put(Event(event_type, payload if payload is not None else {}))

    def _dispatch_loop(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                self._queue.task_done()
                break

            with self._subs_lock:
                subs = list(self._subs.get(event.type, ())) + list(
                    self._subs.get(WILDCARD, ())
                )

            async_count = 0
            for sub in subs:
                if sub.mode == "immediate":
                    self._run_immediate(sub.callback, event)
                else:
                    assert self._executor is not None
                    async_count += 1
                    self._executor.submit(
                        self._run_callback_and_done, sub.callback, event
                    )

            # If no async callbacks, mark done now;
            # otherwise mark done in callback wrapper
            if async_count == 0:
                self._queue.task_done()

    def _run_callback_and_done(
        self, callback: Callable[[Event], None], event: Event
    ) -> None:
        """Run callback and mark queue task as done after async callback completes."""
        try:
            self._run_safe(callback, event)
        finally:
            self._queue.task_done()

    def _run_immediate(self, callback: Callable[[Event], None], event: Event) -> None:
        started = time.monotonic()
        self._run_safe(callback, event)
        elapsed = time.monotonic() - started
        if elapsed > self._immediate_warn_threshold:
            log.warning(
                "Immediate-callback %r for %r was running for %.1f ms and blocking "
                "dispatcher -- try mode='worker'",
                getattr(callback, "__name__", callback),
                event.type,
                elapsed * 1000,
            )

    @staticmethod
    def _run_safe(callback: Callable[[Event], None], event: Event) -> None:
        try:
            callback(event)
        except Exception:
            log.exception("Callback error for event type: %r", event.type)
