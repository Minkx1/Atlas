# events.py

import json
import queue
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    name: str
    content: Any = None
    timestamp: float = 0


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

    def emit(self, name: str | None, content: Any = None):
        self.queue.put(Event(name, content, time.time()) if name else None)

    def subscribe(self, event_name: str, callback: Callable):
        if event_name not in self.callbacks:
            self.callbacks[event_name] = []
        self.callbacks[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable):
        if event_name in self.callbacks and callback in self.callbacks[event_name]:
            self.callbacks[event_name].remove(callback)

    def _dispatch_loop(self):
        while True:
            event = self.queue.get()

            if event is None:
                self.queue.task_done()
                break

            callbacks_to_call = self.callbacks.get(event.name, [])
            # event "*" is Any event that was mitted
            callbacks_to_call = callbacks_to_call + self.callbacks.get("*", [])

            for callback in callbacks_to_call:
                try:
                    callback(event)
                except Exception as e:
                    print(f"[!] Error in callback for {event.name}: {e}")

            self.queue.task_done()

    def wait_for(self, event_name: str, timeout: float | None = None) -> Event | None:
        """Blocks thread until event is emitted."""
        wait_event = threading.Event()
        received_event = None

        def _unblock(e: Event):
            nonlocal received_event
            received_event = e
            wait_event.set()

        self.subscribe(event_name, _unblock)
        wait_event.wait(timeout)
        self.unsubscribe(event_name, _unblock)

        return received_event

    def stop(self):
        self.queue.put(None)


def emit_event(name: str | None, content: Any = None):
    EventManager().emit(name, content)


def wait_for(event_name: str, timeout: float | None = None) -> Event | None:
    return EventManager().wait_for(event_name, timeout)


class DebugServer:
    """Local server that streams events into other terminal.

    Run client with:
    nc localhost 9999"""

    def __init__(self, port=9999):
        self.port = port
        self.clients = []
        self.server_thread = threading.Thread(
            target=self._run_server, name="EVENT_DEBUG_SERVER", daemon=True
        )

        EventManager().subscribe("*", self._broadcast_event)

    def start(self):
        self.server_thread.start()

    def _run_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", self.port))
        server.listen(5)

        while True:
            try:
                client, addr = server.accept()
                self.clients.append(client)
            except Exception:
                break

    def _broadcast_event(self, event: Event):
        if not self.clients:
            return

        content_str = (
            str(event.content)[:100] + "..."
            if len(str(event.content)) > 100
            else str(event.content)
        )
        msg = f"[{time.strftime('%H:%M:%S', time.localtime(event.timestamp))}] {event.name}: {content_str}\n"

        for client in list(self.clients):
            try:
                client.send(msg.encode("utf-8"))
            except Exception:
                self.clients.remove(client)
