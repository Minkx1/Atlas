import threading

from atlas.core.events import Event, EventManager


def test_event_and_command_payloads_are_dictionaries(event_manager: EventManager):
    received = []
    manager = event_manager
    manager.subscribe("ui.banner", lambda e: received.append(e))
    manager.subscribe("tts.speak", lambda e: received.append(e))

    manager.emit("ui.banner")
    manager.emit("tts.speak", {"text": "hello"})
    manager._queue.join()

    assert received[0].payload == {}
    assert received[0].type == "ui.banner"
    assert received[1].payload == {"text": "hello"}
    assert received[1].type == "tts.speak"


def test_async_callback_does_not_block_dispatcher_and_is_flushed(event_manager):
    finished = threading.Event()

    def callback(event: Event):
        finished.set()

    manager = event_manager
    manager.subscribe("ui.banner", callback, mode="worker")
    manager.emit("ui.banner", {})
    manager._queue.join()

    assert finished.is_set()
