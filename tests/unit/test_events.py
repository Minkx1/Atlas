import threading

from atlas.core.events import Event, EventManager


def test_event_and_command_payloads_are_dictionaries(event_manager: EventManager):
    received = []
    manager = event_manager
    manager.subscribe("ui.banner", received.append)
    manager.subscribe("tts.speak", received.append)

    manager.emit("ui.banner")
    manager.emit("tts.speak", {"text": "hello"})
    manager._queue.join()

    assert received[0].payload == {}
    assert received[0].kind == "event"
    assert received[1].payload == {"text": "hello"}
    assert received[1].kind == "command"


def test_async_callback_does_not_block_dispatcher_and_is_flushed(event_manager):
    finished = threading.Event()

    def callback(event: Event):
        finished.set()

    manager = event_manager
    manager.subscribe("ui.banner", callback, asynchronous=True)
    manager.emit("ui.banner", {})
    manager.queue.join()

    assert finished.is_set()
