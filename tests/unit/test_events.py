import threading

from src.core.events import (
    CommandType,
    Event,
    EventManager,
    EventType,
    command,
    emit_event,
)


def reset_events():
    manager = EventManager()
    manager.flush_and_stop()


def test_event_and_command_payloads_are_dictionaries():
    received = []
    manager = EventManager()
    manager.subscribe(EventType.UI_BANNER, received.append)
    manager.subscribe(CommandType.TTS_SPEAK, received.append)

    emit_event(EventType.UI_BANNER)
    command(CommandType.TTS_SPEAK, {"text": "hello"})
    manager.queue.join()

    assert received[0].payload == {}
    assert received[0].kind == "event"
    assert received[1].payload == {"text": "hello"}
    assert received[1].kind == "command"
    reset_events()


def test_async_callback_does_not_block_dispatcher_and_is_flushed():
    finished = threading.Event()

    def callback(event: Event):
        finished.set()

    manager = EventManager()
    manager.subscribe(EventType.UI_BANNER, callback, asynchronous=True)
    emit_event(EventType.UI_BANNER, {})
    manager.flush_and_stop()

    assert finished.is_set()
