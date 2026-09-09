import pytest

from atlas.core.events import EventManager
from atlas.core.module import Module, on_event


def require_importable(module_name: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} is not usable: {exc}")


class ProbeModule(Module):
    def __init__(self, events: EventManager):
        self.received = []
        self._register_events(events)

    @on_event("ui.banner")
    def receive(self, value: int = 0, **kwargs):
        self.received.append(value)


def test_on_event_registers_method_on_the_supplied_manager(event_manager):
    probe = ProbeModule(event_manager)

    event_manager.emit("ui.banner", {"value": 42})
    event_manager._queue.join()

    assert probe.received == [42]


def test_op_module_passes_manager_to_child_components(monkeypatch, event_manager):
    require_importable("onnxruntime")
    from atlas.modules.op import module as op_module
    from atlas.modules.op.module import OpModule

    class FakeCommandOperator:
        def __init__(self, events):
            self.events = events

    class FakeLlama:
        def __init__(self, events):
            self.events = events

    monkeypatch.setattr(op_module, "CommandOperator", FakeCommandOperator)
    monkeypatch.setattr(op_module, "Llama", FakeLlama)

    module = OpModule(event_manager)

    assert module.cmd.events is event_manager
    assert module.llm.events is event_manager


def test_stt_module_passes_manager_to_child_components(monkeypatch, event_manager):
    require_importable("sounddevice")
    from atlas.modules.stt import module as stt_module
    from atlas.modules.stt.module import SttModule

    class FakeKws:
        def __init__(self, events):
            self.events = events

    class FakeState:
        def __init__(self, events):
            self.events = events

    class FakeRecognizer:
        def __init__(self, events):
            self.events = events

    class FakeListener:
        def __init__(self, processor, events):
            self.processor = processor
            self.events = events

    monkeypatch.setattr(stt_module, "KeyWordSpotter", FakeKws)
    monkeypatch.setattr(stt_module, "StateMachine", FakeState)
    monkeypatch.setattr(stt_module, "SpeechRecognizer", FakeRecognizer)
    monkeypatch.setattr(stt_module, "Listener", FakeListener)

    module = SttModule(event_manager)

    assert module.kws.events is event_manager
    assert module.state.events is event_manager
    assert module.recognizer.events is event_manager
    assert module.listener.events is event_manager
