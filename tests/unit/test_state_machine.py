from src.core.events import EventManager, EventType
from src.stt.state_machine import State, StateMachine


def test_state_machine_starts_sleeping_when_configured(monkeypatch):
    from src.stt import state_machine

    monkeypatch.setattr(state_machine.cfg.stt, "start_state", "SLEEPING")

    assert StateMachine().state is State.SLEEPING


def test_awake_state_emits_state_events(monkeypatch):
    from src.stt import state_machine

    monkeypatch.setattr(state_machine.cfg.stt, "start_state", "SLEEPING")
    machine = StateMachine()
    received = []
    manager = EventManager()
    manager.subscribe(EventType.STT_CHANGED_STATE, received.append)
    manager.subscribe(EventType.UI_STATE_CHANGE, received.append)

    machine.set_state(State.AWAKE, "wake word")
    manager.queue.join()

    assert machine.state is State.AWAKE
    assert [event.name for event in received] == [
        EventType.STT_CHANGED_STATE.value,
        EventType.UI_STATE_CHANGE.value,
    ]
    assert received[0].payload == {"state": "AWAKE"}
    assert received[1].payload == {"state": "AWAKE", "detail": "wake word"}
