from atlas.modules.stt.state_machine import State, StateMachine


def test_state_machine_starts_sleeping_when_configured(monkeypatch, event_manager):
    from atlas.modules.stt import state_machine

    monkeypatch.setitem(state_machine.cfg, "start_state", "SLEEPING")

    assert StateMachine(event_manager).state is State.SLEEPING


def test_awake_state_emits_state_events(monkeypatch, event_manager):
    from atlas.modules.stt import state_machine

    monkeypatch.setitem(state_machine.cfg, "start_state", "SLEEPING")
    machine = StateMachine(event_manager)
    received = []
    manager = event_manager
    manager.subscribe("stt.changed_state", lambda e: received.append(e))
    manager.subscribe("ui.state_change", lambda e: received.append(e))

    machine.set_state(State.AWAKE, "wake word")
    manager._queue.join()

    assert machine.state is State.AWAKE
    assert [event.type for event in received] == [
        "stt.changed_state",
        "ui.state_change",
    ]
    assert received[0].payload == {"state": "AWAKE"}
    assert received[1].payload == {"state": "AWAKE", "detail": "wake word"}
