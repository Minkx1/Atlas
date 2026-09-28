#
# stt / state_machine.py
# Responsible for proper state changing/saving and callbacks
#

import time
from enum import StrEnum

from atlas.core.events import EventManager

from .config import SttConfig, stt_cfg


class State(StrEnum):
    WAITING = "WAITING"
    SLEEPING = "SLEEPING"
    AWAKE = "AWAKE"
    RECORDING = "RECORDING"


class StateMachine:
    def __init__(
        self,
        events: EventManager | None = None,
        cfg: SttConfig = stt_cfg,
    ) -> None:
        self.events = events or EventManager()

        self.state = State.SLEEPING
        self.awake_timeout = cfg.awake_timeout

        if cfg.start_state == "AWAKE":
            self.state = State.AWAKE
        else:
            self.state = State.SLEEPING

        self.awake_deadline = 0.0

    def update_deadline(self) -> None:
        """Updates deadline to prevent going to sleep during talking or processing."""
        self.awake_deadline = time.monotonic() + self.awake_timeout

    def is_deadline_expired(self) -> bool:
        return time.monotonic() > self.awake_deadline

    def set_state(self, new_state: State, detail: str | None = None) -> None:
        if self.state != new_state:
            self.state = new_state
            if new_state == State.AWAKE:
                self.update_deadline()

            payload = {"state": new_state.value}
            if detail:
                payload["detail"] = detail
            self.events.emit("stt.changed_state", payload)

    def update(self) -> None:
        if self.state == State.WAITING:
            self.update_deadline()
        elif self.state == State.AWAKE and self.is_deadline_expired():
            self.set_state(
                State.SLEEPING,
                detail=f"Timeout ({int(self.awake_timeout)}s)",
            )
        elif self.state == State.RECORDING:
            self.update_deadline()

    def allow_speech_recognition(self) -> bool:
        return self.state not in {State.WAITING, State.SLEEPING}
