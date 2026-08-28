#
#  state_machine.py
#

import sys
import time
from enum import StrEnum
from pathlib import Path

_MAIN = __name__ == "__main__"
if not _MAIN:
    from ..core.config import cfg
    from ..core.events import EventType, emit_event
else:
    # changing execution dir to src/ for proper importing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import cfg
    from core.events import EventType, emit_event


class State(StrEnum):
    WAITING = "WAITING"
    SLEEPING = "SLEEPING"
    AWAKE = "AWAKE"
    RECORDING = "RECORDING"


class StateMachine:
    def __init__(self) -> None:
        self.state = State.SLEEPING

        if cfg.stt.start_state == "AWAKE":
            self.state = State.AWAKE
        else:
            self.state = State.SLEEPING

        self.awake_deadline = 0.0

    def update_deadline(self) -> None:
        """Updates deadline when needed, so it is not reached during talking or processing."""
        self.awake_deadline = time.monotonic() + cfg.stt.awake_timeout

    def is_deadline_expired(self) -> bool:
        return time.monotonic() > self.awake_deadline

    def set_state(self, new_state: State, detail: str | None = None) -> None:
        if self.state != new_state:
            self.state = new_state
            if new_state == State.AWAKE:
                self.update_deadline()

            emit_event(EventType.STT_CHANGED_STATE, new_state.value)

            payload = {"state": new_state.value}
            if detail:
                payload["detail"] = detail
            emit_event(EventType.UI_STATE_CHANGE, payload)

    def update(self) -> None:
        if self.state == State.WAITING:
            self.update_deadline()
        elif self.state == State.AWAKE and self.is_deadline_expired():
            self.set_state(
                State.SLEEPING,
                detail=f"Timeout ({int(cfg.stt.awake_timeout)}s)",
            )
        elif self.state == State.RECORDING:
            self.update_deadline()

    def allow_speech_recognition(self) -> bool:
        return self.state not in {State.WAITING, State.SLEEPING}


if _MAIN:
    ...
