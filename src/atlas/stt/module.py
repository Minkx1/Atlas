#
# stt / module.py
#

from typing import TYPE_CHECKING

from atlas.core.events import CommandType, EventManager, EventType, log
from atlas.core.module import Module, on_event

from .kws import KeyWordSpotter
from .listener import Listener
from .speech_recognition import SpeechRecognizer
from .state_machine import State as SMState
from .state_machine import StateMachine

if TYPE_CHECKING:
    import numpy as np


class SttModule(Module):
    name = "stt"

    def __init__(self, events: EventManager | None = None) -> None:
        self.events = events or EventManager()
        self._register_events(self.events)

        self.kws = KeyWordSpotter(self.events)
        self.state = StateMachine(self.events)
        self.recognizer = SpeechRecognizer(self.events)
        self.listener = Listener(self._processor, self.events)

    def _processor(self, chunk: np.ndarray) -> None:
        kw = self.kws.process_chunk(chunk)
        if kw:
            self.events.emit(EventType.KWS_KEYWORD_DETECTED, {"keyword": kw})
        self.state.update()

        allow_rec = self.state.allow_speech_recognition()
        self.recognizer.process(chunk, allow_rec)

    # Module methods

    def load(self) -> None:
        self.kws.load()
        self.recognizer.load()

    def start(self) -> None:
        self.recognizer.start()
        self.listener.start()

    def close(self) -> None:
        if hasattr(self, "listener"):
            self.listener.close()
        if hasattr(self, "kws"):
            if hasattr(self.kws, "stream"):
                del self.kws.stream
            if hasattr(self.kws, "kws"):
                del self.kws.kws
        if hasattr(self, "recognizer"):
            self.recognizer.close()

    # Events

    @on_event(EventType.TTS_BUSY, EventType.OP_START)
    def waiting_state(self, **kwargs) -> None:
        """Waiting when the TTS is speaking"""
        self.state.set_state(SMState.WAITING)

    @on_event(CommandType.SET_STATE)
    def set_state(self, state: str = "", detail: str | None = None, **kwargs) -> None:
        self.state.set_state(SMState(state), detail)

    @on_event(EventType.STT_CHANGED_STATE)
    def reset_kws(self, state: str = "") -> None:
        if state == "SLEEPING":
            self.kws.reset()

    @on_event(EventType.VAD_START)
    def on_vad_start(self, **kwargs) -> None:
        self.state.set_state(SMState.RECORDING)

    @on_event(EventType.VAD_END)
    def on_vad_end(self, **kwargs) -> None:
        self.state.set_state(SMState.AWAKE)

    @on_event(EventType.TTS_FREE)
    def awaken(self, **kwargs) -> None:
        self.state.set_state(SMState.AWAKE)
        self.state.update_deadline()

    @on_event(EventType.KWS_KEYWORD_DETECTED)
    def handle_kw_detected(self, keyword: str = "", **kwargs):
        if self.state.state == SMState.WAITING:
            self.events.emit(EventType.OP_INTERRUPT, {})
            self.state.set_state(SMState.AWAKE, f"Interrupted: {keyword}")
        else:
            # app.operator.submit("!EVENT_KEYWORD_DETECTED")
            log(f"Keyword detected directly: {keyword}.", level="INFO")
            self.events.emit(EventType.OP_INTENT, {"intent": "greet"})
            self.state.set_state(SMState.AWAKE, f"Keyword: {keyword}")
