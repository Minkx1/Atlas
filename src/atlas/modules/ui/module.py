#
# ui / module.py
#

import logging

from atlas.core.events import EventManager
from atlas.core.module import Module, on_event

from .ui import UI

log = logging.getLogger(__name__)


class UiModule(Module):
    name = "ui"

    def __init__(self, events: EventManager | None = None) -> None:
        super().__init__(events)

        self.ui = UI(self.events)

    # Module methods

    def load(self) -> None:
        """Load and initialize the UI."""
        pass

    def start(self) -> None:
        """Start the UI."""
        self.ui.run()

    def close(self) -> None:
        """Close the UI."""
        if getattr(self.ui, "is_running", False):
            self.ui.exit()

    # Events

    @on_event("stt.changed_state")
    def on_stt_changed_state(self, state: str = "") -> None:
        """Update the UI when the STT state changes."""
        self.ui.update_state(state)

    @on_event("stt.audiowave")
    def on_audio_wave(self, rms: float = 0.0) -> None:
        """Update the audio waveform."""
        self.ui.update_waveform(rms)

    @on_event("stt.transcribed")
    def on_stt_transcribed(self, text: str = "") -> None:
        """Display recognized speech in the dialog."""
        self.ui.add_user_message(text)

    @on_event("op.llm_chunk")
    def on_llm_chunk(self, text: str = "", is_first: bool = False) -> None:
        """Display a streamed LLM response."""
        self.ui.add_llm_chunk(text, is_first)

    @on_event("ui.say")
    def on_assistant_say(self, text: str = "") -> None:
        """Display an assistant message."""
        self.ui.add_assistant_message(text)

    @on_event("core.terminate")
    def on_terminate(self, **kwargs) -> None:
        """Close the UI when Atlas is terminating."""
        self.ui.exit()
