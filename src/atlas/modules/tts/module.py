#
# tts / module.py
#

from pathlib import Path

from atlas.core.events import EventManager
from atlas.core.module import Module, on_event

from .sound_manager import SoundManager
from .text_to_speech import TextToSpeech


class TtsModule(Module):
    name = "tts"

    def __init__(self, events: EventManager | None = None) -> None:
        super().__init__(events)

        self.piper = TextToSpeech(self.events)
        self.sound = SoundManager(self.events)

    # Module methods

    def load(self) -> None:
        self.piper.load()
        self.sound.load()

    def start(self) -> None:
        self.piper.start()

    def close(self) -> None:
        self.piper.close()

    # Events

    @on_event("tts.sounds.play_category")
    def play_category(self, category: str, **kwargs):
        return self.sound.play_category(category)

    @on_event("tts.sounds.generate_sound")
    def generate_sound(self, text: str = "", path: Path = Path(), **kwargs):
        self.piper._text_to_file(text, path)

    @on_event("op.interrupt")
    def interrupt(self, **kwargs) -> None:
        self.piper.interrupt()
        self.sound.interrupt()

    @on_event("tts.speak", "op.llm_chunk")
    def speak(self, text="", **kwargs):
        self.piper.speak(text)

    @on_event("tts.sounds.play")
    def play_sound(self, payload: Path | dict[str, str | Path | None] | None, **kwargs):
        if payload:
            self.sound.play_sound(payload)
