#
# tts / module.py
#

from pathlib import Path

from atlas.core.events import CommandType, EventManager, EventType
from atlas.core.module import Module, on_event

from .sound_manager import SoundManager
from .text_to_speech import TextToSpeech


class TtsModule(Module):
    name = "tts"

    def __init__(self, events: EventManager | None = None) -> None:
        self.events = events or EventManager()
        self._register_events(self.events)

        self.piper = TextToSpeech(self.events)
        self.sound = SoundManager(self.events)

    def play_category(self, category: str):
        return self.sound.play_category(category)

    # Module methods

    def load(self) -> None:
        self.piper.load()
        self.sound.load()

    def start(self) -> None:
        self.piper.start()

    def close(self) -> None:
        self.piper.close()

    # Events

    @on_event(EventType.SOUNDS_GENERATE_SOUND)
    def generate_sound(self, text: str = "", path: Path = Path(), **kwargs):
        self.piper._text_to_file(text, path)

    @on_event(EventType.OP_INTERRUPT)
    def interrupt(self, **kwargs) -> None:
        self.piper.interrupt()
        self.sound.interrupt()

    @on_event(CommandType.TTS_SPEAK, EventType.OP_LLM_CHUNK)
    def speak(self, text="", **kwargs):
        self.piper.speak(text)

    @on_event(CommandType.TTS_PLAY_SOUND)
    def play_sound(self, payload: Path | dict[str, str | Path | None] | None, **kwargs):
        if payload:
            self.sound.play_sound(payload)
