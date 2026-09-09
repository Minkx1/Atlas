from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import TtsModule
    from .sound_manager import SoundManager
    from .text_to_speech import TextToSpeech


__all__ = [
    "SoundManager",
    "TextToSpeech",
    "TtsModule",
]


def __getattr__(name: str):
    if name == "SoundManager":
        from .sound_manager import SoundManager

        return SoundManager
    if name == "TextToSpeech":
        from .text_to_speech import TextToSpeech

        return TextToSpeech
    if name == "TtsModule":
        from .module import TtsModule

        return TtsModule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
