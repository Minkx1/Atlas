from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kws import KeyWordSpotter
    from .listener import Listener
    from .speech_recognition import SpeechRecognizer
    from .state_machine import State, StateMachine

__all__ = ["KeyWordSpotter", "Listener", "SpeechRecognizer", "State", "StateMachine"]


def __getattr__(name: str):
    if name == "KeyWordSpotter":
        from .kws import KeyWordSpotter

        return KeyWordSpotter
    if name == "Listener":
        from .listener import Listener

        return Listener
    if name == "SpeechRecognizer":
        from .speech_recognition import SpeechRecognizer

        return SpeechRecognizer
    if name in {"State", "StateMachine"}:
        from .state_machine import State, StateMachine

        return State if name == "State" else StateMachine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
