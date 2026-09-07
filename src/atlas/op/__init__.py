from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_operator import CommandOperator
    from .llama import Llama
    from .module import OpModule

__all__ = ["CommandOperator", "Llama", "OpModule"]


def __getattr__(name: str):
    if name == "CommandOperator":
        from .cmd_operator import CommandOperator

        return CommandOperator
    if name == "Llama":
        from .llama import Llama

        return Llama
    if name == "OpModule":
        from .module import OpModule

        return OpModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
