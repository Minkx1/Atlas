from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_operator import CommandOperator
    from .global_operator import Operator
    from .llama import Llama

__all__ = ["CommandOperator", "Llama", "Operator"]


def __getattr__(name: str):
    if name == "CommandOperator":
        from .cmd_operator import CommandOperator

        return CommandOperator
    if name == "Llama":
        from .llama import Llama

        return Llama
    if name == "Operator":
        from .global_operator import Operator

        return Operator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
