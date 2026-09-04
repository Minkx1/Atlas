from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.atlas import Atlas

__all__ = ["Atlas"]


def __getattr__(name: str):
    if name == "Atlas":
        from .core.atlas import Atlas

        return Atlas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
