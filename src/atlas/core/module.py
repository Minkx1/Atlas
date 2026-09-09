#
# core / module.py
# Contains base Module class for every module
#

import importlib
import inspect
import logging
import pkgutil
from types import ModuleType

from atlas.core.events import DispatchMode, EventManager

log = logging.getLogger(__name__)


class Module:
    name: str

    def __init__(self, events: EventManager | None, **kwargs) -> None:
        self.events = events or EventManager()
        self._register_events(self.events)

    def _register_events(self, em: EventManager) -> None:
        """Automatically registers and subscribes marked methods."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_subscribe_events") and hasattr(attr, "_dispatch_mode"):
                for event_type in attr._subscribe_events:

                    def make_callback(method):
                        return lambda e: method(**e.payload)

                    em.subscribe(
                        event_type, make_callback(attr), mode=attr._dispatch_mode
                    )

    def load(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def on_event(*event_types: str, mode: DispatchMode = "worker"):
    """Mark method for event subscription"""

    def wrapper(func):
        func._subscribe_events = event_types
        func._dispatch_mode = mode
        return func

    return wrapper


def discover_modules(package: ModuleType) -> dict[str, type[Module]]:
    """
    import atlas.modules
    discover_modules(atlas.modules)

    Exptected structure:
        atlas/modules/<name>/module.py
        atlas/modules/<module>.py (wip)
    """
    found: dict[str, type[Module]] = {}

    for _, dotted_name, is_pkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        entry_point = dotted_name if not is_pkg else f"{dotted_name}.module"

        try:
            mod = importlib.import_module(entry_point)
        except ModuleNotFoundError:
            log.warning("Module `%s` not found -- skipping", dotted_name)
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, Module)
                and obj is not Module
                and obj.__module__ == mod.__name__
                and not inspect.isabstract(obj)
            ):
                found[obj.name] = obj

    return found
