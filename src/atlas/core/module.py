#
# core / module.py
# Contains base Module class for every module
#


from atlas.core.events import EventManager


class Module:
    name: str

    def _register_events(self, em: EventManager) -> None:
        """Automatically registers and subscribes marked methods."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_subscribe_events"):
                for event_type in attr._subscribe_events:

                    def make_callback(method):
                        return lambda e: method(**e.payload)

                    em.subscribe(event_type, make_callback(attr))

    def load(self) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...


def on_event(*event_types: str):
    """Mark method for event subscription"""

    def wrapper(func):
        func._subscribe_events = event_types
        return func

    return wrapper
