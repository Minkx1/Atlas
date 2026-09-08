import pytest

from atlas.core.events import EventManager


@pytest.fixture(autouse=True)
def clean_event_manager():
    manager = EventManager()
    yield
    manager.close()


@pytest.fixture
def event_manager():
    return EventManager()
