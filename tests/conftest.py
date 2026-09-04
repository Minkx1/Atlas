import pytest

from src.core.events import EventManager


@pytest.fixture(autouse=True)
def clean_event_manager():
    EventManager().flush_and_stop()
    yield
    EventManager().flush_and_stop()

