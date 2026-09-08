import time

import pytest

from atlas.core.events import EventManager


@pytest.fixture(autouse=True)
def clean_event_manager():
    manager = EventManager()
    yield
    manager.close()
    time.sleep(0.01)  # Ensure threads have time to clean up


@pytest.fixture
def event_manager():
    manager = EventManager()
    manager.start()
    yield manager
    manager.close()
    time.sleep(0.01)  # Ensure threads have time to clean up
