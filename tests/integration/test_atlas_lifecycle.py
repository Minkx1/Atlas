"""End-to-end tests for Atlas's own orchestration: module discovery, the
start/close lifecycle and event routing between real Module instances --
without touching any real hardware/ML backends (those are faked).
"""

import time

import pytest

from atlas.core import atlas as atlas_module
from atlas.core.atlas import Atlas
from atlas.core.module import Module, on_event


class _Producer(Module):
    name = "producer"

    def load(self) -> None:
        pass

    def start(self) -> None:
        self.events.emit("producer.ping", {"n": 1})

    def close(self) -> None:
        pass


class _Consumer(Module):
    name = "consumer"

    def __init__(self, events) -> None:
        super().__init__(events)
        self.received: list[int] = []
        self.closed = False

    @on_event("producer.ping")
    def on_ping(self, n: int = 0, **kwargs) -> None:
        self.received.append(n)

    def load(self) -> None:
        pass

    def start(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _fake_discover(package):
    return {"producer": _Producer, "consumer": _Consumer}


@pytest.fixture
def atlas_app(tmp_path, monkeypatch):
    # keep logging/data out of the real repo's data/ directory
    monkeypatch.setattr(atlas_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr("atlas.core.module.discover_modules", _fake_discover)

    app = Atlas()
    yield app
    app.close()


def test_atlas_discovers_and_wires_modules(atlas_app):
    assert set(atlas_app.modules) == {"producer", "consumer"}


def test_ignored_modules_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr("atlas.core.module.discover_modules", _fake_discover)

    app = Atlas(ignored_modules=["consumer"])
    try:
        assert set(app.modules) == {"producer"}
    finally:
        app.close()


def test_full_event_roundtrip_between_modules(atlas_app):
    """Producer emits on start(); Consumer is subscribed via @on_event and
    should receive it through the real EventManager dispatch loop."""
    atlas_app.start()
    atlas_app.events._queue.join()

    consumer = atlas_app.modules["consumer"]
    assert consumer.received == [1]


def test_close_actually_closes_modules(atlas_app):
    atlas_app.start()
    atlas_app.close()

    consumer = atlas_app.modules["consumer"]
    assert consumer.closed is True


def test_close_is_idempotent(atlas_app):
    atlas_app.start()
    atlas_app.close()
    atlas_app.close()  # must not raise / must not double-run cleanup


def test_core_terminate_event_stops_the_run_loop_and_close_still_cleans_up(
    atlas_app,
):
    """Regression test: `shutdown()` (triggered by `core.terminate`, e.g. the
    'farewell' voice command or the UI quit action) used to flip `alive` to
    False *before* `run()`'s `finally: self.close()` ran. Since `close()`
    guarded its cleanup on `if self.alive`, modules were never actually
    closed on this -- the normal -- shutdown path.
    """
    atlas_app.start()
    atlas_app.events.emit("core.terminate")

    deadline = time.monotonic() + 2.0
    while atlas_app.alive and time.monotonic() < deadline:
        time.sleep(0.01)
    assert atlas_app.alive is False

    atlas_app.close()

    consumer = atlas_app.modules["consumer"]
    assert consumer.closed is True
