# Modules API

A module is how you add a built-in capability that runs inside Atlas's own
process (as opposed to a [plugin](plugins.md), which runs as an isolated
subprocess). Use a module when you need tight integration with the event bus,
shared state, or something that must run on every startup — not for
one-off voice commands, which almost always belong in a plugin instead.

## Layout convention

```text
src/atlas/modules/
└── my_feature/
    ├── __init__.py
    ├── module.py      # required: the Module subclass, discovered automatically
    └── config.py      # optional: @config dataclasses, if the module has settings
```

`discover_modules()` (see [Lifecycle](architecture/lifecycle.md)) imports
`atlas.modules.my_feature.module` and picks up the one `Module` subclass it
finds there — no manual registration anywhere else.

## Minimal module

```python
# src/atlas/modules/my_feature/module.py
from atlas.core.events import EventManager
from atlas.core.module import Module, on_event


class MyFeatureModule(Module):
    name = "my_feature"  # unique across all modules; used as the dict key in Atlas.modules

    def __init__(self, events: EventManager | None = None) -> None:
        super().__init__(events)
        # construct sub-components here; keep this cheap, no I/O

    def load(self) -> None:
        """Load models/files. Runs once, synchronously, before start()."""

    def start(self) -> None:
        """Start any worker threads. Runs once, after every module has loaded."""

    def close(self) -> None:
        """Release resources. Must be safe to call even if load()/start() never ran."""

    @on_event("stt.transcribed")
    def on_transcribed(self, text: str = "", **kwargs) -> None:
        self.events.emit("op.intent", {"intent": "my_feature.triggered"})
```

`load()`/`start()`/`close()` are required by the `Module` base class — even a
module with nothing to do there needs the (empty) methods, since `Atlas`
calls them unconditionally.

## Adding config

If the module has its own settings, add a `config.py` next to `module.py`
following the pattern in [Config](architecture/config.md):

```python
# src/atlas/modules/my_feature/config.py
from atlas.utils.config import config

@config(file="my_feature.toml", table="my_feature")
class MyFeatureConfig:
    threshold: float = 0.5

my_feature_cfg = MyFeatureConfig.load()
```

Then thread it through as a constructor default, not a module-level global
read inline:

```python
def __init__(self, events=None, cfg: MyFeatureConfig = my_feature_cfg) -> None:
    super().__init__(events)
    self.cfg = cfg
```

This is what makes the module testable without touching disk: a test just
does `MyFeatureModule(cfg=MyFeatureConfig(threshold=0.9))`.

## Rules

- Talk to other modules **only** through events — never import another
  module's classes to call them directly. See [Events](architecture/events.md)
  for the naming convention (`module.event` for notifications,
  `module.command.verb` for directives).
- Keep `__init__` cheap. Model loading, downloads, and anything slow belongs
  in `load()`.
- `close()` must not assume `load()`/`start()` ran — `Atlas` calls `close()`
  on every constructed module during shutdown, including ones that failed
  earlier in `load_modules()`.
- Prefer `mode="worker"` (the `@on_event` default) for handlers that do real
  work; reserve `mode="immediate"` for fast, order-critical state updates.

## Ignoring a module

Any module can be excluded at construction time — this is how the UI stays
off by default:

```python
Atlas(ignored_modules=["ui"])
```
