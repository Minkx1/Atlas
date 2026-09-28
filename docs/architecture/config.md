# Config

`atlas.utils.config` has two layers: a small low-level file loader, and a
`@config` decorator built on top of it for the common case — a typed section
of one module's own settings.

## Layer 1: `Config.load_raw`

The escape hatch. Reads a file under `config/` by name, generating it from an
`origin` string the first time it's missing:

```python
from atlas.utils.config import Config

data = Config.load_raw("keybinds.cfg", "<ctrl>+<alt>+w")
```

Supports `.toml`, `.json`, `.txt`/`.cfg` (the latter two just return
`{"content": "<raw file text>"}`). Use this directly for files that don't map
onto one fixed schema:

- `config/commands.json` — a user-editable, open-ended map of intent → triggers/sounds.
- `config/keybinds.cfg` — a single raw string, not worth a dataclass.

## Layer 2: `@config`

For everything else — a module's own settings, one TOML table:

```python
from atlas.utils.config import config

@config(file="tts.toml", table="tts")
class TtsConfig:
    volume: float = 1.0
    length_scale: float = 0.85

tts_cfg = TtsConfig.load()          # reads config/tts.toml, generates it if missing
tts_cfg = TtsConfig(volume=0.1)     # plain dataclass — no I/O, for tests
```

`@config` turns the class into a `dataclass` and generates the on-disk example
straight from the field defaults — the example can never drift out of sync
with what the code actually reads, because there's no separate example file
to forget to update.

Each module keeps its dataclasses and the module-level singleton in its own
`config.py` (e.g. `atlas/modules/tts/config.py`), and every class that needs
settings takes the config object as a constructor parameter, defaulting to
that singleton:

```python
class TextToSpeech:
    def __init__(self, events=None, cfg: TtsConfig = tts_cfg) -> None:
        self.cfg = cfg
```

This is why tests never need to monkeypatch a module-level dict — they just
construct `TextToSpeech(cfg=TtsConfig(volume=0.1))`.

!!! warning "One dataclass per TOML table, not per class"
    If more than one class reads the same table (`stt.toml`'s `[stt]` table is
    read by the state machine, the Whisper wrapper, and the recognizer), define
    **one** dataclass covering every field any of them needs, and have each
    class take an instance of that same dataclass. Two `@config`-decorated
    classes pointed at the same table would race on generating the file: the
    first one constructed writes an incomplete example, and the second's
    fields silently fall back to their defaults forever.

## Shared config: `general.toml`

Some settings are used by more than one module — the assistant's name/username
(used by both `op` and `tts` for text formatting) and the shared audio capture
format (used throughout `stt`). Instead of duplicating them into every
module's own file, they live once in `config/general.toml`, exposed as:

```python
from atlas.utils.config import general_cfg

general_cfg.identity.name       # "Atlas"
general_cfg.identity.username   # "Sir"
general_cfg.audio.sample_rate   # 16000
```

`GeneralConfig` isn't built with `@config` for the same reason as the warning
above — `general.toml` has two tables (`[identity]`, `[audio]`) read by
different modules, so its example has to be generated for the whole file in
one shot; see `GeneralConfig.load()` in `atlas/utils/config.py` if you need to
add a third shared section.

## See also

- [Config reference](../config-reference.md) — every file and field, with defaults.
- [Modules API](../modules-api.md) — where a new module's `config.py` fits in.
