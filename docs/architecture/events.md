# Events

Modules never call each other's methods directly. They only publish and
subscribe to named events through one shared `EventManager`.

## `EventManager`

```python
from atlas.core.events import EventManager

events = EventManager()
events.start()

events.subscribe("stt.transcribed", lambda e: print(e.payload["text"]))
events.emit("stt.transcribed", {"text": "hello"})
```

`emit()` puts an `Event` (type, payload dict, timestamp) on a queue. A single
dispatcher thread reads the queue and, for each subscriber, either:

- runs the callback **immediately**, on the dispatcher thread (`mode="immediate"`) — use
  for fast, order-critical work; a callback that takes too long logs a warning
  instead of silently stalling the whole bus;
- submits it to a small **worker** thread pool (`mode="worker"`, the default) —
  use for anything that touches I/O, models, or subprocesses.

A callback that raises is logged and does not crash the dispatcher or other
subscribers.

## Subscribing from a `Module`

Don't call `events.subscribe()` by hand inside a module — use the `@on_event`
decorator from `atlas.core.module`. `Module.__init__` scans the instance for
decorated methods and registers them automatically:

```python
from atlas.core.module import Module, on_event

class TtsModule(Module):
    name = "tts"

    @on_event("op.command.interrupt")
    def interrupt(self, **kwargs) -> None:
        self.piper.interrupt()

    @on_event("tts.command.speak", "op.llm_chunk")  # one method, multiple events
    def speak(self, text: str = "", **kwargs) -> None:
        self.piper.speak(text)
```

The handler is called as `method(**event.payload)` — always accept `**kwargs`
so adding a new payload field later doesn't break existing handlers.

## Naming convention

Event names are dotted and lowercase, namespaced by the module that owns
them: `<module>.<event>`, with a deeper `<module>.<component>.<event>` for
events that belong to one specific sub-component (e.g. `stt.kws.*`,
`stt.vad.*`).

Within that namespace, a `command` segment marks **directives** — "please do
this" — as opposed to plain **notifications** — "this already happened":

| Kind | Example | Meaning |
| --- | --- | --- |
| Notification | `stt.transcribed` | Speech was transcribed; here's the text |
| Command | `stt.command.set_state` | Please switch the STT state machine |
| Notification | `tts.busy` / `tts.free` | TTS started/stopped speaking |
| Command | `tts.command.speak` | Please synthesize and speak this text |

This distinction is a naming convention, not something the `EventManager`
enforces — there's no separate transport for commands vs. events, and nothing
stops a handler from subscribing to either kind.

## Current event catalog

### Notifications

| Event | Payload | Emitted by |
| --- | --- | --- |
| `stt.kws.keyword_detected` | `{keyword: str}` | `stt` (wake word / hotkey), `keybinds` |
| `stt.vad.start` / `stt.vad.end` | `{}` | `stt` (speech boundary) |
| `stt.transcribed` | `{text: str}` | `stt` (Whisper result) |
| `stt.changed_state` | `{state: str, detail?: str}` | `stt` (state machine) |
| `stt.audiowave` | `{rms: float}` | `stt` (listener, UI meter) |
| `op.intent` | `{intent: str}` | `op` (matched built-in intent) |
| `op.llm_chunk` | `{text: str, is_first: bool}` | `op` (streamed LLM sentence) |
| `tts.busy` / `tts.free` | `{}` | `tts` (synthesis/playback state) |
| `tts.sounds.generate_sound` | `{text: str, path: Path}` | `tts` (sound cache needs a file) |
| `ui.say` | `{text: str}` | `tts`, plugins (UI display only) |

### Commands

| Event | Payload | Handled by |
| --- | --- | --- |
| `core.command.terminate` | `{}` | `Atlas` (starts shutdown) |
| `stt.command.set_state` | `{state: str, detail?: str}` | `stt` |
| `op.command.interrupt` | `{}` | `op`, `tts` |
| `tts.command.speak` | `{text: str}` | `tts` |
| `tts.sounds.command.play_category` | `{category: str}` | `tts` |
| `tts.sounds.play` | `path`/`sound`/`text` dict | `tts` (currently only reachable from a plugin's `event` message) |

This table reflects the code, not a frozen protocol — if you add a module,
extend it, and keep the notification/command naming convention.

## Plugins can emit arbitrary events

A plugin's `event`-type protocol message (see [Plugin API](../plugins.md)) is
forwarded to the bus under whatever name the plugin gives it. There's no
allow-list — a misbehaving plugin can emit any event name, so only install
plugins you trust.
