# Architecture

Atlas is a local, event-driven voice pipeline. Components have narrow responsibilities, communicate through queues and the event bus, and keep expensive work away from microphone callbacks.

## Runtime at a glance

```mermaid
flowchart LR
    MIC([Microphone]) --> L[Listener]
    L --> K[KWS]
    L --> V[VAD]
    V --> S[SpeechRecognizer]
    S --> W[Faster-Whisper]
    W --> R[Operator]
    R --> C[CommandOperator]
    C -->|intent match| B[Built-in command]
    C -->|plugin intent| P[Plugin process]
    C -->|no match| Q[Local LLM]
    B --> O[Output]
    P --> O
    Q --> O
    O --> T[TTS / sounds]
    O --> U[Textual UI]
```

| Layer | Responsibility | Main modules |
| --- | --- | --- |
| Input | Capture audio and keyboard wake-up | `Listener`, `KeyBindManager` |
| Recognition | Detect wake word, speech boundaries and text | `KeyWordSpotter`, `VAD`, `Whisper` |
| State | Decide whether Atlas may record and when it sleeps | `StateMachine` |
| Decision | Match built-in intents, start plugins, fall back to LLM | `CommandOperator`, `Operator`, `Llama` |
| Output | Synthesize speech, play cached sounds and render UI | `TextToSpeech`, `SoundManager`, `UI` |
| Coordination | Queue notifications and operation requests | `EventManager` |

## Lifecycle

`Atlas` constructs components and registers subscriptions before starting runtime workers.

```mermaid
stateDiagram-v2
    [*] --> Constructed
    Constructed --> Loaded: load_models()
    Loaded --> Running: start workers
    Running --> ShuttingDown: UI exit / fatal error
    ShuttingDown --> Closed: close components
    Closed --> [*]
```

The intended lifecycle vocabulary is:

- `load()` loads models or static resources;
- `start()` starts worker threads or streams;
- `close()` releases resources and is safe to call during shutdown.

Model loading is synchronous during startup. Audio processing, event dispatch, operator work, TTS and plugin execution use independent workers or subprocesses.

## Events and commands

Atlas uses one queue-based transport for two different concepts:

| Concept | Meaning | API |
| --- | --- | --- |
| Event | A fact that already happened | `emit_event(EventType, payload)` |
| Command | A request for a component to perform an operation | `command(CommandType, payload)` |

Both payloads are dictionaries. Empty payloads use `{}`. `Event` exposes the event/command name, `payload`, timestamp and `kind` (`event` or `command`).

```mermaid
sequenceDiagram
    participant Producer
    participant Bus as EventManager
    participant Ordered as Ordered callback
    participant Async as Async callback

    Producer->>Bus: emit_event(..., {payload})
    Bus->>Ordered: invoke in dispatch order
    Bus-->>Async: submit to callback executor
    Async-->>Bus: complete independently
```

Synchronous callbacks preserve ordering and are appropriate for state mutation. Subscribers may opt into asynchronous execution for independent work. Callback failures are logged by the event layer instead of terminating the dispatcher.

### Current command payloads

| Command | Payload |
| --- | --- |
| `TTS_SPEAK` | `{"text": str}` |
| `TTS_PLAY_SOUND` | `{"path": str or Path, "text": str or None}` |
| `OP_SUBMIT` | `{"text": str}` |

### Current event payloads

| Event | Payload |
| --- | --- |
| `KWS_KEYWORD_DETECTED` | `{"keyword": str}` |
| `STT_TRANSCRIBED` | `{"text": str}` |
| `STT_AUDIOWAVE` | `{"rms": float}` |
| `STT_CHANGED_STATE` | `{"state": str}` |
| `UI_STATE_CHANGE` | `{"state": str, "detail": str or optional}` |
| `UI_TRANSCRIPTION` | `{"text": str}` |
| `UI_LLM_CHUNK` | `{"text": str, "is_first": bool}` |
| `UI_ASSISTANT_SAY` | `{"text": str}` |
| `LLM_RESPONSE` | `{"text": str}` |
| `DEBUG_LOG` | `{"message": str, "source": str, "level": str}` |
| `*_LOADED`, `*_START`, `*_FINISH`, `TTS_BUSY`, `TTS_FREE` | `{}` |

Timing and benchmark measurements are intentionally not part of normal UI/event payloads. Benchmark tooling can measure model and pipeline performance separately without making runtime contracts noisy.

## Speech pipeline

```mermaid
flowchart TD
    A[Audio block] --> B[Pre-roll buffer]
    B --> C[Silero VAD]
    C -->|speech start| D{State allows recording?}
    D -->|no| B
    D -->|yes| E[Speech buffer]
    E -->|speech end| F[Whisper queue]
    F --> G[Transcription]
    G --> H[STT_TRANSCRIBED]
```

The `StateMachine` is the authority for `SLEEPING`, `AWAKE`, `RECORDING` and `WAITING`. VAD detects acoustic speech; it does not decide whether Atlas should accept it.

## Decision pipeline

```mermaid
flowchart TD
    A[Transcribed text] --> B[Operator queue]
    B --> C[Semantic matching]
    C -->|confidence passes| D{Intent owner}
    D -->|built-in| E[Sound / assistant action]
    D -->|plugin| F[Start isolated process]
    C -->|no match| G{LLM available?}
    G -->|yes| H[Stream local response]
    G -->|no| I[idk_cmd response]
    H --> J[TTS + UI]
    E --> J
    F --> J
    I --> J
```

`CommandOperator` loads built-in commands from `config/commands.json`, discovers plugins and compares user text with trigger embeddings. `Operator` owns the queue and selects command or LLM execution.

## Output

`TextToSpeech` synthesizes streamed text and `SoundManager` handles cached sound categories and playback. Output events update the UI; output commands request work from TTS or sound playback.

```mermaid
flowchart LR
    A[Operator / plugin] -->|TTS_SPEAK| T[TextToSpeech]
    A -->|TTS_PLAY_SOUND| S[SoundManager]
    T --> B[TTS_BUSY]
    T --> C[TTS_FREE]
    S --> B
    S --> C
    A --> U[UI_LLM_CHUNK / UI_ASSISTANT_SAY]
```

The distinction matters: an event reports `TTS_BUSY`; a command asks Atlas to speak.

## Plugins and isolation

Plugins are subprocesses. Atlas sends invocation context through `stdin`, reads protocol messages from `stdout`, and reads structured logs from `stderr`. See [Plugin API](plugins.md) for the wire format.

This boundary protects the main process from plugin-specific dependencies and makes language-independent plugins possible. It is not a security sandbox; install only plugins you trust.

## Observability

`EventLogger` subscribes to `DEBUG_LOG` and writes daily files under `data/logs/`. The useful diagnostic categories are model loading, state transitions, command matching, plugin execution, output, and exceptions.

## Current boundaries and next steps

The architecture is deliberately in transition during v0.6:

- lifecycle contracts are being standardized;
- event payloads are dictionary-based;
- command identity is separate from event identity;
- built-in commands and plugins are candidates for a shared capability model;
- plugin protocol versioning is planned for v0.7;
- benchmark measurements belong in manual, fixture-driven tooling.

The stable rule for new integrations is simple: publish facts as events, request work as commands, keep payloads small and structured, and avoid reaching into another component's private state.
