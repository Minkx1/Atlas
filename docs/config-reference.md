# Config reference

Every file below lives under `config/` and is generated with these defaults
the first time it's missing. See [Config](architecture/config.md) for how the
loading mechanism works.

## `general.toml`

Shared across modules — see [Config](architecture/config.md#shared-config-generaltoml).

**`[identity]`**

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | `"Atlas"` | What the assistant calls itself |
| `username` | `"Sir"` | How the assistant addresses you |

**`[audio]`**

| Field | Default | Meaning |
| --- | --- | --- |
| `sample_rate` | `16000` | Capture sample rate (Hz) |
| `channels` | `1` | 1 = mono, 2 = stereo |
| `blocksize` | `512` | Samples per audio block |
| `dtype` | `"float32"` | Sample format |

## `stt.toml`

**`[kws]`** — wake word (Sherpa-ONNX)

| Field | Default | Meaning |
| --- | --- | --- |
| `model_dir` | `"models/sherpa_onnx_kws"` | Model directory under `data/` |
| `keywords_file` | `"keywords.txt"` | Keyword/phoneme definitions, under `config/` |
| `num_threads` | `1` | CPU threads |
| `score_threshold` | `0.12` | Minimum confidence to accept a wake word |

**`[vad]`** — Silero voice activity detection

| Field | Default | Meaning |
| --- | --- | --- |
| `model_path` | `"models/vad/silero_vad.onnx"` | Model path under `data/` |
| `threshold` | `0.5` | Speech-probability threshold |
| `min_silence_duration_ms` | `600` | Silence needed to end a speech segment |
| `preroll_blocks` | `6` | Audio blocks buffered before speech starts |

**`[stt]`** — state machine + Whisper

| Field | Default | Meaning |
| --- | --- | --- |
| `start_state` | `"SLEEPING"` | Initial state (`SLEEPING` or `AWAKE`) |
| `model_size` | `"small"` | Faster-Whisper model size |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `download_root` | `"models/faster-whisper"` | Model cache, under `data/` |
| `beam_size` | `5` | Whisper beam search width |
| `cpu_threads` | `6` | CPU threads for transcription |
| `awake_timeout` | `15.0` | Seconds awake before returning to sleep |
| `min_command_ms` | `600.0` | Minimum recorded duration sent to Whisper |
| `language` | `"en"` | Whisper language hint |
| `initial_prompt` | `"Terms: Atlas."` | Extra context for uncommon words |

## `tts.toml`

**`[tts]`** — Piper synthesis + cached sounds

| Field | Default | Meaning |
| --- | --- | --- |
| `model_path` | `"models/piper/en_US-ryan-medium.onnx"` | Piper voice, under `data/` |
| `use_cuda` | `false` | GPU acceleration |
| `volume` | `1.0` | Output volume multiplier |
| `length_scale` | `0.85` | Speaking speed (lower = faster) |
| `noise_scale` | `1.0` | Audio variation |
| `noise_w_scale` | `1.0` | Phoneme-duration variation |
| `normalize_audio` | `false` | Normalize output level |
| `silence_duration` | `0.5` | Silence padding before playback, seconds |

Changing any of these regenerates every cached sound in `data/sounds/` on
next start (`SoundManager` compares this table against the settings recorded
in `data/sounds/manifest.json`).

## `op.toml`

**`[op]`** — intent matching

| Field | Default | Meaning |
| --- | --- | --- |
| `intent_threshold` | `0.60` | Minimum cosine similarity to accept an intent |
| `margin` | `0.05` | Required lead over the second-best match |

## `llama.toml`

**`[llama]`** — local LLM fallback

| Field | Default | Meaning |
| --- | --- | --- |
| `model_path` | `"models/llm_models/Llama-3.2-3B-Instruct-Q5_K_M.gguf"` | GGUF model, under `data/` |
| `initial_prompt` | *(assistant persona)* | System prompt |
| `context_tokens` | `2048` | Context window |
| `max_msg_tokens` | `128` | Max tokens per response |
| `temperature` | `0.7` | Sampling temperature |

## Not `@config`-managed

| File | Loaded via | Why |
| --- | --- | --- |
| `commands.json` | `Config.load_raw` / `write_from_example` | Open-ended intent → triggers/sounds map, not a fixed schema |
| `keybinds.cfg` | `Config.load_raw` | A single raw string, not a table |
| `config.toml` | *(nothing — legacy)* | Superseded by the per-module files above; kept only so old installs aren't surprised by a missing file |

!!! note "Two fields with no code behind them yet"
    `[kws].awake_keybind` and `[vad].speech_pad_ms` are present in
    `stt.toml` but not currently read by any module — reserved, not wired up.
