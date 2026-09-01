## v0.6.3 (2026-09-01)

### Fix

- Update keybinds.py with lazy pynput import

## v0.6.2 (2026-09-01)

### Fix

- Moved unnecessary breaking imports from build.py

## v0.6.1 (2026-08-30)

### Refactor

- **events**: Refactor of core Event system

## v0.6.0 (2026-08-28)

### Feat

- Added assistant awake keybind

## v0.5.0 (2026-08-28)

### Feat

- Added ability to barge-in mid-speech.

## v0.4.10 (2026-08-28)

### Fix

- Fixed Windows symlink warning from HF Hub

## v0.4.9 (2026-08-28)

### Refactor

- Complete refactor of OP -> TTS pipeline and other general things

## v0.4.8 (2026-08-26)

### Fix

- Added audio-resampling for TTS

## v0.4.7 (2026-08-26)

### Fix

- Added audio-resampling for Listener

## v0.4.6 (2026-08-26)

### Refactor

- **Plugin**: Moved plugin 'timeout' from [plugin] to [execution], style changes to CMD_OP

## v0.4.5 (2026-08-25)

### Fix

- Fixed SSL Certificate error in frezzed builds

## v0.4.4 (2026-08-25)

### Fix

- Fixed Callable syntax error & added onnxruntime to dependencies in pyproject.toml

## v0.4.3 (2026-08-25)

### Refactor

- **STT**: Made SpeechRecognizer more independent

## v0.4.2 (2026-08-25)

### Refactor

- **core**: Reorganized 'src/' into logical sub-modules

## v0.4.1 (2026-08-24)

### BREAKING CHANGE

- Project directory structure has changed.  Users must move their config files to /config and plugins to /plugins.

### Feat

- **Plugin**: Full implementation of plugin system
- **OP**: Added 'Command not recognized' option for operator to say if LLM model was not loaded.

### Fix

- **Plugin**: Fixed log protocol issues
- **Plugin**: Fixed log protocol issues

### Refactor

- **STT**: Moved event subscription from STT to Atlas
- **core**: split data directory into config, plugins, and data
- **STT**: Merged VAD and Whisper into one SpeechRecognizer and made stt components more discrete

## v0.3.5 (2026-08-19)

### Fix

- fix GitHub Actions build

## v0.3.4 (2026-08-18)
