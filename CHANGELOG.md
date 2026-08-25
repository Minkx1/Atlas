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
