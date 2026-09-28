#
# stt / config.py
#
# One dataclass per TOML table in stt.toml, even where a table is read by
# more than one class (`[vad]` by both VAD and SpeechRecognizer; `[stt]` by
# StateMachine, Whisper and SpeechRecognizer) -- see the note in
# `atlas.utils.config.config` for why that matters.
#

from atlas.utils.config import config


@config(file="stt.toml", table="kws")
class KwsConfig:
    model_dir: str = "models/sherpa_onnx_kws"
    keywords_file: str = "keywords.txt"
    num_threads: int = 1
    score_threshold: float = 0.12


@config(file="stt.toml", table="vad")
class VadConfig:
    model_path: str = "models/vad/silero_vad.onnx"
    threshold: float = 0.5
    min_silence_duration_ms: int = 600
    preroll_blocks: int = 6


@config(file="stt.toml", table="stt")
class SttConfig:
    start_state: str = "SLEEPING"
    model_size: str = "small"
    device: str = "cpu"
    download_root: str = "models/faster-whisper"
    beam_size: int = 5
    cpu_threads: int = 6
    awake_timeout: float = 15.0
    min_command_ms: float = 600.0
    language: str = "en"
    initial_prompt: str = "Terms: Atlas."


kws_cfg = KwsConfig.load()
vad_cfg = VadConfig.load()
stt_cfg = SttConfig.load()
