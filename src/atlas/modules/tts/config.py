#
# tts / config.py
#
# TtsConfig is read both by TextToSpeech (actual synthesis parameters) and by
# SoundManager (to detect when those parameters changed, and pre-generated
# sounds need regenerating) -- one dataclass for the whole `[tts]` table,
# shared by both.
#

from atlas.utils.config import config


@config(file="tts.toml", table="tts")
class TtsConfig:
    model_path: str = "models/piper/en_US-ryan-medium.onnx"
    use_cuda: bool = False
    volume: float = 1.0
    length_scale: float = 0.85
    noise_scale: float = 1.0
    noise_w_scale: float = 1.0
    normalize_audio: bool = False
    silence_duration: float = 0.5


tts_cfg = TtsConfig.load()
