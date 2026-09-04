import numpy as np

from src.op.global_operator import Operator
from src.stt.speech_recognition import VAD


def test_sentence_chunker_keeps_punctuation_and_flushes_tail():
    tokens = ["First", " sentence. ", "Second", "!", " tail"]

    assert list(Operator._sentence_chunker(tokens)) == [
        "First sentence.",
        "Second!",
        "tail",
    ]


def test_vad_normalizes_mono_int16_audio():
    normalized = VAD._normalize_chunk(np.array([0, 16384, -32768], dtype=np.int16))

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 3)
    np.testing.assert_allclose(normalized, [[0.0, 0.5, -1.0]])
