#
# core/schema.py
#

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import NotRequired, TypedDict

# Payloads


class EmptyPayload(TypedDict):
    """Events without data (OP_START, VAD_START, TTS_FREE etc)."""


class TextPayload(TypedDict):
    text: str


class IntentPayload(TypedDict):
    intent: str


class LLMChunkPayload(TypedDict):
    text: str
    is_first: bool


class KeywordPayload(TypedDict):
    keyword: str


class AudioWavePayload(TypedDict):
    rms: float


class SetStatePayload(TypedDict):
    state: str
    detail: NotRequired[str | None]


class SoundGenerationPayload(TypedDict):
    text: str
    path: Path


class SoundPlaybackPayload(TypedDict):
    path: NotRequired[str | Path | None]
    sound: NotRequired[str | Path | None]
    text: NotRequired[str | None]


class LogPayload(TypedDict):
    message: str
    source: str
    level: str


# Identifiers


class EventType(StrEnum):
    TTS_LOADED = "TTS_LOADED"
    SOUNDS_GENERATE_SOUND = "SOUNDS_GENERATE_SOUND"
    TTS_BUSY = "TTS_BUSY"
    TTS_FREE = "TTS_FREE"

    KWS_LOADED = "KWS_LOADED"
    KWS_KEYWORD_DETECTED = "KWS_KEYWORD_DETECTED"

    VAD_LOADED = "VAD_LOADED"
    VAD_START = "VAD_START"
    VAD_END = "VAD_END"

    WHISPER_LOADED = "WHISPER_LOADED"
    STT_MUTE = "STT_MUTE"
    STT_UNMUTE = "STT_UNMUTE"
    STT_AUDIOWAVE = "STT_AUDIOWAVE"
    STT_CHANGED_STATE = "STT_CHANGED_STATE"
    STT_TRANSCRIBED = "STT_TRANSCRIBED"
    STT_START = "STT_START"
    STT_FINISH = "STT_FINISH"

    OP_INTERRUPT = "OP_INTERRUPT"
    OP_INTENT = "OP_INTENT"
    OP_START = "OP_LLM_START"
    OP_LLM_CHUNK = "OP_LLM_CHUNK"
    OP_FINISH = "OP_LLM_FINISH"

    UI_BANNER = "UI_BANNER"
    UI_STATE_CHANGE = "UI_STATE_CHANGE"
    UI_TRANSCRIPTION = "UI_TRANSCRIPTION"
    UI_LLM_CHUNK = "UI_LLM_CHUNK"
    UI_LLM_RESPONSE = "UI_LLM_RESPONSE"
    UI_LLM_RESPONSE_DONE = "UI_LLM_RESPONSE_DONE"
    UI_ASSISTANT_SAY = "UI_ASSISTANT_SAY"

    DEBUG_LOG = "DEBUG_LOG"

    LLM_RESPONSE = "LLM_RESPONSE"
    LLM_LOADED = "LLM_LOADED"

    WILDCARD = "*"


class CommandType(StrEnum):
    TTS_SPEAK = "TTS_SPEAK"
    TTS_PLAY_SOUND = "TTS_PLAY_SOUND"
    OP_SUBMIT = "OP_SUBMIT"
    SET_STATE = "SET_STATE"
