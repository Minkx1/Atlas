# newt.py

import sys
import threading

from .config import cfg
from .events import DebugServer, EventManager, emit_event
from .operator import Operator
from .profiler import profiler
from .speech_to_text import KeyWordSpotter, Listener, Whisper
from .text_to_speech import TextToSpeech
from .ui import AssistantUI, console


class Newt:
    def __init__(self) -> None:
        self.events = EventManager()
        self.alive = True

        if cfg.debug_server:
            self.debug_server = DebugServer()

        self.tts = TextToSpeech()
        self.operator = Operator()
        match cfg.stt.pipeline_mode:
            case "KWS":
                self.kws = KeyWordSpotter()
                self.stt = Whisper()
                self.listener = Listener(self.stt, self.kws)
            case "DIRECT":
                stt = Whisper()
                self.listener = Listener(stt)

        self._setup_subscriptions()

    def _setup_subscriptions(self):
        # TTS
        self.events.subscribe("TTS_SPEAK", lambda e: self.tts.speak(e.content))
        self.events.subscribe(
            "TTS_PLAY_SOUND", lambda e: self.tts.play_sound(e.content)
        )

        # STT
        self.events.subscribe(
            "STT_CHANGED_STATE", lambda e: emit_event("PROFILER_SET_STATE", e.content)
        )
        self.events.subscribe(
            "STT_TRANSCRIBE", lambda e: emit_event("OP_RECEIVE_CMD", e.content)
        )
        self.events.subscribe(
            "STT_KEYWORD_DETECTED",
            lambda: emit_event("OP_RECEIVE_CMD", "!EVENT_KEYWORD_DETECTED"),
        )

        # Operator
        self.events.subscribe("OP_ASK_FINISH", lambda e: self._shutdown())
        self.events.subscribe(
            "OP_RECEIVE_CMD", lambda e: self.operator.submit(e.content)
        )
        self.events.subscribe("OP_READY", lambda e: emit_event("STT_CONTINUE"))

        # UI
        self.events.subscribe("UI_BANNER", lambda e: AssistantUI.print_banner())
        self.events.subscribe(
            "UI_STATE_CHANGE", lambda e: AssistantUI.print_state_change(**e.content)
        )
        self.events.subscribe(
            "UI_TRANSCRIPTION", lambda e: AssistantUI.print_transcription(**e.content)
        )
        self.events.subscribe(
            "UI_LLM_CHUNK", lambda e: AssistantUI.print_llm_chunk(**e.content)
        )
        self.events.subscribe(
            "UI_LLM_RESPONSE", lambda e: AssistantUI.print_llm_response(**e.content)
        )

        # Profiler
        def _prof_start():
            if cfg.profiler:
                profiler.start()

        self.events.subscribe("PROFILER_START", lambda: _prof_start)
        self.events.subscribe(
            "PROFILER_SET_STATE", lambda e: profiler.set_state(e.content)
        )
        self.events.subscribe("PROFILER_FINISH", self._prof_finish)

    def _shutdown(self):
        self.alive = False

    def _prof_finish(self):
        if cfg.profiler:
            profiler.stop()
            AssistantUI.print_benchmark_report(profiler.get_summary())

    def close(self):
        self._prof_finish()
        self.operator.close()
        self.tts.close()
        self.listener.close()
        self.events.stop()

    def main(self):
        if self.debug_server:
            self.debug_server.start()

        self.tts.start()
        self.operator.start()
        self.listener.start()

        emit_event("UI_BANNER")

        while self.alive:
            threading.Event().wait(1.0)

    def start(self):
        try:
            self.main()
        except KeyboardInterrupt:
            console.print("[dim]Stopping assistant...[/]")
        except Exception as e:
            console.print("[bold red]][!] ERROR[/]: " + str(e))
        finally:
            self.close()
            sys.exit(0)
