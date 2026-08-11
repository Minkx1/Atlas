# newt.py

import signal
import sys
import threading

from .config import cfg
from .events import (
    Event,
    EventLogger,
    EventManager,
    EventType,
    emit_event,
)
from .operator import Operator
from .profiler import profiler
from .speech_to_text import KeyWordSpotter, Listener, Whisper
from .text_to_speech import TextToSpeech
from .ui import AssistantUI, console


class Newt:
    def __init__(self) -> None:
        self.events = EventManager()
        self.alive = True
        self.logger = None

        if cfg.log:
            self.logger = EventLogger()

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
        """Subscribe all nececessary callbacks for events."""
        event_manager = EventManager()
        app = self

        # TTS
        event_manager.subscribe(EventType.TTS_SPEAK, lambda e: app.tts.speak(e.content))
        event_manager.subscribe(
            EventType.TTS_PLAY_SOUND, lambda e: app.tts.play_sound(e.content)
        )

        event_manager.subscribe(
            EventType.STT_CHANGED_STATE,
            lambda e: emit_event(EventType.PROFILER_SET_STATE, e.content),
        )
        event_manager.subscribe(
            EventType.STT_TRANSCRIBED,
            lambda e: emit_event(EventType.OP_RECEIVE_CMD, e.content),
        )
        event_manager.subscribe(
            EventType.STT_KEYWORD_DETECTED,
            lambda e: emit_event(EventType.OP_RECEIVE_CMD, "!EVENT_KEYWORD_DETECTED"),
        )

        event_manager.subscribe(EventType.OP_ASK_FINISH, lambda e: app._shutdown())
        event_manager.subscribe(
            EventType.OP_RECEIVE_CMD, lambda e: app.operator.submit(e.content)
        )
        event_manager.subscribe(
            EventType.OP_READY, lambda e: emit_event(EventType.STT_CONTINUE)
        )

        event_manager.subscribe(
            EventType.UI_BANNER, lambda e: AssistantUI.print_banner()
        )
        event_manager.subscribe(
            EventType.UI_STATE_CHANGE,
            lambda e: AssistantUI.print_state_change(**e.content),
        )
        event_manager.subscribe(
            EventType.UI_TRANSCRIPTION,
            lambda e: AssistantUI.print_transcription(**e.content),
        )
        event_manager.subscribe(
            EventType.UI_LLM_CHUNK,
            lambda e: AssistantUI.print_llm_chunk(**e.content),
        )
        event_manager.subscribe(
            EventType.UI_LLM_RESPONSE,
            lambda e: AssistantUI.print_llm_response(**e.content),
        )

        def _prof_start(event: Event | None = None):
            if cfg.profiler:
                profiler.start()

        event_manager.subscribe(EventType.PROFILER_START, lambda e: _prof_start(e))
        event_manager.subscribe(
            EventType.PROFILER_SET_STATE,
            lambda e: profiler.set_state(e.content),
        )
        event_manager.subscribe(EventType.PROFILER_FINISH, app._prof_finish)

    def _shutdown(self):
        self.alive = False

    def _prof_finish(self, *args):
        if cfg.profiler:
            profiler.stop()
            AssistantUI.print_benchmark_report(profiler.get_summary())

    def close(self):
        self._prof_finish()
        if getattr(self, "operator", None):
            self.operator.close()
        if getattr(self, "tts", None):
            self.tts.close()
        if getattr(self, "listener", None):
            self.listener.close()
        self.events.stop()

        self._shutdown()

    def main(self):
        self.tts.start()
        self.operator.start()
        self.listener.start()

        emit_event(EventType.UI_BANNER)

        while self.alive:
            threading.Event().wait(1.0)

    def start(self):
        def _handle_sigint(signum, frame):
            self._shutdown()
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGINT, _handle_sigint)
        except ValueError:
            pass

        try:
            self.main()
        except KeyboardInterrupt:
            console.print("[dim]Stopping assistant...[/]")
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            console.print("[bold red]][!] ERROR[/]: " + str(e))
            sys.exit(1)
        finally:
            self.close()
