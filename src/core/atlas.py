# atlas.py

import sys
import time

from ..op import CommandOperator, Llama, Operator
from ..stt import KeyWordSpotter, Listener, SpeechRecognizer, State, StateMachine
from ..tts import SoundManager, TextToSpeech
from .config import cfg
from .events import (
    EventLogger,
    EventManager,
    EventType,
    emit_event,
    log,
)
from .keybinds import KeyBindManager

# from .ui import AssistantUI, console
from .ui import UI


class Atlas:
    def __init__(self) -> None:
        # light components
        self.events = EventManager()
        self.alive = True
        self.logger = None

        if cfg.log:
            self.logger = EventLogger()
            self.logger._write_file(
                f" ===== New Session: [{time.strftime('%H:%M:%S', time.localtime(time.time()))}] | SUCCESS ===== \n",
                time.time(),
            )

        self.keybinds = KeyBindManager()
        self.keybinds.register_keybind(
            cfg.kws.awake_keybind,
            lambda: emit_event(EventType.KWS_KEYWORD_DETECTED, "{HotKey}"),
        )

        # STT Pipeline
        self.kws = KeyWordSpotter()
        self.sr = SpeechRecognizer()
        self.sm = StateMachine()

        def audio_process(audio_chunk):
            kw = self.kws.process_chunk(audio_chunk)
            if kw:
                emit_event(EventType.KWS_KEYWORD_DETECTED, kw)
            self.sm.update()

            allow_rec = self.sm.allow_speech_recognition()
            self.sr.process(audio_chunk, allow_rec)

        self.listener = Listener(audio_process)

        # TTS
        self.sound_manager = SoundManager()
        self.tts = TextToSpeech()

        # Operator
        self.cmd = CommandOperator()
        self.llama = Llama()
        self.operator = Operator(self.cmd, self.llama)

        self._setup_subscriptions()

    def _shutdown(self):
        self.alive = False
        if hasattr(self, "ui") and getattr(self.ui, "is_running", False):
            self.ui.call_from_thread(self.ui.exit)

    def load_models(self):
        try:
            log("Starting model loading...", "ATLAS", "INFO")
            self.kws.load()
            self.sr.load()

            self.tts.load()
            self.sound_manager.load()

            self.cmd.load()
            self.llama.load()

            log("All models loaded successfully.", "ATLAS", "SUCCESS")
        except Exception as e:
            log(
                f"Error loading models: {type(e).__name__}: {e}",
                "ATLAS",
                "ERROR",
            )
            raise

    def _setup_subscriptions(self):
        """Subscribe all nececessary callbacks for events."""
        em = EventManager()
        app = self

        # TTS
        em.subscribe(
            EventType.SOUNDS_GENERATE_SOUND,
            lambda e: app.tts._text_to_file(**e.content),
        )

        em.subscribe(EventType.TTS_BUSY, lambda e: app.sm.set_state(State.WAITING))

        def handle_tts_free(e):
            app.sm.set_state(State.AWAKE)
            app.sm.update_deadline()

        em.subscribe(EventType.TTS_FREE, handle_tts_free)

        def handle_interrupt(e):
            app.tts.interrupt()
            app.sound_manager.interrupt()
            app.operator.interrupt()

        em.subscribe(EventType.OP_INTERRUPT, handle_interrupt)
        em.subscribe(EventType.OP_START, lambda e: self.sm.set_state(State.WAITING))

        def handle_intent(event):
            intent: str = event.content
            app.sound_manager.play_category(intent)

            if intent == "farewell":
                app._shutdown()
            if intent == "sleep":
                app.sm.set_state(State.SLEEPING)

        em.subscribe(EventType.OP_INTENT, handle_intent)

        em.subscribe(EventType.OP_LLM_CHUNK, lambda e: app.tts.speak(e.content))

        # STT
        em.subscribe(
            EventType.STT_CHANGED_STATE,
            lambda e: app.kws.reset() if e.content == "SLEEPING" else None,
        )

        def handle_kw_detected(e):
            if app.sm.state == State.WAITING:
                emit_event(EventType.OP_INTERRUPT)
                app.sm.set_state(State.AWAKE, f"Interrupted: {e.content}")
            else:
                # app.operator.submit("!EVENT_KEYWORD_DETECTED")
                log(f"Keyword detected directly: {e.content}.", level="INFO")
                emit_event(EventType.OP_INTENT, "greet")
                app.sm.set_state(State.AWAKE, f"Keyword: {e.content}")

        em.subscribe(EventType.KWS_KEYWORD_DETECTED, handle_kw_detected)

        em.subscribe(EventType.VAD_START, lambda e: app.sm.set_state(State.RECORDING))
        em.subscribe(EventType.VAD_END, lambda e: app.sm.set_state(State.AWAKE))

        em.subscribe(
            EventType.STT_TRANSCRIBED,
            lambda e: app.operator.submit(e.content),
        )

    def _close(self):
        try:
            log("Shutting down assistant...", "ATLAS", "INFO")

            self.keybinds.close()

            if getattr(self, "listener", None):
                self.listener.close()
                log("Listener closed.", "ATLAS", "DEBUG")

            if getattr(self, "operator", None):
                self.operator.close()
                log("Operator closed.", "ATLAS", "DEBUG")
            if getattr(self, "tts", None):
                self.tts.close()
                log("TTS closed.", "ATLAS", "DEBUG")

            if hasattr(self, "kws"):
                if hasattr(self.kws, "stream"):
                    del self.kws.stream
                if hasattr(self.kws, "kws"):
                    del self.kws.kws
            if hasattr(self, "sr"):
                self.sr.close()

            self._shutdown()
            self.events.flush_and_stop()
            log("Shutdown complete.", "ATLAS", "INFO")
        except Exception as e:  # noqa: BLE001
            log(f"Error during shutdown: {type(e).__name__}: {e}", "ATLAS", "ERROR")
        finally:
            import sys

            sys.stdout.write(  # Textual ui fix
                "\x1b[?1000l"
                "\x1b[?1003l"
                "\x1b[?1015l\x1b[?1006l"
                "\x1b[?25h"
                "\x1b[=0u"
                "\x1b[<u"
                "\x1b[>4m"
                "\x1b[?2004l"
            )
            sys.stdout.flush()

    def _main(self):
        self.load_models()

        self.keybinds.start()
        self.sr.start()
        self.listener.start()
        self.tts.start()
        self.operator.start()

        emit_event(EventType.UI_BANNER)

        self.ui = UI(app=self)
        self.ui.run()  # this blocks main thread

        # from threading import Event
        # while self.alive:
        #     Event().wait(1.0)

    def start(self):
        """Starts Atlas Assistant."""
        try:
            self._main()
        except Exception as e:  # noqa: BLE001
            print(f"[!] FATAL ERROR: {e}")
            sys.exit(1)
        finally:
            self._close()
