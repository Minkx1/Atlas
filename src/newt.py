# newt.py

import sys
import threading

from .config import cfg
from .events import Event, EventManager
from .operator import Operator
from .profiler import profiler
from .speach_to_text import KeyWordSpotter, Listener, Whisper
from .ui import AssistantUI, console


class Newt:
    def __init__(self) -> None:
        self.events = EventManager.get_instace()

        # thrading.Event for blocking STT thread while LLM+TTS is working
        self.events.set_flag("stt_runtime", True)

        self.operator = Operator()
        match cfg.stt.pipeline_mode:
            case "KWS":
                self.kws = KeyWordSpotter()
                self.stt = Whisper()
                self.listener = Listener(self.stt, self.kws)
            case "DIRECT":
                stt = Whisper()
                self.listener = Listener(stt)

    def _parse_event(self, e: Event):
        match e.type:
            case "STT_TRANSCRIBE":
                # STT finished transribing text(e.content)
                self.operator.operate(e.content)
            case "STT_RESUME":
                self.events.set_flag("stt_runtime", True)
            case "STT_FINISH":
                ...
            # Profiler
            case "PROFILER_START":
                if cfg.profiler:
                    profiler.start()
            case "PROFILER_SET_STATE":
                profiler.set_state(e.content)
            case "PROFILER_FINISH":
                if cfg.profiler:
                    profiler.stop()
                    AssistantUI.print_benchmark_report(profiler.get_summary())

            # UI
            case "UI_BANNER":
                AssistantUI.print_banner()
            case "UI_STATE_CHANGE":
                AssistantUI.print_state_change(**e.content)
            case "UI_TRANSCRIPTION":
                AssistantUI.print_transcription(**e.content)
            case "UI_LLM_RESPONSE":
                AssistantUI.print_llm_response(**e.content)

            case _:
                pass

    def close(self):
        self.operator.close()
        self.listener.close()

    def _start_threads(self):
        stt_thread = threading.Thread(
            target=self.listener.start, name="STT_THREAD", daemon=True
        )
        stt_thread.start()

    def main(self):
        self._start_threads()

        while True:
            event = self.events.get_next_event()
            self._parse_event(event)
            self.events.queue.task_done()

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
