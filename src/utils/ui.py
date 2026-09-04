# ui.py
import time
from datetime import datetime

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.events import Resize
from textual.reactive import reactive
from textual.widgets import Input, Label, RichLog, Static

from ..core.config import cfg
from ..core.events import Event, EventManager, EventType, emit_event, log


class AudioWaveform(Static):
    is_listening = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wave_history: list[int] = [0] * 10
        self._current_level: int = 0

        if cfg.stt.start_state in {"SLEEPING", "WAITING"}:
            self.is_listening = False
        else:
            self.is_listening = True

    def on_mount(self) -> None:
        self.set_interval(0.07, self.update_waveform)

    def push_volume(self, volume: float) -> None:
        if not self.is_listening:
            return

        if isinstance(volume, float) and volume <= 1.0:
            val = int(np.clip(volume, 0.0, 1.0) * 32)
        else:
            val = int(np.clip(volume, 0, 32))

        self._current_level = max(self._current_level, val)

    def update_waveform(self) -> None:
        width = self.size.width
        if width <= 1:
            return

        if not self.is_listening:
            self._current_level = 0

        history_len = (width + 1) // 2

        while len(self.wave_history) < history_len:
            self.wave_history.append(0)
        while len(self.wave_history) > history_len:
            self.wave_history.pop()

        new_val = self._current_level
        self._current_level = int(self._current_level * 0.55)

        self.wave_history.insert(0, new_val)
        self.wave_history.pop()

        if width % 2 == 0:
            left_part = list(reversed(self.wave_history))
        else:
            left_part = list(reversed(self.wave_history))[0:-1]

        full_history = left_part + self.wave_history

        smoothed = [full_history[0]]
        if width > 2:
            for i in range(1, width - 1):
                avg = (full_history[i - 1] + full_history[i] + full_history[i + 1]) // 3
                smoothed.append(avg)
            smoothed.append(full_history[-1])
        else:
            smoothed = full_history

        blocks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        lines = ["", "", "", ""]
        for h in smoothed:
            h = max(0, min(32, h))
            for level in range(4):
                level_val = h - (level * 8)
                if level_val >= 8:
                    char = "█"
                elif level_val <= 0:
                    # '.' for level 0 instead of ' '
                    char = "." if level == 0 else " "
                else:
                    char = blocks[level_val]

                lines[3 - level] += char

        wave_text = (
            f"[#00d7ff]{lines[0]}[/#00d7ff]\n"
            f"[#00afff]{lines[1]}[/#00afff]\n"
            f"[#0087ff]{lines[2]}[/#0087ff]\n"
            f"[#005fdf]{lines[3]}[/#005fdf]"
        )
        self.update(wave_text)


tcss = """
Screen {
    background: #000000;
}

#main-app {
    layout: grid;
    grid-size: 3;
    grid-columns: 1fr 2fr 1fr;
    height: 100%;
}

.small-size #main-app {
    display: none;
}

#size-warning {
    display: none;
    content-align: center middle;
    text-align: center;
    width: 100%;
    height: 100%;
}

.small-size #size-warning {
    display: block;
}

#left-panel, #right-panel, #central-panel {
    height: 100%;
    border: round #1e3668;
    border-title-align: center;
    border-title-color: #00d7ff;
    border-title-style: bold;
    background: transparent;
}

#central-panel {
    align: center middle;
    padding: 1;
    border: round #244687;
}

RichLog {
    height: 1fr;
    background: transparent;
    padding: 0 1;
    scrollbar-background: transparent;
    scrollbar-color: #1e3668;
    scrollbar-size: 1 1;
}

#dialog {
    height: 1fr;
    background: transparent;
    padding: 0 1;
    scrollbar-background: transparent;
    scrollbar-color: #1e3668;
    scrollbar-size: 1 1;
    overflow-x: hidden; /* Забороняємо горизонтальний скролл */
    overflow-y: auto;
}

.chat-message {
    width: 100%;
    height: auto;
}

Input {
    dock: bottom;
    border: none;
    border-top: solid #1e3668;
    background: transparent;
    padding: 0 1;
    width: 100%;
}

Input:focus {
    border-top: solid #00d7ff;
}

#image-box {
    content-align: center middle;
    text-align: center;
    width: 100%;
    height: auto;
    margin-bottom: 2;
}

#status-text {
    content-align: center middle;
    text-align: center;
    width: 100%;
    height: 1;
    margin-bottom: 2;
}

AudioWaveform {
    height: 4;
    content-align: center middle;
    text-align: center;
    width: 100%;
    overflow: hidden;
}
"""


class UI(App):
    TITLE = "Atlas"
    CSS = tcss

    def __init__(self, app=None, **kwargs):
        super().__init__(**kwargs)
        self.atlas = app

    def compose(self) -> ComposeResult:
        with Container(id="main-app"):
            with Container(id="left-panel") as left:
                left.border_title = "LOGS"
                yield RichLog(id="event-log", highlight=True, markup=True, wrap=True)

            with Container(id="central-panel") as center:
                center.border_title = "STATUS"
                logo = r"""[#00d7ff]
    ___    __  __           
   /   |  / / / /___ ______ 
  / /| | / __/ / __ `/ ___/ 
 / ___ |/ /_/ / /_/ (__  )  
/_/  |_|\__/_/\__,_/____/   
[/#00d7ff]"""
                yield Static(logo, id="image-box")
                yield Static("[dim #a0a0a0]SLEEPING[/dim #a0a0a0]", id="status-text")
                yield AudioWaveform(id="audio-waveform")

            with Container(id="right-panel") as right:
                self.right_panel = right
                right.border_title = "DIALOG"
                right.border_subtitle = ""

                yield VerticalScroll(id="dialog")
                yield Input(placeholder="> _", id="command-input")

        yield Static("", id="size-warning")

    def on_mount(self) -> None:
        self.dialog = self.query_one("#dialog", VerticalScroll)
        self.audiowave = self.query_one("#audio-waveform", AudioWaveform)
        self.status_text = self.query_one("#status-text", Static)
        self.event_log = self.query_one("#event-log", RichLog)

        self._current_assistant_label: Label | None = None
        self._current_assistant_text = ""

        self.event_log.write("[dim #a0a0a0]Loading modules...[/dim #a0a0a0]")
        self.set_interval(1.0, self.update_clock)
        self.update_clock()

        em = EventManager()
        em.subscribe(EventType.STT_CHANGED_STATE, self.event_stt_changed_state)
        em.subscribe(EventType.STT_AUDIOWAVE, self.on_audio_wave)
        em.subscribe(EventType.STT_TRANSCRIBED, self.event_on_received_command)

        em.subscribe(EventType.DEBUG_LOG, self.event_on_debug_log)
        em.subscribe(EventType.UI_LLM_CHUNK, self.event_on_llm_chunk)
        em.subscribe(EventType.UI_ASSISTANT_SAY, self.event_on_assistant_say)

    def safe_call(self, fn, *args, **kwargs):
        if getattr(self, "is_running", False):
            try:
                self.call_from_thread(fn, *args, **kwargs)
            except Exception as e:
                log(f"UI thread safe call error: {e}.", "UI", "ERROR")
                raise

    def event_stt_changed_state(self, event: Event):
        def _():
            state_str = event.payload.get("state", "")

            if state_str in {"SLEEPING", "WAITING"}:
                self.audiowave.is_listening = False
            else:
                self.audiowave.is_listening = True

            if state_str == "SLEEPING":
                formatted_state = "[dim #a0a0a0]SLEEPING[/dim #a0a0a0]"
            elif state_str == "AWAKE":
                formatted_state = "[bold #00d7ff]AWAKE[/bold #00d7ff]"
            elif state_str == "RECORDING":
                formatted_state = "[bold #00ff5f]RECORDING[/bold #00ff5f]"
            elif state_str == "WAITING":
                formatted_state = "[bold #ffaf00]WAITING...[/bold #ffaf00]"
            else:
                formatted_state = f"[#00d7ff]{state_str}[/#00d7ff]"

            self.status_text.update(formatted_state)

        self.safe_call(_)

    def on_audio_wave(self, event: Event):
        wave_data = event.payload.get("rms", 0.0)
        self.safe_call(self.audiowave.push_volume, wave_data)

    def event_on_debug_log(self, event: Event):
        """This method is called from EVENT_DISPATCHER thread"""
        level = event.payload.get("level", "INFO").upper()
        source = event.payload.get("source", "SYS")
        message = event.payload.get("message", "")

        timestamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))

        # Colors settings
        color_map = {
            "INFO": "green",
            "DEBUG": "dim #a0a0a0",
            "WARNING": "yellow",
            "ERROR": "bold red",
            "SUCCESS": "bold cyan",
        }
        color = color_map.get(level, "white")

        formatted_msg = f"[[#00d7ff]{timestamp}[/#00d7ff]] [[bold]{source}[/bold]]"
        f"[{color}][{level}][/{color}]: {message}"

        # transport writing to Textual main thread
        self.safe_call(self.event_log.write, formatted_msg)

    def event_on_llm_chunk(self, event: Event):
        def f():
            chunk_text = event.payload["text"]
            is_first = event.payload["is_first"]

            if is_first:
                self._current_assistant_text = rf": {chunk_text}"
                self._current_assistant_label = Label(
                    self._current_assistant_text, classes="chat-message"
                )
                self.dialog.mount(self._current_assistant_label)
                self._current_assistant_label.scroll_visible()
            else:
                self._current_assistant_text += chunk_text
                if self._current_assistant_label:
                    self._current_assistant_label.update(self._current_assistant_text)
                    self.dialog.scroll_end(animate=False)

        self.safe_call(f)

    def event_on_assistant_say(self, event: Event):
        def f():
            text = rf": {event.payload['text']}"
            msg_label = Label(text, classes="chat-message")
            self.dialog.mount(msg_label)
            msg_label.scroll_visible()

        self.safe_call(f)

    def event_on_received_command(self, event: Event):
        def f():
            user_text = rf"> [#00d7ff]{event.payload['text']}[/#00d7ff]"

            msg_label = Label(user_text, classes="chat-message")
            self.dialog.mount(msg_label)
            msg_label.scroll_visible()  # auto-scroll

        self.safe_call(f)

    def on_resize(self, event: Resize) -> None:
        req_w, req_h = 80, 24
        if event.size.width < req_w or event.size.height < req_h:
            self.add_class("small-size")
            warning = self.query_one("#size-warning", Static)
            warning.update(
                f"\n\n\n\n\n[bold red]Terminal size too small:[/bold red]\n"
                f"Width = {event.size.width} Height = {event.size.height}\n\n"
                f"[bold #00d7ff]Needed for current config:[/bold #00d7ff]\n"
                f"Width = {req_w} Height = {req_h}"
            )
        else:
            self.remove_class("small-size")

    def update_clock(self) -> None:
        now = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self.right_panel.border_subtitle = f"[#00d7ff]{now}[/#00d7ff]"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.input.value = ""
        emit_event(EventType.STT_TRANSCRIBED, {"text": event.value})


if __name__ == "__main__":
    UI().run()
