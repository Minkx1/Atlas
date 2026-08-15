import random
import time
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.events import Resize
from textual.reactive import reactive
from textual.widgets import Input, RichLog, Static

if __name__ == "__main__":
    from events import Event, EventManager, EventType, emit_event
else:
    from .events import Event, EventManager, EventType, emit_event


def _get_wave() -> float:
    return random.random()

class AudioWaveform(Static):
    is_listening = reactive(True)
    
    wave_history = [0] * 10

    def on_mount(self) -> None:
        self.set_interval(0.07, self.update_waveform)

    def update_waveform(self) -> None:
        width = self.size.width
        if width <= 1:
            return  

        if not self.is_listening:
            self.update("\n\n[dim]# Microphone Offline #[/dim]\n")
            return

        history_len = (width + 1) // 2

        while len(self.wave_history) < history_len:
            self.wave_history.append(0)
        while len(self.wave_history) > history_len:
            self.wave_history.pop()

        # chance = random.random()

        chance = _get_wave()

        if chance > 0.8:
            new_val = random.randint(20, 32)
        elif chance > 0.4:
            new_val = random.randint(8, 20)
        else:
            new_val = random.randint(0, 7)

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
                avg = (full_history[i-1] + full_history[i] + full_history[i+1]) // 3
                smoothed.append(avg)
            smoothed.append(full_history[-1])
        else:
            smoothed = full_history

        blocks = [' ', ' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        lines = ["", "", "", ""]
        for h in smoothed:
            h = max(0, min(32, h))
            for level in range(4):
                level_val = h - (level * 8)
                if level_val >= 8:
                    char = '█'
                elif level_val <= 0:
                    char = ' '
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
    margin-bottom: 4; 
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
    TITLE = "Newt"
    CSS = tcss

    def __init__(self, newt_app=None, **kwargs):
        super().__init__(**kwargs)
        self.newt = newt_app

    def compose(self) -> ComposeResult:
        with Container(id="main-app"):
            with Container(id="left-panel") as left:
                left.border_title = "LOGS"
                yield RichLog(id="event-log", highlight=True, markup=True)

            with Container(id="central-panel") as center:
                center.border_title = "NEWT STATUS"
                logo = r"""[#00d7ff]
 _   _               _   
| \ | | _____      _| |_ 
|  \| |/ _ \ \ /\ / / __|
| |\  |  __/\ V  V /| |_ 
|_| \_|\___| \_/\_/  \__|
[/#00d7ff]"""
                yield Static(logo, id="image-box")
                yield AudioWaveform()
                
            with Container(id="right-panel") as right:
                self.right_panel = right
                right.border_title = "DIALOG"
                right.border_subtitle = "" 
                
                yield RichLog(id="dialog", highlight=True, markup=True)
                yield Input(placeholder="> _", id="command-input")
        
        yield Static("", id="size-warning")

    def on_mount(self) -> None:
        self.dialog_log = self.query_one("#dialog", RichLog)
        self.dialog_log.write("[bold #00ff00]System Initialized![/bold #00ff00]") 
        self.dialog_log.write("[#a0a0a0]Waiting...[/#a0a0a0]")

        self.event_log = self.query_one("#event-log", RichLog)
        self.event_log.write("[dim #a0a0a0]Loading modules...[/dim #a0a0a0]")
        
        self.set_interval(1.0, self.update_clock)
        self.update_clock() 

        EventManager().subscribe(EventType.DEBUG_LOG, self.on_debug_log)

    def on_debug_log(self, event: Event):
        """This method is called from EVENT_DISPATCHER thread"""
        if not isinstance(event.content, dict):
            return

        level = event.content.get("level", "INFO").upper()
        source = event.content.get("source", "SYS")
        message = event.content.get("message", "")
        
        timestamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))

        # Налаштування кольорів для RichLog
        color_map = {
            "INFO": "green",
            "DEBUG": "dim #a0a0a0",
            "WARNING": "yellow",
            "ERROR": "bold red",
            "SUCCESS": "bold cyan"
        }
        color = color_map.get(level, "white")

        formatted_msg = f"[[#00d7ff]{timestamp}[/#00d7ff]] [[bold]{source}[/bold]] [{color}][{level}][/{color}]: {message}"

        # transport writing to Textual main thread
        self.call_from_thread(self.event_log.write, formatted_msg)

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
        self.dialog_log.write(f"[User]: [#00d7ff]{event.value}[/#00d7ff]")
        event.input.value = ""
        self.dialog_log.write("[System]: [dim]NO RESPONSE.[/dim]")
        # TODO: ADD 'OP_RECEIVE_CMD' event emitment and proper flushing() llm chunks or response from OP.

if __name__ == "__main__":
    UI().run()