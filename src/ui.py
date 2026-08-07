# ui.py
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

if __name__ == "__main__":
    MAIN = True
    from config import cfg
    from utils import __version__
else:
    MAIN = False
    from .config import cfg
    from .utils import __version__

console = Console()


class AssistantUI:
    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005

    @staticmethod
    def _format_rtf(rtf: float) -> str:
        """Returns RTF efficiency color (lesser - faster)."""
        if rtf < 0.5:
            return f"[bold green]{rtf:.2f}x (Fast)[/bold green]"
        if rtf < 1.2:
            return f"[bold yellow]{rtf:.2f}x (Normal)[/bold yellow]"
        return f"[bold red]{rtf:.2f}x (Slow)[/bold red]"

    @staticmethod
    def print_banner():
        banner_text = (
            f"[bold cyan]🦎 [bright_green]NEWT[/bright_green] Voice Assistant[/bold cyan] "
            f"[dim]• v{__version__}[/dim]\n"
            f"[dim]Waiting for [/dim][white]'{cfg.name}'[/white][dim]... "
            f"Press [bold]Ctrl+C[/bold] to exit.[/dim]"
        )
        if cfg.profiler:
            banner_text += "\n[bold yellow]⚡ PROFILER MODE ENABLED[/bold yellow]"

        console.print(Panel.fit(banner_text, border_style="cyan"))

    @staticmethod
    def print_state_change(state: str, detail: str = ""):
        if not cfg.profiler:
            return

        time_str = AssistantUI._get_timestamp()
        if state == "AWAKE":
            console.print(
                f"[dim][{time_str}][/dim] [bold green]✦ AWAKE[/bold green]    [dim]{detail}[/dim]"
            )
        elif state == "RECORDING":
            console.print(
                f"[dim][{time_str}][/dim] [bold yellow]● RECORDING[/bold yellow] [dim]Listening to command...[/dim]"
            )
        elif state == "SLEEPING":
            console.print(
                f"[dim][{time_str}][/dim] [dim blue]zZz SLEEPING[/dim blue] [dim]{detail}[/dim]"
            )

    @staticmethod
    def print_transcription(
        text: str, listen_ms: float = 0.0, recog_ms: float = 0.0, rtf: float = 0.0
    ):
        if cfg.profiler:
            console.print(
                f"\n[bold magenta]╭─ SpeechToText[/bold magenta] [dim]({AssistantUI._get_timestamp()})[/dim]"
            )
            console.print(
                f'[bold magenta]├─[/bold magenta] [bold white]"{text}"[/bold white]'
            )
            console.print(
                f"[bold magenta]╰─[/bold magenta] [dim]Audio: [white]{listen_ms:.0f} ms[/white] │ "
                f"STT Latency: [white]{recog_ms:.0f} ms[/white] │ "
                f"RTF:[/dim] {AssistantUI._format_rtf(rtf)}"
            )
        else:
            console.print(f'\n[bold cyan][User]:[/bold cyan] [white]"{text}"[/white]')

    @staticmethod
    def print_llm_chunk(text: str, is_first: bool = False):
        if not text:
            return

        if cfg.profiler:
            if is_first:
                console.print(
                    f"\n[bold yellow]╭─ LLM[/bold yellow] [dim]({AssistantUI._get_timestamp()})[/dim]"
                )
                console.print(
                    '[bold yellow]├─[/bold yellow] [bold white]"[/bold white]', end=""
                )
            console.print(text, end=" ", style="bold white")
        else:
            if is_first:
                console.print(
                    f"\n[bold bright_green][{cfg.name}]:[/bold bright_green] ",
                    end="",
                )
            console.print(text, end=" ", style="white")

    @staticmethod
    def print_llm_response(
        text=None,
        gen_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        if cfg.profiler:
            console.print('"', style="bold white")

            total_tokens = prompt_tokens + completion_tokens
            speed = (
                (completion_tokens / (gen_ms / 1000.0))
                if gen_ms > 0 and completion_tokens > 0
                else 0.0
            )

            if total_tokens > 0:
                console.print(
                    f"[bold yellow]├─[/bold yellow] [dim]Tokens: [white]{prompt_tokens}[/white] prompt + "
                    f"[white]{completion_tokens}[/white] gen = [bold white]{total_tokens}[/bold white] total[/dim]"
                )

            console.print(
                f"[bold yellow]╰─[/bold yellow] [dim]Gen Time: [white]{gen_ms:.0f} ms[/white]"
                + (
                    f" │ Speed: [bold green]{speed:.1f} tok/s[/bold green][/dim]"
                    if speed > 0
                    else "[/dim]"
                )
            )
        else:
            console.print()

    @staticmethod
    def print_tts_status(syn_ms: float = 0.0, audio_ms: float = 0.0, rtf: float = 0.0):
        if not cfg.profiler:
            return

        console.print("[bold cyan]╭─ TextToSpeech[/bold cyan]")
        console.print(
            f"[bold cyan]╰─[/bold cyan] [dim]Synth Time: [white]{syn_ms:.0f} ms[/white] │ "
            f"Audio Duration: [white]{audio_ms:.0f} ms[/white] │ "
            f"RTF:[/dim] {AssistantUI._format_rtf(rtf)}"
        )

    @staticmethod
    def print_benchmark_report(summary: dict[str, dict[str, float]]):
        if not cfg.profiler:
            return

        from rich.table import Table

        table = Table(
            title="\n[bold cyan]─── Resource Profiler Summary ───[/bold cyan]",
            header_style="bold white",
        )
        table.add_column("State", style="bold")
        table.add_column("Time (s)", justify="right")
        table.add_column("Avg CPU (%)", justify="right", style="yellow")
        table.add_column("Peak RAM (MB)", justify="right", style="green")

        for state, stats in summary.items():
            table.add_row(
                state,
                f"{stats['duration_sec']:.1f}s",
                f"{stats['avg_cpu']}%",
                f"{stats['peak_ram_mb']} MB",
            )

        console.print(table)
        console.print()

    @staticmethod
    def print_error(msg: str):
        console.print(f"[bold red][!] ERROR:[/bold red] {msg}")
