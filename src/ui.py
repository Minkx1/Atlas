# ui.py
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from .config import cfg
from .profiler import ResourceProfiler
from .utils import __version__

console = Console()


class AssistantUI:
    @staticmethod
    def _resource_badge() -> str:
        if not cfg.profiler_debug:
            return ""
        cpu, ram = ResourceProfiler.get_instant_stats()
        return f"[dim]RAM: {ram:.0f}MB │ CPU: {cpu:.1f}%[/dim]"

    @staticmethod
    def print_banner():
        console.print(
            Panel.fit(
                f"[bold cyan]🦎 Newt Voice Assistant[/bold cyan] [dim]• v{__version__}[/dim]\n"
                "[dim]Listening for keywords... Press [bold]Ctrl+C[/bold] to exit.[/dim]",
                border_style="cyan",
            )
        )

    @staticmethod
    def print_state_change(state: str, detail: str = ""):
        time_str = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
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
    def print_transcription(text: str, listen_ms: float, recog_ms: float, rtf: float):
        console.print(
            f'[bold magenta]╭─ [/bold magenta] [bold white]"{text}"[/bold white]'
        )
        console.print(
            f"[bold magenta]╰─ [/bold magenta] [dim]{listen_ms:.0f} ms audio  │  "
            f"{recog_ms:.0f} ms STT  │  RTF:[/dim] [bold green]{rtf:.2f}x[/bold green]"
        )
        console.print()

    @staticmethod
    def print_benchmark_report(summary: dict[str, dict[str, float]]):
        if not cfg.profiler_debug:
            return
        console.print("\n[bold cyan]─── Resource  Summary ───[/bold cyan]")
        for state, stats in summary.items():
            console.print(
                f" • [bold]{state:<10}[/bold] -> "
                f"Avg CPU: [yellow]{stats['avg_cpu']}%[/yellow] │ "
                f"Peak RAM: [green]{stats['peak_ram_mb']} MB[/green]"
            )
        console.print()

    @staticmethod
    def print_error(msg: str):
        console.print(f"[bold red]✖ ERROR:[/bold red] {msg}")
