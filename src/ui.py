# ui.py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class AssistantUI:
    @staticmethod
    def print_banner():
        console.print(
            Panel.fit(
                "[bold cyan]Newt Voice Assistant[/bold cyan]\n"
                "[dim]Listening for wake word... Press Ctrl+C to exit.[/dim]",
                border_style="cyan",
            )
        )

    @staticmethod
    def print_state_change(state: str, detail: str = ""):
        if state == "AWAKE":
            console.print(f"[bold green][✦] AWAKE[/bold green] | [dim]{detail}[/dim]")
        elif state == "RECORDING":
            console.print(
                "[bold yellow][●] RECORDING[/bold yellow] | [dim]Listening to command...[/dim]"
            )
        elif state == "SLEEPING":
            console.print(f"[dim blue][zZz] SLEEPING[/dim blue] | [dim]{detail}[/dim]")

    @staticmethod
    def print_transcription(text: str, listen_ms: float, recog_ms: float, rtf: float):
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            width=75,
        )
        table.add_column("Recognized Command", style="bold white", ratio=3)
        table.add_column("Audio", justify="right", style="dim", ratio=1)
        table.add_column("STT Time", justify="right", style="dim", ratio=1)
        table.add_column("RTF", justify="right", style="bold green", ratio=1)

        table.add_row(
            f'"{text}"',
            f"{listen_ms:.0f} ms",
            f"{recog_ms:.0f} ms",
            f"{rtf:.2f}x",
        )
        console.print(table)
        console.print()

    @staticmethod
    def print_error(msg: str):
        console.print(f"[bold red]ERROR:[/bold red] {msg}")
