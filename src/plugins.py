import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from .events import EventType, emit_event, log


@dataclass
class PluginManifest:
    """General Plugin template"""

    id: str
    description: str = "Unknown."
    exec_type: str = "binary"
    exec_file: str = ""
    timeout: float = 0.0
    triggers: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: Path) -> "PluginManifest":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        plugin = data.get("plugin", {})
        execution = data.get("execution", {})
        return cls(
            id=plugin["id"],
            description=plugin.get("description", "Unknown."),
            triggers=plugin.get("triggers", []),
            timeout=plugin.get("timeout", 0.0),
            exec_type=execution.get("type", "binary"),
            exec_file=execution.get("file", ""),
        )


class Plugin:
    def __init__(self, root: Path, manifest: PluginManifest):
        self.root = root
        self.manifest = manifest

    def _build_command(self) -> list[str]:
        """Gets CMD for subprocess Popen."""
        exe = self.root / self.manifest.exec_file
        if self.manifest.exec_type == "python":
            import sys

            return [sys.executable, str(exe)]
        return [str(exe)]  # bin / shebang-script

    def _pump_stderr(self, stderr):
        """logs all errors/logs from stderr"""
        for raw in stderr:
            raw: str = raw.rstrip("\n")
            if not raw:
                continue
            try:
                msg = json.loads(raw)
                if msg.get("type") == "log":
                    log(
                        msg.get("message", ""),
                        msg.get("source", self.manifest.id),
                        msg.get("level", "INFO"),
                    )
                    continue
            except json.JSONDecodeError:
                log(raw, self.manifest.id, "DEBUG")

    def run(self, origin: str) -> None:
        # !NOTE
        # ALWAYS call from separate PLUGIN-THREAD!!!

        # creating process
        try:
            proc = subprocess.Popen(
                self._build_command(),
                cwd=self.root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            log(
                f"Unable to run plugin '{self.manifest.id}': {e}",
                "PLUGIN",
                "ERROR",
            )
            return

        # giving context
        try:
            proc.stdin.write(json.dumps({"origin": origin}) + "\n")  # type: ignore
            proc.stdin.close()  # type: ignore
        except (BrokenPipeError, OSError):
            pass

        threading.Thread(
            target=self._pump_stderr, args=(proc.stderr,), daemon=True
        ).start()

        timer = None
        if self.manifest.timeout > 0:
            timer = threading.Timer(self.manifest.timeout, proc.kill)
            timer.start()
        try:
            for line in proc.stdout:  # type: ignore
                self._handle_line(line)
        finally:
            if timer:
                timer.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1)

    def _handle_line(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log(
                f"Plugin '{self.manifest.id}' submited invalid line: {line!r}",
                "PLUGIN",
                "WARN",
            )
            return

        match msg.get("type"):
            case "say":
                emit_event(EventType.TTS_SPEAK, msg.get("text", ""))
                emit_event(EventType.UI_ASSISTANT_SAY, {"text": msg.get("text", "")})
            case "event":
                self._forward_event(msg)
            case "done":
                pass
            case other:
                log(
                    f"Unknown message type '{self.manifest.id}': {other}",
                    "PLUGIN",
                    "WARN",
                )

    def _forward_event(self, msg: dict):
        name = msg.get("name")
        try:
            emit_event(EventType(name), msg.get("content"))
        except ValueError:
            log(
                f"Plugin '{self.manifest.id}' emitted uknown event: {name}",
                "PLUGIN",
                "WARN",
            )
