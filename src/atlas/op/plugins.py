#
# op / plugins.py
# Plugin loading and runtime system
#

import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from atlas.core.events import CommandType, EventManager, EventType

log = logging.getLogger(__name__)


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
            timeout=execution.get("timeout", 0.0),
            exec_type=execution.get("type", "binary"),
            exec_file=execution.get("file", ""),
        )


class Plugin:
    def __init__(
        self, root: Path, manifest: PluginManifest, events: EventManager | None = None
    ):
        self.events = events or EventManager()

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
                    level = getattr(log, msg.get("level", "INFO").lower(), log.info)
                    level("%s", msg.get("message", ""))
                    continue
            except json.JSONDecodeError:
                log.debug("[%s] %s", self.manifest.id, raw)

    def run(self, origin: str) -> bool:
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
        except OSError:
            log.exception("Unable to run plugin '%s'", self.manifest.id)
            return False

        # giving context
        try:
            proc.stdin.write(json.dumps({"origin": origin}) + "\n")  # type: ignore
            proc.stdin.close()  # type: ignore
        except (BrokenPipeError, OSError):
            log.warning(
                "Plugin '%s' closed stdin before receiving origin", self.manifest.id
            )

        threading.Thread(
            target=self._pump_stderr, args=(proc.stderr,), daemon=True
        ).start()

        timer = None
        if self.manifest.timeout > 0:
            timer = threading.Timer(self.manifest.timeout, proc.kill)
            timer.start()
        timed_out = False
        try:
            for line in proc.stdout:  # type: ignore
                self._handle_line(line)
        finally:
            if timer:
                timed_out = proc.poll() is None and timer.is_alive() is False
                timer.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1)

        if timed_out:
            log.error(
                "Plugin '%s' exceeded timeout of %.1fs",
                self.manifest.id,
                self.manifest.timeout,
            )
        elif proc.returncode:
            log.error(
                "Plugin '%s' exited with status %s",
                self.manifest.id,
                proc.returncode,
            )
        return not timed_out and proc.returncode == 0

    def _handle_line(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log.warning(
                "Plugin '%s' submitted invalid JSON line: %r",
                self.manifest.id,
                line,
            )
            return

        match msg.get("type"):
            case "say":
                # mimics originally-designed event to call `tts.speak(...)`
                self.events.emit_command(
                    CommandType.TTS_SPEAK, {"text": msg.get("text", "")}
                )
                self.events.emit(
                    EventType.UI_ASSISTANT_SAY, {"text": msg.get("text", "")}
                )
            case "event":
                self._forward_event(msg)
            case "done":
                pass
            case other:
                log.warning(
                    "Unknown message type from plugin '%s': %s",
                    self.manifest.id,
                    other,
                )

    def _forward_event(self, msg: dict):
        name = msg.get("name")
        try:
            self.events.emit(EventType(name), msg.get("content") or {})
        except ValueError:
            log.warning("Plugin '%s' emitted unknown event: %s", self.manifest.id, name)
