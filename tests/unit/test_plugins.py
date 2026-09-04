from pathlib import Path

from src.core.events import CommandType, EventManager, EventType
from src.op.plugins import Plugin, PluginManifest


def test_plugin_manifest_reads_toml(tmp_path: Path):
    manifest_path = tmp_path / "plugin.toml"
    manifest_path.write_text(
        """
[plugin]
id = "demo"
description = "Demo plugin"
triggers = ["run demo"]

[execution]
type = "python"
file = "main.py"
timeout = 3.5
""",
        encoding="utf-8",
    )

    manifest = PluginManifest.from_toml(manifest_path)

    assert manifest == PluginManifest(
        id="demo",
        description="Demo plugin",
        triggers=["run demo"],
        exec_type="python",
        exec_file="main.py",
        timeout=3.5,
    )


def test_plugin_say_message_emits_tts_command_and_ui_event(tmp_path: Path):
    plugin = Plugin(tmp_path, PluginManifest(id="demo"))
    received = []
    manager = EventManager()
    manager.subscribe(CommandType.TTS_SPEAK, received.append)
    manager.subscribe(EventType.UI_ASSISTANT_SAY, received.append)

    plugin._handle_line('{"type": "say", "text": "hello"}')
    manager.queue.join()

    assert [event.name for event in received] == [
        CommandType.TTS_SPEAK.value,
        EventType.UI_ASSISTANT_SAY.value,
    ]
    assert all(event.payload == {"text": "hello"} for event in received)
