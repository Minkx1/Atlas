from pathlib import Path

from atlas.core.events import EventManager
from atlas.op.plugins import Plugin, PluginManifest


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


def test_plugin_say_message_emits_tts_command_and_ui_event(
    tmp_path: Path, event_manager: EventManager
):
    plugin = Plugin(tmp_path, PluginManifest(id="demo"), event_manager)
    received = []
    manager = event_manager
    manager.subscribe("tts.speak", received.append)
    manager.subscribe("ui.say", received.append)

    plugin._handle_line('{"type": "say", "text": "hello"}')
    manager._queue.join()

    assert [event.name for event in received] == ["tts.speak", "ui.say"]
    assert all(event.payload == {"text": "hello"} for event in received)
