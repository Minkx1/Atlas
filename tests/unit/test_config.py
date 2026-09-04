from pathlib import Path

from src.core import config


def test_load_config_reads_nested_values(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
name = "Test Atlas"
username = "tester"
log = true

[audio]
sample_rate = 8000
channels = 2

[stt]
start_state = "SLEEPING"
""",
        encoding="utf-8",
    )

    loaded = config.load_config(str(config_path))

    assert loaded.name == "Test Atlas"
    assert loaded.username == "tester"
    assert loaded.log is True
    assert loaded.audio.sample_rate == 8000
    assert loaded.audio.channels == 2
    assert loaded.stt.start_state == "SLEEPING"


def test_load_commands_filters_invalid_entries(monkeypatch, tmp_path: Path):
    commands_path = tmp_path / "commands.json"
    commands_path.write_text(
        '{"greet": {"triggers": ["hello"], "sounds": []}, "invalid": "ignored"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    commands = config.OPConfig(commands="commands.json").load_commands()

    assert commands == {"greet": {"triggers": ["hello"], "sounds": []}}
