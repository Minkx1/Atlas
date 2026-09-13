import json

import pytest


def require_importable(module_name: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} is not usable: {exc}")


def test_op_load_commands_generates_example_on_fresh_install(tmp_path, monkeypatch):
    """Regression test: cmd_operator's bundled example file used to be named
    'commands_example.json' in code but didn't exist on disk under
    modules/op/, so a fresh install (no config/commands.json yet) crashed
    with FileNotFoundError the first time Atlas started.
    """
    require_importable("onnxruntime")
    from atlas.modules.op import cmd_operator
    from atlas.modules.op.cmd_operator import CommandOperator

    monkeypatch.setattr(cmd_operator, "CONFIG_DIR", tmp_path)

    commands = CommandOperator.load_commands()

    assert (tmp_path / "commands.json").exists()
    assert "greet" in commands
    assert "triggers" in commands["greet"]
    assert "sounds" in commands["greet"]


def test_tts_load_commands_generates_example_on_fresh_install(tmp_path, monkeypatch):
    """Same regression as above, for sound_manager's copy of the loader."""
    require_importable("sounddevice")
    from atlas.modules.tts import sound_manager
    from atlas.modules.tts.sound_manager import SoundManager

    monkeypatch.setattr(sound_manager, "CONFIG_DIR", tmp_path)

    commands = SoundManager.load_commands()

    assert (tmp_path / "commands.json").exists()
    assert "greet" in commands


def test_load_commands_filters_invalid_entries(tmp_path, monkeypatch):
    require_importable("onnxruntime")
    from atlas.modules.op import cmd_operator
    from atlas.modules.op.cmd_operator import CommandOperator

    monkeypatch.setattr(cmd_operator, "CONFIG_DIR", tmp_path)
    (tmp_path / "commands.json").write_text(
        json.dumps(
            {"greet": {"triggers": ["hello"], "sounds": []}, "invalid": "ignored"}
        ),
        encoding="utf-8",
    )

    commands = CommandOperator.load_commands()

    assert commands == {"greet": {"triggers": ["hello"], "sounds": []}}
