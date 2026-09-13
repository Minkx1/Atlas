import pytest

from atlas.utils import config as config_module
from atlas.utils.config import Config


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    return tmp_path


def test_load_config_creates_file_from_origin_when_missing(isolated_config_dir):
    origin = '[app]\nname = "Test"\n'

    loaded = Config.load_config("app.toml", origin)

    created = isolated_config_dir / "app.toml"
    assert created.read_text(encoding="utf-8") == origin
    assert loaded == {"app": {"name": "Test"}}


def test_load_config_does_not_overwrite_existing_file(isolated_config_dir):
    existing = isolated_config_dir / "app.toml"
    existing.write_text('[app]\nname = "Existing"\n', encoding="utf-8")

    loaded = Config.load_config("app.toml", '[app]\nname = "Origin"\n')

    assert loaded == {"app": {"name": "Existing"}}


def test_load_config_reads_json(isolated_config_dir):
    (isolated_config_dir / "data.json").write_text(
        '{"greet": {"triggers": ["hello"]}}', encoding="utf-8"
    )

    assert Config.load_config("data.json") == {"greet": {"triggers": ["hello"]}}


@pytest.mark.parametrize("suffix", [".txt", ".cfg"])
def test_load_config_reads_plain_text_as_content(isolated_config_dir, suffix):
    (isolated_config_dir / f"keybinds{suffix}").write_text(
        "<ctrl>+<alt>+w", encoding="utf-8"
    )

    loaded = Config.load_config(f"keybinds{suffix}")

    assert loaded == {"content": "<ctrl>+<alt>+w"}


def test_load_config_unsupported_extension_returns_empty(isolated_config_dir):
    (isolated_config_dir / "unknown.yaml").write_text("a: 1", encoding="utf-8")

    assert Config.load_config("unknown.yaml") == {}


def test_load_config_null_id_returns_empty_without_touching_disk(isolated_config_dir):
    assert Config.load_config() == {}
    assert list(isolated_config_dir.iterdir()) == []


def test_write_from_eample_copies_content_and_returns_it(tmp_path):
    example = tmp_path / "example.toml"
    example.write_text('[op]\nname = "Atlas"\n', encoding="utf-8")
    target = tmp_path / "op.toml"

    result = Config.write_from_eample(target, example)

    assert target.read_text(encoding="utf-8") == example.read_text(encoding="utf-8")
    assert result == example.read_text(encoding="utf-8")
