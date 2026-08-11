import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import llama_cpp  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - exercised in minimal environments
    fake_llama_cpp = types.ModuleType("llama_cpp")

    class _FakeLlama:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            return None

    fake_llama_cpp.Llama = _FakeLlama
    sys.modules.setdefault("llama_cpp", fake_llama_cpp)

from src import config as config_module
from src.config import load_config
from src.operator import CommandOperator


class TodoFixesTests(unittest.TestCase):
    def test_load_config_creates_default_file_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            cfg = load_config(str(config_path))

            self.assertTrue(config_path.exists())
            self.assertEqual(cfg.name, "Newt")
            self.assertEqual(cfg.op.cmd_trigers, "cmd_trigers.json")

    def test_command_operator_reads_triggers_from_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trigger_path = Path(tmpdir) / "custom_triggers.json"
            trigger_path.write_text(
                json.dumps(
                    {
                        "greet": ["hi there"],
                        "farewell": ["see you later"],
                    }
                ),
                encoding="utf-8",
            )

            config_module.cfg.op.cmd_trigers = str(trigger_path)
            operator = CommandOperator()

            self.assertEqual(operator.trigers["greet"], ["hi there"])
            self.assertEqual(operator.trigers["farewell"], ["see you later"])


if __name__ == "__main__":
    unittest.main()
