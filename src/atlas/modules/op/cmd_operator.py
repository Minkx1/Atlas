#
# op / cmd_operator.py
# Loads and manipulaties commands and plugins
#

import json
import logging
import re
import threading
from pathlib import Path

import numpy as np

from atlas.core.events import EventManager
from atlas.utils.config import CONFIG_DIR, DATA_DIR, PLUGINS_DIR, Config

from .plugins import Plugin, PluginManifest
from .sentence_transformer import ONNXSentenceTransformer

log = logging.getLogger(__name__)

CONFIG_EXAMPLE = Path(__file__).parent / "op_example.toml"
cfg = Config.load_config("op.toml", CONFIG_EXAMPLE.read_text())["op"]


class CommandOperator:
    def __init__(
        self,
        events: EventManager | None = None,
    ) -> None:
        self.events = events or EventManager()

        self.history: list[str] = []
        self.commands: dict[str, dict[str, list[dict[str, str]] | list[str]]] = {}
        self.plugins: dict[str, Plugin] = {}

        self.triggers: dict[str, list[str]] = {}

        self.intent_threshold = 0.60
        self.margin = 0.05

    def load(self):
        self.model = ONNXSentenceTransformer(  # embedding model
            "all-MiniLM-L6-v2", DATA_DIR / "models" / "sentence-transformer"
        )
        self.model.load()

        self.trigger_embeddings: dict[str, np.ndarray] = {}

        self._load_commands()
        self._load_plugins()
        self._precompute_embeddings()

        log.info("Embeddings and commands loaded")

    @staticmethod
    def load_commands() -> dict[str, dict[str, str | list[str] | None]]:
        path = CONFIG_DIR / "commands.json"
        if not path.exists():
            Config.write_from_eample(
                path, Path(__file__).parent / "commands_example.json"
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return {}

        commands: dict[str, dict[str, str | list[str] | None]] = {}
        for intent, values in payload.items():
            if not isinstance(values, dict):
                continue

            commands[str(intent)] = {
                "sounds": values.get("sounds", []),
                "triggers": values.get("triggers", []),
            }

        return commands

    def _load_commands(self) -> None:
        """Loads all triggers and intents from commands config."""
        self.commands = self.load_commands() or {}  # type: ignore

        log.debug("Loaded intents: %s", list(self.commands.keys()))

        def _format_triggers(triggers: list[str]) -> list[str]:
            res = []
            for trig in triggers:
                new = trig.format(username=cfg["username"], name=cfg["name"])
                res.append(new)
            return res

        for intent, data in self.commands.items():
            if isinstance(data, dict):
                if "triggers" in data:
                    self.triggers[intent] = _format_triggers(data["triggers"])  # type: ignore
            else:
                log.warning(
                    "Data for intent '%s' is not a dict: %s", intent, type(data)
                )

    def _load_plugins(self) -> None:
        if not PLUGINS_DIR.exists():
            return

        for d in PLUGINS_DIR.iterdir():
            if not d.is_dir():
                continue
            toml_path = d / "plugin.toml"
            if not toml_path.exists():
                continue
            try:
                manifest = PluginManifest.from_toml(toml_path)
            except Exception:
                log.exception("Unable to parse %s", toml_path)
                continue

            self.plugins[manifest.id] = Plugin(d, manifest, self.events)
            self.triggers[manifest.id] = manifest.triggers
            log.info("Loaded plugin: %s", manifest.id)

    def _precompute_embeddings(self) -> None:
        """Precomputes embeddings for triggers."""
        log.debug("Precomputing trigger embeddings")
        for intent, triggers in self.triggers.items():
            if not triggers:
                log.warning("Intent '%s' has empty triggers; skipping", intent)
                continue
            vectors = self._get_embedd_vec(triggers)
            self.trigger_embeddings[intent] = vectors
        log.debug("Embeddings precomputed")

    def _get_embedd_vec(self, phrase: str | list[str]) -> np.ndarray:
        return self.model.encode(
            phrase, normalize_embeddings=True, convert_to_numpy=True
        )

    @staticmethod
    def _eval_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2))

    def operate(self, cmd: str) -> str | None:
        """Returns tuple: command_type"""
        self.history.append(cmd)

        cmd_clean = re.sub(r"[^\w\s]", "", cmd.lower()).strip()
        if not cmd_clean:
            return None

        intent = self._detect_intent(cmd_clean)

        if not intent:
            return None

        self.exec_command(intent)
        return "command"

    def _detect_intent(self, cmd_clean: str) -> str | None:
        if not cmd_clean:
            return None

        cmd_vec = self._get_embedd_vec(cmd_clean)

        scores = []

        for intent, vectors in self.trigger_embeddings.items():
            for trigger_vec in vectors:
                score = self._eval_cosine_similarity(cmd_vec, trigger_vec)
                scores.append((score, intent))

        if not scores:
            return None

        intent_best_scores = {}  # best scores for every intent
        for score, intent in scores:
            if intent not in intent_best_scores or score > intent_best_scores[intent]:
                intent_best_scores[intent] = score

        sorted_intents = sorted(
            intent_best_scores.items(), key=lambda x: x[1], reverse=True
        )

        best_intent, best_score = sorted_intents[0]

        # Seeking second best ONLY from other scores
        second_best_score = sorted_intents[1][1] if len(sorted_intents) > 1 else 0.0
        second_best_intent = sorted_intents[1][0] if len(sorted_intents) > 1 else "None"

        log.debug(
            "Intent check '%s': best %s (%.3f), second %s (%.3f)",
            cmd_clean,
            best_intent,
            best_score,
            second_best_intent,
            second_best_score,
        )

        if best_intent == "llm_query":
            log.debug("Intent is 'llm_query'; passing to LLM")
            return None

        if best_score >= self.intent_threshold:
            margin = best_score - second_best_score
            is_confident = margin >= self.margin

            if is_confident:
                log.debug(
                    "Found confident intent: %s (score %.3f, margin %.3f)",
                    best_intent,
                    best_score,
                    margin,
                )
                return best_intent
            else:
                log.debug(
                    "Rejected intent '%s': margin too low (%.3f < %.3f)",
                    best_intent,
                    margin,
                    self.margin,
                )
        else:
            log.debug(
                "Rejected intent '%s': score too low (%.3f < %.3f)",
                best_intent,
                best_score,
                self.intent_threshold,
            )

        return None

    def exec_command(self, intent: str) -> dict[str, str | None] | None:
        log.debug("Executing intent: %s", intent)
        if intent in self.plugins:
            origin = self.history[-1] if self.history else ""

            def run_plugin() -> None:
                try:
                    self.plugins[intent].run(origin)
                except Exception:
                    log.exception("Plugin '%s' failed unexpectedly", intent)

            # running plugin process in separate thread
            threading.Thread(
                target=run_plugin,
                name=f"PLUGIN_{intent}",
                daemon=True,
            ).start()
            return None

        self.events.emit("op.intent", {"intent": intent})
        log.info(f"Intent: {intent}")
