import re
import threading
import time

import numpy as np

from ..core.config import DATA_DIR, PLUGINS_DIR, cfg
from ..core.events import EventType, emit_event, log
from .plugins import Plugin, PluginManifest
from .sentence_transformer import ONNXSentenceTransformer


class CommandOperator:
    def __init__(self) -> None:
        self.history: list[str] = []
        self.commands: dict[str, dict[str, list[dict[str, str]] | list[str]]] = {}
        self.plugins: dict[str, Plugin] = {}

        self.triggers: dict[str, list[str]] = {}

        self.intent_threshold = 0.60
        self.margin = 0.05

    def load(self):
        _start = time.perf_counter()

        self.model = ONNXSentenceTransformer(  # embedding model
            "all-MiniLM-L6-v2", DATA_DIR / "models" / "sentence-transformer"
        )
        self.model.load()

        self.trigger_embeddings: dict[str, np.ndarray] = {}

        self._load_commands()
        self._load_plugins()
        self._precompute_embeddings()

        log(
            f"Embeddings & commands loaded in: {(time.perf_counter() - _start) * 1000}ms",
            "OP",
            "INFO",
        )

    def _load_commands(self) -> None:
        """Loads all triggers and intents from commands config."""
        self.commands = cfg.op.load_commands() or {}  # type: ignore

        log(
            f"Loaded intents: {list(self.commands.keys())}",
            "OP",
            "DEBUG",
        )

        def _format_triggers(triggers: list[str]) -> list[str]:
            res = []
            for trig in triggers:
                new = trig.format(username=cfg.username, name=cfg.name)
                res.append(new)
            return res

        for intent, data in self.commands.items():
            if isinstance(data, dict):
                if "triggers" in data:
                    self.triggers[intent] = _format_triggers(data["triggers"])  # type: ignore
            else:
                log(
                    f"Data for intent '{intent}' is not a dict. Type: {type(data)}",
                    "OP",
                    "WARN",
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
            except Exception as e:  # noqa: BLE001
                log(f"Unable to parse {toml_path}: {e}", "OP", "ERROR")
                continue

            self.plugins[manifest.id] = Plugin(d, manifest)
            self.triggers[manifest.id] = manifest.triggers
            log(f"Loaded plugin: {manifest.id}", "OP", "INFO")

    def _precompute_embeddings(self) -> None:
        """Precomputes embeddings for triggers."""
        log("Precomputing trigger embeddings...", "OP", "DEBUG")
        for intent, triggers in self.triggers.items():
            if not triggers:
                log(f"Intent '{intent}' has empty triggers. Skipping.", "OP", "WARN")
                continue
            vectors = self._get_embedd_vec(triggers)
            self.trigger_embeddings[intent] = vectors
        log("Embeddings precomputed.", "OP", "DEBUG")

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

        if cmd == "!EVENT_KEYWORD_DETECTED":
            log("Keyword detected directly!", "OP", "DEBUG")
            self.exec_command("greet")
            return "command"

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

        log(
            f"Intent check '{cmd_clean}': Best: {best_intent} ({best_score:.3f}), 2nd: {second_best_intent} ({second_best_score:.3f})",
            "OP",
            "DEBUG",
        )

        if best_intent == "llm_query":
            log("Intent is 'llm_query', passing to LLM.", "OP", "DEBUG")
            return None

        if best_score >= self.intent_threshold:
            margin = best_score - second_best_score
            is_confident = margin >= self.margin

            if is_confident:
                log(
                    f"Found confident intent: {best_intent} (Score: {best_score:.3f}, Margin: {margin:.3f})",
                    "OP",
                    "DEBUG",
                )
                return best_intent
            else:
                log(
                    f"Rejected intent '{best_intent}': Margin too low ({margin:.3f} < {self.margin})",
                    "OP",
                    "DEBUG",
                )
        else:
            log(
                f"Rejected intent '{best_intent}': Score too low ({best_score:.3f} < {self.intent_threshold})",
                "OP",
                "DEBUG",
            )

        return None

    def exec_command(self, intent: str) -> dict[str, str | None] | None:
        log(f"Executing intent: {intent}", "OP", "DEBUG")
        if intent in self.plugins:
            origin = self.history[-1] if self.history else ""
            # running plugin process in separate thread
            threading.Thread(
                target=self.plugins[intent].run, args=(origin,), daemon=True
            ).start()
            return None

        self._play_random_sound(intent)

        if intent == "farewell":
            emit_event(EventType.OP_ASK_FINISH)
        elif intent == "sleep":
            emit_event(EventType.STT_SET_STATE, "SLEEPING")

    def _play_random_sound(self, category: str):
        emit_event(EventType.SM_PLAY_CATEGORY, category)
