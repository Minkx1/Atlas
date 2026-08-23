import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import DATA_DIR, cfg
from .events import EventType, emit_event, log


class ONNXSentenceTransformer:
    """Lightweight replacement for sentence-transformers using ONNX & Rust tokenizers."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", download_dir: Path | None = None
    ):
        self.model_name = model_name

        self.download_dir = download_dir or DATA_DIR / "models" / model_name
        self.onnx_path = self.download_dir / "model.onnx"
        self.tokenizer_path = self.download_dir / "tokenizer.json"

    def _download_file(self, url: str, dest: Path):
        if dest.exists():
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        log(f"Downloading {dest.name}...", "OP", "INFO")
        urllib.request.urlretrieve(url, dest)

    def _download_model(self):
        base_url = f"https://huggingface.co/Xenova/{self.model_name}/resolve/main"

        try:
            self._download_file(f"{base_url}/onnx/model.onnx", self.onnx_path)
            self._download_file(f"{base_url}/tokenizer.json", self.tokenizer_path)
        except (URLError, OSError) as e:
            if self.onnx_path.exists():
                self.onnx_path.unlink()
            if self.tokenizer_path.exists():
                self.tokenizer_path.unlink()
            raise RuntimeError(f"[!] Failed to download {self.model_name}: {e}") from e

    def load(self):
        if not self.onnx_path.exists() or not self.tokenizer_path.exists():
            self._download_model()

        self.tokenizer: Tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_padding(
            direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]"
        )
        self.tokenizer.enable_truncation(max_length=256)

        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )

    def encode(
        self, sentences: str | list[str], normalize_embeddings: bool = True, **kwargs
    ) -> np.ndarray:
        """Mimics the original SentenceTransformer.encode() behavior."""
        if not hasattr(self, "session"):
            raise RuntimeError("Model was used before load() was called.")

        is_single_string = isinstance(sentences, str)
        if is_single_string:
            sentences = [sentences]

        encoded = self.tokenizer.encode_batch(sentences)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        ort_outs = self.session.run(None, ort_inputs)
        token_embeddings = ort_outs[0]

        # mean pooling
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)  # type: ignore
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # normalization
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

        if is_single_string:
            return embeddings[0]

        return embeddings


class CommandOperator:
    class Plugin:
        """Command class describes plugins: directories in /data/commands/ containing file `plugin.toml`"""

        def __init__(self, root: Path, config: Path) -> None:
            self.root: Path = root
            self.config_path = config

            self.id = None
            self.description = "Unknown."

            self.parse_config(self.config_path)

        def parse_config(self, path: Path) -> None: ...

    def __init__(self) -> None:
        self.history: list[str] = []
        self.commands: dict[str, dict[str, list[dict[str, str]] | list[str]]] = {}

        self.triggers: dict[str, list[str]] = {
            "llm_query": [
                "what is",
                "how to",
                "tell me about",
                "explain",
                "write a code",
                "can you",
            ],
        }

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
                    f"Warning: Data for intent '{intent}' is not a dict. Type: {type(data)}",
                    "OP",
                    "WARNING",
                )

    def _load_plugins(self) -> None:
        cmd_dir = DATA_DIR / "commands"

        if not cmd_dir.exists():
            cmd_dir.mkdir(parents=True, exist_ok=True)
            return

        l: list[CommandOperator.Plugin] = []

        for dir in cmd_dir.iterdir():
            if dir.is_dir():
                toml = dir / "plugin.toml"
                if toml.exists() and toml.is_file():
                    l.append(CommandOperator.Plugin(cmd_dir, toml))

    def _precompute_embeddings(self) -> None:
        """Precomputes embeddings for triggers."""
        log("Precomputing trigger embeddings...", "OP", "DEBUG")
        for intent, triggers in self.triggers.items():
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

    def operate(self, cmd: str) -> tuple[str | None, dict | None]:
        """Returns tuple: (command_type, payload)"""
        self.history.append(cmd)

        if cmd == "!EVENT_KEYWORD_DETECTED":
            log("Keyword detected directly!", "OP", "DEBUG")
            payload = self.exec_command("greet")
            return "command", payload

        cmd_clean = re.sub(r"[^\w\s]", "", cmd.lower()).strip()
        if not cmd_clean:
            return None, None

        intent = self._detect_intent(cmd_clean)

        if not intent:
            return None, None
        elif intent.startswith("!"):  # user command
            self.exec_user(intent, cmd)
            return "user", None
        else:
            payload = self.exec_command(intent)
            return "command", payload

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
        payload = self._play_random_sound(intent)

        if intent == "farewell":
            emit_event(EventType.OP_ASK_FINISH)
        elif intent == "sleep":
            emit_event(EventType.STT_SET_STATE, "SLEEPING")

        return payload

    def _play_random_sound(self, category: str) -> dict[str, str | None] | None:
        conf = self.commands.get(category, {})
        sounds = conf.get("sounds", [])

        log(
            f"Fetching sound for '{category}'.",
            "OP",
            "DEBUG",
        )

        if isinstance(sounds, list) and sounds:
            sound = random.choice(sounds)

            path_str = ""
            text_str = ""

            if isinstance(sound, dict):
                path_str = sound.get("path", "")
                text_str = sound.get("text", "")
            elif isinstance(sound, str):
                path_str = sound
            else:
                log(
                    f"Invalid sound type in config for '{category}': {type(sound)}",
                    "OP",
                    "WARNING",
                )

            if text_str:
                try:
                    text_str = text_str.format(username=cfg.username, name=cfg.name)
                except KeyError as e:
                    log(
                        f"Formatting text failed for sound '{text_str}': Missing key {e}",
                        "OP",
                        "DEBUG",
                    )

            if path_str:
                path = Path(path_str)
                if not path.is_absolute():
                    path = DATA_DIR / "sounds" / path

                log(
                    f"Playing sound payload: {path_str} | text: {text_str}",
                    "OP",
                    "DEBUG",
                )
                payload = {"path": str(path), "text": text_str if text_str else None}
                emit_event(EventType.TTS_PLAY_SOUND, payload)
                return payload

        log(f"No sounds available for category: {category}", "OP", "WARNING")
        return None

    def exec_user(self, intent: str, cmd: str) -> bool:
        return False
