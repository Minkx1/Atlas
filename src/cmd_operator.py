#
# cmd_operator.py
# Operator: processes and operates commands
#

import os
import queue
import random
import re
import threading
import time
from pathlib import Path

import llama_cpp
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, cfg
from .events import EventType, emit_event, log, wait_for

# Llama-cpp traceback fix
_orig_llama_del = getattr(llama_cpp.Llama, "__del__", None)
if _orig_llama_del:

    def _silent_llama_del(self):
        try:
            _orig_llama_del(self)  # type: ignore
        except (TypeError, AttributeError, NameError, ImportError):
            pass

    llama_cpp.Llama.__del__ = _silent_llama_del


class CommandOperator:
    class Command:
        """Command class describes user commands: directories in directory commands/ containing file `command.toml`"""

        def __init__(self, dir: Path, config: Path) -> None:
            self.root = dir
            self.config_path = config

    def __init__(self) -> None:
        self.history: list[str] = []
        self.builtin_commands: dict[
            str, dict[str, list[dict[str, str]] | list[str]]
        ] = {}

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

        self.intent_threshold = 0.65
        self.margin = 0.08
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.trigger_embeddings: dict[str, np.ndarray] = {}

        self._load_builtin_commands()
        self._load_user_commands()
        self._precompute_embeddings()

    def _load_builtin_commands(self) -> None:
        self.builtin_commands = cfg.op.load_builtin_commands() or {}

        for intent, data in self.builtin_commands.items():
            if isinstance(data, dict) and "triggers" in data:
                self.triggers[intent] = data["triggers"]  # type: ignore

    def _load_user_commands(self) -> None:
        cmd_dir = DATA_DIR / "commands"

        if not cmd_dir.exists():
            return

        l: list[CommandOperator.Command] = []

        for file in cmd_dir.iterdir():
            if file.is_dir():
                toml = file / "command.toml"
                if toml.exists() and toml.is_file():
                    l.append(CommandOperator.Command(cmd_dir, toml))

    def _precompute_embeddings(self) -> None:
        """Precomputes embeddings for triggers."""
        for intent, triggers in self.triggers.items():
            vectors = self._get_embedd_vec(triggers)
            self.trigger_embeddings[intent] = vectors

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
            payload = self.exec_builtin("greet")
            return "builtin", payload

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
            payload = self.exec_builtin(intent)
            return "builtin", payload

    def _detect_intent(self, cmd_clean: str) -> str | None:
        if not cmd_clean:
            return None

        cmd_vec = self._get_embedd_vec(cmd_clean)

        best_intent: str | None = None
        best_score = 0.0
        second_best_score = 0.0

        for intent, vectors in self.trigger_embeddings.items():
            for trigger_vec in vectors:
                score = self._eval_cosine_similarity(cmd_vec, trigger_vec)

                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_intent = intent
                elif score > second_best_score:
                    second_best_score = score

        if best_intent == "llm_query":
            return None

        if best_intent and best_score >= self.intent_threshold:
            margin = best_score - second_best_score
            is_confident = margin >= self.margin

            if is_confident:
                log(f"Found intent: {best_intent} (Score: {best_score:.2f})", "CMD_OP")
                return best_intent

        return None

    def exec_builtin(self, intent: str) -> dict[str, str | None] | None:
        payload = self._play_random_sound(intent)

        if intent == "farewell":
            emit_event(EventType.OP_ASK_FINISH)
        elif intent == "sleep":
            emit_event(EventType.STT_SET_STATE, "SLEEPING")

        return payload

    def _play_random_sound(self, category: str) -> dict[str, str | None] | None:
        conf = self.builtin_commands.get(category, {})
        sounds = conf.get("sounds", [])

        if isinstance(sounds, list) and sounds:
            sound = random.choice(sounds)

            path_str = ""
            text_str = ""

            if isinstance(sound, dict):
                path_str = sound.get("path", "")
                text_str = sound.get("text", "")
            elif isinstance(sound, str):
                path_str = sound

            if text_str:
                try:
                    text_str = text_str.format(username=cfg.username, name=cfg.name)
                except KeyError:
                    pass

            if path_str:
                path = Path(path_str)
                if not path.is_absolute():
                    path = DATA_DIR / "sounds" / path

                payload = {"path": str(path), "text": text_str if text_str else None}
                emit_event(EventType.TTS_PLAY_SOUND, payload)
                return payload

        log(f"No sounds available for category: {category}", "CMD_OP", "WARNING")
        return None

    def exec_user(self, intent: str, cmd: str) -> bool:
        return False


class LLM:
    class _LLM_Response:
        def __init__(
            self, text: str, prompt_tokens: int, completion_tokens: int, gen_ms: float
        ) -> None:
            self.text = text
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens
            self.gen_ms = gen_ms

    def __init__(self) -> None:
        l = cfg.llm

        self.model_path = DATA_DIR / l.model_path
        self.initial_prompt = l.initial_prompt
        self.context_tokens = l.context_tokens
        self.max_tokens = l.max_msg_tokens
        self.temperature = l.temperature

        self.repeat_penalty = 1.5
        self.stop = ["😊", "\nUser:", "User:", "<|im_end|>"]

        self.history: list[dict[str, str]] = [
            {"role": "system", "content": self.initial_prompt}
        ]

    def load(self):
        _start = time.perf_counter()
        self.llama = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_tokens,
            n_threads=(int(os.cpu_count() or 1)),
            n_gpu_layers=0,
            verbose=False,
        )
        emit_event(EventType.LLM_LOADED, f"{(time.perf_counter() - _start) * 1000}ms")

    def history_add_response(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def get_response(self, message: str) -> _LLM_Response:
        self.history.append({"role": "user", "content": message})
        start_time = time.perf_counter()

        response = self.llama.create_chat_completion(  # type: ignore
            messages=[self.history],  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            stop=self.stop,
        )

        gen_ms = time.perf_counter() - start_time
        text: str = str(response["choices"][0]["message"]["content"])  # type: ignore
        usage = response["usage"]  # type: ignore

        return LLM._LLM_Response(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            gen_ms=gen_ms,
        )

    def stream_response(self, message: str):
        self.history.append({"role": "user", "content": message})

        output = self.llama.create_chat_completion(  # type: ignore
            messages=self.history,  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            stop=self.stop,
            stream=True,
        )

        for chunk in output:
            delta = chunk["choices"][0]["delta"]  # type: ignore
            if "content" in delta:
                yield delta["content"]  # type: ignore

    def close(self):
        if hasattr(self, "llama") and self.llama is not None:
            try:
                self.llama.close()
            except Exception:  # noqa: BLE001, S110
                pass
            finally:
                self.llama = None

    def __del__(self):
        self.close()


class Operator:
    def __init__(self, cmd: CommandOperator, llm: "LLM") -> None:
        self._running = False
        self.cmd = cmd
        self.llm = llm
        self.command_queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._operator_worker, name="OPERATOR_THREAD", daemon=True
        )

    def start(self):
        self._running = True
        self.worker_thread.start()

    def close(self):
        self._running = False
        self.llm.close()

    def submit(self, text: str):
        self.command_queue.put(text)

    @staticmethod
    def _sentence_chunker(token_stream):
        """Generator: gathers tokens into complete sentences."""
        buffer = ""
        for token in token_stream:
            buffer += token
            if any(char in buffer for char in ".?!\n"):
                parts = re.split(r"([.?!]\s+|\n)", buffer, maxsplit=1)
                if len(parts) > 1:
                    sentence = (parts[0] + (parts[1] or "")).strip()
                    if sentence:
                        yield sentence
                    buffer = parts[2] if len(parts) > 2 else ""
        if buffer.strip():
            yield buffer.strip()

    def _operator_worker(self):
        while self._running:
            text = self.command_queue.get(block=True)
            if text is None:
                self.command_queue.task_done()
                break

            self._operate(text)
            self.command_queue.task_done()

    def _operate(self, text: str) -> None:
        if not text:
            return

        emit_event(EventType.PROFILER_SET_STATE, "PROCESSING")
        start_time = time.perf_counter()
        full_response_text = ""

        tts_engaged = False

        try:
            res_type, payload = self.cmd.operate(text)
            emit_event(EventType.OP_CMD_LEVEL, str(res_type))

            if res_type == "builtin" and payload is not None:
                tts_engaged = True

            if not res_type:
                token_stream = self.llm.stream_response(text)

                is_first_chunk = True
                for sentence in self._sentence_chunker(token_stream):
                    full_response_text += sentence + " "

                    emit_event(EventType.TTS_SPEAK, sentence)
                    tts_engaged = True

                    emit_event(
                        EventType.UI_LLM_CHUNK,
                        {"text": sentence, "is_first": is_first_chunk},
                    )
                    is_first_chunk = False

                gen_ms = (time.perf_counter() - start_time) * 1000
                emit_event(
                    EventType.UI_LLM_RESPONSE_DONE,
                    {
                        "text": full_response_text.strip(),
                        "gen_ms": gen_ms,
                    },
                )
                emit_event(EventType.LLM_RESPONSE, full_response_text.strip())
                self.llm.history_add_response(full_response_text.strip())

        finally:
            emit_event(EventType.PROFILER_SET_STATE, "AWAKE")

            if tts_engaged:
                wait_for(EventType.TTS_FREE)

            emit_event(EventType.OP_READY)
