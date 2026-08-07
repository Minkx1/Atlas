#
# operator.py
# Center Of Operations: processes commands from STT
#

import os
import queue
import re
import threading
import time

import llama_cpp

from .config import DATA_DIR, cfg
from .events import emit_event

# Llama-cpp traceback fix
_orig_llama_del = getattr(llama_cpp.Llama, "__del__", None)
if _orig_llama_del:

    def _silent_llama_del(self):
        try:
            _orig_llama_del(self)  # type: ignore
        except (TypeError, AttributeError, NameError, ImportError):
            pass

    llama_cpp.Llama.__del__ = _silent_llama_del


def _sentence_chunker(token_stream):
    """Generator: gathers tokens into a complete sentences."""
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


class Operator:
    def __init__(self) -> None:
        self.llm = LLM()

        self.command_queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._operator_worker, name="OPERATOR_THREAD", daemon=True
        )

    def start(self):
        self.worker_thread.start()

    def submit(self, text: str):
        self.command_queue.put(text)

    def _operator_worker(self):
        while True:
            text = self.command_queue.get()
            if text is None:
                self.command_queue.task_done()
                break

            self._operate(text)
            self.command_queue.task_done()

    def check_builtin_command(self, cmd: str) -> bool:
        """Checks whether command is in a _builtin_ level and if so exutes it."""
        # checking and executing builtin command logic
        return False

    def check_user_command(self, cmd: str) -> bool:
        """Checks whether command is in a _user_ level and if so exutes it."""
        # checking and executing user command logic
        return False

    def _operate(self, text: str) -> None:
        if not text:
            return

        emit_event("PROFILER_SET_STATE", "PROCESSING")
        start_time = time.perf_counter()
        full_response_text = ""

        try:
            if not (self.check_builtin_command(text) or self.check_user_command(text)):
                # The command is not generic
                token_stream = self.llm.stream_response(text)

                is_first_chunk = True
                for sentence in _sentence_chunker(token_stream):
                    full_response_text += sentence + " "

                    emit_event("TTS_SPEAK_CHUNK", sentence)

                    emit_event(
                        "UI_LLM_CHUNK", {"text": sentence, "is_first": is_first_chunk}
                    )
                    is_first_chunk = False

                gen_ms = (time.perf_counter() - start_time) * 1000
                emit_event(
                    "UI_LLM_RESPONSE_DONE",
                    {
                        "text": full_response_text.strip(),
                        "gen_ms": gen_ms,
                    },
                )

        finally:
            emit_event("PROFILER_SET_STATE", "AWAKE")
            emit_event("STT_RESUME")
            emit_event("OPERATOR_FINISHED")

    def close(self):
        self.llm.close()


class _Response:
    def __init__(
        self, text: str, prompt_tokens: int, completion_tokens: int, gen_ms: float
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.gen_ms = gen_ms


class LLM:
    def __init__(self) -> None:
        l = cfg.llm

        self.model_path = DATA_DIR / l.model_path
        self.initial_prompt = l.initial_prompt
        self.context_tokens = l.context_tokens
        self.max_tokens = l.max_msg_tokens
        self.temperature = l.temperature

        self.llama = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_tokens,
            n_threads=(int(os.cpu_count() or 1)),
            n_gpu_layers=0,
            verbose=False,
        )

    def get_response(self, message: str) -> _Response:
        start_time = time.perf_counter()

        response = self.llama.create_chat_completion(  # type: ignore
            messages=[
                {"role": "system", "content": self.initial_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=1.1,  # Token repeatance protection
        )

        gen_ms = time.perf_counter() - start_time

        text: str = str(response["choices"][0]["message"]["content"])  # type: ignore

        usage = response["usage"]  # type: ignore
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return _Response(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            gen_ms=gen_ms,
        )

    def stream_response(self, message: str):
        output = self.llama.create_chat_completion(  # type: ignore
            messages=[
                {"role": "system", "content": self.initial_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=1.1,
            stream=True,  # adding streaming
        )

        for chunk in output:
            delta = chunk["choices"][0]["delta"]  # type: ignore
            if "content" in delta:
                yield delta["content"]  # type: ignore

    def close(self):
        if hasattr(self, "llama") and self.llama is not None:
            try:
                self.llama.close()
            except Exception:
                pass
            finally:
                self.llama = None

    def __del__(self):
        self.close()
