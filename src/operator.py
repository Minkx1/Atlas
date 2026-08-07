#
# operator.py
# Center Of Operations: processes commands from STT
#

import os
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


class Operator:
    def __init__(self) -> None:
        self.llm = LLM()

    def check_builtin_command(self, cmd: str) -> bool:
        """Checks whether command is in a _builtin_ level and if so exutes it.

        Args:
            cmd(str): Input command.
        Returns:
            Is command _builtin_ (bool).
        """
        # checking and executing builtin command logic
        return False

    def check_user_command(self, cmd: str) -> bool:
        """Checks whether command is in a _user_ level and if so exutes it.

        Args:
            cmd(str): Input command.
        Returns:
            Is command _user_ (bool).
        """
        # checking and executing user command logic
        return False

    def operate(self, text: str) -> None:
        text = text.removesuffix(cfg.name)
        if not text:
            return

        emit_event("PROFILER_SET_STATE", "PROCESSING")

        try:
            if not (self.check_builtin_command(text) or self.check_user_command(text)):
                # The command is not generic
                resp = self.llm.get_response(text)
                emit_event(
                    "UI_LLM_RESPONSE",
                    {
                        "text": resp.text,
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "gen_ms": resp.gen_ms,
                    },
                )
                # self.tts.speak_sync(resp.text)
        finally:
            emit_event("PROFILER_SET_STATE", "AWAKE")

            emit_event("STT_RESUME")

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
