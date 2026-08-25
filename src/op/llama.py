#
# llama.py
#

import os
import time

import llama_cpp

from ..core.config import DATA_DIR, cfg
from ..core.events import EventType, emit_event, log

# Llama-cpp traceback fix
_orig_llama_del = getattr(llama_cpp.Llama, "__del__", None)
if _orig_llama_del:

    def _silent_llama_del(self):
        try:
            _orig_llama_del(self)  # type: ignore
        except (TypeError, AttributeError, NameError, ImportError):
            pass

    llama_cpp.Llama.__del__ = _silent_llama_del


class Llama:
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

        self.repeat_penalty = 1.15
        self.stop = ["\nUser:", "User:", "<|im_end|>"]

        self.history: list[dict[str, str]] = [
            {"role": "system", "content": self.initial_prompt}
        ]
        self.no_model = False  # flag that show whether the model file is valid

    def load(self):
        try:
            _start = time.perf_counter()
            log(f"Loading LLM model: {self.model_path.name}...", "LLM", "INFO")
            if not self.model_path.exists():
                log(
                    "LLM .gguf model path is not valid. Running without LLM...",
                    "OP",
                    "WARN",
                )
                self.no_model = True
                return

            self.llama = llama_cpp.Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_tokens,
                n_threads=(int(os.cpu_count() or 1)),
                n_gpu_layers=0,
                verbose=False,
            )
            elapsed = (time.perf_counter() - _start) * 1000
            log(f"LLM model loaded in {elapsed:.0f}ms", "LLM", "INFO")
            emit_event(EventType.LLM_LOADED, f"{elapsed}ms")
        except Exception as e:
            log(
                f"Error loading LLM model: {type(e).__name__}: {e}",
                "LLM",
                "ERROR",
            )
            self.no_model = True
            raise

    def history_add_response(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def get_response(self, message: str) -> _LLM_Response:
        try:
            log(f"LLM: Getting response for: {message}...", "LLM", "DEBUG")
            self.history.append({"role": "user", "content": message})
            start_time = time.perf_counter()

            response = self.llama.create_chat_completion(  # type: ignore
                messages=self.history,  # type: ignore
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                repeat_penalty=self.repeat_penalty,
                stop=self.stop,
            )

            gen_ms = (time.perf_counter() - start_time) * 1000
            text: str = str(response["choices"][0]["message"]["content"])  # type: ignore
            usage = response["usage"]  # type: ignore

            log(
                f"LLM response generated in {gen_ms:.0f}ms ({usage.get('completion_tokens', 0)} tokens)",
                "LLM",
                "DEBUG",
            )
            return Llama._LLM_Response(
                text=text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                gen_ms=gen_ms,
            )
        except Exception as e:
            log(
                f"Error getting LLM response: {type(e).__name__}: {e}",
                "LLM",
                "ERROR",
            )
            raise

    def stream_response(self, message: str):
        try:
            log(f"LLM: Streaming response for: {message}...", "LLM", "DEBUG")
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
                try:
                    delta = chunk["choices"][0]["delta"]  # type: ignore
                    if "content" in delta:
                        yield delta["content"]  # type: ignore
                except Exception as e:
                    log(
                        f"Error processing stream chunk: {type(e).__name__}: {e}",
                        "LLM",
                        "ERROR",
                    )
                    raise
        except Exception as e:
            log(
                f"Error streaming LLM response: {type(e).__name__}: {e}",
                "LLM",
                "ERROR",
            )
            raise

    def close(self):
        if hasattr(self, "llama") and self.llama is not None:
            try:
                log("Closing LLM model...", "LLM", "DEBUG")
                self.llama.close()
            except Exception as e:
                log(
                    f"Error closing LLM: {type(e).__name__}: {e}",
                    "LLM",
                    "ERROR",
                )
                raise
            finally:
                self.llama = None
                log("LLM model closed.", "LLM", "DEBUG")

    def __del__(self):
        self.close()
