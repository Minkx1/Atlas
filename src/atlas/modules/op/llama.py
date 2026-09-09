#
# op / llama.py
# Wrapper for Llama-cpp-python
#

import logging
import os
from contextlib import suppress

import llama_cpp

from atlas.core.config import DATA_DIR, cfg
from atlas.core.events import EventManager

log = logging.getLogger(__name__)

# Llama-cpp traceback fix
_orig_llama_del = getattr(llama_cpp.Llama, "__del__", None)
if _orig_llama_del:

    def _silent_llama_del(self):
        with suppress(TypeError, AttributeError, NameError, ImportError):
            _orig_llama_del(self)  # type: ignore

    llama_cpp.Llama.__del__ = _silent_llama_del


class Llama:
    class _LLM_Response:
        def __init__(
            self, text: str, prompt_tokens: int, completion_tokens: int
        ) -> None:
            self.text = text
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens

    def __init__(self, events: EventManager | None = None) -> None:
        self.events = events or EventManager()

        self.model_path = DATA_DIR / cfg.llm.model_path
        self.initial_prompt = cfg.llm.initial_prompt
        self.context_tokens = cfg.llm.context_tokens
        self.max_tokens = cfg.llm.max_msg_tokens
        self.temperature = cfg.llm.temperature

        self.repeat_penalty = 1.15
        self.stop = ["\nUser:", "User:", "<|im_end|>"]

        self.history: list[dict[str, str]] = [
            {"role": "system", "content": self.initial_prompt}
        ]
        self.no_model = False  # flag that show whether the model file is valid

    def load(self):
        try:
            log.info("Loading LLM model: %s", self.model_path.name)
            if not self.model_path.exists():
                log.warning("LLM model path is invalid; running without LLM")
                self.no_model = True
                return

            self.llama = llama_cpp.Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_tokens,
                n_threads=(int(os.cpu_count() or 1)),
                n_gpu_layers=0,
                verbose=False,
            )
        except Exception:
            log.exception("Error loading LLM model")
            self.no_model = True
            raise

    def history_add_response(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def get_response(self, message: str) -> _LLM_Response:
        try:
            log.debug("Getting LLM response for: %s", message)
            self.history.append({"role": "user", "content": message})
            response = self.llama.create_chat_completion(  # type: ignore
                messages=self.history,  # type: ignore
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                repeat_penalty=self.repeat_penalty,
                stop=self.stop,
            )

            text: str = str(response["choices"][0]["message"]["content"])  # type: ignore
            usage = response["usage"]  # type: ignore

            return Llama._LLM_Response(
                text=text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        except Exception:
            log.exception("Error getting LLM response")
            raise

    def stream_response(self, message: str):
        try:
            log.debug("Streaming LLM response for: %s", message)
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
                except Exception:
                    log.exception("Error processing LLM stream chunk")
                    raise
        except Exception:
            log.exception("Error streaming LLM response")
            raise

    def close(self):
        if hasattr(self, "llama") and self.llama is not None:
            try:
                log.debug("Closing LLM model")
                self.llama.close()
            except Exception:
                log.exception("Error closing LLM")
                raise
            finally:
                self.llama = None
                log.debug("LLM model closed")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    llm = Llama()
    print("Loading model...")
    llm.load()
    print("Loading complete!")
    while True:
        try:
            for tok in llm.stream_response(input("> ")):
                print(tok, sep="", end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nQuiting...")
            break

    llm.close()
