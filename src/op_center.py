#
# op_center.py
# Center Of Operations: processes commands from STT
#

import os
import time

import llama_cpp

if __name__ == "__main__":
    MAIN = True
    from config import DATA_DIR, cfg
    from profiler import profiler
    from text_to_speach import TextToSpeech
    from ui import AssistantUI
else:
    MAIN = False
    from .config import DATA_DIR, cfg
    from .profiler import profiler
    from .text_to_speach import TextToSpeech
    from .ui import AssistantUI


class Operator:
    def __init__(self) -> None:
        self.tts = TextToSpeech()
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

        profiler.set_state("PROCESSING")

        try:
            if not self.check_builtin_command(text):
                if not self.check_user_command(text):
                    # The command is not generic
                    resp = self.llm.get_response(text)
                    self.tts.speak_sync(resp.text)
        finally:
            profiler.set_state("AWAKE")

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

        response = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": self.initial_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=1.1,  # Token repeatance protection
        )

        gen_ms = time.perf_counter() - start_time

        text: str = str(response["choices"][0]["message"]["content"])

        usage = response["usage"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        AssistantUI.print_llm_response(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            gen_ms=gen_ms,
        )

        return _Response(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            gen_ms=gen_ms,
        )

    def close(self):
        """Safe C++ context closing with Python 3.14 GC protection"""
        if (
            not getattr(self, "_is_closed", True)
            and hasattr(self, "llama")
            and self.llama is not None
        ):
            try:
                self.llama.close()
            except Exception:
                pass
            finally:
                self.llama = None
                self._is_closed = True

    def __del__(self):
        self.close()
