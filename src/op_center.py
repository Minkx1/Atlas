#
# op_center.py
# Center Of Operations: processes commands from STT
#

from os import cpu_count

from llama_cpp import Llama

if __name__ == "__main__":
    MAIN = True
    from config import DATA_DIR, cfg
else:
    from .config import DATA_DIR, cfg


class Operator:
    def __init__(self) -> None:
        pass

    def operate(self, text: str) -> None: ...


class LLM:
    def __init__(self) -> None:
        l = cfg.llm

        self.model_path = DATA_DIR / l.model_path
        self.initial_prompt = l.initial_prompt
        self.context_tokens = l.context_tokens
        self.max_tokens = l.max_msg_tokens
        self.temperature = l.temperature

        self.llama = Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_tokens,
            n_threads=(int(cpu_count() or 1)),
            n_gpu_layers=0,
            verbose=False,
        )

    def get_response(self, message: str) -> str:
        response = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": self.initial_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=1.1,  # Token repeatance protection
        )
        return str(response["choices"][0]["message"]["content"])  # type: ignore

    def stream(self) -> None:
        from rich import print as rprint

        while True:
            try:
                rprint("[NEWT]: " + self.get_response(input(">>> ")))
            except KeyboardInterrupt:
                print("\nQuiting...")
                return


if MAIN:
    llm = LLM()
    llm.stream()
