#
# op / config.py
#

from atlas.utils.config import config


@config(file="op.toml", table="op")
class OpConfig:
    intent_threshold: float = 0.60
    margin: float = 0.05


@config(file="llama.toml", table="llama")
class LlamaConfig:
    model_path: str = "models/llm_models/Llama-3.2-3B-Instruct-Q5_K_M.gguf"
    initial_prompt: str = (
        "You are Atlas, a highly efficient AI voice assistant.\n"
        "Your absolute priority is to provide plain, spoken-language responses.\n"
        "\n"
        "Rules:\n"
        '1. Always call the user "Sir".\n'
        "2. Respond strictly in English.\n"
        "3. Use ONLY plain text (letters, numbers, and basic punctuation). \n"
        "4. Keep answers concise, limited to 2-5 sentences.\n"
    )
    context_tokens: int = 2048
    max_msg_tokens: int = 128
    temperature: float = 0.7


op_cfg = OpConfig.load()
llama_cfg = LlamaConfig.load()
