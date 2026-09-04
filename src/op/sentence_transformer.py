#
# op / sentence_transformer.py
# Own ONNX-runtime wrapper for SentenceTransformer models using `tokenizers`
#

from pathlib import Path
from urllib.error import URLError

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from ..core.config import DATA_DIR
from ..core.events import log


class ONNXSentenceTransformer:
    """Lightweight replacement for sentence-transformers using ONNX & Rust tokenizers"""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", download_dir: Path | None = None
    ):
        self.model_name = model_name

        self.download_dir = download_dir or DATA_DIR / "models" / model_name
        self.onnx_path = self.download_dir / "model.onnx"
        self.tokenizer_path = self.download_dir / "tokenizer.json"

    def _download_file(self, url: str, dest: Path):
        import shutil
        import ssl
        import urllib.request

        # SSL Certificate fix
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        if dest.exists():
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        log(f"Downloading {dest.name}...", "OP", "INFO")

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with (
            urllib.request.urlopen(req, context=ctx) as response,
            open(dest, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

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

        if not sentences:  # prot from [] triggers
            return np.array([])

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
