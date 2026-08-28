#
#  kws.py
#

import os
import sys
import time
from pathlib import Path

import numpy as np

_MAIN = __name__ == "__main__"
if not _MAIN:
    from ..core.config import CONFIG_DIR, DATA_DIR, cfg
    from ..core.events import EventType, emit_event, log
else:
    # changing execution dir to src/ for proper importing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import CONFIG_DIR, DATA_DIR, cfg
    from core.events import EventType, emit_event, log


class KeyWordSpotter:
    """Sherpa-ONNX Keyword Spotter model.

    Usage:
    ```python
    kws = KeyWordSpotter()

    kw = kws.process_chunk(audio_chunk)
    if kw:
        print("Keyword was detected: " + kw)
    ```
    """

    def __init__(
        self,
        model_dir: str = cfg.kws.model_dir,
        keywords_file: Path = CONFIG_DIR / cfg.kws.keywords_file,
        num_threads: int = cfg.kws.num_threads,
        keywords_threshold: float = cfg.kws.score_threshold,
    ):
        path: Path = DATA_DIR / model_dir
        self.tokens = str(path / "tokens.txt")
        self.encoder = str(path / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        self.decoder = str(path / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        self.joiner = str(path / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")

        self.num_threads: int = num_threads
        self.keywords_threshold: float = keywords_threshold
        self.keywords_file: str = str(keywords_file)

        if not os.path.exists(self.tokens):
            log(
                f"No Sherpa model in: {path}. Donwloading...",
                source="KWS",
                level="WARN",
            )
            self._download_sherpa_onnx_model(path)

    def load(self):
        import sherpa_onnx

        try:
            _start = time.perf_counter()

            log("Loading Sherpa-ONNX KWS model...", "KWS", "INFO")
            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=self.tokens,
                encoder=self.encoder,
                decoder=self.decoder,
                joiner=self.joiner,
                keywords_file=f"{self.keywords_file}",
                num_threads=self.num_threads,
                keywords_threshold=self.keywords_threshold,
                feature_dim=80,
            )

            self.stream = self.kws.create_stream()

            elapsed = (time.perf_counter() - _start) * 1000
            log(f"KWS model loaded in {elapsed:.0f}ms", "KWS", "SUCCESS")
            emit_event(EventType.KWS_LOADED, f"{elapsed}ms")
        except Exception as e:
            log(
                f"Error loading KWS model: {type(e).__name__}: {e}",
                "KWS",
                "ERROR",
            )
            raise

    @staticmethod
    def _download_sherpa_onnx_model(model_path: Path):
        import shutil
        import ssl
        import tarfile
        import urllib.request
        from urllib.error import URLError

        url = (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
        )
        archive_name = "sherpa_kws_temp.tar.bz2"
        archive_path = model_path.parent / archive_name
        extracted_folder_name = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
        extracted_path = model_path.parent / extracted_folder_name

        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[I] Downloading Sherpa-ONNX KWS model from {url}...")

            # SSL Certificate fix
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with (
                urllib.request.urlopen(url, context=ctx) as response,
                open(archive_path, "wb") as out_file,
            ):
                shutil.copyfileobj(response, out_file)

            print("[I] Extracting model archive...")
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(path=model_path.parent)

            if model_path.exists():
                shutil.rmtree(model_path)
            extracted_path.rename(model_path)

            print(f"[I] Model successfully installed to {model_path}")

        except (URLError, tarfile.TarError, OSError) as e:
            if extracted_path.exists():
                shutil.rmtree(extracted_path, ignore_errors=True)
            raise RuntimeError(
                f"[!] Failed to download or extract Sherpa-ONNX model: {e}"
            ) from e
        finally:
            if archive_path.exists():
                archive_path.unlink()

    def process_chunk(self, chunk_np: np.ndarray) -> str | None:
        """Processes audio chunk. If keyword was spotted: returns it. Else: returns None."""
        if not hasattr(self, "kws"):
            raise RuntimeError("KWS was used before kws.load()")

        chunk_np = chunk_np.squeeze(1) if chunk_np.ndim > 1 else chunk_np
        self.stream.accept_waveform(cfg.audio.sample_rate, chunk_np)
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            result = self.kws.get_result(self.stream)
            if result:
                keyword = result.strip()
                self.reset()
                return keyword
        return None

    def reset(self):
        if not hasattr(self, "kws"):
            raise RuntimeError("KWS was used before kws.load()")
        self.stream = self.kws.create_stream()


if _MAIN:
    print("Testing 'KWS' module...")
