#
# op / global_operator.py
# Global Operator: processes and operates all commands, that come from STT
#

from __future__ import annotations

import queue
import re
import threading
from typing import TYPE_CHECKING

from ..core.events import EventType, emit_event

if TYPE_CHECKING:
    from .cmd_operator import CommandOperator
    from .llama import Llama


class Operator:
    def __init__(self, cmd: CommandOperator, llm: Llama) -> None:
        self._running = False
        self.cmd = cmd
        self.llm = llm
        self.command_queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._operator_worker, name="OPERATOR_THREAD", daemon=True
        )

        self.interrupt_flag = threading.Event()

    def start(self):
        self._running = True
        self.worker_thread.start()

    def close(self):
        self._running = False
        self.command_queue.put(None)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.llm.close()

    def submit(self, text: str):
        self.command_queue.put(text)

    def interrupt(self):
        self.interrupt_flag.set()

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

    def _stream_llm_response(self, text: str):
        full_response_text = ""

        self.interrupt_flag.clear()
        token_stream = self.llm.stream_response(text)

        is_first_chunk = True
        for sentence in self._sentence_chunker(token_stream):
            if self.interrupt_flag.is_set():  # Interruption
                break

            full_response_text += sentence + " "

            emit_event(EventType.OP_LLM_CHUNK, {"text": sentence})

            emit_event(
                EventType.UI_LLM_CHUNK,
                {"text": sentence, "is_first": is_first_chunk},
            )
            is_first_chunk = False

        emit_event(
            EventType.UI_LLM_RESPONSE_DONE,
            {
                "text": full_response_text.strip(),
            },
        )
        emit_event(EventType.LLM_RESPONSE, {"text": full_response_text.strip()})
        self.llm.history_add_response(full_response_text.strip())

    def _operate(self, text: str) -> None:
        if not text:
            return

        emit_event(EventType.OP_START, {})
        res_type = self.cmd.operate(text)

        if not res_type:  # LLM
            if self.llm.no_model:  # LLM model was not load for some reason
                emit_event(EventType.OP_INTENT, {"intent": "idk_cmd"})
            else:
                self._stream_llm_response(text)

        emit_event(EventType.OP_FINISH, {})
