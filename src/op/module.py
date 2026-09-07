#
# op / module.py
#

# from __future__ import annotations

import queue
import re
import threading

from ..core.events import CommandType, EventManager, EventType
from ..core.module import Module, on_event
from .cmd_operator import CommandOperator
from .llama import Llama


class OpModule(Module):
    name = "op"

    def __init__(self, events: EventManager | None = None) -> None:
        self.events = events or EventManager()
        self._register_events(self.events)

        self._running = False
        self.cmd = CommandOperator(self.events)
        self.llm = Llama(self.events)
        self.command_queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._operator_worker, name="OPERATOR_THREAD", daemon=True
        )

        self.interrupt_flag = threading.Event()

    # Class methods

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

            self.events.emit(EventType.OP_LLM_CHUNK, {"text": sentence})

            self.events.emit(
                EventType.UI_LLM_CHUNK,
                {"text": sentence, "is_first": is_first_chunk},
            )
            is_first_chunk = False

        self.events.emit(
            EventType.UI_LLM_RESPONSE_DONE,
            {
                "text": full_response_text.strip(),
            },
        )
        self.events.emit(EventType.LLM_RESPONSE, {"text": full_response_text.strip()})
        self.llm.history_add_response(full_response_text.strip())

    def _operate(self, text: str) -> None:
        if not text:
            return

        self.events.emit(EventType.OP_START, {})
        res_type = self.cmd.operate(text)

        if not res_type:  # LLM
            if self.llm.no_model:  # LLM model was not load for some reason
                self.events.emit(EventType.OP_INTENT, {"intent": "idk_cmd"})
            else:
                self._stream_llm_response(text)

        self.events.emit(EventType.OP_FINISH, {})

    # Module methods

    def start(self):
        self._running = True
        self.worker_thread.start()

    def load(self):
        self.llm.load()
        self.cmd.load()

    def close(self):
        self._running = False
        self.command_queue.put(None)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.llm.close()

    # Events

    @on_event(CommandType.OP_SUBMIT, EventType.STT_TRANSCRIBED)
    def submit(self, text: str = "", **kwargs):
        self.command_queue.put(text)

    @on_event(EventType.OP_INTERRUPT)
    def interrupt(self, **kwargs):
        self.interrupt_flag.set()
