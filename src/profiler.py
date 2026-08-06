# profiler.py
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import psutil

_process = psutil.Process(os.getpid())
_process.cpu_percent(interval=None)


@dataclass
class StateMetrics:
    cpu_samples: list[float] = field(default_factory=list)
    ram_peak_mb: float = 0.0
    duration_sec: float = 0.0


class ResourceProfiler:
    def __init__(self, sample_interval: float = 0.5):
        self.interval = sample_interval
        self.current_state = "SLEEPING"
        self.metrics: dict[str, StateMetrics] = defaultdict(StateMetrics)
        self._running = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def get_instant_stats() -> tuple[float, float]:
        cpu = _process.cpu_percent(interval=None)
        ram_mb = _process.memory_info().rss / (1024 * 1024)
        return cpu, ram_mb

    def set_state(self, new_state: str):
        self.current_state = new_state

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor_loop(self):
        while self._running:
            cpu, ram_mb = self.get_instant_stats()
            state_data = self.metrics[self.current_state]

            state_data.cpu_samples.append(cpu)
            state_data.ram_peak_mb = max(ram_mb, state_data.ram_peak_mb)
            state_data.duration_sec += self.interval  # <--- Оновлення часу

            time.sleep(self.interval)

    def get_summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for state, data in self.metrics.items():
            avg_cpu = (
                sum(data.cpu_samples) / len(data.cpu_samples)
                if data.cpu_samples
                else 0.0
            )
            summary[state] = {
                "avg_cpu": round(avg_cpu, 1),
                "peak_ram_mb": round(data.ram_peak_mb, 1),
                "duration_sec": round(data.duration_sec, 1),
            }
        return summary


profiler = ResourceProfiler()
