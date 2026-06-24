"""Timing primitives. perf_counter-based; all durations in seconds (float)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def now() -> float:
    return time.perf_counter()


@dataclass
class Span:
    name: str
    elapsed: float


class Timer:
    """Accumulates named spans within one turn. Use .span() as a context manager."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._t0 = now()

    @contextmanager
    def span(self, name: str):
        start = now()
        try:
            yield
        finally:
            self.spans.append(Span(name, now() - start))

    def mark(self, name: str, elapsed: float) -> None:
        """Record a span whose duration was measured elsewhere (e.g. TTFT)."""
        self.spans.append(Span(name, elapsed))

    def get(self, name: str) -> float | None:
        for s in self.spans:
            if s.name == name:
                return s.elapsed
        return None

    @property
    def total(self) -> float:
        return now() - self._t0

    def as_dict(self) -> dict[str, float]:
        d = {s.name: s.elapsed for s in self.spans}
        d["total"] = self.total
        return d
