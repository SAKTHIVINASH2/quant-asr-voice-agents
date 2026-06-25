"""LLM stage. Streaming generation with explicit TTFT measurement.

Interface returns a GenResult carrying the text plus the measurements RQ2/RQ3 need:
time-to-first-token, total decode time, and token count.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.utils.timing import now


@dataclass
class GenResult:
    text: str
    ttft_s: float            # time to first token
    decode_s: float          # total generation wall time
    n_tokens: int
    # RQ3 only — populated by speculative-decoding backends, else None:
    n_draft: int | None = None
    n_accept: int | None = None

    @property
    def acceptance_rate(self) -> float | None:
        if self.n_draft and self.n_draft > 0 and self.n_accept is not None:
            return self.n_accept / self.n_draft
        return None

    @property
    def tokens_per_s(self) -> float:
        """Overall throughput incl. prefill — length/prompt dependent."""
        return self.n_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def decode_tps(self) -> float | None:
        """PURE inter-token decode speed (excludes prefill/TTFT). This is the
        length-independent metric for comparing quant levels — standard
        llama.cpp 'eval tok/s'."""
        gen_time = self.decode_s - self.ttft_s
        if self.n_tokens > 1 and gen_time > 0:
            return (self.n_tokens - 1) / gen_time
        return None


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None,
                 temperature: float = 0.0) -> GenResult: ...

    def first_sentence(self, prompt: str, max_tokens: int = 256) -> GenResult:
        """Generate only until the first sentence boundary (voice TTFS metric).
        Default impl stops at the first '. ! ?'."""
        return self.generate(prompt, max_tokens=max_tokens, stop=[". ", "! ", "? ", "\n"])

    def close(self) -> None:
        pass


class DummyLLM(LLM):
    """Emits canned tokens with a realistic per-token delay so TTFT/decode are real."""

    def __init__(self, tok_delay_s: float = 0.03, first_tok_s: float = 0.12) -> None:
        self.tok_delay_s = tok_delay_s
        self.first_tok_s = first_tok_s

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None,
                 temperature: float = 0.0) -> GenResult:
        canned = ("Sure, I can help with that. Let me check the details "
                  "and get back to you shortly.").split()
        t0 = now()
        time.sleep(self.first_tok_s)
        ttft = now() - t0
        out = []
        for i, w in enumerate(canned[:max_tokens]):
            if i > 0:
                time.sleep(self.tok_delay_s)
            out.append(w)
            if stop and any(s.strip() and (w.endswith(s.strip())) for s in stop):
                break
        return GenResult(" ".join(out), ttft_s=ttft, decode_s=now() - t0, n_tokens=len(out))


class LlamaCppLLM(LLM):
    """llama-cpp-python backend. Streams to capture TTFT.

    Pass a `draft_model` (a llama_cpp LlamaDraftModel) to enable speculative
    decoding for RQ3. NOTE: the Python bindings don't cleanly expose per-call
    n_draft/n_accept — see rq3_specdec.py for the recommended acceptance-rate
    measurement via the `llama-speculative` CLI."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int | None = None,
                 draft_model=None, verbose: bool = False) -> None:
        from llama_cpp import Llama  # lazy import
        self.llm = Llama(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
            n_gpu_layers=0,            # force CPU — this is the whole point
            draft_model=draft_model, verbose=verbose,
        )

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None,
                 temperature: float = 0.0) -> GenResult:
        t0 = now()
        ttft = None
        pieces: list[str] = []
        n = 0
        stream = self.llm.create_completion(
            prompt=prompt, max_tokens=max_tokens, stop=stop or [], stream=True,
            temperature=temperature,   # 0.0 = greedy/deterministic for eval
        )
        for chunk in stream:
            tok = chunk["choices"][0]["text"]
            if ttft is None:
                ttft = now() - t0
            pieces.append(tok)
            n += 1
        if ttft is None:
            ttft = now() - t0
        return GenResult("".join(pieces), ttft_s=ttft, decode_s=now() - t0, n_tokens=n)


def build_llm(dummy: bool, model_path: str | None = None, **kw) -> LLM:
    if dummy:
        return DummyLLM()
    if not model_path:
        raise ValueError("model_path required for real LLM (got None). "
                         "Fill src/config.py or pass --model-path.")
    return LlamaCppLLM(model_path=model_path, **kw)
