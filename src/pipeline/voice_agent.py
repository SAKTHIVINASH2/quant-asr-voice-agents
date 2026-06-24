"""Voice agent orchestrator: ASR -> LLM -> TTS, fully instrumented.

This is the shared backbone for all three experiments. It records a per-turn
latency breakdown so RQ2 can answer "where does the time go?" directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.asr import ASR
from src.pipeline.llm import LLM, GenResult
from src.pipeline.tts import TTS, TTSResult
from src.utils.timing import Timer


@dataclass
class TurnResult:
    transcript: str
    response: str
    # latency breakdown (seconds)
    asr_s: float
    llm_ttft_s: float
    llm_decode_s: float
    tts_ttfa_s: float
    tts_total_s: float
    total_s: float
    # derived perceptual metrics
    ttft_s: float            # time-to-first-token (ASR + LLM TTFT)
    ttfa_s: float            # time-to-first-audio (ASR + LLM-first-sentence + TTS TTFA)
    gen: GenResult = field(repr=False, default=None)

    def as_row(self) -> dict:
        return {
            "asr_s": self.asr_s,
            "llm_ttft_s": self.llm_ttft_s,
            "llm_decode_s": self.llm_decode_s,
            "tts_ttfa_s": self.tts_ttfa_s,
            "tts_total_s": self.tts_total_s,
            "total_s": self.total_s,
            "ttft_s": self.ttft_s,
            "ttfa_s": self.ttfa_s,
            "n_tokens": self.gen.n_tokens if self.gen else None,
        }


class VoiceAgent:
    def __init__(self, asr: ASR, llm: LLM, tts: TTS,
                 system_prompt: str = "You are a helpful voice assistant. "
                                      "Answer concisely.") -> None:
        self.asr, self.llm, self.tts = asr, llm, tts
        self.system_prompt = system_prompt

    def build_prompt(self, transcript: str, history: list[tuple[str, str]] | None = None) -> str:
        # Minimal chat formatting; swap for the model's chat template when using
        # a real LLM (TODO: use llama_cpp create_chat_completion for correct template).
        lines = [self.system_prompt]
        for u, a in (history or []):
            lines += [f"User: {u}", f"Assistant: {a}"]
        lines += [f"User: {transcript}", "Assistant:"]
        return "\n".join(lines)

    def run_turn(self, audio_path: str, history=None,
                 max_tokens: int = 128) -> TurnResult:
        timer = Timer()

        with timer.span("asr"):
            transcript = self.asr.transcribe(audio_path)

        prompt = self.build_prompt(transcript, history)

        # Full response (for quality eval) — TTFT/decode captured inside.
        gen = self.llm.generate(prompt, max_tokens=max_tokens)
        timer.mark("llm_decode", gen.decode_s)
        timer.mark("llm_ttft", gen.ttft_s)

        out_wav = audio_path + ".out.wav"
        tts_res: TTSResult = self.tts.synthesize(gen.text, out_wav)
        timer.mark("tts_total", tts_res.total_s)
        timer.mark("tts_ttfa", tts_res.ttfa_s)

        asr_s = timer.get("asr") or 0.0
        # TTFA in a streaming agent = ASR + LLM-time-to-first-sentence + TTS-TTFA.
        # We approximate LLM-first-sentence by TTFT here; rq2 can measure it exactly
        # via llm.first_sentence(). (TODO: wire streaming LLM->TTS for true TTFA.)
        ttft_s = asr_s + gen.ttft_s
        ttfa_s = asr_s + gen.ttft_s + tts_res.ttfa_s

        return TurnResult(
            transcript=transcript, response=gen.text,
            asr_s=asr_s, llm_ttft_s=gen.ttft_s, llm_decode_s=gen.decode_s,
            tts_ttfa_s=tts_res.ttfa_s, tts_total_s=tts_res.total_s,
            total_s=timer.total, ttft_s=ttft_s, ttfa_s=ttfa_s, gen=gen,
        )

    def close(self) -> None:
        for c in (self.asr, self.llm, self.tts):
            c.close()
