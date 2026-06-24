"""TTS stage. Sentence-by-sentence synthesis with time-to-first-audio capture.

Interface synthesizes text to a wav file and reports TTFA (time until the first
sentence's audio is ready) — the metric that matters for perceived voice latency.
"""
from __future__ import annotations

import re
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.utils.timing import now

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


@dataclass
class TTSResult:
    wav_path: str | None
    ttfa_s: float       # time to first sentence's audio
    total_s: float      # time to synthesize everything
    n_sentences: int


class TTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, out_path: str) -> TTSResult: ...

    def close(self) -> None:
        pass


class DummyTTS(TTS):
    """Sleeps proportional to text length per sentence; writes a tiny silent wav
    so the ASR-noise round-trip has a real file to read (with a .txt sidecar)."""

    def __init__(self, per_char_s: float = 0.002, base_s: float = 0.05) -> None:
        self.per_char_s = per_char_s
        self.base_s = base_s

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        t0 = now()
        sents = split_sentences(text) or [text]
        ttfa = None
        for s in sents:
            time.sleep(self.base_s + self.per_char_s * len(s))
            if ttfa is None:
                ttfa = now() - t0
        _write_silence(out_path, seconds=max(0.1, 0.05 * len(sents)))
        with open(f"{out_path}.txt", "w", encoding="utf-8") as fh:
            fh.write(text)  # sidecar so DummyASR can round-trip the text
        return TTSResult(out_path, ttfa_s=ttfa or 0.0, total_s=now() - t0,
                         n_sentences=len(sents))


class Pyttsx3TTS(TTS):
    """Windows-friendly TTS via the OS speech engine (SAPI5 on Windows).

    Offline, no special wheels — the robust choice on native Windows where Piper's
    phonemizer wheels are flaky. Note: pyttsx3 does not stream, so TTFA is
    APPROXIMATED from the first sentence's text fraction (documented limitation).
    total_s is measured honestly (single synthesis pass)."""

    def __init__(self) -> None:
        import pyttsx3  # lazy import
        self.engine = pyttsx3.init()

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        t0 = now()
        sents = split_sentences(text) or [text]
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()
        total = now() - t0
        # Approximate TTFA: pyttsx3 can't stream, so scale by first-sentence length.
        frac = len(sents[0]) / max(1, len(text))
        return TTSResult(out_path, ttfa_s=total * frac, total_s=total,
                         n_sentences=len(sents))


class PiperTTS(TTS):
    def __init__(self, voice_path: str) -> None:
        from piper.voice import PiperVoice  # lazy import
        self.voice = PiperVoice.load(voice_path)

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        import wave as _wave
        t0 = now()
        sents = split_sentences(text) or [text]
        ttfa = None
        with _wave.open(out_path, "wb") as wf:
            for i, s in enumerate(sents):
                self.voice.synthesize(s, wf)
                if ttfa is None:
                    ttfa = now() - t0
        return TTSResult(out_path, ttfa_s=ttfa or 0.0, total_s=now() - t0,
                         n_sentences=len(sents))


def _write_silence(path: str, seconds: float = 0.2, rate: int = 16000) -> None:
    n = int(seconds * rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n)


def build_tts(dummy: bool, voice_path: str | None = None,
              backend: str | None = None) -> TTS:
    """backend: 'piper' (default, Linux/CPU), 'pyttsx3' (Windows SAPI5), or
    'dummy'. Falls back to env VA_TTS_BACKEND. dummy=True overrides all."""
    import os
    if dummy:
        return DummyTTS()
    backend = backend or os.environ.get("VA_TTS_BACKEND", "piper")
    if backend == "pyttsx3":
        return Pyttsx3TTS()
    if backend == "dummy":
        return DummyTTS()
    from src import config
    vp = voice_path or str(config.PIPER_VOICE)
    return PiperTTS(vp)
