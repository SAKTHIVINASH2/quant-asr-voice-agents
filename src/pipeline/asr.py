"""ASR stage. Common interface + faster-whisper impl + dummy impl.

Returns transcript text and leaves timing to the caller (voice_agent).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class ASR(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe a wav file to text."""

    def close(self) -> None:  # optional cleanup
        pass


class DummyASR(ASR):
    """No model. Sleeps a realistic CPU latency and echoes a canned transcript.

    Reads a sidecar `<audio_path>.txt` if present (used by the ASR-noise round-trip
    to carry the 'spoken' text through without a real model)."""

    def __init__(self, latency_s: float = 0.10) -> None:
        self.latency_s = latency_s

    def transcribe(self, audio_path: str) -> str:
        time.sleep(self.latency_s)
        sidecar = f"{audio_path}.txt"
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return "this is a dummy transcript"


class FasterWhisperASR(ASR):
    def __init__(self, model_size: str = "small", compute_type: str = "int8",
                 device: str = "cpu") -> None:
        from faster_whisper import WhisperModel  # lazy import
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self.model.transcribe(audio_path, beam_size=1)
        return "".join(seg.text for seg in segments).strip()


def build_asr(dummy: bool, **kw) -> ASR:
    if dummy:
        return DummyASR()
    from src import config
    return FasterWhisperASR(
        model_size=kw.get("model_size", config.ASR_MODEL),
        compute_type=kw.get("compute_type", config.ASR_COMPUTE_TYPE),
    )
