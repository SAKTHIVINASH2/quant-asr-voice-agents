"""ASR-noise injection — the core novel manipulation for RQ1.

Take clean text -> synthesize speech (TTS) -> re-transcribe (ASR) -> noisy text.
The delta between model behaviour on clean vs. noisy text, as a function of
quantization level, is RQ1's headline contribution. No prior work does this.

Two modes:
  - 'roundtrip': real TTS->ASR (needs real models; most faithful noise).
  - 'synthetic': inject controlled character/word errors at a target WER
    (model-free; lets you sweep noise level precisely and run on CPU instantly).
"""
from __future__ import annotations

import random
import string
import tempfile
from dataclasses import dataclass

from src.eval.metrics import wer


@dataclass
class NoisedText:
    clean: str
    noisy: str
    wer: float
    mode: str


def roundtrip_noise(text: str, tts, asr, tmp_dir: str | None = None) -> NoisedText:
    """Real TTS->ASR round-trip. `tts` and `asr` are pipeline components."""
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir, delete=False) as f:
        wav = f.name
    tts.synthesize(text, wav)
    noisy = asr.transcribe(wav)
    return NoisedText(text, noisy, wer(text, noisy), "roundtrip")


def synthetic_noise(text: str, target_wer: float = 0.10,
                    rng: random.Random | None = None) -> NoisedText:
    """Model-free controlled corruption to a target WER. Useful for a clean
    noise-level sweep (e.g. WER in {0, 0.05, 0.10, 0.20}) without TTS/ASR."""
    rng = rng or random.Random(0)
    words = text.split()
    if not words:
        return NoisedText(text, text, 0.0, "synthetic")
    n_edits = max(0, round(target_wer * len(words)))
    idxs = rng.sample(range(len(words)), min(n_edits, len(words)))
    for i in idxs:
        op = rng.choice(["sub", "del", "homophone"])
        if op == "del":
            words[i] = ""
        elif op == "homophone":
            words[i] = _homophone(words[i], rng)
        else:  # substitution: perturb a couple of characters
            words[i] = _char_perturb(words[i], rng)
    noisy = " ".join(w for w in words if w)
    return NoisedText(text, noisy, wer(text, noisy), "synthetic")


# A few common ASR confusions; extend with a real confusion set if desired.
_HOMOPHONES = {
    "to": "two", "two": "to", "their": "there", "there": "their",
    "your": "you're", "no": "know", "know": "no", "buy": "by",
    "for": "four", "won": "one", "here": "hear", "write": "right",
}


def _homophone(w: str, rng: random.Random) -> str:
    key = w.lower().strip(string.punctuation)
    return _HOMOPHONES.get(key, _char_perturb(w, rng))


def _char_perturb(w: str, rng: random.Random) -> str:
    if len(w) < 2:
        return w
    i = rng.randrange(len(w))
    repl = rng.choice(string.ascii_lowercase)
    return w[:i] + repl + w[i + 1:]
