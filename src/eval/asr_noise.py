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


# --------------------------------------------------------------------------- #
# Phonetically-grounded noise (validity extension for RQ1)                    #
# --------------------------------------------------------------------------- #
# Real ASR errors are ACOUSTIC, not random: the recognizer confuses
# similar-sounding phonemes, drops final/unstressed consonants, and swaps
# reduced vowels. `synthetic_noise` uses random-character substitution, which a
# reviewer can dismiss as unrealistic. `phonetic_noise` instead perturbs each
# chosen word along acoustically-plausible axes while staying WER-controllable
# (the round-trip TTS->ASR path is faithful but its WER is whatever the pipeline
# yields, usually too mild to hit a 0.30 stress level). This is a GRAPHEME-level
# approximation of phonetic confusion (no G2P dependency) — motivated by, not a
# substitute for, a true phoneme confusion matrix.
_VOWELS = "aeiou"
# Bidirectional acoustically-similar grapheme confusions (voiced/unvoiced pairs,
# nasals, sibilants, liquids, and reduced-vowel swaps).
_PHON_CONFUSE = {
    "b": "pv", "p": "bf", "d": "t", "t": "d", "g": "kj", "k": "gc",
    "v": "fb", "f": "vp", "z": "s", "s": "zc", "m": "n", "n": "ml",
    "c": "ks", "j": "g", "l": "rn", "r": "l", "w": "v",
    "a": "eo", "e": "ia", "i": "ey", "o": "ua", "u": "o", "y": "i",
}


def _phonetic_perturb(w: str, rng: random.Random) -> str:
    """One acoustically-plausible edit: confuse a similar phoneme, swap a
    reduced vowel, or drop a final consonant. Preserves trailing punctuation
    and leading capitalization."""
    core, suffix = w, ""
    while core and core[-1] in string.punctuation:
        suffix, core = core[-1] + suffix, core[:-1]
    if len(core) < 2:
        return w
    cap = core[0].isupper()
    chars = list(core.lower())
    op = rng.choices(["sub", "final", "vowel"], weights=[0.5, 0.25, 0.25])[0]
    if op == "final" and len(chars) >= 3 and chars[-1] not in _VOWELS:
        chars.pop()                                   # dropped final consonant
    else:
        pool = _VOWELS if op == "vowel" else "".join(_PHON_CONFUSE)
        cand = [k for k, c in enumerate(chars) if c in pool and c in _PHON_CONFUSE]
        if not cand:
            cand = [k for k, c in enumerate(chars) if c in _PHON_CONFUSE]
        if cand:
            k = rng.choice(cand)
            chars[k] = rng.choice(_PHON_CONFUSE[chars[k]])
    out = "".join(chars)
    out = out.capitalize() if cap else out
    return out + suffix


def phonetic_noise(text: str, target_wer: float = 0.10,
                   rng: random.Random | None = None) -> NoisedText:
    """Acoustically-motivated corruption to a target WER: homophone swaps,
    phoneme confusions, and occasional function-word drops. Realistic *and*
    WER-controllable, unlike the round-trip path."""
    rng = rng or random.Random(0)
    words = text.split()
    if not words:
        return NoisedText(text, text, 0.0, "phonetic")
    n_edits = max(0, round(target_wer * len(words)))
    idxs = rng.sample(range(len(words)), min(n_edits, len(words)))
    for i in idxs:
        op = rng.choices(["phon", "homophone", "del"], weights=[0.7, 0.2, 0.1])[0]
        if op == "del":
            words[i] = ""                             # ASR drops a short word
        elif op == "homophone":
            key = words[i].lower().strip(string.punctuation)
            words[i] = _HOMOPHONES.get(key) or _phonetic_perturb(words[i], rng)
        else:
            words[i] = _phonetic_perturb(words[i], rng)
    noisy = " ".join(w for w in words if w)
    return NoisedText(text, noisy, wer(text, noisy), "phonetic")
