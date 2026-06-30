# Research Proposal

## When Voice Agents Get Cheap: How Quantization and Latency-Optimization Break Down in On-Device Spoken Dialogue

**Author:** B. Sakthi Vinash
**Date:** 2026-06-24
**Status:** Draft v1

---

## 1. One-sentence pitch

Every team shipping an on-device voice agent has to pick a quantization level and a latency-optimization strategy — but nobody has measured *what actually breaks* (tool-calling, multi-turn coherence, time-to-first-token) when you do this **inside a real ASR → LLM → TTS loop on CPU**, because all existing evidence is text-only. This paper measures it.

---

## 2. Motivation & gap (what the literature scan established)

The intersection of **quantization × low-latency inference × spoken dialogue** is an explicit, verified open gap. Specifically:

- Quantization-degradation studies (IFEval, MT-Bench multi-turn, tool-calling via ACBench) are **all text-only** — never run inside a voice pipeline where turns are shorter and ASR injects noise.
- Latency-aware speculative decoding for voice exists (**PredGen**, COLM 2025) but assumes a full-precision draft model; **draft-model quantization on CPU is unmeasured.**
- A claim that a full ASR→4-bit-LLM→TTS pipeline hits 0.94s with TTS dominating latency was **refuted (0–3)** during fact-checking — so the latency split is *not* established and is itself worth measuring.

The unfair advantage: this requires understanding **both** speech UX (latency budgets, turn-taking, ASR noise) **and** LLM inference internals. Few researchers have both.

---

## 3. Research questions

- **RQ1 (core).** At which quantization level (FP16 → Q5 → Q4 → Q3 → Q2) does *tool-calling* and *multi-turn coherence* degrade, and is the degradation **worse inside a voice pipeline** (with ASR transcription noise) than on clean text input?
- **RQ2.** Where does latency actually go in an on-device ASR→LLM→TTS loop on CPU — is the LLM stage or the TTS stage the bottleneck? (Directly tests the refuted claim.)
- **RQ3.** When you quantize the **draft model** for speculative decoding, how do acceptance rate and net time-to-first-token change on CPU? Is there a quant level that maximizes net speedup?

---

## 4. Closest prior work (cite + differentiate)

| Paper | Venue | What it does | How we differ |
|---|---|---|---|
| PredGen (arXiv 2506.15556) | COLM 2025 | Input-time speculation, optimizes TTFS | We add **draft-model quantization** + CPU setting |
| ACBench (arXiv 2505.19433) | 2025 | 4-bit degrades agentic accuracy 10–15% | We measure it **inside a voice loop**, not text |
| Quant. vulnerability (arXiv 2409.11055) | IJCAI 2025 | IFEval/multi-turn most quant-sensitive | We add **ASR-noise interaction** + tool-calling |
| llama.cpp K-quant study (arXiv 2601.14277) | Jan 2026 | IFEval across K-quants; **no tool-calling** | We **add tool-calling** + voice context |
| SpecASR (arXiv 2507.18181) | DAC 2025 | Task-aware spec-decoding for ASR | We target the **LLM turn**, not ASR |
| Gatekeeper / cascade-routing (2502.19335 / 2410.10347) | ICML/ICLR 2025 | Learned deferral, text-only | Out of scope v1; possible extension |

**Novelty statement:** First systematic measurement of quantization's effect on tool-calling and multi-turn coherence *within a spoken-dialogue pipeline*, plus the first characterization of draft-model quantization for CPU speculative decoding in a voice latency budget.

**Novelty CONFIRMED via full-paper deep-read of the two closest priors (2026-06-24):**
- *vs ACBench (2505.19433, RQ1):* ACBench is 100% clean text, only 4-bit+ GPTQ/AWQ on vLLM (no GGUF, no sub-4-bit results, no CPU), uses T-Eval/WorfBench/AgentBoard (not BFCL/MT-Bench/IFEval), and does not isolate multi-turn coherence. **Position:** "ACBench establishes that 4-bit quantization preserves clean-text tool-use (1–3%) while degrading real-world agentic execution 10–15% on GPU; we extend it in three directions it does not address — GGUF/sub-4-bit on CPU, multi-turn tool-calling coherence, and the interaction with noisy ASR-transcribed input in a voice pipeline." Do NOT claim "first to study compression + agentic skills" (ACBench owns that).
- *vs DuoDecoding (2503.00784, RQ3):* draft on CPU but target mandatorily on GPU (A800), draft fixed at Q5_K_M (never swept), acceptance rate never reported, zero voice framing. **Position:** "Unlike DuoDecoding, which fixes the CPU draft at a single Q5_K_M quantization and requires the target on GPU, we sweep draft bit-width (Q8_0–Q2_K), report the acceptance-rate/latency tradeoff, and run a fully on-device CPU pipeline under voice time-to-first-sentence budgets."
- **Hardest-to-scoop core (lean the framing here):** the **ASR-noise × quantization interaction** in a voice pipeline. Treat CPU / GGUF / sub-4-bit as supporting deltas. For RQ3, foreground the *both-on-CPU* constraint and *time-to-first-sentence* metric (the deltas DuoDecoding's GPU-target framework cannot reach by simple extension).

---

## 5. Method / experimental design

### 5.1 Models (all GGUF, CPU via llama.cpp)
- **Target LLMs:** Qwen2.5-7B-Instruct, Llama-3.2-3B-Instruct (both strong tool-callers, small enough for CPU).
- **Draft models (for RQ3):** Qwen2.5-0.5B / 1.5B-Instruct.
- **Quant levels:** FP16 (baseline), Q8_0, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K — using llama.cpp K-quants.

### 5.2 Pipeline
- **ASR:** `faster-whisper` (small/base) — CPU, streaming.
- **LLM:** llama.cpp server, one quant level at a time.
- **TTS:** Piper (fast CPU TTS) or Kokoro — sentence-by-sentence streaming.
- Instrument every stage boundary with timestamps.

### 5.3 Datasets / tasks
- **Tool-calling:** Berkeley Function-Calling Leaderboard (BFCL) — subset, multi-turn category.
- **Multi-turn coherence:** MT-Bench (use turn-1 vs turn-2 deltas, the established sensitive metric).
- **Spoken-dialogue realism:** synthesize spoken versions (TTS the user prompts → re-transcribe with Whisper) to inject **realistic ASR noise**, then feed both clean-text and ASR-noised versions. The *delta between them* is a core contribution.

### 5.4 Metrics
- **Quality:** BFCL tool-call accuracy (AST + executable), MT-Bench turn-1/turn-2 scores (LLM-as-judge), instruction-following (IFEval subset).
- **Latency:** time-to-first-token (TTFT), time-to-first-audio (TTFA), inter-token latency, per-stage breakdown (ASR/LLM/TTS).
- **Spec-decoding (RQ3):** token acceptance rate, net TTFT speedup vs draft quant level.

### 5.5 Baselines / conditions
- FP16 text (gold) → each quant level on clean text → each quant level on ASR-noised input.
- Spec-decoding: FP16 draft vs Q8/Q4/Q2 draft, acceptance × speedup curves.

---

## 6. Expected contributions

1. A **"what breaks first" curve** for voice agents across quant levels — actionable deployment guidance.
2. Evidence on whether **ASR noise amplifies or masks** quantization degradation (novel interaction effect).
3. The first **draft-quantization acceptance/speedup characterization** for CPU speculative decoding.
4. An **open-source benchmark harness** (reproducibility = citations).

---

## 7. Feasibility (your constraints: CPU / Colab)

- Everything runs on CPU — GGUF/llama.cpp is built for it. **No GPU training required.**
- LLM-as-judge (MT-Bench) can use an API (cheap, small volume).
- Estimated compute: a laptop CPU or free Colab is sufficient for inference sweeps; the only cost is wall-clock time.
- **Risk:** running the full sweep is slow on CPU → mitigate by scripting overnight batch runs and caching.

---

## 8. Target venues & timeline

| Venue | Type | Typical deadline | Fit |
|---|---|---|---|
| **Interspeech 2027** | Speech | ~Feb/Mar 2027 | Strong — voice + latency |
| **EMNLP 2026 (Findings)** | NLP | ~May 2026 *(likely passed)* / industry track | Strong — eval/benchmark |
| **ICASSP 2027** | Signal/speech | ~Sep 2026 | Good — on-device |
| **arXiv preprint first** | — | anytime | **Do this immediately** for the product-edge / visibility goal |

**Suggested plan:** arXiv preprint as soon as RQ1+RQ2 results are solid (gets you the product-edge visibility fast), then extend with RQ3 for a conference submission.

### Rough schedule
- **Weeks 1–2:** Build the ASR→LLM→TTS harness; reproduce a baseline latency split (RQ2).
- **Weeks 3–5:** Quant sweep on BFCL + MT-Bench, clean text (RQ1 part A).
- **Weeks 6–7:** ASR-noise injection + delta analysis (RQ1 part B — the novel bit).
- **Weeks 8–9:** Draft-quantization spec-decoding experiments (RQ3).
- **Weeks 10–12:** Write-up, arXiv, polish for venue.

---

## 9. Scope discipline (what to cut if time-constrained)

- **Minimum viable paper:** RQ1 + RQ2 only (quantization × voice pipeline). Drop spec-decoding.
- **Full paper:** add RQ3.
- **Do NOT** add routing/cascades (Direction 3) in v1 — it's a separate paper.

---

## 10. Open questions to resolve before starting

- Confirm BFCL multi-turn subset size is tractable on CPU.
- Pick the LLM-as-judge model (cost vs reliability).
- Decide whether to include a 2-bit condition (likely collapses — but the collapse point is interesting).

---

## Senior-reviewer analysis (deep-research, 2026-06-30)

**Verdict: workshop-ready now; one experiment from main-conference (Interspeech/ICASSP).**

- **Novelty CONFIRMED un-scooped:** no 2024–26 paper measures quant-level × ASR-noise
  interaction on tool-calling. Closest: ICINCO 2025 (Murgia, faster-whisper→4-bit GGUF
  correction, single 4-bit, no sweep, no tools); arXiv 2605.15104 (voice tool-calling,
  zero quantization); text-only quant×perturbation (Nature s41598-025-29770-0,
  OpenReview QhkW8xPH1v).
- **Top reviewer attacks:**
  1. Synthetic word-noise is a coarse proxy — field prefers REAL-ASR-grounded:
     MEDSAGE (arXiv 2408.14418, LLM-ICL ASR errors from real audio) and ICASSP-2024
     lattice/word-confusion-network inputs (arXiv 2401.02921). THE decisive fix.
  2. RQ2 "no quant speedup" tensions with arXiv 2505.06461 (Q4 DOES speed up CPU) +
     ARM iPhone Q4 1.5-2.5x (memory-bound) + Kurt 2601.14277 (low-bit faster on CPU).
     Null is x86-specific, valid only for Q5→Q2 PLATEAU. FIX: add Q8_0 to sweep to show
     Q8-slow→Q4-fast→Q2-plateau curve (reconciles). Mechanism (dequant overhead, T-MAC
     arXiv 2407.00088) established but "2-bit slower on x86" datapoints unverified.
  3. Robustness×compression literature is SPLIT (many find quantized MORE robust) —
     frame null as a voice-domain datapoint, NOT settled consensus.
- **RQ2 latency direction corroborated** (telecom H100 arXiv 2508.04721: LLM 72%), but
  GPU + meets budget; CPU over-budget regime is the novel part.
- **Prioritized to-do:** (1) real-ASR-grounded noise = paper-v2, decisive for main conf;
  (2) add Q8_0 to RQ2; (3) one scale axis (7B or AWQ/GPTQ); (4) reframe robustness null.
