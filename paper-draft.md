# Quantization and ASR Noise in On-Device Voice Agents: Independent, Not Compounding

**B. Sakthi Vinash**
Draft v0.1 — 2026-06-26

> Status: working draft. Numbers are from Qwen2.5-3B-Instruct (GGUF, llama.cpp)
> on an 8-core commodity CPU. `[TODO]` marks gaps to fill before submission.

---

## Abstract

On-device voice assistants increasingly pair a small quantized LLM with streaming
ASR and TTS, yet two deployment fears are largely untested *together*: that
aggressive quantization cripples the agent, and that quantization and noisy ASR
input compound into a quality cliff. We measure both inside a CPU ASR→LLM→TTS
pipeline. We find (1) the LLM stage dominates end-to-end latency (76–84%) and no
quantization level meets a 1-second conversational budget — and aggressive
quantization yields no significant *latency* benefit on CPU, where thermal
throttling dominates bit-width; and (2) on tool-calling (BFCL, n=400), 2-bit
quantization is statistically indistinguishable from 5-bit on both clean and
ASR-noised input — there is no quantization main effect and no quantization×noise
interaction. ASR noise alone roughly halves argument slot-filling accuracy,
uniformly across quantization levels, while function selection and JSON-format
fidelity remain robust. The practical implication is that on-device voice agents
can be quantized aggressively with negligible tool-calling cost; the binding
constraints are latency and ASR quality, not weight precision.

---

## 1. Introduction

[TODO: 3 paragraphs]
- Voice agents are moving on-device for privacy/latency/cost; the standard stack is
  streaming ASR → small LLM → streaming TTS on constrained hardware.
- Two intuitions drive deployment decisions: "quantize to go fast/fit" and the fear
  that "aggressive quantization + noisy speech will compound into failure." Neither
  is measured *inside a voice pipeline on CPU*.
- We test both. Contributions:
  1. A CPU latency decomposition showing the LLM dominates and quantization does
     not buy meaningful latency (RQ2).
  2. The first measurement of quantization × ASR-noise *interaction* on tool-calling
     in a voice context, showing the two are independent (RQ1).
  3. An open, reproducible harness (dummy-mode + GGUF) for on-device voice-agent
     evaluation.

## 2. Related Work

[Pull from literature-review.md §1–5; key anchors below]
- **Speculative decoding / latency:** PredGen (TTFS framing), DuoDecoding (Q5_K_M CPU
  draft), AdaSpec. We differ: fully on-CPU, latency *decomposition* not acceleration.
- **Quantization degradation:** Lee et al. 2409.11055 (IFEval/multi-turn sensitive),
  Dutta 2407.09141 ("accuracy is not all you need"; flips), Song 2505.15030 (4-bit
  sweet spot, sub-4-bit collapse), Kurt 2601.14277 (GGUF K-quants). All text-only.
- **Tool-calling under compression:** ACBench 2505.19433 (4-bit keeps tool-use 1–3%
  but real-world agentic −10–15%), all clean text, GPU. BFCL benchmark.
- **Gap we fill:** none evaluate quantization (a) inside a voice/ASR pipeline,
  (b) on tool-calling under ASR noise, (c) for the quant×noise interaction, (d) on CPU.

## 3. Method

### 3.1 Pipeline
Streaming ASR (faster-whisper) → LLM (Qwen2.5-3B-Instruct, GGUF via llama.cpp,
CPU-only) → TTS (Piper / SAPI5). Per-stage timing instrumentation captures
time-to-first-token, time-to-first-audio, and per-stage latency. [TODO: hardware
spec table — 8-core CPU, RAM, OS.]

### 3.2 Quantization levels
GGUF K-quants: Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (≈5.5 → 2.6 effective bits/weight).

### 3.3 RQ2 — latency
31 representative voice-assistant utterances; pinned response length
(max_tokens=100) for cross-quant comparability; decode speed reported as pure
inter-token tok/s (prefill excluded); interleaved trials with ±std to control for
laptop thermal throttling.

### 3.4 RQ1 — quality under noise
Tool-calling on **BFCL v3 `simple`** (function selection + AST argument match) with
each task's own tool catalog in the prompt; greedy decoding (temperature 0) for
determinism. **ASR noise** injected as controlled word-level corruption at target
WER (0.0 = clean, 0.30 = heavy). Metrics: `name_match` (function selection),
`valid_json` (format), `exact` (function + all arguments). [TODO: note synthetic-
noise limitation + phonetic-noise future work.]

## 4. Results

### 4.1 RQ2 — the LLM dominates and quantization doesn't help latency

Per-stage latency (Qwen2.5-3B, CPU), mean across utterances:

| Stage | share |
|---|---|
| ASR | ~18% |
| **LLM decode** | **76–84% (dominant)** |
| TTS | ~2–3% |

Every quantization level exceeds the 1s conversational budget (TTFA ~1.9–2.8s).
Decode speed across quants (pure tok/s, interleaved, ±std):

| Q5_K_M | Q4_K_M | Q3_K_M | Q2_K |
|---|---|---|---|
| 10.0±1.7 | 12.0±1.6 | 11.1±1.0 | 11.8±0.9 |

All ±std bands overlap; the same model varies more by thermal position (Q5_K_M:
10.0 tok/s run first vs 7.3 run last) than quants differ from each other. **No
significant latency benefit from aggressive quantization on CPU.**

### 4.2 RQ1 — no quantization×noise interaction (n=400 BFCL)

| quant | name_match (clean→noisy) | exact clean | exact noisy | Δ(noise) |
|---|---|---|---|---|
| Q5_K_M | 0.96 → 0.96 | 0.58 | 0.29 | 0.28 |
| Q2_K | 0.96 → 0.95 | 0.55 | 0.28 | 0.27 |

Q2_K and Q5_K_M are statistically identical on every measure (differences ~0.01,
within the ±0.03 noise floor at n=400). **No quantization main effect, no
quant×noise interaction.** ASR noise alone roughly halves `exact` slot-filling
(0.58→0.29) — uniformly across quant levels. `name_match` and `valid_json` stay
~0.96 throughout. [TODO: add the 4-quant n=80 table + the methodological note that
its apparent monotonic Δ was a sampling artifact — good rigor anecdote for
discussion.]

## 5. Discussion

- **Quantization is cheap for tool-calling quality, useless for CPU latency.** The
  two deployment levers point opposite ways from intuition.
- **ASR noise, not weight precision, is the quality bottleneck** — and it hits
  argument slot-filling specifically (entity recovery), not function selection.
  Implication: invest in ASR quality / noise-robust slot extraction, not higher
  bit-width.
- **Methodological:** decode-latency benchmarks of quantized LLMs on laptops must
  control thermal throttling; quality metrics must use greedy decoding and decompose
  Δ into clean/noisy components (an apparent n=80 interaction vanished at n=400).

## 6. Limitations

Single model family (Qwen2.5-3B); single CPU; synthetic (not phonetic/real-ASR)
noise; BFCL `simple` (single-turn, small per-task catalogs make `name_match`
saturate); latency numbers from a thermally-variable laptop. [TODO: which to
address pre-submission — second model (Llama-3.2-3B) is the highest value.]

## 7. Conclusion

On-device voice agents can be quantized aggressively (to 2-bit) with negligible
tool-calling cost, clean or noisy; quantization and ASR noise degrade quality
through independent channels, not a compounding cliff. The binding constraints are
latency (LLM-dominated, over budget, not helped by quantization on CPU) and ASR
quality (the true driver of slot-filling errors).

---

## Appendix / repro
Harness: `src/` (pipeline, eval, experiments). Run RQ2: `scripts/quant_sweep.py`;
RQ1: `python -m src.experiments.rq1_quality --dataset bfcl ...`. All results in
`results/`. [TODO: release repo + commit hash.]
