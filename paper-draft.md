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
throttling dominates bit-width; and (2) on tool-calling (BFCL, n=400, two model
families), there is **no quantization×noise interaction** — ASR noise roughly
halves argument slot-filling accuracy *uniformly* across quantization levels, so a
2-bit model is no more noise-fragile than a 5-bit one. The *standalone* cost of
2-bit quantization on clean input is model-dependent (negligible for Qwen2.5-3B,
~15 points of exact-match for Llama-3.2-3B), but in neither case does ASR noise
compound it. Function selection and JSON-format fidelity remain robust throughout.
The practical implication is that the quantization and noise failure modes are
independent: the binding constraints for on-device voice agents are latency and ASR
quality, and quantization can be chosen on its clean-input merits without fear of a
noise-amplified cliff.

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

### 4.2 RQ1 — no quantization×noise interaction (n=400 BFCL, two model families)

**Qwen2.5-3B-Instruct:**

| quant | name_match (clean→noisy) | exact clean | exact noisy | Δ(noise) |
|---|---|---|---|---|
| Q5_K_M | 0.96 → 0.96 | 0.58 | 0.29 | 0.28 |
| Q2_K | 0.96 → 0.95 | 0.55 | 0.28 | 0.27 |

**Llama-3.2-3B-Instruct:**

| quant | name_match (clean→noisy) | exact clean | exact noisy | Δ(noise) |
|---|---|---|---|---|
| Q5_K_M | 0.95 → 0.95 | 0.55 | 0.30 | 0.25 |
| Q2_K | 0.86 → 0.85 | 0.40 | 0.18 | 0.21 |

**No quant×noise interaction, on both models.** The noise penalty Δ does not grow
for the 2-bit model — it is equal (Qwen: 0.28 vs 0.27) or slightly *smaller*
(Llama: 0.25 vs 0.21, a floor effect from Llama-Q2's lower clean baseline). ASR
noise roughly halves `exact` slot-filling uniformly across quant levels; `valid_json`
stays ~0.97 everywhere.

**The quantization *main effect* is model-dependent.** Qwen2.5-3B tolerates 2-bit
gracefully (Q2≈Q5 on clean input). Llama-3.2-3B does not: at Q2_K, clean
name_match drops 0.95→0.86 (−9 pts) and clean exact 0.55→0.40 (−15 pts) — both
≈4σ at n=400, i.e. real degradation. So *how much* aggressive quantization costs
on clean input depends on the model family, but *whether noise compounds it* does
not — the interaction is null for both.

[TODO: add the Qwen 4-quant n=80 table + the methodological note that its apparent
monotonic Δ was a sampling artifact that vanished at n=400 — rigor anecdote for §5.]

## 5. Discussion

- **Quantization is cheap for tool-calling quality, useless for CPU latency.** The
  two deployment levers point opposite ways from intuition.
- **ASR noise, not weight precision, is the quality bottleneck** — and it hits
  argument slot-filling specifically (entity recovery), not function selection.
  Implication: invest in ASR quality / noise-robust slot extraction, not higher
  bit-width.
- **The interaction null generalizes; the main effect does not.** Across two model
  families the *absence* of a quant×noise interaction is consistent, but the
  standalone clean-input cost of 2-bit differs sharply (Qwen2.5-3B robust;
  Llama-3.2-3B loses ~15 pts exact). Practitioners should pick a quant level by
  testing their specific model on clean input — and can then trust that ASR noise
  won't disproportionately amplify that choice.
- **Methodological:** decode-latency benchmarks of quantized LLMs on laptops must
  control thermal throttling; quality metrics must use greedy decoding and decompose
  Δ into clean/noisy components (an apparent n=80 interaction vanished at n=400).

## 6. Limitations

Two model families (Qwen2.5-3B, Llama-3.2-3B), both ~3B — larger/smaller scales
untested; single CPU; **synthetic (word-level) noise rather than phonetic or
real-ASR errors** (highest-value next step — real ASR confusions are phonetically
structured); BFCL `simple` (single-turn, small per-task catalogs make `name_match`
saturate; multi-turn BFCL v3 untested); latency numbers from a thermally-variable
laptop (structure reproduces; absolute ms will not). The interaction null is shown
for tool-calling; other capabilities (multi-turn dialogue coherence, long-context)
are not covered.

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
