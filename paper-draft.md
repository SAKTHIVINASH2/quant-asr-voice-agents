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

Voice assistants are increasingly being pushed onto the device rather than the
cloud, motivated by privacy, offline availability, per-request cost, and the tight
latency budget of natural conversation, where the gap between turns in human
dialogue centers on roughly 200 ms and delays beyond about one second are perceived
as broken [Stivers et al., 2009]. The dominant architecture for such systems is a
cascade of streaming automatic speech recognition (ASR), a language model that
interprets the request and emits a response or a tool call, and streaming
text-to-speech (TTS), all running on constrained consumer hardware — frequently a
commodity CPU rather than a datacenter GPU. The enabling technology for fitting a
capable LLM into this setting is aggressive post-training quantization: GGUF
K-quants served by `llama.cpp` routinely compress a multi-billion-parameter model to
two-to-five effective bits per weight so that it runs at all on a laptop CPU.

Two intuitions drive how practitioners configure these systems, and neither has
been tested inside an actual voice pipeline. The first is that quantization is a
*latency* lever — that lower bit-widths make the agent fast enough for real-time
conversation. The second is a *fear*: aggressive quantization is known to degrade
LLM quality, and ASR delivers noisy, error-laden transcripts to the LLM, so the two
might *compound* into a quality cliff — a quantized model that copes with clean text
but collapses on the garbled input a real microphone produces. This compounding
fear is plausible precisely because the evidence base does not address it: studies
of quantization-induced degradation are almost exclusively conducted on clean,
written-text benchmarks and, typically, on GPUs [Lee et al., 2024; Dutta et al.,
2024; Song et al., 2025], while evaluations of tool-calling under compression are
likewise clean-text and GPU-bound [Dong et al., 2025]. Whether quantization and ASR
noise interact — whether low-bit models are *disproportionately* fragile to noisy
speech — has, to our knowledge, never been measured.

We measure both intuitions directly, inside a CPU ASR→LLM→TTS pipeline, asking:
**(RQ2)** Where does latency actually go on commodity CPU, and does aggressive
quantization meaningfully reduce it? **(RQ1)** Does aggressive quantization degrade
tool-calling *more* under ASR-noised input than under clean input — i.e., is there a
quantization×noise interaction?

Our findings contradict both intuitions. (1) The LLM stage dominates end-to-end
latency (76–84%), every quantization level we test exceeds the one-second
conversational budget, and aggressive quantization yields no statistically
significant latency benefit on CPU, where thermal throttling dominates bit-width.
(2) On tool-calling (BFCL, n=400, two model families), there is no quantization×noise
interaction: ASR noise roughly halves argument slot-filling accuracy *uniformly*
across quantization levels, so a 2-bit model is no more noise-fragile than a 5-bit
one. The *standalone* cost of 2-bit quantization on clean input is model-dependent
(negligible for Qwen2.5-3B, ~15 points for Llama-3.2-3B), but in neither case does
noise compound it. The practical consequence is that the two failure modes are
independent: practitioners can choose a quantization level on its clean-input merits
without fear of a noise-amplified cliff, while the binding constraints — latency and
ASR quality — lie elsewhere. We release an open, reproducible harness (with a
model-free "dummy" mode for pipeline validation) for on-device voice-agent
evaluation.

## 2. Related Work

**On-device LLM quantization.** GGUF K-quants served by `llama.cpp` are the de-facto
format for running LLMs on commodity CPUs, compressing weights to roughly 2.6–5.5
effective bits. Systematic on-device evaluations identify 4-bit as a practical
"sweet spot" — near-lossless accuracy with substantial throughput and memory savings
— and report a sharp quality fall-off below roughly 3.5 bits per weight [Song et
al., 2025; Kurt, 2026]. These studies establish *that* and *where* quantization
costs accuracy, but measure it on clean, single-turn text benchmarks rather than in
a deployed pipeline.

**Quality degradation under quantization.** A line of work shows that aggregate
accuracy understates quantization's behavioral effects: compressed models that match
their originals on average nonetheless "flip" a meaningful fraction of individual
answers, and degradation concentrates in specific capabilities — strict
instruction-following and multi-turn coherence are especially sensitive, sometimes
inverting model-size advantages [Lee et al., 2024; Dutta et al., 2024; Li et al.,
2024]. We adopt this study's central methodological lesson — accuracy alone is
misleading — but extend the question to a setting these works do not touch: noisy,
spoken-language input.

**Tool-calling and compression.** Function-calling is now a core capability of
deployed agents, with the Berkeley Function-Calling Leaderboard (BFCL) providing
AST- and state-based evaluation of tool selection and argument filling [Patil et
al., 2024]. ACBench [Dong et al., 2025] is the closest prior work on the effect of
compression on agentic ability, finding that 4-bit quantization preserves isolated
tool-use to within 1–3% while degrading long-horizon agentic accuracy by 10–15%.
Crucially, ACBench — like all the above — evaluates only clean, typed text on GPU;
it does not consider ASR-noised input, sub-4-bit GGUF quantization, CPU deployment,
or the noise×quantization interaction, which is precisely the gap we address.

**Low-latency voice agents.** Recent systems target the conversational latency
budget through speculative and pipeline techniques: PredGen reframes the metric
around time-to-first-sentence and speculates during the user's speech [Li & Grover,
2025]; DuoDecoding runs a Q5_K_M-quantized draft model on CPU alongside a GPU target
[Lv et al., 2025]; AdaSpec adapts speculation to latency SLOs [Huang et al., 2025];
and ChipChat demonstrates a fully on-device cascaded agent at sub-second latency on
high-end Apple Silicon [Likhomanenko et al., 2025]. These works *accelerate*
inference; we instead *decompose* where latency goes on commodity CPU and ask whether
quantization is an effective latency lever there at all. Notably, the one prior data
point on a quantized CPU draft (DuoDecoding) still pairs it with a datacenter GPU
target and fixes the bit-width, leaving the fully-on-CPU, bit-width-swept regime
unmeasured.

**Gap.** Across these threads, no prior work evaluates quantization (a) inside a
voice/ASR pipeline, (b) on tool-calling under ASR-style input noise, (c) for the
quantization×noise *interaction*, or (d) on CPU. This paper addresses all four.

## 3. Method

### 3.1 Pipeline
Streaming ASR (faster-whisper, `base`, INT8) → LLM (Qwen2.5-3B-Instruct and
Llama-3.2-3B-Instruct, GGUF via `llama.cpp`, CPU-only, `n_gpu_layers=0`) → TTS
(Piper on Linux / SAPI5 on Windows). Per-stage timing instrumentation captures
time-to-first-token (TTFT), time-to-first-audio (TTFA), and per-stage wall time.

**Hardware.** All experiments run on CPU: [TODO: exact CPU model], 8 cores,
~7.7 GB RAM, Windows 11 (Anaconda, Python 3.13). No GPU is used at any stage.
Absolute latencies are hardware- and thermal-dependent; the *structure* of the
results (stage dominance, budget violation, overlapping cross-quant speed bands)
is what reproduces.

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
`valid_json` (format), `exact` (function + all arguments). The noise model is
synthetic word-level corruption (substitutions, deletions, and homophone swaps) to a
target WER; this is a deliberate simplification — real ASR errors are phonetically
structured — and we treat a phonetically-grounded noise model as the primary
validity extension (§6).

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

> **Figure 1** (`results/rq2_quant_sweep.png`): stacked per-stage latency by quant
> level with the 1 s budget line — LLM dominates, all levels over budget.
> **Figure 2** (`results/rq2_quant_speed.png`): pure decode speed (tok/s) by quant
> with ±std error bars — bands overlap, no significant quant effect.

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

> **Figure 3** (`results/rq1_interaction.png`): tool-call accuracy vs quant level,
> clean vs ASR-noised — the clean and noisy curves stay parallel (constant gap)
> rather than diverging at low bits, the visual signature of a null interaction.

**A cautionary intermediate result.** An earlier 4-quant Qwen sweep at n=80 produced
an apparently *monotonic* interaction — exact Δ(noise) of 0.21, 0.26, 0.29, 0.31 for
Q5/Q4/Q3/Q2 — that looked like clear support for the low-bit-fragility hypothesis.
Decomposition showed it was an artifact: the noisy values had converged (Q5 and Q2
both ≈0.34), and Q2's larger Δ rode entirely on an implausibly high *clean* baseline
(Q2 clean exact 0.65 > Q5's 0.55) — sampling noise at n=80. Scaling to n=400 (above)
erased the effect: clean baselines equalize and the Δ's match. We report this
because it illustrates a methodological point (§5): interaction effects must be
decomposed into their clean and noisy components and powered adequately, or
proportion noise at small n masquerades as a trend.

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

## References

- Dong et al. (2025). *Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression* (ACBench). arXiv:2505.19433.
- Dutta et al. (2024). *Accuracy is Not All You Need.* NeurIPS 2024. arXiv:2407.09141.
- Huang et al. (2025). *AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware LLM Serving.* ACM SoCC. arXiv:2503.05096.
- Kurt (2026). *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct.* arXiv:2601.14277.
- Lee et al. (2024). *A Comprehensive Evaluation of Quantized Instruction-Tuned LLMs: An Experimental Analysis up to 405B.* arXiv:2409.11055.
- Li & Grover (2025). *PredGen: Accelerated Inference of LLMs through Input-Time Speculation for Real-Time Speech Interaction.* COLM. arXiv:2506.15556.
- Li et al. (2024). *Evaluating Quantized Large Language Models.* ICML. arXiv:2402.18158.
- Likhomanenko et al. (2025). *ChipChat: Low-Latency Cascaded Conversational Agent in MLX.* arXiv:2509.00078.
- Lv et al. (2025). *DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting.* arXiv:2503.00784.
- Patil et al. (2024). *Berkeley Function-Calling Leaderboard (BFCL).* gorilla.cs.berkeley.edu.
- Song et al. (2025). *A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources.* arXiv:2505.15030.
- Stivers et al. (2009). *Universals and cultural variation in turn-taking in conversation.* PNAS 106(26):10587–10592.

> Citation keys are author-year for the draft; convert to the venue's bib style at submission. Full per-paper notes (numbers, methods, limitations) are in `literature-review.md`.

## Appendix / repro
Harness: `src/` (pipeline, eval, experiments). Run RQ2: `scripts/quant_sweep.py`;
RQ1: `python -m src.experiments.rq1_quality --dataset bfcl ...`. All results in
`results/`. Exact commands for every table are in `REPRODUCE.md`. The code and
results in this paper are frozen at git tag `paper-v1`. [TODO: public repo URL on
release.]
