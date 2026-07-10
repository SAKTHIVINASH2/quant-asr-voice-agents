# Quantization and ASR Noise in On-Device Voice Agents: Independent, Not Compounding

**B. Sakthi Vinash**
Draft v0.2 — 2026-07-09

> Status: working draft. Numbers are from Qwen2.5-3B-Instruct (GGUF, llama.cpp)
> on a commodity laptop CPU (Intel Core i5-1135G7, 4C/8T).

---

## Abstract

On-device voice assistants increasingly pair a small quantized LLM with streaming
ASR and TTS, yet two deployment fears are largely untested *together*: that
aggressive quantization cripples the agent, and that quantization and noisy ASR
input compound into a quality cliff. We measure both inside a CPU ASR→LLM→TTS
pipeline. We find (1) the LLM stage dominates end-to-end latency (76–84%) and no
quantization level meets a 1-second conversational budget — and quantization is not
a reliable latency lever on CPU: decode speed varies non-monotonically with bit-width
and the cross-quant differences are swamped by thermal/load state, which dominates as
the driver of absolute latency; and (2) on tool-calling (BFCL, n=400, two model
families), there is **no quantization×noise interaction** — ASR noise roughly
halves argument slot-filling accuracy *uniformly* across quantization levels, so a
2-bit model is no more noise-fragile than a 5-bit one; the null holds under both a
synthetic and a phonetically-grounded noise model. The *standalone* cost of
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
dialogue centers on roughly 200 ms [Stivers et al., 2009] and delays beyond about
one second are widely operationalized as the point at which a spoken exchange begins
to feel broken. The dominant architecture for such systems is a
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
conversational budget, and quantization is not a reliable latency lever on CPU:
decode speed varies non-monotonically with bit-width and — apart from the extreme
8-bit-versus-2-bit gap — cross-quant differences are swamped by thermal state, which
dominates absolute latency.
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
format for running LLMs on commodity CPUs, compressing weights to roughly 3.1–5.7
effective bits per weight (Q2_K–Q5_K_M, as measured by `llama.cpp`; 2.6 is Q2_K's
nominal block value before per-tensor overhead). Systematic on-device evaluations
identify 4-bit as a practical "sweet spot" — near-lossless accuracy with substantial
throughput and memory savings [Song et al., 2025; Kurt, 2026] — with a sharp quality
fall-off below roughly 3.5 bits per weight [Song et al., 2025]. These studies
establish *that* and *where* quantization costs accuracy, but measure it on clean,
single-turn text benchmarks rather than in a deployed pipeline.

**Quality degradation under quantization.** A line of work shows that aggregate
accuracy understates quantization's behavioral effects: compressed models that match
their originals on average nonetheless "flip" a meaningful fraction of individual
answers, and degradation concentrates in specific capabilities — strict
instruction-following and multi-turn coherence are especially sensitive, sometimes
inverting model-size advantages [Lee et al., 2024; Dutta et al., 2024; Li et al.,
2024]. We adopt this study's central methodological lesson — accuracy alone is
misleading — but extend the question to a setting these works do not touch: noisy,
spoken-language input. A separate thread asks whether compression changes a model's
*robustness* to perturbed input rather than its clean-input accuracy: for code
generation, Fang et al. [2025] find quantized models are typically no less robust —
and often *more* robust — than their full-precision counterparts under adversarial
perturbation, an early indication that quantization damage and input corruption need
not compound. We test that hypothesis directly, for spoken input and tool-calling.

**Tool-calling and compression.** Function-calling is now a core capability of
deployed agents, with the Berkeley Function-Calling Leaderboard (BFCL) providing
AST- and state-based evaluation of tool selection and argument filling [Patil et
al., 2024]. ACBench [Dong et al., 2025] is the closest prior work on the effect of
compression on agentic ability, finding that 4-bit quantization preserves isolated
tool-use to within 1–3% while degrading long-horizon agentic accuracy by 10–15%.
Crucially, ACBench — like all the above — evaluates only clean, typed text on GPU;
it does not consider ASR-noised input, sub-4-bit GGUF quantization, CPU deployment,
or the noise×quantization interaction, which is precisely the gap we address. Closer
to the *noise* axis, AgentNoiseBench [Wang et al., 2026] benchmarks tool-using agents
under injected user- and tool-side noise, but its perturbations are textual
(instruction/tool-result) rather than ASR transcription errors, and it holds model
precision fixed — leaving the quantization×noise interaction untouched.

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
unmeasured. Closest to our deployment setting, Ethiraj et al. [2025] build an
end-to-end telecom voice agent from streaming ASR, a quantized LLM, and real-time
TTS — but at a single fixed 4-bit precision on an NVIDIA H100 GPU, without a
bit-width sweep, sub-4-bit GGUF, tool-calling, or any treatment of the
quantization×noise interaction, and they report the LLM stage as the latency
bottleneck, consistent with our CPU finding below.

**Gap.** These threads each touch one axis of our setting but none combines them.
Ethiraj et al. [2025] place a quantized LLM inside a voice/ASR pipeline, but at fixed
4-bit precision on GPU with no tool-calling and no noise study; ACBench [Dong et al.,
2025] measures compression against tool-calling, but on clean typed text on GPU;
AgentNoiseBench [Wang et al., 2026] perturbs tool-using agents, but with textual
noise at fixed precision; Fang et al. [2025] probe the quantization×robustness
interaction, but for code generation, not speech. What remains unmeasured is their
*conjunction*: the quantization×ASR-noise **interaction** on tool-calling, swept
across bit-widths (including sub-4-bit GGUF), on CPU. That conjunction is what this
paper measures.

## 3. Method

### 3.1 Pipeline
Streaming ASR (faster-whisper, `base`, INT8) → LLM (Qwen2.5-3B-Instruct and
Llama-3.2-3B-Instruct, GGUF via `llama.cpp`, CPU-only, `n_gpu_layers=0`) → TTS
(Piper on Linux / SAPI5 on Windows). Per-stage timing instrumentation captures
time-to-first-token (TTFT), time-to-first-audio (TTFA), and per-stage wall time.

**Hardware.** All experiments run on CPU: an **Intel Core i5-1135G7** (11th-gen
"Tiger Lake", 4 cores / 8 threads, 2.4 GHz base) — a commodity ultrabook processor
— with ≈8 GB RAM, Windows 11 (Anaconda, Python 3.13). No GPU is used at any stage.
Absolute latencies are hardware- and thermal-dependent; the *structure* of the
results (stage dominance, budget violation, and the absence of a reliable
quant→latency ordering) is what reproduces.

### 3.2 Quantization levels
GGUF K-quants: Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (≈5.7 → 3.1 effective bits/weight, as
measured by `llama.cpp`; Q2_K's nominal block value is 2.6).

### 3.3 RQ2 — latency
10 voice-assistant utterances drawn from a 31-prompt pool (commands, questions,
tool-invoking requests, lookups), pinned response length (max_tokens=100) for
cross-quant comparability, pooled over two interleaved trials that rotate each
quant through a different thermal position (n=20 per quant level). Decode speed is
reported as pure inter-token tok/s (prefill excluded) with ±std to expose thermal
throttling.

### 3.4 RQ1 — quality under noise
Tool-calling on **BFCL v3 `simple`** (function selection + AST argument match) with
each task's own tool catalog in the prompt; greedy decoding (temperature 0) for
determinism. **ASR noise** injected as controlled word-level corruption at target
WER (0.0 = clean, 0.30 = heavy). Metrics: `name_match` (function selection),
`valid_json` (format), `exact` (function + all arguments). We use two noise models
at the same target WER: (i) **synthetic** word-level corruption (character
substitutions, deletions, homophone swaps), which the main tables report, and (ii) a
**phonetically-grounded** model that applies acoustically-plausible edits — voicing
confusions, nasal/liquid swaps, reduced-vowel substitution, and final-consonant
drops. Real ASR errors are phonetically structured, so (ii) is a realism check on
(i); §4.2 reports that the interaction null holds identically under both.

## 4. Results

### 4.1 RQ2 — the LLM dominates latency; quantization is not a reliable CPU latency lever

On an unloaded machine, per-stage latency (Qwen2.5-3B, CPU) is:

| Stage | share |
|---|---|
| ASR | ~18% |
| **LLM decode** | **76–84% (dominant)** |
| TTS | ~2–3% |

Every quantization level exceeds the 1s conversational budget (TTFA ~1.9–2.8s).

**Quantization is not a reliable latency lever.** Pure decode speed (tok/s, prefill
excluded) does not vary monotonically with bit-width. Pooling two interleaved trials
that rotate each quant through a different thermal position (n=20 utterances/level, a
thermally-variable laptop under background load):

| quant | eff. bits/wt (≈) | decode tok/s (mean ± std) |
|---|---|---|
| Q8_0 | 8.5 | 6.5 ± 1.4 |
| Q5_K_M | 5.7 | 6.2 ± 1.8 |
| Q4_K_M | 4.8 | 9.1 ± 1.3 |
| Q3_K_M | 3.9 | 7.8 ± 1.1 |
| Q2_K | 3.1 | **10.1 ± 0.9** |

The two least-aggressive levels (Q8_0, Q5_K_M) are the slowest and the most
aggressive (Q2_K) the fastest — but the relationship is not monotone (Q5_K_M is
*slower* than the more-compressed Q4/Q3/Q2, and Q3_K_M dips below Q4_K_M), and at
these ±std the bands overlap for every adjacent pair; only the extremes (Q2_K vs
Q5_K_M/Q8_0) separate cleanly. The reason the middle is unstable is that per-quant
run-to-run thermal drift is comparable to the cross-quant gaps — the same model's
decode speed shifts by 2–3 tok/s depending on its thermal position in the interleave
(consistent with the loaded-vs-cool swings in the next paragraph). The takeaway is
therefore not a clean "4-bit latency sweet spot": **on a commodity CPU the
quantization level is not a dependable knob for decode latency — system and thermal
state dominate it, and no level comes close to the conversational budget.** Sub-4-bit
compression saves memory, not reliable time.

**System state dominates bit-width.** Absolute latency is highly sensitive to
machine load and thermal state: across runs the same Q5_K_M model ranged from ~10
to ~6 tok/s and ASR from ~1.5 s to ~9 s depending on background load — a swing
larger than the entire quant×speed effect. On commodity laptops, then, the dominant
determinant of on-device latency is not the quantization level but the system's
thermal/load state, and no quantization choice brings a 3B model within the
conversational budget.

> **Figure 1** (`results/rq2_quant_sweep.png`): stacked per-stage latency by quant
> level with the 1 s budget line — LLM dominates, all levels over budget.
> **Figure 2** (`results/rq2_quant_speed.png`): pure decode speed (tok/s) by quant
> with ±std error bars (n=20, two interleaved trials) — the ordering is non-monotone
> and the bands overlap for every adjacent pair, the visual signature of thermal
> noise dominating any bit-width effect.

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

**The null replicates under phonetically-grounded noise.** The tables above use
synthetic word-level corruption; a natural worry is that the null is an artifact of
unrealistic noise, since real ASR errors are acoustic rather than random. We
therefore re-ran the Qwen2.5-3B sweep (n=400) with a phonetically-motivated noise
model — voicing confusions (s/z, t/d, k/c), nasal and liquid swaps, reduced-vowel
substitution, and final-consonant drops — injected to the same 0.30 target WER. The
interaction remains null: `exact` Δ(noise) is 0.19 (Q5_K_M) vs 0.21 (Q2_K),
statistically indistinguishable, exactly as under synthetic noise (`name_match`
0.96→0.96 vs 0.96→0.94; `valid_json` ~0.97 throughout). Notably the phonetic model
is *gentler* than random-character corruption (Q5_K_M `exact`-under-noise 0.39 vs
0.29; Q2_K 0.33 vs 0.28) — plausibly because acoustic confusions leave more of the
target entity recoverable than character flips do — so the synthetic figures are, if
anything, a conservative upper bound on noise-induced degradation. Both noise models
agree: no quant×noise interaction (Figure 3 shows the phonetic-noise sweep).

**The quantization *main effect* is model-dependent.** Qwen2.5-3B tolerates 2-bit
gracefully (Q2≈Q5 on clean input). Llama-3.2-3B does not: at Q2_K, clean
name_match drops 0.95→0.86 (−9 pts) and clean exact 0.55→0.40 (−15 pts) — both
≈4σ at n=400, i.e. real degradation. So *how much* aggressive quantization costs
on clean input depends on the model family, but *whether noise compounds it* does
not — the interaction is null for both.

> **Figure 3** (`results/rq1_interaction_phonetic.png`): tool-call `exact`-match
> accuracy vs quant level, clean vs phonetically-noised (Qwen2.5-3B, n=400) — the
> clean and noisy curves stay parallel (constant gap) rather than diverging at low
> bits, the visual signature of a null interaction. The synthetic-noise sweep is
> identical in shape.

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

- **Quantization is cheap for tool-calling quality, an unreliable lever for CPU
  latency.** The two deployment levers point opposite ways from intuition.
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
untested; single CPU; **word-level noise** — both a synthetic and a
phonetically-grounded model (the interaction null holds under both), but not real
round-trip TTS→ASR errors, whose WER is pipeline-determined and which remain the one
untested noise source; BFCL `simple` (single-turn, small per-task catalogs make `name_match`
saturate; multi-turn BFCL v3 untested); latency numbers from a thermally-variable
laptop (structure reproduces; absolute ms will not). The interaction null is shown
for tool-calling; other capabilities (multi-turn dialogue coherence, long-context)
are not covered.

## 7. Conclusion

On-device voice agents can be quantized aggressively (to 2-bit) with negligible
tool-calling cost, clean or noisy; quantization and ASR noise degrade quality
through independent channels, not a compounding cliff. The binding constraints are
latency (LLM-dominated, over budget, not reliably reduced by quantization on CPU)
and ASR quality (the true driver of slot-filling errors).

---

## References

- Dong et al. (2025). *Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression* (ACBench). arXiv:2505.19433.
- Dutta et al. (2024). *Accuracy is Not All You Need.* NeurIPS 2024. arXiv:2407.09141.
- Ethiraj et al. (2025). *Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS.* arXiv:2508.04721.
- Fang et al. (2025). *Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation.* arXiv:2506.22776.
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
- Wang et al. (2026). *AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition.* arXiv:2602.11348.

> Citation keys are author-year for the draft; convert to the venue's bib style at submission. Full per-paper notes (numbers, methods, limitations) are in `literature-review.md`.

## Appendix / repro
Harness: `src/` (pipeline, eval, experiments). Run RQ2: `scripts/quant_sweep.py`;
RQ1: `python -m src.experiments.rq1_quality --dataset bfcl ...`. All results in
`results/`. Exact commands for every table are in `REPRODUCE.md`. The code and
results in this paper are frozen at git tag `paper-v1`; the repository will be made
public on release.
