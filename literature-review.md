# Literature Review

## Quantization × Low-Latency Inference in On-Device Voice Agents

**Author:** B. Sakthi Vinash
**Date:** 2026-06-24
**Status:** Draft v1
**Companion doc:** `research-proposal.md`

---

## 0. How to read this document

This review surveys five bodies of work that the proposed paper sits between:

1. **Speculative decoding** — the mechanism behind low-latency LLM inference (foundations → latency-aware → speech-specific).
2. **Quantization** — the methods that make on-device inference feasible, and the studies measuring what quality they cost.
3. **Tool-calling / agentic benchmarks** — how we will *measure* whether a compressed model still works.
4. **On-device / voice-agent latency** — where the time actually goes, and the human budget we must hit.
5. **Routing / cascades** — an adjacent lever, scoped as a *future v2 extension*, not part of v1.

Each section ends with a thematic synthesis. Section 6 consolidates the **gap** and maps each research question (RQ1–RQ3) to the prior work it extends. A consolidated reference list is in Section 7.

**Recurring finding across all five areas:** the relevant evidence is almost entirely **text-only, single-turn, and GPU-measured**. Nobody has measured quantization's quality cost *inside a voice/ASR pipeline*, on tool-calling, across many dialogue turns, on CPU. That intersection is the contribution.

---

## 1. Speculative Decoding: Foundations → Latency-Aware → Speech

### 1A. Foundations
- **Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (ICML 2023, arXiv:2211.17192).** A small draft model proposes γ tokens; the large target verifies all γ in one parallel pass; a modified rejection-sampling rule yields output **identical in distribution** to the target (lossless). **2×–3×** on T5-XXL (11B). Server/GPU, full-precision. *The lossless guarantee our RQ3 inherits.*
- **Chen et al., *Speculative Sampling* (DeepMind, arXiv:2302.01318).** Same idea at scale — **2×–2.5×** on 70B Chinchilla, no distribution change.
- **Medusa (arXiv:2401.10774).** Self-drafting via extra decoding heads on the target (no separate draft model); **2.2×–3.6×**. A quantized external draft is the alternative that sidesteps head-training.
- **EAGLE / EAGLE-2 (arXiv:2401.15077 / 2406.16858).** Drafts at the feature layer with dynamic draft trees; **3×**, then **3.05×–4.26×**. State-of-the-art draft *quality* — but on GPU, full-precision.
- **Lookahead Decoding (arXiv:2402.02057, ICML 2024).** Draft-free via Jacobi iteration; **1.8×** (up to 4× multi-GPU). Gains rely on *abundant* compute — the opposite of the CPU regime.

### 1B. Latency-aware & SLO-aware
- **AdaSpec (SoCC 2025, arXiv:2503.05096).** Dynamically adapts draft length to a **TPOT/SLO** target rather than throughput; **up to 66%** faster while honoring SLOs. But the budget is a *server SLO*, draft is *full-precision*, no CPU/voice.
- **Speculative Edge-Cloud Decoding w/ Early Exits (arXiv:2505.21594).** Edge runs a tiny draft (Vicuna-68M), cloud runs the target (Llama2-7B) with early-exit adapters + preemptive pre-drafting. **Up to 35% latency / 52% cost** reduction; but the early-exit contribution itself is marginal (**~1.06×** on top of a 1.94× base). Needs a network link; draft largely full-precision.
- **DuoDecoding (arXiv:2503.00784) — ⭐ closest prior to RQ3.** Places a **Q5_K_M-quantized GGUF draft on CPU** alongside a GPU target, running drafting/verification in parallel. **2.61×** generation speedup; **TTFT cut to 83%**. *But:* it pairs the CPU draft with a datacenter **A800 GPU target** (not a fully on-device pipeline), and **fixes** quantization at Q5_K_M without sweeping bit-width against acceptance rate or first-sentence latency. (Related: Dovetail, arXiv:2412.18934.)

### 1C. Speech / ASR-specific
- **PredGen (COLM 2025, arXiv:2506.15556) — names our exact metric.** Speculates candidate responses *while the user is still speaking* (input-time speculation), because sentence-by-sentence TTS needs the **first sentence** before audio starts. **~2× TTFS** reduction (Qwen2.5-7B: Lmsys 488ms→171–248ms). Attacks the *input* side; does not quantize the draft or run CPU-only.
- **SpecASR (DAC 2025, arXiv:2507.18181).** Task-aware spec-decoding for *ASR* (audio→text): adaptive draft length + draft recycling + sparse token-tree. **3.04×–3.79×** over AR, **1.25×–1.84×** over generic spec-decoding, no WER loss. Targets recognition, not the generative LLM turn; full-precision draft.

### Synthesis (§1)
The field moved **throughput on big GPUs → latency under SLO budgets → voice on-device**. DuoDecoding already proves a **Q5_K_M CPU draft** is viable and even cuts TTFT, but pairs it with a GPU target and fixes the bit-width. AdaSpec/SpecASR contribute adaptive-draft-length machinery but assume full-precision drafts. PredGen/SpecASR own the voice latency framing but optimize input-time or recognition. **No work treats draft-model quantization as a controlled variable on CPU under a voice latency budget — RQ3's exact hole.**

---

## 2. Quantization: Methods and Quality Degradation

### 2A. Methods
- **GPTQ (ICLR 2023, arXiv:2210.17323).** One-shot layer-wise PTQ using Hessian info; 3–4 bit weight-only, ~lossless perplexity. One of the two dominant 4-bit formats.
- **AWQ (MLSys 2024, arXiv:2306.00978).** Protects ~1% salient channels (chosen by *activation* magnitude). At 4-bit, +0.5–0.6 perplexity vs FP16 where GPTQ degrades 2.3–4.9 under shifted calibration — **more calibration-robust** (matters when runtime input = noisy ASR transcripts).
- **SmoothQuant (ICML 2023, arXiv:2211.10438).** W8A8 by migrating activation-outlier difficulty to weights; 1.56× speedup. 8-bit, so less relevant to aggressive on-device compression; collapses at 405B.
- **llama.cpp K-quants / GGUF — the format on-device agents actually run.** Super-blocks of 256 weights with quantized sub-block scales → non-integer effective bits: **Q2_K (~2.6 bpw), Q3_K (~3.4), Q4_K_M (~4.5, the "sweet spot"), Q5_K (~5.5), Q8_0 (8-bit)**. Quality only characterized post-hoc (see 2B).
- **QLoRA / NF4 (NeurIPS 2023, arXiv:2305.14314).** 4-bit NormalFloat + double quantization; preserves 16-bit task performance (Guanaco = 99.3% of ChatGPT on Vicuna). Mostly a finetuning-memory result; NF4 is the bitsandbytes inference datatype.

### 2B. Degradation studies
- **Lee et al., *Comprehensive Eval up to 405B* (arXiv:2409.11055).** 4-bit near-lossless on average (Llama-3.1-8B: −1.42% GPTQ / −1.58% AWQ; 70B −0.31%). **Inversions** (4-bit-big > FP16-small) hold *generally* — **but reverse on IFEval**, where FP16-7B beats 4-bit-13B. **IFEval + TruthfulQA most sensitive; MT-Bench turn-2 worse than turn-1.** Text-only, no tool-calling.
- **Kurt, *Which Quantization Should I Use?* (Jan 2026, arXiv:2601.14277).** GGUF K-quants on Llama-3.1-8B. **Q4_K_S: ~0.43% AvgLoss at 70.8% size reduction**; **Q3_K_S: ~5.73% AvgLoss**; **GSM8K most sensitive (Q3_K_S −9.32 points)**. CPU, text-only, single-turn, **no tool-calling, did NOT test Q2_K**.
- **Dutta et al., *Accuracy is Not All You Need* (NeurIPS 2024, arXiv:2407.09141) — methodological backbone.** Introduces **"flips"** (% answers switching correct↔incorrect). On MMLU, accuracy moves <1% yet **3–14% of answers flip**. **MT-Bench turn-2 drops 7.35→6.81 (W8A8) / 6.71 (W4A4) (~10%)** while turn-1 barely moves. *Proves accuracy is the wrong metric and multi-turn is where quantization bites.*
- **Song et al., *Systematic Eval of On-Device LLMs* (arXiv:2505.15030) — core CPU evidence.** On laptop/server CPU via GGUF: **4-bit = optimal operating point (30–50% throughput gain, near-lossless), quality floor ~3.5 bpw**. Sub-4-bit collapse: 1.5B GSM8K **60.8% (fp16) → 52.8% (q4_k) → 21.5% (q2_k)**. CPU decode realistically **~18–38 TPS**. Text-only, single-turn.
- **Li et al., *Evaluating Quantized LLMs* (ICML 2024, arXiv:2402.18158).** Broadest sweep (weight/activation/KV-cache separately). **Activation & KV-cache quant far more fragile than weight-only; emergent abilities, dialogue, long-context degrade first.**

### Synthesis (§2)
Two-part consensus: **(1) 4-bit weight-only ≈ near-lossless on average** (three independent studies agree); **(2) sub-4-bit collapses, and damage concentrates in instruction-following (IFEval), multi-step reasoning (GSM8K), and multi-turn (turn-2) — not knowledge (MMLU).** Dutta proves accuracy *understates* the damage. **Unanimous gap:** every study is text-only/single-turn-dominant; **none tests tool-calling; none measures inside a voice/ASR pipeline.** That is RQ1's territory.

---

## 3. Tool-Calling / Agentic Benchmarks and Compression

### 3A. Benchmarks (how we measure)
- **BFCL v1–v3 (Berkeley Function-Calling Leaderboard) — our primary tool axis.** v1: AST matching + executable checks. **v3: 1,000 multi-turn entries** (Base, Missing-Parameter, Missing-Function, Long-Context, Composite) across 8 API domains, scored on **backend state** after execution (tolerates multiple valid trajectories). Multi-step ≠ multi-turn. Text-only, clean input — *never* ASR-noised.
- **ToolBench/ToolLLM (ICLR 2024, arXiv:2307.16789):** 16k+ real APIs, pass/win-rate — too online/flaky for a controlled CPU latency study. **API-Bank (EMNLP 2023, arXiv:2304.08244):** 1,888 multi-turn tool dialogues; good secondary set. **T-Eval (ACL 2024, arXiv:2312.14033):** six competencies (Plan/Reason/Retrieve/Understand/Instruct/Review) — the right lens for *attributing* where quantization breaks tool use.
- **MT-Bench (NeurIPS 2023, arXiv:2306.05685).** 80 questions × 2 turns, GPT-4 judge (**85% human agreement**, > inter-human 81%). Our **multi-turn quality** axis; note verbosity/position/self-enhancement judge biases. Judge runs offline, not in the latency loop.
- **IFEval (arXiv:2311.07911).** ~500 prompts, 25 *programmatically verifiable* instruction types; strict/loose accuracy, **zero judge cost** — ideal cheap CPU probe for the same skill that produces valid tool-call JSON.

### 3B. Compression × agentic
- **ACBench, *Can Compressed LLMs Truly Act?* (arXiv:2505.19433) — the single most important prior for RQ1.** 12 tasks × 15 models, GPTQ/AWQ 4-bit + Wanda/SparseGPT pruning. **4-bit preserves isolated tool-use/workflow generation (~1–3% loss) but drops real-world agentic accuracy ~10–15%.** I.e., per-call accuracy is a *misleading proxy* — compounded multi-step chains degrade an order of magnitude more.
- **Supporting evidence.** Red Hat's 500k-eval study: 4-bit recovers >99% on academic benchmarks — *but tests no function-calling*, exactly the gap ACBench exposes. Practitioner reports: quantized function-call **JSON stays valid** while the **agentic reasoning on top collapses**; an 8B-INT4 chain failed where a smaller unquantized model held.

### Synthesis (§3)
Tool-call evaluation matured from single-call AST matching to **stateful multi-turn** (BFCL v3). ACBench is the corrective to "4-bit is free": **1–3% on isolated calls vs 10–15% on real agentic chains.** **Gap:** all clean typed text — never ASR-noised spoken input. **Recommended CPU eval set:** BFCL v3 multi-turn (Base + Missing-Parameter, report *state-based end-to-end success*) + MT-Bench (offline judge) + IFEval (zero-cost format fidelity), then **re-run BFCL turns through an ASR layer (clean vs WER-perturbed)** for the never-measured quantization × ASR-noise × multi-turn-tool-use deltas.

---

## 4. On-Device / Voice-Agent Latency: Where the Time Goes

### 4A. The human budget
- **Stivers et al. (PNAS 2009).** Across 10 languages, the modal between-turn gap is **~200 ms** (200–600 ms typical) — near human reaction-time floor, implying speakers *predict* turn-ends. Operationalized in practice as: **<300 ms feels natural, >1 s feels broken** (users start repeating). This is the hard ceiling an on-device ASR→LLM→TTS loop must clear.

### 4B. Pipelines and measured breakdowns
- **Telecom voice agent (arXiv:2508.04721) — RQ2 evidence + the refuted-claim correction.** Streaming ASR + 4-bit (BitsAndBytes) 2B LLM + streaming TTS. Total **0.934 s**, of which **LLM = 0.670 s (~72%)**, TTS = 0.286 s, ASR = 0.049 s; TTFT 0.106 s, TTFA 0.678 s. **⚠️ The fact-checked claim that "TTS dominates" is FALSE — the paper's own table shows the LLM dominates.** Caveat: measured on an **H100, not CPU** — the LLM share would likely *grow* on CPU.
- **ChipChat (Apple, arXiv:2509.00078).** Full cascaded agent at **~920 ms** entirely on a Mac Studio (M2 Ultra, MLX) — largest contributor is the **~560 ms LLM state-inference step**. Proves on-device voice agents are feasible *today*, but on high-end unified-memory Apple Silicon, not commodity CPU. (Brief's id "2508.11269" was wrong.)

### 4C. Latency-cutting levers
- **Endpoint anticipation (arXiv:2606.13450).** Predicts end-of-turn and **speculatively launches LLM+TTS during the user's speech**; trades realized latency against wasted (discarded) compute. Complements quantization: a cheaper LLM is cheaper to throw away.
- **PSLM (arXiv:2406.12428).** Generates text + speech tokens **in parallel**, removing the "finish text first" tax: **0.92 s → 0.54 s (−41%)** (A100). Stacks with quantization.
- **WavChat survey (arXiv:2411.13577).** Cascaded (ASR→LLM→TTS) vs end-to-end taxonomy; "cascaded = quality but latency"; stresses chunk-streaming + full-duplex. (SLED, arXiv:2506.09397, is a distributed *speculative-decoding* edge system, not a survey.)

### 4D. CPU substrate
- **Song et al. (arXiv:2505.15030, again).** The CPU quantization-vs-latency curve for the dominant LLM stage: 4-bit = sweet spot, ~18–38 TPS decode, ~3.5 bpw floor.
- **Edge–cloud collaboration survey (arXiv:2507.16731).** When local quantized inference can't meet the budget, offload — but offload reintroduces network latency, often blowing the 1 s ceiling. Makes the on-CPU quantization tradeoff a *gatekeeper*.
- **CPU ASR/TTS engines.** whisper.cpp / faster-whisper (INT8, RTF ≪ 1, ~100 ms) and Piper / Kokoro TTS (RTF ≪ 1, streaming) are **not** the bottleneck — they leave nearly the whole budget to the LLM.

### Synthesis (§4)
On-device voice agents are feasible today (ChipChat ~920 ms), CPU ASR/TTS run faster than real-time, and the budget is tight (~300 ms natural, 1 s broken). **The decisive RQ2 finding: the LLM stage dominates** (telecom 0.67 s ≈ 72%; ChipChat ~560 ms) — and the "TTS dominates" reading is refuted. Since published budgets are GPU/Apple-Silicon numbers, the open question is sharp: **how does aggressive quantization trade decode latency against quality specifically on commodity CPU, and is there an operating point where an on-CPU ASR→LLM→TTS agent stays inside the human turn-taking budget?**

---

## 5. Routing and Cascades (Adjacent — scoped as future v2)

- **FrugalGPT (arXiv:2305.05176).** Canonical LLM cascade: confidence-scored escalation cheap→expensive. **GPT-4 quality at up to 98% lower cost.** Sequential → worst case pays every tier's latency (hostile to a turn budget). Text/dollar-cost framed.
- **Hybrid LLM (ICLR 2024, arXiv:2404.14618).** Pre-generation *router* predicts difficulty: **22% routed to small model with <1% quality loss; 40% fewer large calls.** Text-only, call-count cost.
- **Unified Routing & Cascading (ETH, arXiv:2410.10347).** Proves both are special cases of cost-budgeted linear optimization; **cascade-routing +1–4% AUC on RouterBench.** Static cost model, no latency, text-only.
- **Speculative Cascades (Google, ICLR 2025, arXiv:2405.19261) — closest to our concern.** Cascade deferral via speculative verification (token-level deferral accepts a *good* small answer even off the large model's argmax); the **only** method treating latency as first-class — but server-side decoding latency, full-precision cloud tiers, text-only.
- **Gatekeeper (Google, arXiv:2502.19335).** Learned **confidence-tuning** deferral loss, beating fixed thresholds. Latency-agnostic, text-only, no quantized tier.
- **Survey (arXiv:2603.04445, 2026).** Confirms the whole field is **cost/throughput-framed; quantization undiscussed, voice out of scope.**

### Synthesis (§5)
Routing/cascades are mature for *cost* reduction but uniformly **text-and-throughput framed** — none under a conversational turn-latency budget, none with an aggressively-quantized on-device tier. Position as a clear **v2 extension**: on-device quantized small tier for the common case + confidence-tuned deferral (Gatekeeper) + speculative verification (speculative cascades) escalating to cloud only when the latency budget and quantization quality-floor jointly demand it. **Not a v1 baseline.**

---

## 6. Consolidated Gap and RQ Mapping

| RQ | Question | Extends | Precise gap it fills |
|---|---|---|---|
| **RQ1** | Does quantization break tool-calling + multi-turn coherence *worse inside a voice pipeline* (ASR noise) than on clean text? | ACBench (1–3% vs 10–15%), Dutta "flips"/turn-2, Lee IFEval sensitivity | All prior work is **text-only, clean input, no tool-calling-in-pipeline**. The ASR-noise × quantization × multi-turn-tool interaction is unmeasured. |
| **RQ2** | Where does latency actually go in ASR→LLM→TTS **on CPU** — LLM or TTS? | Telecom (LLM=72%, GPU), ChipChat (~560ms LLM, MLX), Song CPU curves | Bottleneck breakdowns exist only on **GPU/Apple Silicon**; the "TTS dominates" claim is **refuted**. The CPU split is unmeasured and likely *more* LLM-dominated. |
| **RQ3** | How does **quantizing the draft model** affect spec-decoding acceptance + TTFT on CPU under a voice budget? | DuoDecoding (Q5_K_M CPU draft, but GPU target, fixed bit-width), AdaSpec (adaptive length, full-precision), PredGen (voice TTFS) | No work treats **draft-model quantization as a swept variable on CPU under a voice latency budget**. |

**One-sentence gap:** The entire stack — quantization-degradation studies, tool-calling benchmarks, and routing/cascade methods — has been evaluated **on clean text, on GPUs, single-turn**; this paper measures it **inside an on-device CPU voice pipeline**, where ASR noise, sub-second latency budgets, and multi-turn tool-calling interact.

---

## 7. Key References

**Speculative decoding:** Leviathan 2211.17192 · Chen 2302.01318 · Medusa 2401.10774 · EAGLE 2401.15077 / 2406.16858 · Lookahead 2402.02057 · AdaSpec 2503.05096 · Edge-Cloud Early-Exit 2505.21594 · **DuoDecoding 2503.00784** · Dovetail 2412.18934 · **PredGen 2506.15556** · SpecASR 2507.18181

**Quantization:** GPTQ 2210.17323 · AWQ 2306.00978 · SmoothQuant 2211.10438 · QLoRA/NF4 2305.14314 · GGUF K-quants (llama.cpp quantize README) · Lee 2409.11055 · Kurt 2601.14277 · Dutta 2407.09141 · Song 2505.15030 · Li 2402.18158

**Tool-calling / benchmarks:** BFCL (gorilla.cs.berkeley.edu) · ToolLLM 2307.16789 · API-Bank 2304.08244 · T-Eval 2312.14033 · MT-Bench 2306.05685 · IFEval 2311.07911 · **ACBench 2505.19433** · Red Hat quantized-LLM evals (2024)

**Voice / on-device latency:** Stivers PNAS 2009 · Telecom 2508.04721 · ChipChat 2509.00078 · Endpoint anticipation 2606.13450 · PSLM 2406.12428 · WavChat 2411.13577 · SLED 2506.09397 · Edge-cloud survey 2507.16731

**Routing / cascades:** FrugalGPT 2305.05176 · Hybrid LLM 2404.14618 · Unified routing/cascading 2410.10347 · Speculative Cascades 2405.19261 · Gatekeeper 2502.19335 · Survey 2603.04445

---

## 8. Corrections logged during review

- **ChipChat** is **arXiv:2509.00078**, not 2508.11269.
- The telecom paper (2508.04721) claim that **"TTS dominates latency" is refuted** by its own table — the **LLM dominates (0.67 s ≈ 72%)**; the 0.934 s total is on an **H100, not CPU**.
- **DuoDecoding (2503.00784)** already runs a **Q5_K_M-quantized CPU draft** — closest prior to RQ3; it fixes the bit-width (doesn't sweep it) and pairs with a GPU target.
- **Kurt (2601.14277)** did **not** evaluate Q2_K — sub-3-bit K-quant data comes from Song et al. (2505.15030).
- **PSLM (2406.12428)** is a parallel-generation *method*, not a survey; **SLED (2506.09397)** is a speculative-decoding edge *system*, not a survey. The actual spoken-dialogue survey is **WavChat (2411.13577)**.
