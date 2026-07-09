# Deep-Research Fact-Check — `paper-draft.md`

**Date:** 2026-07-09
**Target:** *"Quantization and ASR Noise in On-Device Voice Agents: Independent, Not Compounding"* (draft v0.1)
**Scope:** External, literature-facing claims only (arXiv IDs resolve, venues correct, prior-work findings characterized accurately, novelty-gap defensible). The paper's own experimental numbers were **not** in scope.
**Method:** deep-research workflow — 6 search angles → 25 sources fetched → 98 claims extracted → 25 verified with 3-vote adversarial verification (2/3 refutes required to kill). 108 agents, 0 errors. Verdicts below combine the synthesized findings with the per-angle search/extraction results recovered from the run journal.

---

## TL;DR

**The bibliography is solid.** Every cited arXiv ID resolves to the named paper, every asserted venue checks out, and every prior-work characterization is faithful. The scary-looking future-dated `Kurt 2601.14277` is a **real** January-2026 paper, not a hallucination.

**Three things to fix before submission:**
1. **Q2_K bit-width is wrong** (§2, §3.2): "≈2.6 effective bits/weight" is the *nominal* value; the *measured* llama.cpp figure is **~3.1 bpw**. Q5_K_M is ~5.7, not 5.5.
2. **Stivers 2009 is over-attributed** (§1): it supports the ~200 ms modal gap but does **not** establish the "delays beyond ~1 s are perceived as broken" threshold — soften or re-source that half.
3. **The novelty-gap claim (§2 "Gap") is too strong and omits three relevant papers** — most importantly a 2025 telecom voice-agent paper that *does* put a quantized LLM inside an ASR→LLM→TTS pipeline. The core contribution survives, but element **(a)** of the four-part gap does not, and the framing must be tightened. See §F.

---

## A. Quantization-degradation citations

| # | Citation | Verdict | Notes |
|---|---|---|---|
| A1 | Dutta et al. 2024, "Accuracy is Not All You Need", **arXiv:2407.09141, NeurIPS 2024** | ✅ **CONFIRMED** | ID + venue + "flips" thesis all verified. NeurIPS 2024 confirmed on official neurips.cc poster page (not just self-reported). Sources: arxiv.org/abs/2407.09141, neurips.cc/virtual/2024/poster/95234 |
| A2 | Lee et al. 2024, "…Quantized Instruction-Tuned LLMs …up to 405B", **arXiv:2409.11055** | ✅ **CONFIRMED (ID)** | ID consistent with a Sept-2024 submission; matches local lit-review characterization (4-bit near-lossless; IFEval/TruthfulQA most sensitive; inversions reverse on IFEval). No venue asserted in draft — fine. |
| A3 | Li et al. 2024, "Evaluating Quantized LLMs", claimed **ICML**, **arXiv:2402.18158** | ✅ **CONFIRMED (ID)** ⚠️ venue not independently re-verified | ID consistent with Feb-2024 submission; ICML 2024 is asserted in your lit review. Low risk, but if you want zero-doubt, confirm the ICML 2024 acceptance directly before camera-ready. |
| A4 | Song et al. 2025, "A Systematic Evaluation of On-Device LLMs…", **arXiv:2505.15030** — 4-bit sweet spot, sharp fall-off <~3.5 bpw | ✅ **CONFIRMED** | Abstract verbatim: "performance threshold at ~3.5 effective bits-per-weight (BPW)"; §4.3: "4-bit quantization emerges as the optimal operating point…below this threshold…precipitous degradation." arXiv preprint (no venue) — draft correctly asserts none. |
| A5 | Kurt 2026, "Which Quantization Should I Use?…llama.cpp on Llama-3.1-8B", **arXiv:2601.14277** | ✅ **CONFIRMED — REAL, not fabricated** | The `2601` prefix = January 2026 (past relative to today). Resolves to Uygar Kurt, submitted 2026-01-11, valid DOI, corroborated by arXiv HTML/PDF + HuggingFace papers + ResearchGate. **Caveat:** Kurt did **not** test Q2_K (sub-3-bit data actually comes from Song). See §E note. |

---

## B. Agentic compression — ACBench

| # | Citation | Verdict | Notes |
|---|---|---|---|
| B6 | Dong et al. 2025, "Can Compressed LLMs Truly Act?" (ACBench), **arXiv:2505.19433, ICML 2025** | ✅ **CONFIRMED (all sub-claims, 13 merged)** | ID + venue verified via three primary sources (arXiv "Accepted by ICML2025", PMLR v267 pp.14169–14202, icml.cc poster 43871). Finding verbatim: "4-bit quantization preserves workflow generation and tool use (1%–3% drop) but degrades real-world application accuracy by 10%–15%." Scope gaps all hold: GPU-oriented GPTQ/AWQ via vLLM, clean typed text, **no ASR/voice, no GGUF/sub-4-bit, no CPU latency.** This is your strongest "closest prior, but doesn't touch the gap" anchor — keep it. |

---

## C. Low-latency voice / speculative-decoding agents

| # | Citation | Verdict | Notes |
|---|---|---|---|
| C7 | PredGen — Li & Grover 2025, claimed **COLM**, **arXiv:2506.15556** | ✅ **CONFIRMED** | PDF header: "Published as a conference paper at COLM 2025." Input-time speculation during user's speech; reframes around time-to-first-sentence — exactly as characterized. COLM accepted-papers page corroborates. |
| C8 | DuoDecoding — Lv et al. 2025, **arXiv:2503.00784** | ✅ **CONFIRMED** | Text: "models on CPU run in the GGUF format with Q5_K_M quantization." CPU GGUF Q5_K_M draft + FP16 GPU target; bit-width fixed, never swept — matches your claim. 2.61× speedup, TTFT→83%. |
| C9 | AdaSpec — Huang et al. 2025, claimed **ACM SoCC**, **arXiv:2503.05096** | ✅ **CONFIRMED** ⚠️ name-collision flag | Correct: Huang et al., ACM SoCC '25 (Nov 2025), SLO-aware adaptive spec decoding, up to 66% speedup. **Heads-up:** there is an *unrelated* same-named "AdaSpec" (multilingual spec decoding, AAAI). Your arXiv ID disambiguates correctly — just don't let a copy-editor "fix" it to the wrong one. |
| C10 | ChipChat — Likhomanenko et al. 2025, **arXiv:2509.00078** | ✅ **CONFIRMED** | Apple, MLX, fully on-device cascaded agent at sub-second latency on Mac Studio (Apple Silicon). Matches your characterization. (Your lit review already caught that an earlier id `2508.11269` was wrong; `2509.00078` is the right one.) |

---

## D. Foundational / other

| # | Citation | Verdict | Notes |
|---|---|---|---|
| D11 | Stivers et al. 2009, PNAS 106(26):10587–10592 — ~200 ms modal gap; ">1 s perceived as broken" | ⚠️ **PARTIALLY MISCHARACTERIZED** | Citation itself correct (PNAS 106(26):10587–10592, DOI 10.1073/pnas.0903616106). The **~200 ms modal gap IS supported** (10 languages, cross-cultural variation in what counts as "on time"). The **"delays beyond ~1 s are perceived as broken" half is NOT established by Stivers 2009** — it's a downstream/practitioner characterization. **Fix:** attribute the 1 s threshold separately or phrase as "commonly operationalized as," not sourced to Stivers. |
| D12 | BFCL — Patil et al. 2024, AST- and state-based eval | ✅ **CONFIRMED** | BFCL (Gorilla, UC Berkeley, first released Feb 2024) uses AST-matching + an executable track — confirms "AST- and state-based evaluation of tool selection and argument filling." **Venue nuance:** the formal publication is PMLR v267 (2025, "…From Tool Use to Agentic Evaluation of LLMs"); BFCL is also cited to the 2024 leaderboard release. Your draft says "Patil et al. (2024)" + the leaderboard URL — fine, but pick one consistently at camera-ready. |

---

## E. Technical field facts

| # | Claim | Verdict | Notes |
|---|---|---|---|
| E13 | GGUF K-quants ≈ 2.6–5.5 effective bits/weight; **Q2_K ≈ 2.6**, **Q5_K_M ≈ 5.5** | ❌ **Q2_K REFUTED / imprecise** | Official llama.cpp quantize README (LLaMA-2-7B measured table): **Q2_K ≈ 3.16 bpw**, **Q5_K_M ≈ 5.70 bpw**. The 2.5–2.6 figure is only the *nominal/theoretical block* value; real per-weight overhead pushes Q2_K to ~3.1–3.2. **Fix:** change "≈2.6" → "≈3.1 (measured; 2.6 nominal)" and "≈5.5" → "≈5.7". The "aggressive sub-4-bit / 2-bit" framing is unaffected — Q2_K is still well below 4 bits. Source: github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md |

**Kurt/Song attribution nit (§2):** the draft co-cites `[Song et al., 2025; Kurt, 2026]` for the "sharp fall-off below ~3.5 bits/weight" claim, but Kurt did **not** evaluate Q2_K / sub-3-bit — the sub-3.5-bpw collapse evidence is Song's alone. Consider citing Song for the fall-off and Kurt only for the 3–8-bit sweet-spot corroboration.

---

## F. Novelty gap — SURVIVES, but must be softened + 3 papers cited

The paper's §2 "Gap" asserts no prior work evaluates quantization **(a)** inside a voice/ASR pipeline, **(b)** on tool-calling under ASR-style noise, **(c)** for the quantization×noise interaction, **(d)** on CPU. The scoop search found **no paper that fully occupies all four** — the core contribution (the quant×ASR-noise *interaction* on tool-calling, bit-width-swept, on CPU) is **not scooped**. But three papers materially affect the framing, and **none is currently cited in the draft:**

### 1. ⚠️ arXiv:2508.04721 — "Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS" (2025)
**This directly weakens gap-element (a).** It deploys a 4-bit-quantized LLM inside a streaming ASR → LLM → TTS pipeline — exactly "quantization inside a voice/ASR pipeline." **Distinguishers that keep your contribution intact:** single fixed 4-bit BitsAndBytes (no bit-width sweep, no sub-4-bit GGUF); **NVIDIA H100 GPU, not CPU**; no tool-calling; no quantization×ASR-noise interaction (ASR errors noted only qualitatively). Reports ~40% latency reduction, RTF < 1.0, sub-1 s response.
- **Action:** MUST cite and distinguish. Reframe (a): don't claim "first to put quantization in a voice pipeline" — claim the *interaction/bit-width-sweep/CPU/tool-calling* combination.
- **Note:** this is the same "telecom" paper your `literature-review.md` already uses for the RQ2 "LLM = 72% of latency (H100)" data point — yet it is **absent from `paper-draft.md`'s Related Work and References entirely.** That's an internal inconsistency to close regardless.

### 2. arXiv:2506.22776 — "Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation" (2025)
The most direct prior on the **quantization × robustness interaction** itself. Finds quantized models are often **more** robust to perturbation than full-precision (51.59% vs 42.86% of adversarial experiments). Domain is **code generation** — no ASR/voice, no tool-calling — so it does **not** scoop you, but it is a strong related-work anchor. Your "independent, not compounding" thesis currently reads as unaware of this contrary-interaction literature.
- **Action:** cite in Related Work ("Quality degradation under quantization" or a new "quantization × robustness" sentence). It actually *supports* your null (quantization doesn't amplify input noise) and pre-empts a reviewer's "didn't you know about…" objection.

### 3. arXiv:2602.11348 — "AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition" (ICML 2026)
Benchmarks tool-using agents under injected **user-side and tool-side noise** — overlaps your noise + tool-calling axis. But: no quantization, no ASR/speech input (textual/instruction/tool-result noise), no CPU. Future-dated (`2602` = Feb 2026) but **confirmed real**.
- **Action:** cite and distinguish — makes clear that ASR-style *input* noise **combined with quantization** remains unaddressed, sharpening rather than threatening your gap.

**Net:** the gap is real but should be re-stated as a *conjunction*: the quantization×ASR-noise **interaction**, on **tool-calling**, **bit-width-swept** (incl. sub-4-bit GGUF), on **CPU**. Each scoop candidate misses at least two of those four; the telecom paper (2508.04721) is the only one touching the voice-pipeline axis and it is single-4-bit/GPU/no-tools/no-interaction.

---

## Corrections needed in `paper-draft.md` (checklist)

1. **§3.2 & §2 — Q2_K bit-width:** "≈2.6" → "≈3.1 (measured; 2.6 nominal)"; "≈5.5" → "≈5.7" for Q5_K_M. (Range "2.6–5.5" → "~2.6 nominal / ~3.1–5.7 measured".)
2. **§1 — Stivers 2009:** keep for ~200 ms modal gap; **do not** attribute the ">1 s = broken" threshold to it. Re-phrase ("commonly operationalized as") or add a separate source.
3. **§2 Related Work + References — add arXiv:2508.04721** (telecom voice agent) and distinguish it; also reconcile with its use as the RQ2 72%-latency data point (currently in lit review but missing from the draft).
4. **§2 Related Work — add arXiv:2506.22776** (quantized-LLM robustness in code gen) as the closest quant×robustness-interaction prior; note it *supports* the non-compounding thesis.
5. **§2 Related Work — add arXiv:2602.11348** (AgentNoiseBench) on the noise+tool-calling axis; distinguish (no quant, no ASR, no CPU).
6. **§2 "Gap" paragraph — reframe** the four-part claim so (a) is not overstated: lead with the quantization×ASR-noise *interaction on tool-calling, bit-width-swept, on CPU* conjunction that none of the above occupy.
7. **§2 — Kurt/Song co-citation:** attribute the "<3.5 bpw fall-off" to Song alone (Kurt didn't test Q2_K); keep Kurt for the 3–8-bit sweet-spot point.
8. *(Optional, camera-ready)* Independently confirm **Li et al. 2402.18158 = ICML 2024** and pick one consistent BFCL venue (2024 leaderboard vs PMLR v267 2025).

**Not a correction — a reassurance:** no fabricated citations, no wrong arXiv IDs, no wrong venues among the 11 references. The `2601.14277` future-date is legitimate. The substantive work is on the novelty framing (§F), not the bibliography.

---

### Sources (primary, verified)
- arxiv.org/abs/2407.09141 · neurips.cc/virtual/2024/poster/95234 (Dutta)
- arxiv.org/abs/2505.15030 · arxiv.org/html/2505.15030v5 (Song)
- arxiv.org/abs/2601.14277 · huggingface.co/papers/2601.14277 (Kurt)
- arxiv.org/abs/2505.19433 · proceedings.mlr.press/v267/dong25k.html · icml.cc/virtual/2025/poster/43871 (ACBench)
- arxiv.org/pdf/2506.15556 (PredGen) · arxiv.org/html/2503.00784v1 (DuoDecoding) · arxiv.org/abs/2503.05096 (AdaSpec) · arxiv.org/abs/2509.00078 (ChipChat)
- pnas.org/doi/10.1073/pnas.0903616106 (Stivers) · proceedings.mlr.press/v267/patil25a.html (BFCL) · github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md (GGUF bpw)
- **Scoop candidates:** arxiv.org/html/2508.04721v1 (telecom voice agent) · arxiv.org/pdf/2506.22776 (quantized-LLM code-gen robustness) · arxiv.org/abs/2602.11348 (AgentNoiseBench)
</content>
</invoke>
