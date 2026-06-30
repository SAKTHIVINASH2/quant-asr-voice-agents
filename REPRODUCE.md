# Reproducing the Results

This document lists the **exact commands** used to generate every result in
`paper-draft.md`. The code is frozen at git tag **`paper-v1`** (commit `ecf6ca3`) —
check it out to validate against the published numbers:

```bash
git checkout paper-v1
```

All experiments run on **CPU**. Hardware used for the paper: Intel Core i5-1135G7
(4 cores / 8 threads), ≈8 GB RAM, Windows 11 (Anaconda Python 3.13) / WSL2 Ubuntu.
Models: Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct (GGUF).

---

## 0. Environment setup

### Windows (Anaconda PowerShell)
```powershell
conda activate automation
pip install -r requirements.txt
# prebuilt CPU wheel avoids the Windows long-path source build:
pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install faster-whisper soundfile pyttsx3 huggingface_hub
# keep the machine awake for long runs:
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

### Linux / WSL
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install faster-whisper piper-tts soundfile huggingface_hub
```

### Models (downloaded into ./models/)
```python
from huggingface_hub import hf_hub_download
for q in ["Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
    hf_hub_download("bartowski/Qwen2.5-3B-Instruct-GGUF",
                    f"Qwen2.5-3B-Instruct-{q}.gguf", local_dir="models")
```
BFCL data (RQ1) and faster-whisper/Piper voices download automatically on first run.

### Smoke test (no models, verifies the pipeline)
```bash
python -m src.experiments.rq2_latency --dummy --n-turns 3
python -m src.experiments.rq1_quality --dummy --limit 3 --noise-levels 0.0 0.3
```

---

## 1. RQ2 — latency decomposition (Table in §4.1)

Single-quant latency split (per-stage ASR/LLM/TTS breakdown + dominance verdict):
```powershell
python -m src.experiments.rq2_latency --quant Q4_K_M --model-path models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --n-turns 5
```

Quant sweep with thermal control (decode tok/s ±std, interleaved trials) — produces
`results/rq2_quant_sweep.{csv,png}` and `results/rq2_quant_speed.png`:
```powershell
python scripts\quant_sweep.py --quants Q5_K_M Q4_K_M Q3_K_M Q2_K --n-turns 10 --max-tokens 100 --trials 2 --tts pyttsx3
```
(Windows uses `--tts pyttsx3`; on Linux omit it to use Piper. The reported decode
speeds are pure inter-token tok/s, prefill excluded.)

---

## 2. RQ1 — quantization × ASR-noise interaction (Table in §4.2)

**Headline result (n=400, BFCL simple, Q5 vs Q2 — the confirmed null):**
```powershell
python -m src.experiments.rq1_quality --quants Q5_K_M Q2_K --dataset bfcl --limit 400 --noise-levels 0.0 0.3 --skip-ifeval
```

**Four-quant sweep (n=80, shows flat interaction across all levels):**
```powershell
python -m src.experiments.rq1_quality --quants Q5_K_M Q4_K_M Q3_K_M Q2_K --dataset bfcl --limit 80 --noise-levels 0.0 0.3 --skip-ifeval
```

Notes for validation:
- **Greedy decoding** (temperature 0, set in code) makes clean cells deterministic —
  reruns reproduce clean numbers exactly. Noisy cells vary only by the corruption
  seed (`--seed`, default 0).
- `--skip-ifeval` drops the IFEval metric (high-variance, slow); the paper's RQ1
  uses tool-calling only.
- Output prints `MAIN EFFECT` (clean accuracy by quant) and `INTERACTION CHECK`
  (Δ(noise) by quant). The key metric on BFCL is `exact` (function + arguments);
  `name_match` saturates because BFCL-simple tasks have tiny per-task tool catalogs.
- Results written to `results/rq1_quality_real.jsonl` (one row per quant×noise cell)
  and figure `results/rq1_interaction.png`.

### Curated-set variant (sanity / ablation)
The 42-task curated set (`data/tool_tasks.json`) instead of BFCL:
```powershell
python -m src.experiments.rq1_quality --quants Q5_K_M Q4_K_M Q3_K_M Q2_K --noise-levels 0.0 0.15 0.3 --seeds 5 --skip-ifeval
```

---

## 3. RQ3 — speculative decoding (harness only, not in paper v1)
```bash
python -m src.experiments.rq3_specdec --dummy
```
Real acceptance-rate measurement requires the llama.cpp `llama-speculative` CLI
(set `LLAMA_SPECULATIVE_BIN`) and 0.5B draft GGUFs. See `src/experiments/rq3_specdec.py`.

---

## Notes on determinism & known variance
- **RQ1 quality** numbers are deterministic per (quant, noise-seed) under greedy
  decoding — they reproduce exactly.
- **RQ2 latency** numbers are hardware/thermal dependent; absolute ms will differ by
  machine, but the *structure* (LLM dominates, all over 1s budget, overlapping tok/s
  bands across quants) reproduces. Use `--trials` to average thermal drift.
- The earlier n=80 4-quant run showed an apparent monotonic `exact` Δ that did NOT
  survive to n=400 (sampling artifact) — see paper §5. Always decompose Δ into clean
  and noisy components.
