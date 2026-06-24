# Quantization × Latency in On-Device Voice Agents

Research code for measuring how aggressive quantization and latency-optimization
interact in an on-device (CPU) ASR → LLM → TTS voice pipeline.

**Companion docs:** `research-proposal.md`, `literature-review.md`

## Research questions
- **RQ1** — Does quantization break tool-calling + multi-turn coherence *worse* on
  ASR-noised input than on clean text?  (`src/experiments/rq1_quality.py`)
- **RQ2** — Where does latency actually go on CPU — LLM or TTS?  (`src/experiments/rq2_latency.py`)
- **RQ3** — How does quantizing the *draft* model affect speculative-decoding
  acceptance rate and TTFT on CPU?  (`src/experiments/rq3_specdec.py`)

## Quickstart (no downloads — verify the plumbing first)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core only; heavy deps are optional extras

# Everything runs in DUMMY mode with zero model downloads:
python -m src.experiments.rq2_latency --dummy
python -m src.experiments.rq1_quality  --dummy
python -m src.experiments.rq3_specdec  --dummy
```

Dummy components sleep for realistic durations and return canned outputs, so you can
confirm the full pipeline, timing instrumentation, and results files work before
touching real models.

## Running for real

1. Install the heavy optional deps:
   ```bash
   pip install -r requirements-models.txt
   ```
2. Download models (see `scripts/download_models.sh`): GGUF LLMs at several quant
   levels, a faster-whisper model, and a Piper voice.
3. Point `src/config.py` (or `--config`) at the model paths.
4. Drop `--dummy` from the commands above.

## Colab

Open `notebooks/colab_runner.ipynb` (or paste `scripts/colab_setup.sh` into a cell).
The same modules run unchanged; Colab just provides the pip installs and a place to
mount Drive for caching models.

## Layout

```
src/
  config.py            # model registry, quant levels, paths — EDIT THIS
  pipeline/            # the voice agent: asr.py, llm.py, tts.py, voice_agent.py
  eval/                # asr_noise.py, metrics.py, scoring.py
  experiments/         # rq1_quality.py, rq2_latency.py, rq3_specdec.py
results/               # JSONL/CSV outputs (gitignored)
data/                  # datasets: BFCL, MT-Bench, IFEval (gitignored)
scripts/               # download_models.sh, colab_setup.sh
```

## Status / TODO
Search the tree for `TODO:` — the pipeline + timing + RQ2 are runnable; RQ1 scoring
(BFCL state-based, MT-Bench judge) and RQ3 acceptance-rate readout have clearly
marked extension points.
