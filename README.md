# Quantization and ASR Noise in On-Device Voice Agents: Independent, Not Compounding

Code, data and results for the paper by **Sakthivinash Balu** (Independent Researcher, ORCID [0009-0009-4091-9795](https://orcid.org/0009-0009-4091-9795)).

> Does aggressive LLM quantization make a voice agent's tool-calling *more fragile*
> to speech-recognition errors? **No.** On BFCL (n=400) with Qwen2.5-3B and
> Llama-3.2-3B GGUF models on CPU, ASR noise roughly halves exact argument
> slot-filling *uniformly* across bit-widths (Q5_K_M vs Q2_K); the quantization × noise
> interaction stays within ±0.04, and every 95% CI excludes an interaction larger than
> ~0.13. The finding replicates under two noise models (random-character and phonetic).
> On CPU, the LLM dominates turn latency, and decode speed is not monotone in bit-width.

- **Paper:** [`paper.tex`](paper.tex) (LaTeX) · [`paper-draft.md`](paper-draft.md) (Markdown)
- **Reproduce every number:** [`REPRODUCE.md`](REPRODUCE.md)
- **Background:** [`literature-review.md`](literature-review.md), [`research-proposal.md`](research-proposal.md)

## Quickstart (no model downloads)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.experiments.rq1_quality --dummy
python -m src.experiments.rq2_latency --dummy
```

Dummy mode checks the whole pipeline with canned components. For real runs install
`requirements-models.txt`, download GGUF models (`scripts/download_models.sh`), and
follow `REPRODUCE.md`.

## Layout

```
paper.tex, paper-draft.md   the paper
src/
  pipeline/                 ASR → LLM → TTS voice agent (faster-whisper, llama.cpp, Piper)
  eval/                     ASR-noise models, BFCL loader, scoring
  experiments/              rq1_quality.py (quant × noise), rq2_latency.py, rq3_specdec.py
scripts/                    model download, quant sweep, plots, interaction CIs
data/                       BFCL v3 simple subset + curated task sets
results/                    raw JSONL/CSV results and the paper's figures
notebooks/                  Windows / Colab / local runners
```

Models (`models/`, ~14 GB of GGUF files) are not included; see `REPRODUCE.md`.

## Citation

```bibtex
@misc{balu2026quantasr,
  author = {Balu, Sakthivinash},
  title  = {Quantization and {ASR} Noise in On-Device Voice Agents: Independent, Not Compounding},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code: MIT (see [`LICENSE`](LICENSE)). Paper text and figures: CC BY 4.0.
`data/bfcl/` is from the Berkeley Function-Calling Leaderboard (Apache-2.0).
