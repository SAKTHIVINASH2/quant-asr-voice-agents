"""Regenerate the RQ1 interaction figure from an existing results JSONL — no LLM
re-run needed. Plots `exact` (function + args) clean vs ASR-noised per quant; the
constant vertical gap (parallel curves) is the visual signature of the null
interaction. `name_match` saturates ~0.96 and is uninformative, so we plot `exact`.

Usage (run from repo root, on a machine with matplotlib):
    python scripts/plot_rq1_interaction.py                       # synthetic run
    python scripts/plot_rq1_interaction.py --in results/rq1_quality_real_phonetic.jsonl \
        --out results/rq1_interaction_phonetic.png --label "phonetic noise"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from src import config                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(config.RESULTS_DIR / "rq1_quality_real.jsonl"),
                    help="RQ1 results JSONL (one row per quant×noise cell)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "rq1_interaction.png"))
    ap.add_argument("--label", default="",
                    help="noise-model label for the title, e.g. 'phonetic noise'")
    args = ap.parse_args()

    cells = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if not cells:
        print(f"no cells in {args.inp}")
        return
    quants: list[str] = []
    for c in cells:
        if c["quant"] not in quants:
            quants.append(c["quant"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wer in sorted({c["wer"] for c in cells}):
        ys = [next((c["exact"] for c in cells
                    if c["quant"] == q and c["wer"] == wer), float("nan"))
              for q in quants]
        ax.plot(quants, ys, marker="o",
                label="clean" if wer == 0 else f"ASR noise (wer={wer})")
    ax.set_ylabel("tool-call exact-match accuracy (function + args)")
    ax.set_xlabel("quantization (more aggressive →)")
    suffix = f" [{args.label}]" if args.label else ""
    ax.set_title("RQ1: tool-calling exact-match vs quant, clean vs ASR-noised"
                 f"{suffix}\nparallel curves (constant gap) = null interaction")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
