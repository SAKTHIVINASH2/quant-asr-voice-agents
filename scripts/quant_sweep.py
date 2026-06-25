"""RQ2 quant sweep — the latency-vs-quantization curve (the real RQ2 figure).

Downloads each quant of the 3B model (if missing), runs the latency experiment
per level, and writes a combined CSV + a comparison plot to results/.

Usage:
    python scripts/quant_sweep.py                       # default sweep, 3B
    python scripts/quant_sweep.py --quants Q4_K_M Q3_K_M Q2_K --n-turns 5
    python scripts/quant_sweep.py --dummy               # plumbing check, no downloads

Run from the project root. On Windows set VA_TTS_BACKEND=pyttsx3 first
(or pass --tts pyttsx3); on Linux/WSL the default Piper is used.
"""
from __future__ import annotations

import argparse
import os
import sys

# allow running as `python scripts/quant_sweep.py` from repo root
sys.path.insert(0, os.getcwd())

from src import config                       # noqa: E402
from src.eval.metrics import summarize       # noqa: E402
import src.experiments.rq2_latency as rq2    # noqa: E402

REPO_3B = "bartowski/Qwen2.5-3B-Instruct-GGUF"
FILE_TMPL = "Qwen2.5-3B-Instruct-{quant}.gguf"
DEFAULT_QUANTS = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def ensure_model(quant: str) -> str | None:
    """Download the GGUF for `quant` into models/ if absent. Returns local path."""
    dest = config.MODELS_DIR / FILE_TMPL.format(quant=quant)
    if dest.exists():
        return str(dest)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"  [skip {quant}] huggingface_hub not installed and file missing")
        return None
    try:
        print(f"  downloading {quant} ...")
        p = hf_hub_download(repo_id=REPO_3B, filename=FILE_TMPL.format(quant=quant),
                            local_dir=str(config.MODELS_DIR))
        return p
    except Exception as e:                    # noqa: BLE001
        print(f"  [skip {quant}] download failed: {e}")
        return None


def read_rows(quant: str, dummy: bool) -> list[dict]:
    import json
    tag = "dummy" if dummy else quant
    path = config.RESULTS_DIR / f"rq2_latency_{tag}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", nargs="*", default=DEFAULT_QUANTS)
    ap.add_argument("--n-turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="pin response length so decode time is comparable across quants")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--tts", default=None, help="override VA_TTS_BACKEND (e.g. pyttsx3)")
    args = ap.parse_args()
    if args.tts:
        os.environ["VA_TTS_BACKEND"] = args.tts

    quants = ["dummyA", "dummyB"] if args.dummy else args.quants
    summary_rows = []
    for q in quants:
        print(f"\n### sweep: {q} ###")
        mp = None
        if not args.dummy:
            mp = ensure_model(q)
            if mp is None:
                continue
        # fresh results file per quant is handled inside rq2.run
        rq2.run(dummy=args.dummy, quant=q, model_path=mp, n_turns=args.n_turns,
                max_tokens=args.max_tokens)
        rows = read_rows(q, args.dummy)
        if not rows:
            continue
        tps = [r.get("toks_per_s") for r in rows if r.get("toks_per_s")]
        summary_rows.append({
            "quant": q,
            "asr_ms": summarize([r["asr_s"] for r in rows])["mean"] * 1000,
            "llm_ms": summarize([r["llm_decode_s"] for r in rows])["mean"] * 1000,
            "tts_ms": summarize([r["tts_total_s"] for r in rows])["mean"] * 1000,
            "ttfa_ms": summarize([r["ttfa_s"] for r in rows])["mean"] * 1000,
            "total_ms": summarize([r["total_s"] for r in rows])["mean"] * 1000,
            "toks_per_s": summarize(tps)["mean"] if tps else float("nan"),
            "ms_per_tok": (1000 / summarize(tps)["mean"]) if tps else float("nan"),
            "avg_tokens": summarize([r.get("n_tokens") for r in rows])["mean"],
        })

    _write_and_plot(summary_rows, args.dummy)


def _write_and_plot(rows: list[dict], dummy: bool) -> None:
    if not rows:
        print("no results to summarize.")
        return
    import csv
    out_csv = config.RESULTS_DIR / "rq2_quant_sweep.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    print(f"\n{'quant':10s} {'tok/s':>6s} {'ms/tok':>7s} {'ASR':>7s} {'TTS':>6s} "
          f"{'TTFA':>7s}  budget   (tok/s = the quant-comparison metric)")
    for r in rows:
        flag = "OVER" if r["ttfa_ms"] > config.BUDGET_BROKEN_S * 1000 else "ok"
        print(f"{r['quant']:10s} {r['toks_per_s']:6.1f} {r['ms_per_tok']:7.1f} "
              f"{r['asr_ms']:6.0f}m {r['tts_ms']:5.0f}m {r['ttfa_ms']:6.0f}m  {flag}")
    print("  NOTE: compare quants by tok/s (length-independent). Raw decode time / "
          "TTFA still vary with\n        response length unless --max-tokens pins it.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r["quant"] for r in rows]
        bottoms = [0] * len(rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, color, name in [("asr_ms", "#4C72B0", "ASR"),
                                 ("llm_ms", "#C44E52", "LLM"),
                                 ("tts_ms", "#55A868", "TTS")]:
            vals = [r[key] for r in rows]
            ax.bar(labels, vals, bottom=bottoms, label=name, color=color)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.axhline(config.BUDGET_BROKEN_S * 1000, ls="--", color="k",
                   label="1s 'broken' budget")
        ax.set_ylabel("mean latency (ms)")
        ax.set_title("RQ2: latency vs quantization (CPU, Qwen2.5-3B)")
        ax.legend()
        plt.tight_layout()
        out_png = config.RESULTS_DIR / "rq2_quant_sweep.png"
        plt.savefig(out_png, dpi=130)
        print(f"wrote {out_png}")

        # Second figure: decode speed (tok/s) — the length-independent quant metric.
        tps = [r["toks_per_s"] for r in rows]
        if any(v == v for v in tps):  # not all NaN
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(labels, tps, color="#C44E52")
            ax2.set_ylabel("decode speed (tokens/sec)")
            ax2.set_title("RQ2: decode speed vs quantization (CPU, Qwen2.5-3B)")
            for i, v in enumerate(tps):
                if v == v:
                    ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom")
            plt.tight_layout()
            out_png2 = config.RESULTS_DIR / "rq2_quant_speed.png"
            plt.savefig(out_png2, dpi=130)
            print(f"wrote {out_png2}")
    except Exception as e:                    # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
