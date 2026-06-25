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


def summarize_quant(quant: str, rows: list[dict]) -> dict:
    dtps = [r.get("decode_tps") for r in rows if r.get("decode_tps")]
    d_stat = summarize(dtps) if dtps else {"mean": float("nan"), "std": float("nan")}
    return {
        "quant": quant,
        "n": len(rows),
        "asr_ms": summarize([r["asr_s"] for r in rows])["mean"] * 1000,
        "llm_ms": summarize([r["llm_decode_s"] for r in rows])["mean"] * 1000,
        "tts_ms": summarize([r["tts_total_s"] for r in rows])["mean"] * 1000,
        "ttfa_ms": summarize([r["ttfa_s"] for r in rows])["mean"] * 1000,
        "total_ms": summarize([r["total_s"] for r in rows])["mean"] * 1000,
        "decode_tps": d_stat["mean"],
        "decode_tps_std": d_stat.get("std", float("nan")),
        "avg_tokens": summarize([r.get("n_tokens") for r in rows])["mean"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", nargs="*", default=DEFAULT_QUANTS)
    ap.add_argument("--n-turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="pin response length so decode time is comparable across quants")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--tts", default=None, help="override VA_TTS_BACKEND (e.g. pyttsx3)")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat the sweep N times, ROTATING quant order each trial so "
                         "thermal drift cancels out; results are pooled per quant")
    args = ap.parse_args()
    if args.tts:
        os.environ["VA_TTS_BACKEND"] = args.tts

    quants = ["dummyA", "dummyB"] if args.dummy else args.quants
    from collections import defaultdict
    pooled: dict[str, list] = defaultdict(list)

    for t in range(args.trials):
        # rotate order so each quant occupies a different thermal position per trial
        order = quants[t % len(quants):] + quants[:t % len(quants)]
        if args.trials > 1:
            print(f"\n===== TRIAL {t+1}/{args.trials} — order: {' '.join(order)} =====")
        for q in order:
            print(f"\n### sweep: {q} (trial {t+1}) ###")
            mp = None
            if not args.dummy:
                mp = ensure_model(q)
                if mp is None:
                    continue
            rq2.run(dummy=args.dummy, quant=q, model_path=mp, n_turns=args.n_turns,
                    max_tokens=args.max_tokens)
            pooled[q].extend(read_rows(q, args.dummy))  # pool across trials

    summary_rows = [summarize_quant(q, pooled[q]) for q in quants if pooled[q]]
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

    print(f"\n{'quant':10s} {'decode tok/s (±std)':>20s} {'ASR':>7s} {'TTS':>6s} "
          f"{'TTFA':>7s}  budget")
    for r in rows:
        flag = "OVER" if r["ttfa_ms"] > config.BUDGET_BROKEN_S * 1000 else "ok"
        spd = f"{r['decode_tps']:.1f} ± {r['decode_tps_std']:.1f}"
        print(f"{r['quant']:10s} {spd:>20s} {r['asr_ms']:6.0f}m {r['tts_ms']:5.0f}m "
              f"{r['ttfa_ms']:6.0f}m  {flag}")
    print("  decode tok/s EXCLUDES prefill (length-independent). If the ±std bands\n"
          "  overlap across quants, the differences are NOT significant — likely\n"
          "  thermal/system noise on a laptop, not a real quantization effect.\n"
          "  For a clean claim: --trials N (interleaves quants to cancel thermal drift).")

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

        # Second figure: pure decode speed with std error bars — so overlapping
        # bands make non-significant quant differences visually obvious.
        tps = [r["decode_tps"] for r in rows]
        errs = [r.get("decode_tps_std", 0) for r in rows]
        if any(v == v for v in tps):  # not all NaN
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(labels, tps, yerr=errs, capsize=5, color="#C44E52")
            ax2.set_ylabel("pure decode speed (tokens/sec)")
            ax2.set_title("RQ2: decode speed vs quantization (CPU, Qwen2.5-3B)\n"
                          "error bars = std; overlap ⇒ no significant difference")
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
