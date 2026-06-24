"""RQ3 — How does quantizing the DRAFT model affect speculative-decoding
acceptance rate and TTFT on CPU, under a voice latency budget?

The sweep: fix the target quant, vary the DRAFT quant (Q8_0 -> Q2_K), measure
  (a) token acceptance rate, (b) TTFT / time-to-first-sentence, (c) net speedup
vs. no-draft autoregressive decoding — both models on CPU (no GPU).

This is the delta vs DuoDecoding (2503.00784), which fixes the draft at Q5_K_M,
never reports acceptance rate, and requires the target on GPU.

ACCEPTANCE-RATE MEASUREMENT — IMPORTANT:
  The llama-cpp-python bindings don't cleanly expose per-call n_draft/n_accept.
  The reliable path is the llama.cpp `llama-speculative` CLI, which prints
  "n_drafted = X / n_accept = Y" to stderr. `measure_acceptance_cli()` below
  shells out to it and parses those. Fill LLAMA_SPECULATIVE_BIN. The pure-Python
  path here measures end-to-end latency/speedup (real and useful) but reports
  acceptance only when the CLI is available.

Run:  python -m src.experiments.rq3_specdec --dummy
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess

from src import config
from src.eval.metrics import append_jsonl, summarize
from src.pipeline.llm import build_llm, LlamaCppLLM

LLAMA_SPECULATIVE_BIN = os.environ.get("LLAMA_SPECULATIVE_BIN")  # path to llama-speculative

PROMPTS = [
    "Summarize the benefits of regular exercise in two sentences.",
    "Write a short polite reply declining a meeting invitation.",
    "Explain what an API is to a beginner.",
]


def measure_acceptance_cli(target_gguf: str, draft_gguf: str, prompt: str,
                           n_predict: int = 128, threads: int | None = None) -> dict | None:
    """Run llama.cpp `llama-speculative` and parse acceptance stats from stderr."""
    if not LLAMA_SPECULATIVE_BIN:
        return None
    cmd = [LLAMA_SPECULATIVE_BIN, "-m", target_gguf, "-md", draft_gguf,
           "-p", prompt, "-n", str(n_predict), "-t", str(threads or os.cpu_count())]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stderr + proc.stdout
    n_draft = _grep_int(out, r"n_drafted\s*=\s*(\d+)")
    n_accept = _grep_int(out, r"n_accept\s*=\s*(\d+)")
    if n_draft and n_accept is not None:
        return {"n_draft": n_draft, "n_accept": n_accept,
                "acceptance_rate": n_accept / n_draft}
    return None


def _grep_int(text: str, pat: str) -> int | None:
    m = re.search(pat, text)
    return int(m.group(1)) if m else None


def _build_draft(dummy: bool, draft_quant: str):
    """Build a llama_cpp draft model object for speculative decoding (real mode)."""
    if dummy:
        return None
    from llama_cpp import Llama
    spec = config.draft_spec(draft_quant)
    if not spec.exists:
        raise FileNotFoundError(f"draft GGUF missing for {draft_quant}: {spec.path}")
    # llama-cpp-python accepts a draft Llama via the LlamaDraftModel protocol in
    # recent versions; if your version differs, prefer the CLI path above.
    return Llama(model_path=str(spec.path), n_ctx=spec.n_ctx, n_gpu_layers=0, verbose=False)


def run(dummy: bool, target_quant: str, draft_quants: list[str]) -> None:
    results_path = config.RESULTS_DIR / f"rq3_specdec_{'dummy' if dummy else target_quant}.jsonl"
    target_path = None if dummy else str(config.target_spec(target_quant).path)

    # Baseline: no speculative decoding.
    base_llm = build_llm(dummy, model_path=target_path)
    base = _bench(base_llm, "none")
    base_llm.close()

    rows = [base]
    for dq in draft_quants:
        if dummy:
            # Simulate: better drafts (higher bits) accept more, but cost more/token.
            llm = build_llm(True)
            r = _bench(llm, dq)
            # fabricate a plausible acceptance curve so the report shape is visible
            fake_acc = {"Q8_0": 0.78, "Q5_K_M": 0.74, "Q4_K_M": 0.68,
                        "Q3_K_M": 0.55, "Q2_K": 0.38,
                        "dummyA": 0.7, "dummyB": 0.5}.get(dq, 0.6)
            r.update({"acceptance_rate": fake_acc, "n_draft": 100,
                      "n_accept": int(fake_acc * 100)})
            llm.close()
        else:
            draft = _build_draft(False, dq)
            llm = LlamaCppLLM(model_path=target_path, draft_model=draft)
            r = _bench(llm, dq)
            acc = measure_acceptance_cli(target_path, str(config.draft_spec(dq).path),
                                         PROMPTS[0])
            if acc:
                r.update(acc)
            llm.close()
        r["target_quant"] = "dummy" if dummy else target_quant
        r["draft_quant"] = dq
        append_jsonl(r, results_path)
        rows.append(r)

    _report(rows, base)


def _bench(llm, draft_quant: str) -> dict:
    ttfts, decodes, tps = [], [], []
    for p in PROMPTS:
        g = llm.generate(p, max_tokens=128)
        ttfts.append(g.ttft_s)
        decodes.append(g.decode_s)
        tps.append(g.tokens_per_s)
    return {
        "draft_quant": draft_quant,
        "ttft_s": summarize(ttfts)["mean"],
        "decode_s": summarize(decodes)["mean"],
        "tokens_per_s": summarize(tps)["mean"],
    }


def _report(rows: list[dict], base: dict) -> None:
    print("\n=== RQ3: draft-quant sweep (both models on CPU) ===")
    print(f"{'draft':10s} {'accept_rate':>11s} {'TTFT(ms)':>9s} "
          f"{'tok/s':>7s} {'speedup':>8s}")
    base_tps = base["tokens_per_s"] or 1e-9
    for r in rows:
        acc = r.get("acceptance_rate")
        acc_s = f"{acc:.2f}" if acc is not None else "  n/a"
        speedup = r["tokens_per_s"] / base_tps
        print(f"{r['draft_quant']:10s} {acc_s:>11s} {r['ttft_s']*1000:9.1f} "
              f"{r['tokens_per_s']:7.1f} {speedup:7.2f}x")
    print("\n  Sweet spot = draft quant maximizing speedup while acceptance stays high.")
    print("  The acceptance-rate column is RQ3's novel measurement (DuoDecoding omits it).")
    if any(r.get("acceptance_rate") is None and r["draft_quant"] != "none" for r in rows):
        print("  NOTE: acceptance n/a — set LLAMA_SPECULATIVE_BIN for real acceptance stats.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--target-quant", default="Q4_K_M")
    ap.add_argument("--draft-quants", nargs="*", default=None)
    args = ap.parse_args()
    dq = args.draft_quants or (["dummyA", "dummyB"] if args.dummy else config.QUANT_LEVELS)
    run(args.dummy, args.target_quant, dq)


if __name__ == "__main__":
    main()
