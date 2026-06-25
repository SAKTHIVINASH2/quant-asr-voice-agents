"""RQ1 — Does aggressive quantization break tool-calling and instruction-following
WORSE under ASR-noised input than on clean text?

The core novelty: measure quality as a function of BOTH quantization level AND
input noise (clean text vs ASR-style errors), and report the INTERACTION —
does noise amplify quantization damage as bits drop? No prior work (ACBench,
Dutta, Lee) does this; all are clean-text only.

Design:
   for quant in QUANTS:
     for wer in NOISE_LEVELS:          # 0.0 = clean, 0.15, 0.30 = ASR-noised
       tool-calling accuracy (function-selection, valid-JSON, exact)
       instruction-following pass rate (IFEval-style)
   -> interaction = degradation(noisy) grows as quant drops?

Metrics per (quant, wer) cell:
   - name_match : did the model pick the right function?  (intent recognition)
   - valid_json : did it emit parseable JSON?               (format fidelity)
   - exact      : function + all args correct               (full correctness)
   - ifeval     : verifiable-instruction pass rate

Run:  python -m src.experiments.rq1_quality --dummy
      python -m src.experiments.rq1_quality --quants Q5_K_M Q4_K_M Q3_K_M Q2_K
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from src import config
from src.eval.asr_noise import synthetic_noise, roundtrip_noise
from src.eval.metrics import append_jsonl, summarize
from src.eval.scoring import ToolCall, score_tool_call, check_instruction
from src.pipeline.asr import build_asr
from src.pipeline.llm import build_llm
from src.pipeline.tts import build_tts

NOISE_LEVELS = [0.0, 0.15, 0.30]   # target WER: clean, moderate, heavy ASR noise

# Filled at runtime from the task set so the model picks from KNOWN function
# names (like real function-calling). Without this, name-matching is impossible.
TOOL_PROMPT = (
    "You are a function-calling assistant. Choose exactly one function from the "
    "list and fill its arguments based on the user request.\n"
    "Available functions:\n{catalog}\n"
    'Respond with ONLY a JSON object: {{"name": <function_name>, '
    '"arguments": {{...}}}}. No explanation, no markdown.\n'
    "User: {user}\nJSON:"
)


def build_catalog(tool_tasks: list[dict]) -> str:
    """One line per unique function: name(arg1, arg2, ...) from the gold calls."""
    seen: dict[str, list[str]] = {}
    for t in tool_tasks:
        g = t["gold"]
        if g.name not in seen:
            seen[g.name] = list(g.args.keys())
    return "\n".join(f"- {name}({', '.join(keys)})" for name, keys in seen.items())


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_tool_tasks() -> list[dict]:
    p = config.DATA_DIR / "tool_tasks.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [{"user": t["user"],
                 "gold": ToolCall(t["gold"]["name"], t["gold"].get("arguments", {}))}
                for t in raw]
    return [{"user": "What's the weather in Tokyo?",
             "gold": ToolCall("get_weather", {"city": "Tokyo"})}]


def load_ifeval_tasks() -> list[dict]:
    p = config.DATA_DIR / "ifeval_tasks.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return [{"prompt": "Describe coffee in at least 20 words.",
             "instruction": {"type": "min_words", "n": 20}}]


# --------------------------------------------------------------------------- #
# Noise application                                                           #
# --------------------------------------------------------------------------- #
def noisify(text: str, wer: float, use_roundtrip: bool, asr, tts, rng) -> str:
    if wer <= 0:
        return text
    if use_roundtrip and asr is not None:
        # most faithful: real TTS->ASR errors (slow; ignores the exact wer target)
        return roundtrip_noise(text, tts, asr).noisy
    # fast, controllable: inject errors to a target WER (default)
    return synthetic_noise(text, target_wer=wer, rng=rng).noisy


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def build_tool_items(dataset: str, limit: int | None, category: str) -> list[dict]:
    """Normalize either dataset to items of {user, catalog, score(out)->dict}."""
    if dataset == "bfcl":
        from src.eval.bfcl import load_bfcl_tasks, score_bfcl
        tasks = load_bfcl_tasks(category, limit)
        return [{"user": t.user, "catalog": t.catalog,
                 "score": (lambda tt: lambda out: score_bfcl(out, tt))(t)}
                for t in tasks]
    # curated set: global catalog, exact-match scorer
    tool_tasks = load_tool_tasks()
    catalog = build_catalog(tool_tasks)
    if limit:
        tool_tasks = tool_tasks[:limit]
    return [{"user": t["user"], "catalog": catalog,
             "score": (lambda g: lambda out: score_tool_call(out, g))(t["gold"])}
            for t in tool_tasks]


def eval_cell_raw(llm, tool_items, ifeval_tasks, wer, use_roundtrip,
                  asr, tts, rng) -> dict:
    """Return RAW per-task 0/1 score lists (pooled across seeds by the caller).
    Greedy decoding (temperature=0) so the only randomness is the noise seed."""
    raw = {"name_match": [], "valid_json": [], "exact": [], "ifeval": []}
    for item in tool_items:
        user = noisify(item["user"], wer, use_roundtrip, asr, tts, rng)
        out = llm.generate(TOOL_PROMPT.format(catalog=item["catalog"], user=user),
                           max_tokens=96, temperature=0.0).text
        sc = item["score"](out)
        for k in ("name_match", "valid_json", "exact"):
            raw[k].append(int(sc[k]))
    for t in ifeval_tasks:
        prompt = noisify(t["prompt"], wer, use_roundtrip, asr, tts, rng)
        out = llm.generate(prompt, max_tokens=160, temperature=0.0).text
        raw["ifeval"].append(int(check_instruction(out, t["instruction"])))
    return raw


def run(dummy: bool, quants: list[str], seed: int = 0,
        limit: int | None = None, noise_levels: list[float] | None = None,
        roundtrip: bool = False, seeds: int = 1, skip_ifeval: bool = False,
        dataset: str = "curated", bfcl_category: str = "simple") -> None:
    levels = noise_levels if noise_levels is not None else NOISE_LEVELS
    tool_items = build_tool_items(dataset, limit, bfcl_category)
    ifeval_tasks = [] if skip_ifeval else load_ifeval_tasks()
    if limit:
        ifeval_tasks = ifeval_tasks[:limit]
    print(f"  dataset={dataset}  tool_items={len(tool_items)}  "
          f"ifeval={len(ifeval_tasks)}")
    asr = build_asr(dummy) if (roundtrip and not dummy) else None
    tts = build_tts(dummy) if roundtrip else None
    results_path = config.RESULTS_DIR / f"rq1_quality_{'dummy' if dummy else 'real'}.jsonl"
    results_path.unlink(missing_ok=True)
    metrics = ("name_match", "valid_json", "exact", "ifeval")

    cells = []
    for quant in quants:
        mp = None if dummy else str(config.target_spec(quant).path)
        llm = build_llm(dummy, model_path=mp)
        for wer in levels:
            # Greedy decoding is deterministic, so seeds only vary the noise
            # pattern. Clean cells (wer=0) need just one seed.
            n_seeds = 1 if wer <= 0 else seeds
            pooled = {m: [] for m in metrics}
            for s in range(n_seeds):
                rng = random.Random(seed + 1000 * s)
                raw = eval_cell_raw(llm, tool_items, ifeval_tasks, wer,
                                    roundtrip, asr, tts, rng)
                for m in metrics:
                    pooled[m].extend(raw[m])
            cell = {"quant": quant, "wer": wer, "seeds": n_seeds,
                    "n_tool": len(tool_items), "n_ifeval": len(ifeval_tasks)}
            for m in metrics:
                st = summarize(pooled[m])
                cell[m] = st.get("mean", float("nan"))
                cell[m + "_std"] = st.get("std", 0.0)
            append_jsonl(cell, results_path)
            cells.append(cell)
            print(f"  {quant:8s} wer={wer:.2f}  "
                  f"name={cell['name_match']:.2f} json={cell['valid_json']:.2f} "
                  f"exact={cell['exact']:.2f} ifeval={cell['ifeval']:.2f}")
        llm.close()

    _report(cells)
    _plot(cells)


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _report(cells: list[dict]) -> None:
    quants = []
    for c in cells:
        if c["quant"] not in quants:
            quants.append(c["quant"])
    print("\n=== RQ1: quantization × ASR-noise interaction ===")
    print("  (interaction = how much MORE noise hurts as bits drop)\n")
    print(f"{'quant':8s} {'metric':11s} {'clean':>7s} {'noisy':>7s} "
          f"{'Δ(noise)':>9s}")
    interaction = {}
    for q in quants:
        clean = next((c for c in cells if c["quant"] == q and c["wer"] == 0.0), None)
        noisy = max((c for c in cells if c["quant"] == q and c["wer"] > 0),
                    key=lambda c: c["wer"], default=None)
        if not clean or not noisy:
            continue
        for metric in ("name_match", "valid_json", "exact", "ifeval"):
            d = clean[metric] - noisy[metric]
            interaction.setdefault(metric, []).append((q, d))
            print(f"{q:8s} {metric:11s} {clean[metric]:7.2f} {noisy[metric]:7.2f} "
                  f"{d:9.2f}")
        print()
    # Main effect of quantization on CLEAN input (the most reliable signal).
    print("  MAIN EFFECT — clean-input accuracy by quant (does quality drop as")
    print("  bits drop, independent of noise?):")
    clean_cells = {c["quant"]: c for c in cells if c["wer"] == 0.0}
    for metric in ("name_match", "exact", "ifeval"):
        seq = "  ".join(
            f"{q}:{clean_cells[q][metric]:.2f}±{clean_cells[q].get(metric+'_std',0):.2f}"
            for q in quants if q in clean_cells)
        print(f"    {metric:11s} {seq}")
    # The interaction: does Δ(noise) grow as quant drops?
    print("\n  INTERACTION CHECK — Δ(noise) by quant (should GROW if low-bit models")
    print("  are more fragile to ASR noise):")
    for metric, pairs in interaction.items():
        seq = "  ".join(f"{q}:{d:+.2f}" for q, d in pairs)
        print(f"    {metric:11s} {seq}")
    print("\n  Read: if MAIN EFFECT declines monotonically but INTERACTION is flat,")
    print("  quantization hurts quality on its own but noise doesn't hit low-bit")
    print("  models disproportionately. ±std bands that overlap = not significant.")


def _plot(cells: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # noqa: BLE001
        print(f"(plot skipped: {e})")
        return
    quants = []
    for c in cells:
        if c["quant"] not in quants:
            quants.append(c["quant"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wer in sorted({c["wer"] for c in cells}):
        ys = [next((c["name_match"] for c in cells
                    if c["quant"] == q and c["wer"] == wer), float("nan"))
              for q in quants]
        ax.plot(quants, ys, marker="o",
                label=f"clean" if wer == 0 else f"ASR noise (wer={wer})")
    ax.set_ylabel("tool-call function-selection accuracy")
    ax.set_xlabel("quantization (more aggressive →)")
    ax.set_title("RQ1: tool-calling accuracy vs quant, clean vs ASR-noised\n"
                 "widening gap at low bits = the interaction effect")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = config.RESULTS_DIR / "rq1_interaction.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--quants", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N tasks of each set (faster first pass)")
    ap.add_argument("--noise-levels", nargs="*", type=float, default=None,
                    help="override WER levels, e.g. 0.0 0.3 for a quick 2-point sweep")
    ap.add_argument("--roundtrip", action="store_true",
                    help="use real TTS->ASR noise instead of fast synthetic (slow)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="noise seeds to average per noisy cell (averages out which "
                         "words get corrupted; clean cells always use 1)")
    ap.add_argument("--skip-ifeval", action="store_true",
                    help="skip the (noisy, slow) IFEval metric; tool-calling only")
    ap.add_argument("--dataset", choices=["curated", "bfcl"], default="curated",
                    help="tool-calling task source (bfcl = hundreds of real items)")
    ap.add_argument("--bfcl-category", default="simple",
                    help="BFCL category: simple, parallel, multiple, ...")
    args = ap.parse_args()
    quants = args.quants or (["dummyA", "dummyB"] if args.dummy else config.QUANT_LEVELS)
    run(args.dummy, quants, args.seed, args.limit, args.noise_levels, args.roundtrip,
        args.seeds, args.skip_ifeval, args.dataset, args.bfcl_category)


if __name__ == "__main__":
    main()
