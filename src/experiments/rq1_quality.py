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

TOOL_PROMPT = (
    "You are a function-calling assistant. Respond with ONLY a JSON object "
    'of the form {{"name": <function_name>, "arguments": {{<args>}}}}. '
    "No other text.\nUser: {user}\nJSON:"
)


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
def noisify(text: str, wer: float, dummy: bool, asr, tts, rng) -> str:
    if wer <= 0:
        return text
    if dummy or asr is None:
        return synthetic_noise(text, target_wer=wer, rng=rng).noisy
    # real ASR round-trip ignores the exact wer target (uses real TTS->ASR errors)
    return roundtrip_noise(text, tts, asr).noisy


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def eval_cell(llm, tool_tasks, ifeval_tasks, quant, wer, dummy, asr, tts, rng) -> dict:
    tool_scores = {"name_match": [], "valid_json": [], "exact": []}
    for t in tool_tasks:
        user = noisify(t["user"], wer, dummy, asr, tts, rng)
        out = llm.generate(TOOL_PROMPT.format(user=user), max_tokens=96).text
        sc = score_tool_call(out, t["gold"])
        for k in tool_scores:
            tool_scores[k].append(int(sc[k]))

    ifeval_pass = []
    for t in ifeval_tasks:
        prompt = noisify(t["prompt"], wer, dummy, asr, tts, rng)
        out = llm.generate(prompt, max_tokens=160).text
        ifeval_pass.append(int(check_instruction(out, t["instruction"])))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "quant": quant, "wer": wer,
        "n_tool": len(tool_tasks), "n_ifeval": len(ifeval_tasks),
        "name_match": mean(tool_scores["name_match"]),
        "valid_json": mean(tool_scores["valid_json"]),
        "exact": mean(tool_scores["exact"]),
        "ifeval": mean(ifeval_pass),
    }


def run(dummy: bool, quants: list[str], seed: int = 0,
        limit: int | None = None, noise_levels: list[float] | None = None) -> None:
    rng = random.Random(seed)
    levels = noise_levels if noise_levels is not None else NOISE_LEVELS
    tool_tasks = load_tool_tasks()
    ifeval_tasks = load_ifeval_tasks()
    if limit:
        tool_tasks = tool_tasks[:limit]
        ifeval_tasks = ifeval_tasks[:limit]
    asr = build_asr(dummy) if not dummy else None
    tts = build_tts(dummy)
    results_path = config.RESULTS_DIR / f"rq1_quality_{'dummy' if dummy else 'real'}.jsonl"
    results_path.unlink(missing_ok=True)

    cells = []
    for quant in quants:
        mp = None if dummy else str(config.target_spec(quant).path)
        llm = build_llm(dummy, model_path=mp)
        for wer in levels:
            cell = eval_cell(llm, tool_tasks, ifeval_tasks, quant, wer,
                             dummy, asr, tts, rng)
            append_jsonl(cell, results_path)
            cells.append(cell)
            print(f"  {quant:8s} wer={wer:.2f}  name_match={cell['name_match']:.2f} "
                  f"valid_json={cell['valid_json']:.2f} exact={cell['exact']:.2f} "
                  f"ifeval={cell['ifeval']:.2f}")
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
    # The headline: does Δ(noise) grow as quant drops?
    print("  INTERACTION CHECK — Δ(noise) by quant (should grow if low-bit models")
    print("  are more fragile to ASR noise):")
    for metric, pairs in interaction.items():
        seq = "  ".join(f"{q}:{d:+.2f}" for q, d in pairs)
        print(f"    {metric:11s} {seq}")
    print("\n  (dummy numbers are placeholders — run with real GGUF models.)")


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
    args = ap.parse_args()
    quants = args.quants or (["dummyA", "dummyB"] if args.dummy else config.QUANT_LEVELS)
    run(args.dummy, quants, args.seed, args.limit, args.noise_levels)


if __name__ == "__main__":
    main()
