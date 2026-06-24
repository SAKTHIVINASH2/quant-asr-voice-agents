"""RQ1 — Does quantization break tool-calling + multi-turn coherence WORSE on
ASR-noised input than on clean text?

Design (the 2x2x-sweep that is the paper's core):
   for quant in QUANT_LEVELS:
     for input_kind in {clean, asr-noised}:
       score tool-calling (BFCL-style) + multi-turn quality (MT-Bench-style)
   -> report the INTERACTION: degradation(noisy) - degradation(clean) per quant.

The headline number is the *interaction* term: does noise amplify quantization
damage? Prior work (ACBench, Dutta) only has the clean-text column.

Run:  python -m src.experiments.rq1_quality --dummy
"""
from __future__ import annotations

import argparse

from src import config
from src.eval.asr_noise import synthetic_noise, roundtrip_noise
from src.eval.metrics import append_jsonl, summarize
from src.eval.scoring import ToolCall, score_tool_call, judge_response
from src.pipeline.asr import build_asr
from src.pipeline.llm import build_llm
from src.pipeline.tts import build_tts

# Tiny illustrative task set. TODO: load real BFCL v3 multi-turn (Base +
# Missing-Parameter) and MT-Bench (80 Q x 2 turns). See data/README.
TOOL_TASKS = [
    {
        "user": 'Book a table for 2 at "Trattoria Roma" at 7pm.',
        "gold": ToolCall("book_restaurant",
                         {"name": "Trattoria Roma", "party_size": 2, "time": "19:00"}),
    },
    {
        "user": "What's the weather in Tokyo?",
        "gold": ToolCall("get_weather", {"city": "Tokyo"}),
    },
]
CHAT_TASKS = [
    {"turn1": "Explain what a transformer is in one sentence.",
     "turn2": "Now rewrite it for a 10-year-old."},
]


def _eval_input_kind(llm, tasks, chats, kind: str, quant: str,
                     asr=None, tts=None, dummy=True) -> list[dict]:
    rows = []

    # --- Tool-calling ---
    for t in tasks:
        user = t["user"]
        if kind == "noisy":
            user = (synthetic_noise(user, target_wer=0.15).noisy if dummy
                    else roundtrip_noise(user, tts, asr).noisy)
        prompt = (f"You are a tool-calling assistant. Respond ONLY with JSON "
                  f'{{"name": ..., "arguments": {{...}}}}.\nUser: {user}\nJSON:')
        gen = llm.generate(prompt, max_tokens=128)
        sc = score_tool_call(gen.text, t["gold"])
        rows.append({"quant": quant, "input_kind": kind, "task": "tool",
                     "exact": int(sc["exact"]), "valid_json": int(sc["valid_json"]),
                     "name_match": int(sc["name_match"])})

    # --- Multi-turn coherence (turn-1 vs turn-2 quality) ---
    for c in chats:
        q1 = c["turn1"]
        if kind == "noisy":
            q1 = (synthetic_noise(q1, target_wer=0.15).noisy if dummy
                  else roundtrip_noise(q1, tts, asr).noisy)
        a1 = llm.generate(f"User: {q1}\nAssistant:", max_tokens=96).text
        s1 = judge_response(q1, a1, turn=1, model=config.JUDGE_MODEL, dummy=dummy)
        q2 = c["turn2"]
        a2 = llm.generate(f"User: {q1}\nAssistant: {a1}\nUser: {q2}\nAssistant:",
                          max_tokens=96).text
        s2 = judge_response(q2, a2, turn=2, model=config.JUDGE_MODEL, dummy=dummy)
        rows.append({"quant": quant, "input_kind": kind, "task": "chat",
                     "turn1_score": s1, "turn2_score": s2,
                     "turn_drop": s1 - s2})
    return rows


def run(dummy: bool, quants: list[str]) -> None:
    asr = build_asr(dummy) if not dummy else None
    tts = build_tts(dummy)  # needed for roundtrip noise in real mode
    results_path = config.RESULTS_DIR / f"rq1_quality_{'dummy' if dummy else 'real'}.jsonl"

    all_rows = []
    for quant in quants:
        mp = None if dummy else str(config.target_spec(quant).path)
        llm = build_llm(dummy, model_path=mp)
        for kind in ("clean", "noisy"):
            rows = _eval_input_kind(llm, TOOL_TASKS, CHAT_TASKS, kind, quant,
                                    asr=asr, tts=tts, dummy=dummy)
            for r in rows:
                append_jsonl(r, results_path)
            all_rows += rows
        llm.close()
    _report(all_rows)


def _report(rows: list[dict]) -> None:
    print("\n=== RQ1: quantization x input-noise interaction ===")
    quants = sorted({r["quant"] for r in rows})
    print(f"{'quant':10s} {'tool_exact(clean)':>17s} {'tool_exact(noisy)':>17s} "
          f"{'Δ(noise)':>9s} {'turn_drop(clean)':>16s} {'turn_drop(noisy)':>16s}")
    for q in quants:
        def tool_acc(kind):
            xs = [r["exact"] for r in rows if r["quant"] == q
                  and r["task"] == "tool" and r["input_kind"] == kind]
            return sum(xs) / len(xs) if xs else float("nan")

        def turn_drop(kind):
            xs = [r["turn_drop"] for r in rows if r["quant"] == q
                  and r["task"] == "chat" and r["input_kind"] == kind]
            return summarize(xs)["mean"] if xs else float("nan")

        tc, tn = tool_acc("clean"), tool_acc("noisy")
        print(f"{q:10s} {tc:17.2f} {tn:17.2f} {tc - tn:9.2f} "
              f"{turn_drop('clean'):16.2f} {turn_drop('noisy'):16.2f}")
    print("\n  The INTERACTION term Δ(noise) growing as quant drops = RQ1's core finding.")
    print("  (dummy numbers are placeholders — wire real models + BFCL/MT-Bench.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--quants", nargs="*", default=None,
                    help="quant levels to sweep (default: config.QUANT_LEVELS)")
    args = ap.parse_args()
    quants = args.quants or (["dummyA", "dummyB"] if args.dummy else config.QUANT_LEVELS)
    run(args.dummy, quants)


if __name__ == "__main__":
    main()
