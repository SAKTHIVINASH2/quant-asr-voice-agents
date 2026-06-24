"""RQ2 — Where does latency go in ASR->LLM->TTS on CPU?

Directly tests the refuted telecom-paper claim ("TTS dominates"): we expect the
LLM stage to dominate, and by a wider margin on CPU than on the paper's H100.

Sweeps quant levels (when real models are present) and prints a per-stage
breakdown + a verdict on which stage dominates and whether the turn fits the
human latency budget (config.BUDGET_*).

Run:  python -m src.experiments.rq2_latency --dummy
      python -m src.experiments.rq2_latency --quant Q4_K_M
"""
from __future__ import annotations

import argparse
import tempfile

from src import config
from src.eval.metrics import append_jsonl, summarize
from src.pipeline.asr import build_asr
from src.pipeline.llm import build_llm
from src.pipeline.tts import build_tts
from src.pipeline.voice_agent import VoiceAgent

# A few representative user turns. TODO: replace with a real spoken-dialogue set.
SAMPLE_TURNS = [
    "What's the weather like in Paris today?",
    "Book me a table for two at an Italian restaurant tonight.",
    "Remind me to call the dentist tomorrow morning.",
    "How many ounces are in a liter?",
    "Cancel my 3pm meeting and let the team know.",
]


def make_audio(text: str, tts) -> str:
    """Render a user turn to a wav (with .txt sidecar for dummy ASR)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    tts.synthesize(text, wav)
    return wav


def run(dummy: bool, quant: str, model_path: str | None, n_turns: int) -> None:
    asr = build_asr(dummy)
    tts = build_tts(dummy)
    mp = model_path or (None if dummy else str(config.target_spec(quant).path))
    llm = build_llm(dummy, model_path=mp)
    agent = VoiceAgent(asr, llm, tts)

    results_path = config.RESULTS_DIR / f"rq2_latency_{'dummy' if dummy else quant}.jsonl"
    rows = []
    for text in SAMPLE_TURNS[:n_turns]:
        wav = make_audio(text, tts)
        turn = agent.run_turn(wav)
        row = {"quant": "dummy" if dummy else quant, "turn_text": text, **turn.as_row()}
        rows.append(row)
        append_jsonl(row, results_path)

    agent.close()
    _report(rows, quant if not dummy else "dummy")


def _report(rows: list[dict], label: str) -> None:
    stages = ["asr_s", "llm_decode_s", "tts_total_s"]
    print(f"\n=== RQ2 latency breakdown [{label}] (n={len(rows)}) ===")
    means = {}
    for s in stages:
        st = summarize([r[s] for r in rows])
        means[s] = st["mean"]
        print(f"  {s:14s} mean={st['mean']*1000:7.1f} ms  p95={st['p95']*1000:7.1f} ms")
    total = summarize([r["total_s"] for r in rows])
    ttft = summarize([r["ttft_s"] for r in rows])
    ttfa = summarize([r["ttfa_s"] for r in rows])
    print(f"  {'total_s':14s} mean={total['mean']*1000:7.1f} ms")
    print(f"  TTFT           mean={ttft['mean']*1000:7.1f} ms")
    print(f"  TTFA           mean={ttfa['mean']*1000:7.1f} ms")

    dominant = max(means, key=means.get)
    share = means[dominant] / sum(means.values()) * 100
    print(f"\n  -> DOMINANT STAGE: {dominant} ({share:.0f}% of stage time)")
    if dominant == "llm_decode_s":
        print("     Consistent with our hypothesis (LLM-dominated), contra 'TTS dominates'.")
    budget = config.BUDGET_BROKEN_S
    print(f"  -> Mean TTFA {ttfa['mean']*1000:.0f} ms vs {budget*1000:.0f} ms 'broken' budget: "
          f"{'WITHIN' if ttfa['mean'] < budget else 'OVER'} budget.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="run with no real models")
    ap.add_argument("--quant", default="Q4_K_M", help="quant level (real mode)")
    ap.add_argument("--model-path", default=None, help="override GGUF path")
    ap.add_argument("--n-turns", type=int, default=5)
    args = ap.parse_args()
    run(args.dummy, args.quant, args.model_path, args.n_turns)


if __name__ == "__main__":
    main()
