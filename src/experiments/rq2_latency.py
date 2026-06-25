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

# Representative voice-assistant turns, varied in type and length (commands,
# questions, tool-invoking requests, info lookups). ~30 prompts so --n-turns can
# go meaningfully higher than the old 5. TODO: swap for a real spoken-dialogue
# corpus (e.g. SpokenWOZ / MultiWOZ utterances) for the final paper.
SAMPLE_TURNS = [
    # short commands
    "Set a timer for ten minutes.",
    "Turn off the living room lights.",
    "Play some jazz music.",
    "Cancel my 3pm meeting.",
    "Mute the volume.",
    # info questions
    "What's the weather like in Paris today?",
    "How many ounces are in a liter?",
    "Who won the World Cup in 2022?",
    "What's the capital of Australia?",
    "How far is the moon from the earth?",
    # reminders / calendar
    "Remind me to call the dentist tomorrow morning.",
    "Add a meeting with Sarah on Friday at 2pm.",
    "What's on my calendar for tomorrow?",
    "Remind me to take my medication every night at nine.",
    "Move my 10am call to the afternoon.",
    # bookings / multi-clause
    "Book me a table for two at an Italian restaurant tonight.",
    "Find a flight from Chennai to Singapore next Monday morning.",
    "Order a large pepperoni pizza and a side of garlic bread.",
    "Schedule a cab to the airport for 6am tomorrow.",
    "Reserve a hotel room in Goa for this weekend under five thousand rupees.",
    # messaging / actions with notification
    "Cancel my 3pm meeting and let the team know I'm running late.",
    "Text Mom that I'll be home by eight.",
    "Send an email to my manager asking for tomorrow off.",
    "Share my location with Dad.",
    # conversions / math / units
    "Convert 75 degrees Fahrenheit to Celsius.",
    "What's 18 percent of 240?",
    "How many tablespoons are in half a cup?",
    # general / longer
    "Explain what machine learning is in simple terms.",
    "Give me a recipe for a quick vegetable stir fry.",
    "Summarize today's top technology news in a few sentences.",
    "What should I pack for a three day trip to a cold city?",
]


def make_audio(text: str, tts) -> str:
    """Render a user turn to a wav (with .txt sidecar for dummy ASR)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    tts.synthesize(text, wav)
    return wav


def run(dummy: bool, quant: str, model_path: str | None, n_turns: int,
        max_tokens: int = 128) -> None:
    asr = build_asr(dummy)
    tts = build_tts(dummy)
    mp = model_path or (None if dummy else str(config.target_spec(quant).path))
    llm = build_llm(dummy, model_path=mp)
    agent = VoiceAgent(asr, llm, tts)

    results_path = config.RESULTS_DIR / f"rq2_latency_{'dummy' if dummy else quant}.jsonl"
    results_path.unlink(missing_ok=True)   # start fresh — don't append to a prior run
    rows = []
    for text in SAMPLE_TURNS[:n_turns]:
        wav = make_audio(text, tts)
        turn = agent.run_turn(wav, max_tokens=max_tokens)
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
    # Length-independent decode speed — the right metric for comparing quants.
    tps = summarize([r.get("toks_per_s") for r in rows])
    ntok = summarize([r.get("n_tokens") for r in rows])
    if tps.get("n"):
        print(f"  decode speed   mean={tps['mean']:7.1f} tok/s  "
              f"({1000/tps['mean']:.1f} ms/tok)   avg {ntok['mean']:.0f} tokens/response")

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
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="pin response length for apples-to-apples quant comparison")
    args = ap.parse_args()
    run(args.dummy, args.quant, args.model_path, args.n_turns, args.max_tokens)


if __name__ == "__main__":
    main()
