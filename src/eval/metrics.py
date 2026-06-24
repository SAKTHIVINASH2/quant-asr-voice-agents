"""Shared metrics + results IO."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median, pstdev


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate. Uses jiwer if available, else a Levenshtein fallback."""
    try:
        import jiwer
        return float(jiwer.wer(reference, hypothesis))
    except Exception:
        return _levenshtein_wer(reference, hypothesis)


def _levenshtein_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def summarize(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    vs = sorted(vals)
    return {
        "n": len(vs),
        "mean": mean(vs),
        "median": median(vs),
        "p95": vs[min(len(vs) - 1, int(0.95 * len(vs)))],
        "std": pstdev(vs) if len(vs) > 1 else 0.0,
        "min": vs[0],
        "max": vs[-1],
    }


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def append_jsonl(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
