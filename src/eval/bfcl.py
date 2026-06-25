"""Berkeley Function-Calling Leaderboard (BFCL) loader + AST scorer.

Downloads the BFCL v3 'simple' / 'parallel' categories (single-turn function
calling with provided tool schemas + gold answers) and maps them to the task
format RQ1 uses. Each BFCL task carries its OWN function list (a per-task
catalog), unlike the curated set's global catalog.

This gives hundreds of independent items — the statistical power needed to
confirm/refute the 2-bit noise cliff. Single-turn AST scoring only (no v3
multi-turn state machine); that's the right scope for RQ1 v1.

Data source: gorilla repo (raw GitHub). Cached under data/bfcl/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src import config

# BFCL data is published as a HuggingFace dataset (more stable than the GitHub
# repo, whose paths move between releases).
HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_DIR = config.DATA_DIR / "bfcl"


@dataclass
class BFCLTask:
    id: str
    user: str
    catalog: str                 # per-task function list for the prompt
    gold_name: str
    gold_args: dict              # param -> list of allowed values (BFCL semantics)


def _download(category: str) -> tuple[Path, Path]:
    """Fetch the question + possible-answer files for a category from HF."""
    from huggingface_hub import hf_hub_download
    BFCL_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"BFCL_v3_{category}.json"
    q = hf_hub_download(HF_REPO, fname, repo_type="dataset", local_dir=str(BFCL_DIR))
    a = hf_hub_download(HF_REPO, f"possible_answer/{fname}", repo_type="dataset",
                        local_dir=str(BFCL_DIR))
    return Path(q), Path(a)


def _catalog_from_functions(functions: list[dict]) -> str:
    lines = []
    for fn in functions:
        params = list((fn.get("parameters", {}).get("properties", {}) or {}).keys())
        lines.append(f"- {fn['name']}({', '.join(params)})")
    return "\n".join(lines)


def load_bfcl_tasks(category: str = "simple", limit: int | None = None) -> list[BFCLTask]:
    """Load and parse a BFCL category into BFCLTask objects."""
    q_path, a_path = _download(category)
    questions = [json.loads(l) for l in open(q_path, encoding="utf-8") if l.strip()]
    answers = {json.loads(l)["id"]: json.loads(l)
               for l in open(a_path, encoding="utf-8") if l.strip()}

    tasks: list[BFCLTask] = []
    for q in questions:
        ans = answers.get(q["id"])
        if not ans:
            continue
        # question is [[{role, content}, ...]]; take the first user content
        turns = q["question"][0] if q["question"] else []
        user = next((t["content"] for t in turns if t.get("role") == "user"), None)
        if not user:
            continue
        gt = ans["ground_truth"][0]      # {func_name: {param: [allowed,...]}}
        gold_name = next(iter(gt))
        gold_args = gt[gold_name]
        tasks.append(BFCLTask(
            id=q["id"], user=user,
            catalog=_catalog_from_functions(q["function"]),
            gold_name=gold_name, gold_args=gold_args,
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks


def score_bfcl(pred_text: str, task: BFCLTask) -> dict:
    """BFCL-style AST scoring: function name + each gold param's value within the
    allowed set. Relaxed type matching (str(5) == 5)."""
    from src.eval.scoring import parse_tool_call
    call = parse_tool_call(pred_text)
    if call is None:
        return {"valid_json": False, "name_match": False,
                "args_match": False, "exact": False}
    name_ok = call.name == task.gold_name

    def val_ok(param, allowed):
        if param not in call.args:
            return "" in allowed or None in allowed   # optional/empty allowed
        v = call.args[param]
        cand = {str(v).lower()} | {str(v)}
        allow = {str(a).lower() for a in allowed} | {str(a) for a in allowed}
        return bool(cand & allow)

    args_ok = all(val_ok(p, a) for p, a in task.gold_args.items())
    return {
        "valid_json": True,
        "name_match": name_ok,
        "args_match": args_ok,
        "exact": name_ok and args_ok,
    }
