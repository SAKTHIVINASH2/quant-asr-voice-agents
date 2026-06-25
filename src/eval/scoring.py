"""Quality scorers for RQ1: tool-calling (BFCL-style), instruction-following
(IFEval-style), and multi-turn chat quality (MT-Bench judge).

These are deliberately lightweight, dependency-free SCAFFOLDS with the real
methodology described in TODOs. They run in --dummy mode and give you the data
shapes; swap in the official harnesses when you scale up.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Tool-calling (BFCL-style AST match)                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    name: str
    args: dict


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} object, after stripping markdown fences.
    Robust to the ```json ... ``` wrapping and preamble that chat models add."""
    text = re.sub(r"```(?:json)?", "", text)   # drop code fences
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract a JSON tool call from model output. Handles code fences, preamble,
    and nested arg objects. Real BFCL also supports Python-AST format.
    TODO: use the official BFCL checker (AST + state-based) for v3 multi-turn."""
    blob = _extract_json_object(text)
    if not blob:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if not name:
        return None
    return ToolCall(name, args if isinstance(args, dict) else {})


def score_tool_call(pred: str, gold: ToolCall) -> dict:
    """AST-style match: correct function name + matching argument keys/values."""
    call = parse_tool_call(pred)
    if call is None:
        return {"valid_json": False, "name_match": False, "args_match": False, "exact": False}
    name_ok = call.name == gold.name
    args_ok = call.args == gold.args
    return {
        "valid_json": True,
        "name_match": name_ok,
        "args_match": args_ok,
        "exact": name_ok and args_ok,
    }


# --------------------------------------------------------------------------- #
# Instruction-following (IFEval-style verifiable checks)                       #
# --------------------------------------------------------------------------- #
def check_instruction(response: str, instruction: dict) -> bool:
    """A handful of programmatically-verifiable constraints (the IFEval idea).
    instruction = {"type": ..., ...}. TODO: use google-research/instruction_following_eval
    for the full 25 instruction types."""
    t = instruction.get("type")
    if t == "min_words":
        return len(response.split()) >= instruction["n"]
    if t == "max_words":
        return len(response.split()) <= instruction["n"]
    if t == "keyword":
        return instruction["word"].lower() in response.lower()
    if t == "json":
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    if t == "ends_with":
        return response.rstrip().endswith(instruction["suffix"])
    return False


# --------------------------------------------------------------------------- #
# Multi-turn chat quality (MT-Bench, LLM-as-judge)                             #
# --------------------------------------------------------------------------- #
def judge_response(question: str, answer: str, turn: int,
                   model: str | None = None, dummy: bool = True) -> float:
    """Return a 1-10 quality score. In dummy mode returns a length-based proxy.
    Real mode calls the Claude API as judge (see config.JUDGE_MODEL).
    TODO: port MT-Bench's single-answer-grading prompt; mind verbosity/position bias."""
    if dummy:
        # crude proxy so the pipeline produces plausible numbers offline
        return max(1.0, min(10.0, 4.0 + len(answer.split()) / 20.0))
    from anthropic import Anthropic  # lazy import
    client = Anthropic()
    prompt = (
        f"You are grading an AI assistant answer on a 1-10 scale. "
        f"Question (turn {turn}): {question}\nAnswer: {answer}\n"
        f"Respond with ONLY an integer 1-10."
    )
    msg = client.messages.create(
        model=model or "claude-opus-4-8", max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = msg.content[0].text.strip()
    m = re.search(r"\d+", txt)
    return float(m.group(0)) if m else 5.0
