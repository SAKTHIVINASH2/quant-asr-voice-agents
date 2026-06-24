"""Central configuration: model registry, quant levels, paths.

EDIT the paths below to point at your downloaded models, or pass --config <yaml>
to an experiment to override. Everything has a dummy fallback so the skeleton runs
with no edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VA_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("VA_RESULTS_DIR", ROOT / "results"))
MODELS_DIR = Path(os.environ.get("VA_MODELS_DIR", ROOT / "models"))

for _d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Quantization levels under test -----------------------------------------
# GGUF K-quant names. The sweep is the independent variable for RQ1 and RQ3.
# Ordered most → least precise. Q2_K is expected to collapse (Song et al. 2505.15030).
QUANT_LEVELS = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
BASELINE_QUANT = "FP16"  # reference; may be a separate full-precision GGUF or HF model


@dataclass
class ModelSpec:
    """One LLM at one quant level. `path` is a GGUF file; None => must download."""
    name: str
    quant: str
    path: Path | None = None
    n_ctx: int = 4096
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 4)

    @property
    def exists(self) -> bool:
        return self.path is not None and Path(self.path).exists()


# --- Model registry ----------------------------------------------------------
# TODO: fill in real GGUF paths (see scripts/download_models.sh). The {quant}
# placeholder is filled per level. Target models = the LLM under test.
TARGET_MODEL_TEMPLATE = {
    "name": "Qwen2.5-7B-Instruct",
    "gguf_glob": str(MODELS_DIR / "qwen2.5-7b-instruct-{quant}.gguf"),
}

# Draft models for RQ3 speculative decoding (small, also swept over quant).
DRAFT_MODEL_TEMPLATE = {
    "name": "Qwen2.5-0.5B-Instruct",
    "gguf_glob": str(MODELS_DIR / "qwen2.5-0.5b-instruct-{quant}.gguf"),
}


def target_spec(quant: str) -> ModelSpec:
    p = Path(TARGET_MODEL_TEMPLATE["gguf_glob"].format(quant=quant))
    return ModelSpec(TARGET_MODEL_TEMPLATE["name"], quant, p if p.exists() else None)


def draft_spec(quant: str) -> ModelSpec:
    p = Path(DRAFT_MODEL_TEMPLATE["gguf_glob"].format(quant=quant))
    return ModelSpec(DRAFT_MODEL_TEMPLATE["name"], quant, p if p.exists() else None, n_ctx=4096)


# --- Speech components -------------------------------------------------------
ASR_MODEL = os.environ.get("VA_ASR_MODEL", "small")          # faster-whisper size
ASR_COMPUTE_TYPE = os.environ.get("VA_ASR_COMPUTE", "int8")  # CPU-friendly
PIPER_VOICE = Path(os.environ.get("VA_PIPER_VOICE", MODELS_DIR / "en_US-amy-medium.onnx"))

# --- Voice latency budget (from Stivers et al. 2009; see literature-review.md)
BUDGET_NATURAL_S = 0.30   # < this feels natural
BUDGET_BROKEN_S = 1.00    # > this feels broken

# --- LLM-as-judge (RQ1 MT-Bench scoring) ------------------------------------
JUDGE_MODEL = os.environ.get("VA_JUDGE_MODEL", "claude-opus-4-8")
