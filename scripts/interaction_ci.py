"""Confidence interval on the RQ1 quantization x noise INTERACTION.

For a null-result paper the key quantity is not each cell's accuracy but the
interaction contrast: does the noise penalty grow for the low-bit model?

    interaction = Delta_Q2 - Delta_Q5,  where  Delta_q = exact(clean) - exact(noisy)

We report it with a 95% CI so the null is a bounded *equivalence* claim ("no
interaction larger than X") rather than a mere failure to reach significance.

The CI treats each cell as an independent binomial at n tasks:
    Var(interaction) = sum_i p_i (1 - p_i) / n_i   over the four cells.
This is CONSERVATIVE: clean and noisy cells are paired on the same tasks, so the
true CI is tighter. A tighter bound would need per-task logging (a re-run); the
conservative bound already excludes any meaningful interaction, so we stop here.

Reads cell means straight from the paper tables (no re-run). Run:
    python scripts/interaction_ci.py
"""
from __future__ import annotations

import math

N = 400  # tasks per cell (BFCL simple, Q5 vs Q2)
Z = 1.96  # 95%

# Per run: exact-match rates (clean, noisy) for SE, and the harness-reported
# Delta(noise) for each quant (full-precision, so the point estimate matches the
# 4.2 tables exactly rather than subtracting the 2-dp display means).
RUNS = {
    #                          Q5: (clean, noisy, delta)   Q2: (clean, noisy, delta)
    "Qwen2.5-3B / synthetic":  {"Q5": (0.58, 0.29, 0.28), "Q2": (0.55, 0.28, 0.27)},
    "Qwen2.5-3B / phonetic":   {"Q5": (0.58, 0.39, 0.19), "Q2": (0.55, 0.33, 0.21)},
    "Llama-3.2-3B / synthetic":{"Q5": (0.55, 0.30, 0.25), "Q2": (0.40, 0.18, 0.21)},
    "Llama-3.2-3B / phonetic": {"Q5": (0.55, 0.39, 0.16), "Q2": (0.40, 0.26, 0.14)},
}


def var_p(p: float, n: int = N) -> float:
    return p * (1.0 - p) / n


def main() -> None:
    print(f"{'run':26s} {'d_Q5':>6s} {'d_Q2':>6s} {'interaction':>12s} "
          f"{'95% CI':>18s}  excludes |x|>")
    worst = 0.0
    for name, cells in RUNS.items():
        q5c, q5n, d_q5 = cells["Q5"]
        q2c, q2n, d_q2 = cells["Q2"]
        inter = d_q2 - d_q5                       # matches the 4.2 tables' Delta column
        se = math.sqrt(var_p(q5c) + var_p(q5n) + var_p(q2c) + var_p(q2n))
        lo, hi = inter - Z * se, inter + Z * se
        bound = max(abs(lo), abs(hi))
        worst = max(worst, bound)
        print(f"{name:26s} {d_q5:6.2f} {d_q2:6.2f} {inter:+12.2f} "
              f"  [{lo:+.2f}, {hi:+.2f}]      ~{bound:.2f}")
    print(f"\ninteraction = Delta(Q2) - Delta(Q5), where Delta(q) = exact(clean) - exact(noisy).")
    print(f"Across all runs the 95% CI excludes any interaction larger than ~{worst:.2f}.")
    print("CI treats cells as independent binomials (conservative; the clean/noisy")
    print("cells are paired on the same 400 tasks, so the true interval is tighter).")


if __name__ == "__main__":
    main()
