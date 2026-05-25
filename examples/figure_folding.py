"""Generate the Ptyche folding-curve figure (examples/folding_curve.png).

Correlation dimension D2 vs embedding dimension m for a generic observable x(t) and a
square-law (intensity) observable x(t)^2 of the same Lorenz attractor. The generic curve sits
near the true D2 ~ 2.06; the square-law curve starts FOLDED at m=3 -- right on the plastic
number 1.3247 (a tempting false discovery) -- and only climbs to the true value as m grows.
That climb is the folding signature Ptyche detects, and the m=3 coincidence is the trap it flags.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ptyche as p

PLASTIC = 1.3247179572
TRUE_D2 = 2.06
M_VALUES = (3, 4, 5, 6, 7, 8, 9, 10)


def lorenz(n=120000, dt=0.01, transient=3000, seed=1):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    x = np.array([1.0, 1.0, 1.0]) + 0.01 * np.random.default_rng(seed).standard_normal(3)

    def f(v):
        X, Y, Z = v
        return np.array([s * (Y - X), X * (r - Z) - Y, X * Y - b * Z])
    out = np.empty((n, 3))
    for i in range(n + transient):
        k1 = f(x); k2 = f(x + .5 * dt * k1); k3 = f(x + .5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= transient:
            out[i - transient] = x
    return out


def main():
    X = lorenz()[:, 0]
    gen = p.embedding_scan(X, m_values=M_VALUES)["d2_by_m"]
    sq = p.embedding_scan(X ** 2, m_values=M_VALUES)["d2_by_m"]
    mg, vg = zip(*sorted(gen.items()))
    ms, vs = zip(*sorted(sq.items()))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(TRUE_D2, ls="--", lw=1.2, color="0.4", label=f"true $D_2 \\approx {TRUE_D2}$")
    ax.axhline(PLASTIC, ls=":", lw=1.2, color="#b22222",
               label=f"plastic number $\\rho = {PLASTIC:.4f}$")
    ax.plot(mg, vg, "o-", color="#1f77b4", lw=2, ms=6,
            label="generic observable  $x(t)$")
    ax.plot(ms, vs, "s-", color="#ff7f0e", lw=2, ms=6,
            label="square-law observable  $x(t)^2$  (folded)")

    # flag the trap: square-law at m=3 lands on the plastic number
    ax.annotate("FALSE-CONSTANT TRAP\n$D_2(m{=}3)\\approx\\rho$ (plastic)",
                xy=(ms[0], vs[0]), xytext=(4.2, 1.18),
                fontsize=9, color="#b22222",
                arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.2))
    ax.annotate("folding: climbs to the\ntrue dimension with $m$",
                xy=(ms[3], vs[3]), xytext=(6.0, 1.45), fontsize=9, color="#cc6600",
                arrowprops=dict(arrowstyle="->", color="#cc6600", lw=1.0))

    ax.set_xlabel("embedding dimension  $m$")
    ax.set_ylabel("correlation dimension  $D_2$")
    ax.set_title("Ptyche: a non-generic (square-law) sensor folds the attractor\n"
                 "and lands $D_2$ on the plastic number at low $m$", fontsize=11)
    ax.set_ylim(1.0, 2.4)
    ax.set_xticks(list(M_VALUES))
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"examples/folding_curve.{ext}", dpi=150, bbox_inches="tight")
    print("generic   D2 by m:", {k: round(v, 3) for k, v in gen.items()})
    print("squarelaw D2 by m:", {k: round(v, 3) for k, v in sq.items()})
    print("wrote examples/folding_curve.png and .pdf")


if __name__ == "__main__":
    main()
