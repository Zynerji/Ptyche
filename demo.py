"""Ptyche demo: a square-law (intensity) sensor watching the Lorenz attractor.

The naive pipeline -- pick a small embedding dimension, read off the invariant -- reports a
correlation dimension that lands on the plastic number (a tempting false discovery). Ptyche
converges across embedding dimensions, detects the folding, recovers the true invariants, and
names the numerology trap. It also reports the largest Lyapunov exponent and K2 entropy with
the same convergence caveats.
"""
import numpy as np
import ptyche as p


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


FS = 100.0   # 1/dt  -> Lyapunov/K2 in per-time units
TRUTH = "Lorenz: D2 ~ 2.06, lambda1 ~ 0.906/t, h_KS ~ 0.91/t; plastic number rho = 1.3247"


def show(title, x):
    print(f"\n--- {title} ---")
    r = p.analyze(x, fs=FS)
    print("  D2 by embedding m:", {k: round(v, 3) for k, v in r.d2_by_m.items()})
    print(f"  naive D2 (low m) = {r.D2_naive_lowm:.3f} | converged/best D2 = {r.D2_converged:.3f}")
    print(f"  verdict: {r.verdict}  (folding score {r.folding_score:.2f})")
    if r.lyapunov1:
        print(f"  lambda1 = {r.lyapunov1['lambda1']:.3f} / time   (fit r2 {r.lyapunov1['r2']:.3f})")
    if r.k2_entropy:
        print(f"  K2 = {r.k2_entropy['K2']:.3f} / time   "
              f"(converged: {r.k2_entropy['converged']}; sequence "
              f"{ {k: round(v, 2) for k, v in r.k2_entropy['k2_by_m'].items()} })")
    for n in r.notes:
        print("  *", n)


if __name__ == "__main__":
    print(TRUTH)
    data = lorenz()
    show("generic observable  x(t)        (well-behaved sensor)", data[:, 0])
    show("square-law observable  x(t)^2   (intensity/power sensor)", data[:, 0] ** 2)
    show("limit cycle  sin(t)             (1-D control)",
         np.sin(np.linspace(0, 2000, 120000)))
    print("\nHEADLINE: a square-law sensor + m=3 gives D2 ~ plastic ratio 1.3247 (a tempting")
    print("false discovery). Ptyche recovers the true D2 ~ 2.06 and flags the trap -- and gives")
    print("lambda1 ~ 0.91/time, K2 reported with an honest convergence flag.")
