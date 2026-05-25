"""Tests for Ptyche. Self-contained (embedded Lorenz / Rossler generators)."""

import numpy as np
import ptyche as p

M = (3, 4, 5, 6, 7, 8)
NP = 4000


def _integ(f, x0, n, dt, transient, seed=1):
    x = np.array(x0, float) + 0.001 * np.random.default_rng(seed).standard_normal(len(x0))
    out = np.empty((n, len(x0)))
    for i in range(n + transient):
        k1 = f(x); k2 = f(x + .5 * dt * k1); k3 = f(x + .5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= transient:
            out[i - transient] = x
    return out


def _lorenz(n=60000):
    return _integ(lambda v: np.array([10 * (v[1] - v[0]), v[0] * (28 - v[2]) - v[1],
                                      v[0] * v[1] - 8 / 3 * v[2]]), [1, 1, 1], n, 0.01, 3000)


_L = _lorenz()
_X = _L[:, 0]


# ----- the dimension pillar (D2) ----------------------------------------------------------- #
def test_false_constant_guard():
    w = p.false_constant_warnings(1.325)
    assert any("plastic" in t["constant"] for t in w)
    assert p.false_constant_warnings(1.99) == []       # 1.99 near nothing exotic
    assert p.false_constant_warnings(2.0) == []        # integers are legit dims, not traps


def test_generic_observable_not_folded():
    rep = p.analyze(_X, m_values=M, n_points=NP, dynamics=False)
    assert not rep.folded
    assert 1.6 < rep.D2_converged < 2.4                # Lorenz ~2.06
    assert rep.false_constant_traps == []


def test_squarelaw_is_folded_and_flags_plastic_trap():
    rep = p.analyze(_X ** 2, m_values=M, n_points=NP, dynamics=False)
    assert rep.folded
    assert rep.D2_converged > rep.D2_naive_lowm + 0.3
    assert any("plastic" in t["constant"] for t in rep.false_constant_traps)


def test_limit_cycle_converges_to_one_without_trap():
    s = np.sin(np.linspace(0, 2000, 60000))
    rep = p.analyze(s, m_values=M, n_points=NP, dynamics=False)
    assert not rep.folded
    assert 0.8 < rep.D2_converged < 1.25
    assert rep.false_constant_traps == []


# ----- the Lyapunov pillar ----------------------------------------------------------------- #
def test_lyapunov_lorenz_matches_truth():
    r = p.lyapunov_rosenstein(_X, fs=100.0)             # fs = 1/dt
    assert 0.75 < r["lambda1"] < 1.05                   # Lorenz lambda1 ~ 0.906
    assert r["r2"] > 0.95                               # clean linear divergence


def test_lyapunov_limit_cycle_is_nonpositive():
    s = np.sin(np.linspace(0, 4000, 60000))
    r = p.lyapunov_rosenstein(s, fs=1.0)
    assert r["lambda1"] < 0.05                          # periodic -> ~0, not positive


# ----- the entropy pillar ------------------------------------------------------------------ #
def test_k2_positive_for_chaos_and_convergence_aware():
    r = p.k2_entropy(_X, fs=100.0)
    assert r["K2"] > 0.3                                # chaotic -> positive entropy
    assert r["descending"]                             # K2(m) decreases toward the true value
    assert isinstance(r["converged"], bool)


def test_k2_near_zero_for_limit_cycle():
    s = np.sin(np.linspace(0, 4000, 60000))
    r = p.k2_entropy(s, fs=1.0)
    assert abs(r["K2"]) < 0.1                           # periodic -> ~0 entropy


# ----- the DSI bonus ----------------------------------------------------------------------- #
def test_logperiodic_squarelaw_doubling():
    u = np.linspace(0, 30, 20000)
    sig = np.cos(2.0 * u + 0.3)
    out = p.logperiodic_squarelaw_check(sig, u)
    assert 1.85 < out["doubling_ratio"] < 2.15
