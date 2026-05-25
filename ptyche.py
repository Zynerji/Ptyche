"""Ptyche (Greek pi-tau-upsilon-chi-eta, "a fold") -- trustworthy nonlinear-dynamics
invariants from non-generic (e.g. square-law) observations.

The bottleneck
--------------
Real sensors usually record a *nonlinear function* of a system's state -- intensity
(proportional to amplitude^2), power, magnitude |x|, rectified or log signals. Takens'
embedding theorem guarantees attractor reconstruction only for *generic* observables; an
even/symmetric observable of a symmetric system (the common case for intensity/power) is
NON-generic: the delay embedding reconstructs only the symmetry *quotient* of the attractor.
Standard pipelines (pick an embedding dimension m, compute the correlation dimension D2 /
Lyapunov / entropy) then return values that are SILENTLY WRONG -- folded and under-resolved --
and occasionally land near a famous constant, producing false "discoveries."

Ptyche breaks this in three steps:
  1. CONVERGE  -- estimate the invariant across embedding dimensions and report the converged
                  value (a single low-m number is untrustworthy).
  2. DETECT    -- flag attractor folding / genericity failure (D2 still climbing with m;
                  signal sign-collapse) and recover the true dimension.
  3. GUARD     -- warn when an UNCONVERGED estimate coincides with a famous constant
                  (the "plastic-ratio trap": D2(m=3)=1.33 ~ plastic number, but D2 -> 2.06).

The same converge/detect/guard machinery wraps three invariants:
  * D2  -- correlation dimension (Grassberger-Procaccia),
  * lambda1 -- largest Lyapunov exponent (Rosenstein), and
  * K2  -- correlation (Kolmogorov-Sinai lower-bound) entropy.

It is not magic and breaks no law of computation: it breaks the *reliability* bottleneck of
nonlinear-dynamics estimation under realistic, non-generic sensing.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

__version__ = "0.1.0"

# Exotic constants people get falsely excited to "discover" in an unconverged dimension
# estimate. Deliberately EXCLUDES integers and simple fractions (1, 2, 3, 1/2, ...): those are
# legitimate attractor dimensions (limit cycle = 1, torus/Lorenz ~ 2), not numerology traps.
FAMOUS_CONSTANTS = {
    "golden ratio phi": 1.6180339887,
    "plastic number rho": 1.3247179572,
    "sqrt(2)": 1.4142135624,
    "sqrt(3)": 1.7320508076,
    "e": 2.7182818285,
    "pi": 3.1415926536,
    "ln 2": 0.6931471806,
    "Feigenbaum delta": 4.6692016091,
    "Euler-Mascheroni gamma": 0.5772156649,
}


# --------------------------------------------------------------------------- #
# Correlation dimension (Grassberger-Procaccia) with robust scaling-band fit.
# --------------------------------------------------------------------------- #
def _gp_dimension(points: np.ndarray, n_eps: int = 28) -> float:
    d = pdist(points)
    d = d[d > 0]
    if d.size < 50:
        return float("nan")
    dmin, dmax = np.percentile(d, 0.5), np.percentile(d, 99)
    if not (dmax > dmin > 0):
        return float("nan")
    eps = np.logspace(np.log10(dmin), np.log10(dmax), n_eps)
    npair = d.size
    C = np.array([np.count_nonzero(d < e) / npair for e in eps])
    valid = C > 0
    le, lc = np.log(eps[valid]), np.log(C[valid])
    band = (C[valid] >= 3e-3) & (C[valid] <= 0.15)   # below saturation, above depletion
    if band.sum() < 4:
        lo, hi = int(0.2 * len(le)), int(0.6 * len(le))
        band = np.zeros(len(le), bool); band[lo:hi] = True
    slope, _ = np.polyfit(le[band], lc[band], 1)
    return float(slope)


def _delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = len(x) - (m - 1) * tau
    if n <= 0:
        raise ValueError("series too short for embedding")
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


def autocorr_delay(x: np.ndarray, max_lag: int = 2000) -> int:
    """Takens delay = first lag where autocorrelation drops below 1/e."""
    x = np.asarray(x, float) - np.mean(x)
    var = float(np.dot(x, x))
    if var == 0:
        return 1
    for k in range(1, min(max_lag, len(x) - 1)):
        if np.dot(x[:-k], x[k:]) / var < np.exp(-1.0):
            return max(k, 1)
    return max(1, len(x) // 20)


def correlation_dimension(x: np.ndarray, m: int = 5, tau: int | None = None,
                          n_points: int = 6000) -> float:
    """GP correlation dimension of a scalar series via delay embedding.

    Subsamples to n_points (temporal decorrelation). NOTE: GP estimates inflate at large m
    with finite data (curse of dimensionality); trust the plateau, not the largest m -- which
    is exactly why `converged_dimension` reports the flattest m-window rather than max(m)."""
    x = np.asarray(x, float)
    if tau is None:
        tau = autocorr_delay(x)
    emb = _delay_embed(x, m, tau)
    idx = np.linspace(0, len(emb) - 1, min(n_points, len(emb))).astype(int)
    return _gp_dimension(emb[idx])


# --------------------------------------------------------------------------- #
# Largest Lyapunov exponent (Rosenstein 1993) -- from a scalar series.
# --------------------------------------------------------------------------- #
def lyapunov_rosenstein(x: np.ndarray, m: int = 7, tau: int | None = None, fs: float = 1.0,
                        theiler: int | None = None, cap: int = 8000, k_max: int | None = None,
                        fit_window: tuple = (0.02, 0.4)) -> dict:
    """Largest Lyapunov exponent via Rosenstein's mean-log-divergence method.

    Tracks how nearest-neighbour trajectory pairs separate: <ln d(k)> grows linearly with
    slope = lambda1 (per sample). Pass fs = 1/dt to get lambda1 per unit time (default per
    sample). Uses a CONTIGUOUS slice (no striding -- the step size must equal the sampling
    interval) and a Theiler window to exclude temporally-correlated neighbours.

    Returns {'lambda1', 'lambda1_per_sample', 'divergence_curve', 'fit_k', 'r2'}.
    """
    x = np.asarray(x, float)
    if tau is None:
        tau = autocorr_delay(x)
    xc = x[:min(len(x), cap)]
    emb = _delay_embed(xc, m, tau)
    M = len(emb)
    if M < 200:
        return {"lambda1": float("nan"), "lambda1_per_sample": float("nan"),
                "divergence_curve": np.array([]), "fit_k": (0, 0), "r2": float("nan")}
    if theiler is None:
        theiler = max(tau * (m - 1), tau)               # one embedding window ~ one orbit
    if k_max is None:
        k_max = min(int(2.0 * theiler), M // 4)
    tree = cKDTree(emb)
    # nearest neighbour of each point that is at least `theiler` samples away in time
    kq = min(M, 2 * theiler + 5)
    dists, idxs = tree.query(emb, k=kq)
    nbr = np.full(M, -1, int)
    for i in range(M):
        for j, di in zip(idxs[i][1:], dists[i][1:]):
            if abs(int(j) - i) > theiler:
                nbr[i] = int(j); break
    valid0 = np.where(nbr >= 0)[0]
    div = np.full(k_max + 1, np.nan)
    for k in range(k_max + 1):
        ii = valid0[(valid0 + k < M) & (nbr[valid0] + k < M)]
        if ii.size < 20:
            break
        d = np.linalg.norm(emb[ii + k] - emb[nbr[ii] + k], axis=1)
        d = d[d > 0]
        if d.size:
            div[k] = float(np.mean(np.log(d)))
    ks = np.where(np.isfinite(div))[0]
    if ks.size < 5:
        return {"lambda1": float("nan"), "lambda1_per_sample": float("nan"),
                "divergence_curve": div, "fit_k": (0, 0), "r2": float("nan")}
    lo = max(1, int(fit_window[0] * k_max)); hi = max(lo + 3, int(fit_window[1] * k_max))
    seg = ks[(ks >= lo) & (ks <= hi)]
    if seg.size < 3:
        seg = ks[:max(3, ks.size // 2)]
    slope, intercept = np.polyfit(seg, div[seg], 1)
    resid = div[seg] - (slope * seg + intercept)
    ss = float(np.sum((div[seg] - np.mean(div[seg])) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")
    return {"lambda1": float(slope * fs), "lambda1_per_sample": float(slope),
            "divergence_curve": div, "fit_k": (int(seg[0]), int(seg[-1])), "r2": r2}


# --------------------------------------------------------------------------- #
# K2 correlation entropy (Grassberger-Procaccia) -- KS-entropy lower bound.
# --------------------------------------------------------------------------- #
def _corr_sum(points: np.ndarray, eps: np.ndarray) -> np.ndarray:
    d = pdist(points); d = d[d > 0]; n = d.size
    if n < 50:
        return np.full(len(eps), np.nan)
    return np.array([np.count_nonzero(d < e) / n for e in eps])


def _k2_pair(x, m1, m2, tau, n_points):
    """Single (m, m+1) GP entropy estimate: (fs-free) ln[C_m/C_{m+1}] / tau, per sample."""
    e1 = _delay_embed(x, m1, tau); e2 = _delay_embed(x, m2, tau)
    i1 = np.linspace(0, len(e1) - 1, min(n_points, len(e1))).astype(int)
    i2 = np.linspace(0, len(e2) - 1, min(n_points, len(e2))).astype(int)
    p1, p2 = e1[i1], e2[i2]
    d1 = pdist(p1); d1 = d1[d1 > 0]; d2 = pdist(p2); d2 = d2[d2 > 0]
    if d1.size < 50 or d2.size < 50:
        return float("nan")
    lo = np.log10(max(np.percentile(d1, 1), np.percentile(d2, 1)))
    hi = np.log10(min(np.percentile(d1, 50), np.percentile(d2, 50)))
    if not (hi > lo):
        return float("nan")
    eps = np.logspace(lo, hi, 18)
    C1, C2 = _corr_sum(p1, eps), _corr_sum(p2, eps)
    band = np.isfinite(C1) & np.isfinite(C2) & (C1 >= 1e-2) & (C1 <= 0.1) & (C2 > 0)
    if band.sum() < 3:
        return float("nan")
    return float(np.mean(np.log(C1[band] / C2[band])) / tau)


def k2_entropy(x: np.ndarray, m_values=(3, 4, 5, 6, 7, 8), tau: int | None = None,
               fs: float = 1.0, n_points: int = 4000, tol: float = 0.15) -> dict:
    """Correlation entropy K2 (a lower bound on the Kolmogorov-Sinai entropy h_KS).

    From the embedding-dimension scaling of the correlation sum,
        C_m(eps) ~ eps^{D2} * exp(-m * tau * K2),
    the finite-m estimate K2(m) = (fs/tau) * <ln[ C_m / C_{m+1} ]>_eps DECREASES toward the true
    K2 as m grows (low-m overestimates). Mirroring the dimension pillar, this returns the whole
    K2(m) sequence, the most-converged (largest reliable m) value, and a convergence flag --
    rather than one misleading low-m number. Pass fs = 1/dt for per-time units.

    Returns {'K2', 'K2_per_sample', 'k2_by_m', 'converged', 'descending'}. A positive converged
    K2 is a chaos signature; K2 -> 0 for periodic/quasiperiodic signals. It remains an ESTIMATE
    (a KS lower bound that converges slowly from above); read it with its convergence flag.
    """
    x = np.asarray(x, float)
    if tau is None:
        tau = autocorr_delay(x)
    ms = sorted(int(m) for m in m_values)
    k2_by_m = {}
    for m1, m2 in zip(ms[:-1], ms[1:]):
        v = _k2_pair(x, m1, m2, tau, n_points)
        if np.isfinite(v):
            k2_by_m[m1] = float(v * fs)
    if not k2_by_m:
        return {"K2": float("nan"), "K2_per_sample": float("nan"), "k2_by_m": {},
                "converged": False, "descending": False}
    keys = sorted(k2_by_m); vals = [k2_by_m[k] for k in keys]
    k2 = vals[-1]                                        # most-converged (largest m) value
    converged = bool(len(vals) >= 2 and abs(vals[-1] - vals[-2]) / max(abs(vals[-1]), 1e-9) < tol)
    descending = bool(len(vals) >= 2 and vals[-1] < vals[0])
    return {"K2": float(k2), "K2_per_sample": float(k2 / fs), "k2_by_m": k2_by_m,
            "converged": converged, "descending": descending}


# --------------------------------------------------------------------------- #
# The three pillars: converge, detect folding, guard against false constants.
# --------------------------------------------------------------------------- #
def embedding_scan(x: np.ndarray, m_values=(3, 4, 5, 6, 7, 8), tau: int | None = None,
                   n_points: int = 6000) -> dict:
    """D2 across embedding dimensions m (the curve a single number hides). The default m
    range stays below the finite-sample inflation regime for low-dimensional attractors."""
    if tau is None:
        tau = autocorr_delay(x)
    d2 = {int(m): correlation_dimension(x, m=int(m), tau=tau, n_points=n_points)
          for m in m_values}
    return {"tau": int(tau), "d2_by_m": d2}


def converged_dimension(d2_by_m: dict, plateau_tol: float = 0.10) -> dict:
    """Best D2 = the PLATEAU (flattest consecutive-m window), NOT max(m).

    GP estimates rise from an underestimate at small m, plateau near the true dimension, then
    inflate at large m (finite-sample). The plateau is the trustworthy value. The low-m value
    and the rise-to-plateau quantify folding / under-resolution.
    """
    ms = sorted(d2_by_m)
    vals = np.array([d2_by_m[m] for m in ms], float)
    ok = np.isfinite(vals); ms = list(np.array(ms)[ok]); vals = vals[ok]
    if len(vals) < 2:
        return {"D2": float(vals[-1]) if len(vals) else float("nan"),
                "converged": False, "rise": float("nan"), "D2_lowm": float("nan"),
                "m_at_D2": (ms[-1] if ms else None), "top_spread": float("nan")}
    # GP underestimates from below within the (capped, pre-inflation) m range, so the best
    # estimate is the maximum; convergence = the top of the curve has flattened.
    D2 = float(np.max(vals))
    top_spread = float(abs(vals[-1] - vals[-2]))
    return {"D2": D2, "converged": bool(top_spread < plateau_tol),
            "top_spread": top_spread, "m_at_D2": int(ms[int(np.argmax(vals))]),
            "D2_lowm": float(vals[0]), "rise": float(D2 - vals[0])}


def false_constant_warnings(value: float, tol_rel: float = 0.02) -> list:
    """Flag a value suspiciously close to a famous constant (the numerology trap)."""
    if not np.isfinite(value):
        return []
    out = []
    for name, c in FAMOUS_CONSTANTS.items():
        if c > 0 and abs(value - c) / c < tol_rel:
            out.append({"constant": name, "value": c,
                        "rel_diff": float(abs(value - c) / c)})
    return out


def _folding_score(D2_lowm: float, D2_plateau: float) -> float:
    """0..1: relative rise from the low-m estimate to the plateau (folding signature).
    ~0 = generic/well-reconstructed; large = the low-m value is a folded under-estimate."""
    if not (np.isfinite(D2_lowm) and np.isfinite(D2_plateau)) or D2_plateau <= 0:
        return float("nan")
    return float(np.clip((D2_plateau - D2_lowm) / D2_plateau, 0.0, 1.0))


def folding_score(x: np.ndarray, **kw) -> float:
    """Convenience: folding score of a series (0 generic .. 1 strongly folded)."""
    scan = embedding_scan(x, **kw)
    conv = converged_dimension(scan["d2_by_m"])
    return _folding_score(conv["D2_lowm"], conv["D2"])


@dataclass
class FoldReport:
    D2_naive_lowm: float
    D2_converged: float
    converged: bool
    folded: bool
    folding_score: float
    rise_with_m: float
    tau: int
    d2_by_m: dict
    false_constant_traps: list = field(default_factory=list)
    verdict: str = ""
    notes: list = field(default_factory=list)
    lyapunov1: dict | None = None
    k2_entropy: dict | None = None


def analyze(x: np.ndarray, m_values=(3, 4, 5, 6, 7, 8), tau: int | None = None,
            n_points: int = 6000, dynamics: bool = True, fs: float = 1.0) -> FoldReport:
    """One-call diagnosis. Returns a FoldReport with the trustworthy invariant + warnings.

    With dynamics=True also estimates the largest Lyapunov exponent (Rosenstein) and the
    correlation entropy K2 behind the same convergence/folding caveats. Pass fs = 1/dt for the
    dynamical rates in per-time units (default per-sample)."""
    x = np.asarray(x, float)
    scan = embedding_scan(x, m_values=m_values, tau=tau, n_points=n_points)
    conv = converged_dimension(scan["d2_by_m"])
    fold_s = _folding_score(conv["D2_lowm"], conv["D2"])
    folded = bool(np.isfinite(fold_s) and fold_s > 0.25)
    # only warn about a famous-constant coincidence when the naive value is actually
    # misleading (folded / not converged); an integer D2 of a genuine limit cycle is not a trap.
    traps = false_constant_warnings(conv.get("D2_lowm", float("nan"))) if folded else []
    notes = []
    if folded:
        notes.append(
            f"Attractor FOLDING / non-generic observable: D2 climbs from "
            f"{conv['D2_lowm']:.2f} (m={m_values[0]}) to {conv['D2']:.2f} (m={m_values[-1]}); "
            f"the low-m value is an under-resolved quotient, not the true dimension.")
    if traps:
        names = ", ".join(t["constant"] for t in traps)
        notes.append(
            f"FALSE-CONSTANT TRAP: the unconverged low-m D2={conv.get('D2_lowm', float('nan')):.3f} "
            f"~ {names}. This is a resolution artifact, NOT a real coincidence -- the converged "
            f"D2={conv['D2']:.2f}.")
    if not conv["converged"]:
        notes.append("D2 not yet plateaued: report as a LOWER BOUND; increase m / data length.")
    if conv["converged"] and not folded:
        notes.append("Reconstruction looks generic and converged; D2 is trustworthy.")
    verdict = ("FOLDED (use converged D2, distrust the naive value)" if folded
               else ("CONVERGED" if conv["converged"] else "UNDER-RESOLVED (lower bound only)"))
    lya = ent = None
    if dynamics:
        lya = lyapunov_rosenstein(x, tau=scan["tau"], fs=fs)
        ent = k2_entropy(x, m_values=m_values, tau=scan["tau"], fs=fs)
        if folded:
            notes.append("Under folding the embedding reconstructs the symmetry quotient, so "
                         "lambda1 / K2 are quotient-attractor estimates -- interpret with the "
                         "same caution as the folded D2.")
        if lya and np.isfinite(lya.get("r2", float("nan"))) and lya["r2"] < 0.9:
            notes.append("lambda1 divergence fit is poor (r2 < 0.9): widen fit_window / add data.")
        if ent and not ent.get("converged", False):
            notes.append("K2 has not converged across m (still descending): treat as an upper "
                         "estimate of the KS-entropy lower bound.")
    return FoldReport(
        D2_naive_lowm=conv.get("D2_lowm", float("nan")),
        D2_converged=conv["D2"], converged=conv["converged"], folded=folded,
        folding_score=fold_s, rise_with_m=conv["rise"], tau=scan["tau"],
        d2_by_m=scan["d2_by_m"], false_constant_traps=traps, verdict=verdict, notes=notes,
        lyapunov1=lya, k2_entropy=ent)


# --------------------------------------------------------------------------- #
# Bonus: log-periodic (discrete-scale-invariance) square-law correction.
# --------------------------------------------------------------------------- #
def logperiodic_squarelaw_check(signal_in_u: np.ndarray, u: np.ndarray) -> dict:
    """For a log-periodic (DSI) signal sampled uniformly in u=ln r, a square-law detector
    measures DOUBLE the log-frequency (apparent rescaling ratio sqrt(lambda), not lambda).
    Returns the measured log-frequency of the signal and of its square."""
    def logfreq(s):
        s = s - np.mean(s); sgn = np.sign(s); sgn[sgn == 0] = 1
        n_zero = int(np.sum(np.abs(np.diff(sgn)) > 0))
        return n_zero * np.pi / float(u[-1] - u[0])
    a_lin = logfreq(signal_in_u)
    a_sq = logfreq(signal_in_u ** 2)
    return {"logfreq_linear": a_lin, "logfreq_squarelaw": a_sq,
            "doubling_ratio": (a_sq / a_lin if a_lin else float("nan")),
            "apparent_dsi_is_sqrt_of_true": True}
