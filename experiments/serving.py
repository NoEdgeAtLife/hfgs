"""Applications 2 and 3 (final).

(2) LLM inference serving: transient overflow of the work backlog = finite
    horizon ruin (Asmussen duality); offload threshold M = excess-of-loss.
(3) The scheme as a differentiable layer: exact gradients that keep the
    fourth-order rate because the *corrector* is differentiated too.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # runnable without pip install -e .
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)

import time
import numpy as np
from hfgs import (SNLP, grid_scheme, midpoints, finite_horizon, threshold,
                  dLxi_dc, dLxi_dlam, euler_invert, boundary_from)

W = np.array([0.70, 0.28, 0.02])
MEANS = np.array([0.15, 0.80, 6.00])
BR = 1.0 / MEANS
LAM, MU_REP, BUF, TWIN, ALPHA, THETA_R = 120.0, 2.0, 60.0, 60.0, 0.005, 0.60
EX = float(W @ MEANS)
J, N = 5, 1 << 14
ib = int(round(BUF * 2 ** J - 0.5))


def EXM(M):
    """E[X ^ M] for the exponential mixture."""
    return float(np.sum(W * (1.0 - np.exp(-BR * M)) / BR)) if np.isfinite(M) else EX


def mk(c, M=np.inf):
    return SNLP(c, LAM, W, BR, M=M)


def prob(c, M=np.inf, T=TWIN, Jl=J, Nl=N):
    return finite_horizon(mk(c, M), T, Jl, Nl)[int(round(BUF * 2 ** Jl - 0.5))]


def dprob_dc(c, M=np.inf, T=TWIN, corrected=True):
    """d/dc of the finite-horizon curve.  The derivative transform is pushed
    through the SAME linear scheme, and its own boundary data is extracted so
    that the fourth-order correction survives differentiation (Theorem 8)."""
    prev = [None]

    def Fhat(s):
        md = mk(c, M)
        rho = md.phi(s, rho0=prev[0]); prev[0] = rho
        dfun = lambda z: dLxi_dc(md, z, s, rho)
        cor = tuple(boundary_from(dfun, K=2)) if corrected else None
        return grid_scheme(dfun, J, N, "shannon", corr=cor) / s
    return np.real(euler_invert(Fhat, T))[ib]


def size(M=np.inf, verbose=False):
    """c* with P = ALPHA: bisection into the well-resolved region, then
    Newton on log P using the exact derivative."""
    lo = 1.001 * LAM * EXM(M)                    # utilisation just under 1
    hi = lo
    while prob(hi, M) > ALPHA:
        hi *= 1.3
    nb = 0
    while hi / lo > 1.02:                        # bisect on the SIGN only
        mid = 0.5 * (lo + hi); nb += 1
        if prob(mid, M) > ALPHA:
            lo = mid
        else:
            hi = mid
    c, nn = hi, 0
    for nn in range(1, 12):
        p = prob(c, M); d = dprob_dc(c, M)
        if verbose:
            print(f"     newton {nn}   c = {c:10.6f}   P = {p:.10e}"
                  f"   dP/dc = {d: .4e}")
        step = (np.log(p) - np.log(ALPHA)) * p / d
        c -= step
        if abs(step) < 1e-11 * c:
            break
    return c, nb, nn


# ---------------------------------------------------------------------------
print("=" * 78)
print("A.  Workload and the buffer ladder")
print("=" * 78)
print(f"  E[work] = {EX:.4f} ktok/request, offered load = {LAM*EX:.2f} ktok/s,"
      f"  P(X > 10 ktok) = {float(W @ np.exp(-BR*10)):.5f}")
t0 = time.time()
lad = finite_horizon(mk(60.0), TWIN, 5, 1 << 16)
el = time.time() - t0
xs = midpoints(5, 1 << 16)
print(f"  R = 30 replicas (utilisation {LAM*EX/60:.4f}); one pass = {el:.2f} s for"
      f" {1<<16} buffer levels ({1e6*el/(1<<16):.1f} us each)")
for b_ in (20, 40, 60, 100, 150, 200):
    print(f"      P(backlog >= {b_:3d} ktok within 60 s) = "
          f"{lad[int(round(b_*32-0.5))]:.6e}")
print(f"  inverted: buffer needed for 0.5% = {threshold(lad, xs, ALPHA):.2f} ktok")

print()
print("=" * 78)
print("B.  Transient overload (utilisation > 1: infinite-horizon theory is void)")
print("=" * 78)
ovc = 52.0
print(f"  c = {ovc} ktok/s vs offered {LAM*EX:.2f}: utilisation "
      f"{LAM*EX/ovc:.3f} > 1, so psi_inf(B) = 1 for every B.")
print(f"  Lundberg root still unique in Re z > 0:  rho(0.1) = "
      f"{complex(mk(ovc).phi(0.1)):.6f}   rho(1+2i) = {complex(mk(ovc).phi(1+2j)):.6f}")
print("     window T (s)    P(>=60 ktok)   P(>=120)     P(>=240)")
for T in (5.0, 15.0, 30.0, 60.0, 120.0):
    vv = finite_horizon(mk(ovc), T, 5, 1 << 14)
    print(f"         {T:5.0f}       {vv[ib]:.5e}    "
          f"{vv[int(round(120*32-0.5))]:.5e}  {vv[int(round(240*32-0.5))]:.5e}")

print()
print("=" * 78)
print("C.  Sizing: bisection on the sign, then Newton on the exact derivative")
print("=" * 78)
t0 = time.time()
cst, nb, nn = size(verbose=True)
print(f"  c* = {cst:.8f} ktok/s = {cst/MU_REP:.4f} replicas   "
      f"[{nb} bisections + {nn} Newton steps, {time.time()-t0:.1f} s]")
print(f"  residual: P(c*) - alpha = {prob(cst) - ALPHA:.2e};  "
      f"utilisation at c* = {LAM*EX/cst:.4f};  integer answer "
      f"{int(np.ceil(cst/MU_REP))} replicas")

print()
print("=" * 78)
print("D.  Offload frontier (overflow pool priced at +60% per unit throughput)")
print("=" * 78)
print("      M (ktok)   E[X^M]   ceded ktok/s    c*(M)    replicas  offload   total")
t0 = time.time()
rows, best = [], (1e18, None)
for M in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 48.0, np.inf):
    cd = LAM * (EX - EXM(M))
    cs, _, _ = size(M)
    off = (1 + THETA_R) * cd / MU_REP
    tot = cs / MU_REP + off
    rows.append((M, cd, cs, tot))
    best = min(best, (tot, M))
    lbl = f"{M:6.1f}" if np.isfinite(M) else "   inf"
    print(f"      {lbl}    {EXM(M):.4f}   {cd:8.3f}   {cs:9.4f}   {cs/MU_REP:7.3f}"
          f"   {off:6.3f}   {tot:7.3f}")
print(f"  optimum M* ~ {best[1]} ktok: {best[0]:.3f} replica-equivalents vs "
      f"{rows[-1][3]:.3f} unmanaged  ({100*(1-best[0]/rows[-1][3]):.1f}% cheaper)")
print(f"  [{time.time()-t0:.1f} s for {len(rows)} complete designs]")
np.save(DATA / "frontier.npy", np.array([[m if np.isfinite(m) else np.inf, c, t]
                                  for m, _, c, t in rows]))

print()
print("=" * 78)
print("E.  Does the gradient keep the order?  (Theorem 8)")
print("=" * 78)
c0 = 68.0
an_c = dprob_dc(c0, corrected=True)
an_u = dprob_dc(c0, corrected=False)
e = 2e-2
d1 = (prob(c0 + e) - prob(c0 - e)) / (2 * e)
d2 = (prob(c0 + e / 2) - prob(c0 - e / 2)) / e
rich = (4 * d2 - d1) / 3
print(f"  Richardson central difference        {rich: .12e}   (reference)")
print(f"  analytic, corrector differentiated   {an_c: .12e}   rel {abs(an_c/rich-1):.1e}")
print(f"  analytic, corrector NOT differentiated {an_u: .12e} rel {abs(an_u/rich-1):.1e}")
print("  -> differentiating the corrector is what preserves the rate.")

print()
print("=" * 78)
print("F.  Atoms belong on the grid: a retention M puts a kink in xi at u = M")
print("=" * 78)
print("     infinite-horizon xi(M+1/2; delta=0.2), error vs a J=14 reference")
for M in (8.0, 8.0 + 1.0 / 3.0):
    md = mk(70.0, M); d = 0.2; rho = md.phi(d)
    fn = lambda z: md.Lxi(z, d, rho)
    cor = tuple(md.boundary_data(d, rho, K=2))
    ref = np.real(grid_scheme(fn, 14, 1 << 22, "shannon", corr=cor))[
        int(round((M + 0.5) * 2 ** 14 - 0.5))]
    out = []
    for Jl in (5, 6, 7, 8):
        v = np.real(grid_scheme(fn, Jl, 1 << 22, "shannon", corr=cor))[
            int(round((M + 0.5) * 2 ** Jl - 0.5))]
        out.append(abs(v - ref))
    tag = "on grid " if abs(M * 32 - round(M * 32)) < 1e-12 else "off grid"
    print(f"     M = {M:8.5f} ({tag}): " +
          "  ".join(f"J={j}:{v:.2e}" for j, v in zip((5, 6, 7, 8), out)) +
          "   orders " + " ".join(f"{np.log2(out[i]/out[i+1]):.2f}"
                                  for i in range(3)))
