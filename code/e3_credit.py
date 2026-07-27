"""Application 1 -- finite-maturity credit under a spectrally negative Levy
firm-value model, with recovery that depends on the overshoot at default."""
import time
import numpy as np
from gsx import SNLP, grid_scheme, midpoints, finite_horizon

rng = np.random.default_rng(20260727)

# ------------------------------------------------------------------ model
r, qdiv, alpha = 0.04, 0.0, 0.30          # rate, dividend, deadweight loss
lam = 0.5                                  # downward jump intensity  (per yr)
wj, bj = [0.90, 0.10], [20.0, 2.0]         # log-jump mixture: means 5% and 50%
sig = 0.20


def make(sigma):
    tmp = SNLP(c=0.0, lam=lam, w=wj, b=bj, sigma=sigma)
    A1 = float(np.real(tmp.A(1.0)))
    c = r - qdiv - 0.5 * sigma ** 2 + A1                 # psi(1) = r - q
    return SNLP(c=c, lam=lam, w=wj, b=bj, sigma=sigma)


mod = make(sig)
print("=" * 78)
print("1.  Risk-neutral calibration")
print("=" * 78)
print(f"  c = {mod.c:.6f},  sigma = {sig},  check psi(1) - (r-q) = "
      f"{float(np.real(mod.psi(1.0))) - (r - qdiv):.2e}")
x0 = np.log(1 / 0.60)                       # leverage D/V = 60%
print(f"  log-leverage x0 = log(1/0.6) = {x0:.6f}")

KAP = ("expo", 1.0)                          # kappa(x,y) = e^{-y};  scale by 1-alpha
REC = 1.0 - alpha

# ------------------------------------------------------------- validation
print()
print("=" * 78)
print("2.  Validation against EXACT Monte Carlo  (sigma = 0, so ruin can only")
print("    happen at a jump epoch and the simulation is unbiased)")
print("=" * 78)
mod0 = make(0.0)
J, N = 6, 1 << 18
h = 2.0 ** -J
m = int(round(x0 / h - 0.5))
xg = (m + 0.5) * h


def mc(T, npath=4_000_000, batch=500_000):
    """(P(tau<=T),  E[e^{-r tau} e^{-|X_tau|} 1{tau<=T}])  exact simulation."""
    pd = 0.0
    rec = 0.0
    n = 0
    while n < npath:
        k = min(batch, npath - n)
        n += k
        t = np.zeros(k)
        x = np.full(k, xg)
        alive = np.ones(k, bool)
        while alive.any():
            idx = np.flatnonzero(alive)
            dt = rng.exponential(1.0 / mod0.lam, idx.size)
            t2 = t[idx] + dt
            over = t2 > T
            x2 = x[idx] + mod0.c * dt
            comp = rng.random(idx.size) < wj[0]
            jump = np.where(comp, rng.exponential(1 / bj[0], idx.size),
                            rng.exponential(1 / bj[1], idx.size))
            x2 = x2 - jump
            ruin = (~over) & (x2 < 0)
            pd += ruin.sum()
            rec += np.sum(np.exp(-r * t2[ruin]) * np.exp(x2[ruin]))
            done = over | ruin
            t[idx] = t2
            x[idx] = x2
            alive[idx[done]] = False
    return pd / npath, rec / npath


print("     T     P(default) scheme      MC (4e6 paths)   |diff|      recovery leg"
      "  scheme      MC          |diff|")
for T in (1.0, 3.0, 5.0):
    p_s = finite_horizon(mod0, T, J, N, delta=0.0)[m]
    rl_s = finite_horizon(mod0, T, J, N, delta=r, kappa=KAP)[m]
    p_m, rl_m = mc(T)
    se_p = np.sqrt(p_m * (1 - p_m) / 4e6)
    print(f"   {T:4.1f}     {p_s:.8f}        {p_m:.8f}   {abs(p_s-p_m):.1e}"
          f"  (2se {2*se_p:.1e})    {rl_s:.8f}  {rl_m:.8f}  {abs(rl_s-rl_m):.1e}")

# ------------------------------------------------- term structure (sigma>0)
print()
print("=" * 78)
print("3.  Term structure at 60% leverage  (sigma = 0.20)")
print("=" * 78)
print("     T     PD(T)      E[e^-rT] surv    recovery leg    bond price   spread bp"
      "   LGD given default")
t0 = time.time()
rows = []
for T in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
    pd_ = finite_horizon(mod, T, J, N, delta=0.0)[m]
    rl = REC * finite_horizon(mod, T, J, N, delta=r, kappa=KAP)[m]
    B = np.exp(-r * T) * (1 - pd_) + rl
    spr = (-np.log(B) / T - r) * 1e4
    lgd = 1 - rl / (np.exp(-r * T) * pd_) if pd_ > 0 else np.nan
    rows.append((T, pd_, spr))
    print(f"   {T:5.2f}   {pd_:.6f}     {np.exp(-r*T)*(1-pd_):.6f}      {rl:.6f}"
          f"      {B:.6f}    {spr:7.2f}       {lgd:.4f}")
print(f"  [{time.time()-t0:.1f} s for the whole curve, each point a full grid solve]")

# --------------------------------------------------------- the spot ladder
print()
print("=" * 78)
print("4.  One pass returns the whole leverage ladder (T = 5y)")
print("=" * 78)
t0 = time.time()
pdv = finite_horizon(mod, 5.0, J, N, delta=0.0)
rlv = REC * finite_horizon(mod, 5.0, J, N, delta=r, kappa=KAP)
el = time.time() - t0
xs = midpoints(J, N)
Bv = np.exp(-r * 5.0) * (1 - pdv) + rlv
sprv = (-np.log(np.maximum(Bv, 1e-300)) / 5.0 - r) * 1e4
print(f"  {el:.2f} s for {N} leverage points  ->  {1e6*el/N:.3f} microsec/point")
print("     D/V     x        PD(5y)     spread bp   dSpread/dx (bp)   [from same grid]")
for lv in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
    xx = np.log(1 / lv)
    i = int(round(xx * 2 ** J - 0.5))
    dsp = (sprv[i + 1] - sprv[i - 1]) / (2 * h)
    print(f"    {lv:.2f}   {xs[i]:.4f}   {pdv[i]:.6f}    {sprv[i]:8.2f}    {dsp:10.2f}")

np.save("credit_ladder.npy", np.vstack([xs[:1 << 12], pdv[:1 << 12], sprv[:1 << 12]]))
np.save("credit_ts.npy", np.array(rows))
