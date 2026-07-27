import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gsx import SNLP, grid_scheme, midpoints, finite_horizon

plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 200, "axes.linewidth": .6,
                     "font.family": "serif"})

# ---------------------------------------------------------------- Figure 1
mod = SNLP(25.0, 20.0, [1.0], [1.0]); rho = mod.phi(0.0)
bd = tuple(mod.boundary_data(0.0, rho, K=2))
f = lambda z: mod.Lxi(z, 0.0, rho)
ex = lambda u: 0.8 * np.exp(-0.2 * u)
NB = 1 << 22
Js = np.arange(3, 12)
E = {k: [] for k in ("node", "haar", "shan", "corr")}
for J in Js:
    h = 2.0 ** -J
    m = int(round(2.0 / h - 0.5)); x = (m + .5) * h
    vh = np.real(grid_scheme(f, J, NB, "haar"))[m]
    vs = np.real(grid_scheme(f, J, NB, "shannon"))[m]
    vc = np.real(grid_scheme(f, J, NB, "shannon", corr=bd))[m]
    E["node"].append(abs(vh / ex(m * h) - 1))
    E["haar"].append(abs(vh / ex(x) - 1))
    E["shan"].append(abs(vs / ex(x) - 1))
    E["corr"].append(abs(vc / ex(x) - 1))

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
sty = [("node", "left endpoint, Haar filter (as published)", "o-", "#b3392b"),
       ("haar", "midpoint, Haar filter", "s-", "#d98c1f"),
       ("shan", "midpoint, $s\\equiv1$", "^-", "#2b6cb3"),
       ("corr", "midpoint, $s\\equiv1$, corrected", "d-", "#1f7a4d")]
for k, lab, ms, cl in sty:
    ax[0].semilogy(Js, E[k], ms, label=lab, color=cl, ms=3.2, lw=1.0)
for p, tx, yy in ((1, "$O(h)$", 3e-3), (2, "$O(h^2)$", 3e-5), (4, "$O(h^4)$", 3e-9)):
    ax[0].semilogy(Js, yy * 2.0 ** (-p * (Js - 3.0)), "k:", lw=.7)
ax[0].set_xlabel("resolution level $J$   ($h=2^{-J}$)")
ax[0].set_ylabel("relative error at $u\\approx2$")
ax[0].set_ylim(1e-16, 1e-1)
ax[0].legend(fontsize=6.2, loc="lower left", framealpha=.9)
ax[0].set_title("(a) convergence, Example 5.1", fontsize=8.5)

# predicted vs measured error curve, level J=8
J = 8; h = 2.0 ** -J
vs = np.real(grid_scheme(f, J, NB, "shannon"))
xs = midpoints(J, NB)
sl = slice(int(0.3 / h), int(12 / h))
mm = np.arange(NB)[sl]
meas = vs[sl] - ex(xs[sl])
pred = -((-1.0) ** mm) * (h ** 2 / np.pi ** 3) * (0.8 / xs[sl] ** 2 - 0.16 / xs[sl])
ax[1].plot(xs[sl], meas / h ** 2, lw=.5, color="#2b6cb3", label="measured")
ax[1].plot(xs[sl], pred / h ** 2, lw=.5, color="#b3392b", ls="--",
           label="Theorem 3 prediction")
ax[1].set_xlabel("$u$")
ax[1].set_ylabel("error $/\\,h^{2}$")
ax[1].set_title("(b) error profile at $J=8$, uncorrected", fontsize=8.5)
ax[1].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig("fig1_convergence.png"); plt.close(fig)
print("fig1 done; error/h^2 envelope max", np.abs(meas / h ** 2).max())

# ---------------------------------------------------------------- Figure 2
ts = np.load("credit_ts.npy"); lad = np.load("credit_ladder.npy")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax[0].plot(ts[:, 0], ts[:, 2], "o-", color="#1f4e79", ms=3, lw=1.1)
ax[0].set_xlabel("maturity $T$ (years)"); ax[0].set_ylabel("credit spread (bp)")
ax[0].set_title("(a) spread term structure, $D/V=60\\%$", fontsize=8.5)
ax[0].set_ylim(bottom=0)
xx, pd_, sp = lad
msk = (xx > 0.05) & (xx < 1.6)
ax[1].plot(np.exp(-xx[msk]), sp[msk], color="#1f4e79", lw=1.1)
ax[1].set_xlabel("leverage $D/V$"); ax[1].set_ylabel("5y spread (bp)")
ax[1].set_title("(b) the whole ladder, one FFT pass", fontsize=8.5)
fig.tight_layout(); fig.savefig("fig2_credit.png"); plt.close(fig)
print("fig2 done")

# ---------------------------------------------------------------- Figure 3
W = np.array([0.70, 0.28, 0.02]); BR = 1.0 / np.array([0.15, 0.80, 6.00])
LAM = 120.0
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
xs = midpoints(5, 1 << 14)
for c, lab, cl in ((68.7, "$c=68.7$ (util. 0.78)", "#1f7a4d"),
                   (60.0, "$c=60.0$ (util. 0.90)", "#2b6cb3"),
                   (52.0, "$c=52.0$ (util. 1.04)", "#b3392b")):
    v = finite_horizon(SNLP(c, LAM, W, BR), 60.0, 5, 1 << 14)
    k = (xs > 5) & (xs < 300)
    ax[0].semilogy(xs[k], np.maximum(v[k], 1e-12), lw=1.1, label=lab, color=cl)
ax[0].axhline(0.005, color="k", ls=":", lw=.8)
ax[0].text(250, 0.0062, "SLO 0.5%", fontsize=6.5, ha="right")
ax[0].set_xlabel("backlog threshold $B$ (ktok)")
ax[0].set_ylabel("$P(\\mathrm{backlog}\\geq B$ in 60 s$)$")
ax[0].set_ylim(1e-6, 1.5)
ax[0].legend(fontsize=6.3); ax[0].set_title("(a) buffer ladder", fontsize=8.5)

fr = np.load("frontier.npy")
fin = np.isfinite(fr[:, 0])
ax[1].plot(fr[fin, 0], fr[fin, 2], "o-", color="#1f4e79", ms=3, lw=1.1,
           label="total")
ax[1].plot(fr[fin, 0], fr[fin, 1] / 2.0, "s--", color="#2b6cb3", ms=2.6, lw=.9,
           label="in-house capacity")
ax[1].plot(fr[fin, 0], fr[fin, 2] - fr[fin, 1] / 2.0, "^--", color="#b3392b",
           ms=2.6, lw=.9, label="offload")
ax[1].axhline(fr[~fin, 2][0], color="k", ls=":", lw=.8)
ax[1].text(46, fr[~fin, 2][0] + .4, "no offloading", fontsize=6.5, ha="right")
ax[1].set_xscale("log"); ax[1].set_xlabel("offload threshold $M$ (ktok)")
ax[1].set_ylabel("cost (replica-equivalents)")
ax[1].legend(fontsize=6.3); ax[1].set_title("(b) offload frontier", fontsize=8.5)
fig.tight_layout(); fig.savefig("fig3_serving.png"); plt.close(fig)
print("fig3 done")
