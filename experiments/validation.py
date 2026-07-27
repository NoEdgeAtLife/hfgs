
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # runnable without pip install -e .
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)

import numpy as np
from scipy.integrate import quad
from hfgs import SNLP, grid_scheme, midpoints, euler_invert, finite_horizon, gaver_digits

# =====================================================================
print("=" * 76)
print("A.  delta > 0, kappa = y^3  (thesis Example 5.13: c=3, lam=1.5, Exp(0.7))")
print("=" * 76)
lam, beta, c, delta = 1.5, 0.7, 3.0, 0.04
mod = SNLP(c=c, lam=lam, w=[1.0], b=[beta])
kap = ("power", 3)
rho = mod.phi(delta)
q = lam / (c * (beta + float(np.real(rho))))
R = beta * (1 - q)
exact = lambda u: (6.0 / beta ** 3) * q * np.exp(-R * u)
print(f"  rho = {float(np.real(rho)):.10f}   xi(0) exact = {exact(0):.6f}"
      f"   decay R = {R:.6f}")
print(f"  free check  Lw(rho)/c = {float(np.real(mod.Lw(rho, kap)))/c:.10f}")
print(f"  free check  Lxi(0)    = "
      f"{float(np.real(mod.Lxi(np.array([0j]), delta, rho, kap)[0])):.6f}"
      f"   (= [Lw(0)-Lw(rho)]/delta = "
      f"{float(np.real(mod.Lw(0.0,kap)-mod.Lw(rho,kap)))/delta:.6f})")
bd = mod.boundary_data(delta, rho, kap, K=2)
print(f"  boundary data  xi(0+) = {bd[0]:.10f} (exact {exact(0):.10f}),"
      f"  xi'(0+) = {bd[1]:.10f} (exact {-R*exact(0):.10f})")

f = lambda z: mod.Lxi(z, delta, rho, kap)
NB = 1 << 22
print("\n   rel. error at u ~ 10:  node/haar    mid/haar    mid/shan   mid/shan+corr")
rows = []
for J in (4, 6, 8, 10):
    h = 2.0 ** -J
    m = int(round(10.0 / h - 0.5)); x = (m + .5) * h
    vh = np.real(grid_scheme(f, J, NB, "haar"))[m]
    vs = np.real(grid_scheme(f, J, NB, "shannon"))[m]
    vc = np.real(grid_scheme(f, J, NB, "shannon", corr=tuple(bd)))[m]
    r = (abs(vh / exact(m * h) - 1), abs(vh / exact(x) - 1),
         abs(vs / exact(x) - 1), abs(vc / exact(x) - 1))
    rows.append(r)
    print(f"      J = {J:2d}          " + "  ".join(f"{v:.3e}" for v in r))
print("   orders per doubling of J: " +
      "  ".join(f"{np.log2(rows[0][k]/rows[1][k])/2:5.2f}" for k in range(4)) +
      "   |  " +
      "  ".join(f"{np.log2(rows[2][k]/rows[3][k])/2:5.2f}" for k in range(4)))

# =====================================================================
print()
print("=" * 76)
print("B.  Gaussian component (creeping):  sigma > 0")
print("=" * 76)
mg = SNLP(c=1.0, lam=0.9, w=[1.0], b=[1.0], sigma=0.5)
print(f"  psi'(0+) = c - mu1 = {mg.c - mg.mu1():.4f}  (positive loading)")
d = 0.0
rg = mg.phi(d)
print(f"  rho(0) = {float(np.real(rg)):.3e}")
# exact 0-scale function by partial fractions: 1/psi(z) has simple poles
#   psi(z) = c z + s^2 z^2/2 - lam z/(z+b)  ->  z * P(z) / (z+b)
sig, b0 = mg.sigma, mg.b[0]
# psi(z) = z*[c + s^2 z/2 - lam/(z+b)] = z*(c(z+b) + s^2 z(z+b)/2 - lam)/(z+b)
poly = np.polynomial.Polynomial
num = poly([mg.c * b0 - mg.lam, mg.c + sig ** 2 * b0 / 2, sig ** 2 / 2])   # in z
roots = num.roots()
print("  roots of the cubic psi(z)=0 :", np.sort(np.append(roots, 0.0)))


def W0(u):
    """0-scale function: L{W}(z) = 1/psi(z) = (z+b)/(z * num(z))."""
    tot = np.zeros_like(np.asarray(u, float))
    # partial fractions of (z+b) / (z * prod(z - r_i))
    allr = np.append(roots, 0.0)
    lead = sig ** 2 / 2
    for r in allr:
        others = [rr for rr in allr if rr != r]
        res = (r + b0) / (lead * np.prod([r - rr for rr in others]))
        tot = tot + np.real(res * np.exp(r * np.asarray(u, float)))
    return tot


uu = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
psi_exact = 1.0 - (mg.c - mg.mu1()) * W0(uu)
print("  u                :", uu)
print("  1-(c-mu1)W(u)    :", np.round(psi_exact, 10))
fg = lambda z: mg.Lxi(z, 0.0, rg)
J, N = 10, 1 << 22
v = np.real(grid_scheme(fg, J, N, "shannon", corr=tuple(mg.boundary_data(0.0, rg, K=2))))
xs = midpoints(J, N)
idx = [int(round(t * 2 ** J - 0.5)) for t in uu]
print("  scheme           :", np.round(v[idx], 10))
print("  |diff|           :", np.abs(v[idx] - (1.0 - (mg.c - mg.mu1()) * W0(xs[idx]))))
print(f"  xi(0+) = kappa(0,0) = 1 ?  boundary_data gives "
      f"{mg.boundary_data(0.0, rg, K=2)[0]:.10f}")

# =====================================================================
print()
print("=" * 76)
print("C.  finite horizon: Euler inversion vs the exact Prabhu/Seal formula")
print("    (lam=0.9, c=1, Exp(1), delta=0, u ~ 10)")
print("=" * 76)
m5 = SNLP(c=1.0, lam=0.9, w=[1.0], b=[1.0])


def exact_surv(u, t, lam=0.9):
    sl = np.sqrt(lam)

    def g(th):
        f1 = lam * np.exp(2 * sl * t * np.cos(th) - (1 + lam) * t
                          + u * (sl * np.cos(th) - 1))
        f2 = np.cos(u * sl * np.sin(th)) - np.cos(u * sl * np.sin(th) + 2 * th)
        return f1 * f2 / (1 + lam - 2 * sl * np.cos(th))
    return 1 - lam * np.exp(-(1 - lam) * u) + quad(g, 0, np.pi, limit=400)[0] / np.pi


J, N = 6, 1 << 18
h = 2.0 ** -J
m = int(round(10.0 / h - 0.5)); x = (m + .5) * h
print("      t        scheme            exact          abs err")
for t in (1.0, 5.0, 10.0, 20.0, 50.0):
    v = finite_horizon(m5, t, J, N)[m]
    ex = 1 - exact_surv(x, t)
    print(f"   {t:5.1f}   {v:.12f}   {ex:.12f}   {abs(v-ex):.2e}")

print("\n  a-priori bound of Thm 6:  psi_inf(u) e^{-A}/(1-e^{-A}) with A=18.4")
psinf = 0.9 * np.exp(-0.1 * x)
print(f"     psi_inf({x:.3f}) = {psinf:.6f} ->  bound = "
      f"{psinf*np.exp(-18.4)/(1-np.exp(-18.4)):.3e}")

print("\n  digits destroyed by cancellation in 2N-term Gaver-Stehfest:")
for N_ in (2, 4, 6, 8, 10, 12, 16, 20):
    print(f"     N = {N_:2d}   log10 sum|a_n(N)|/n = {gaver_digits(N_):6.2f}")
