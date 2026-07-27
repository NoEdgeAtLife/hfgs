import numpy as np
from gsx import SNLP, grid_scheme, midpoints

np.set_printoptions(precision=12)

# ---- Example 5.1 : lam=20, Exp(1), c=25, delta=0  ->  psi(u) = 0.8 e^{-0.2u}
mod = SNLP(c=25.0, lam=20.0, w=[1.0], b=[1.0])
rho = mod.phi(0.0)
print("rho =", rho)
print("xi(0)  = Lw(rho)/c =", float(np.real(mod.Lw(rho))) / mod.c, "  exact 0.8")
print("int xi = Lxi(0)    =", float(np.real(mod.Lxi(np.array([0.0 + 0j]), 0.0, rho)[0])),
      "  exact 4.0")
J0J1 = mod.boundary_data(0.0, rho, K=4)
print("boundary data (xi(0+), xi'(0+), xi''(0+), xi'''(0+)) =", J0J1)
print("        exact                                        =",
      [0.8 * (-0.2) ** k for k in range(4)])

exact = lambda u: 0.8 * np.exp(-0.2 * u)
f = lambda z: mod.Lxi(z, 0.0, rho)

NBIG = 1 << 22
U = 5.0
print("\n rel. error at u ~ 5 (N = 2^22, so aliasing is nil)")
print("   J |  node/haar   mid/haar    mid/shannon  mid/shannon+corr")
rows = []
for J in (4, 5, 6, 7, 8, 9, 10, 11, 12):
    h = 2.0 ** -J
    m = int(round(U / h - 0.5))
    xm = (m + 0.5) * h
    vh = np.real(grid_scheme(f, J, NBIG, "haar"))[m]
    vs = np.real(grid_scheme(f, J, NBIG, "shannon"))[m]
    vc = np.real(grid_scheme(f, J, NBIG, "shannon", corr=tuple(J0J1[:2])))[m]
    e_node = abs(vh / exact(m * h) - 1)
    e_hm = abs(vh / exact(xm) - 1)
    e_sm = abs(vs / exact(xm) - 1)
    e_cm = abs(vc / exact(xm) - 1)
    rows.append((J, e_node, e_hm, e_sm, e_cm))
    print(f"  {J:2d} | {e_node:.4e}  {e_hm:.4e}  {e_sm:.4e}  {e_cm:.4e}")

print("\n observed orders (log2 of successive ratios)")
for i in range(1, len(rows)):
    o = [np.log2(rows[i - 1][k] / rows[i][k]) for k in range(1, 5)]
    print(f"  {rows[i-1][0]}->{rows[i][0]} : " + "  ".join(f"{v:6.3f}" for v in o))

# ---------- Theorem 2/3 : does the predicted error formula match? ----------
print("\n prediction check, filter='shannon', u ~ 5, m even/odd matters")
print("   J |   measured        predicted (Thm 3)   ratio")
xi0, xi1 = 0.8, -0.16
for J in (6, 8, 10, 12, 14):
    h = 2.0 ** -J
    m = int(round(U / h - 0.5))
    x = (m + 0.5) * h
    v = np.real(grid_scheme(f, J, NBIG, "shannon"))[m]
    meas = v - exact(x)
    pred = -((-1.0) ** m) * (h ** 2 / np.pi ** 3) * (xi0 / x ** 2 + xi1 / x)
    print(f"  {J:2d} | {meas: .6e}   {pred: .6e}   {meas/pred if pred!=0 else np.nan:8.4f}")

print("\n same, at a point where the two singular terms do NOT cancel (u ~ 2)")
U2 = 2.0
for J in (6, 8, 10, 12, 14):
    h = 2.0 ** -J
    m = int(round(U2 / h - 0.5))
    x = (m + 0.5) * h
    v = np.real(grid_scheme(f, J, NBIG, "shannon"))[m]
    meas = v - exact(x)
    pred = -((-1.0) ** m) * (h ** 2 / np.pi ** 3) * (xi0 / x ** 2 + xi1 / x)
    print(f"  {J:2d} | {meas: .6e}   {pred: .6e}   {meas/pred:8.4f}")

print("\n filter error check: haar minus shannon should be -h^2 xi''/24 (+O(h^4))")
for J in (6, 8, 10, 12):
    h = 2.0 ** -J
    m = int(round(U / h - 0.5))
    x = (m + 0.5) * h
    vh = np.real(grid_scheme(f, J, NBIG, "haar"))[m]
    vs = np.real(grid_scheme(f, J, NBIG, "shannon"))[m]
    pred = -(h ** 2 / 24.0) * 0.8 * 0.04 * np.exp(-0.2 * x)
    print(f"  J={J:2d}  measured {vh-vs: .6e}   predicted {pred: .6e}"
          f"   ratio {(vh-vs)/pred:8.4f}")
