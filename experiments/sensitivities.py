
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # runnable without pip install -e .
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)

import numpy as np
from hfgs import SNLP, grid_scheme, midpoints, dLxi_dc, dLxi_dlam

print("Exact sensitivities vs closed form / central differences")
print("="*74)
# delta = 0, kappa = 1, Exp(b):  psi(u) = (lam/(c b)) e^{-(b-lam/c)u}
mod = SNLP(c=25.0, lam=20.0, w=[1.0], b=[1.0]); d=0.0
rho = mod.phi(d); J,N = 8, 1<<20
u = midpoints(J,N)
gc = np.real(grid_scheme(lambda z: dLxi_dc(mod,z,d,rho), J,N,"shannon"))
gl = np.real(grid_scheme(lambda z: dLxi_dlam(mod,z,d,rho), J,N,"shannon"))
lam,c,b = 20.0,25.0,1.0
print("      u      d psi/dc  (exact)      rel.err     d psi/dlam (exact)     rel.err")
for uu in (1.,5.,10.,20.):
    i=int(round(uu*2**J-0.5)); x=u[i]
    psi = lam/(c*b)*np.exp(-(b-lam/c)*x)
    ec = psi*(-1.0/c - x*lam/c**2)
    el = psi*( 1.0/lam + x/c)
    print(f"   {x:6.3f}   {gc[i]: .10e} ({ec: .3e})  {abs(gc[i]/ec-1):.1e}"
          f"   {gl[i]: .10e} ({el: .3e})  {abs(gl[i]/el-1):.1e}")

# delta > 0, kappa = y^3  (rho moves with the parameter -- the hard case)
print()
print("delta=0.04, kappa=y^3, c=3, lam=1.5, Exp(0.7):  rho depends on theta")
m2 = SNLP(c=3.0, lam=1.5, w=[1.0], b=[0.7]); d2=0.04; kap=("power",3)
r2 = m2.phi(d2)
J,N = 8, 1<<20; u = midpoints(J,N)
gc2 = np.real(grid_scheme(lambda z: dLxi_dc(m2,z,d2,r2,kap), J,N,"shannon"))
gl2 = np.real(grid_scheme(lambda z: dLxi_dlam(m2,z,d2,r2,kap), J,N,"shannon"))
def exact_xi(uu, c=3.0, lam=1.5, beta=0.7, delta=0.04):
    mm = SNLP(c=c, lam=lam, w=[1.0], b=[beta]); r = float(np.real(mm.phi(delta)))
    q = lam/(c*(beta+r)); return (6/beta**3)*q*np.exp(-beta*(1-q)*uu)
eps=1e-5
print("      u     dxi/dc analytic     central diff        rel      dxi/dlam analytic   central diff        rel")
for uu in (1.,5.,10.,20.):
    i=int(round(uu*2**J-0.5)); x=u[i]
    fc=(exact_xi(x,c=3.0+eps)-exact_xi(x,c=3.0-eps))/(2*eps)
    fl=(exact_xi(x,lam=1.5+eps)-exact_xi(x,lam=1.5-eps))/(2*eps)
    print(f"   {x:6.3f}  {gc2[i]: .10e}  {fc: .10e}  {abs(gc2[i]/fc-1):.1e}"
          f"   {gl2[i]: .10e}  {fl: .10e}  {abs(gl2[i]/fl-1):.1e}")
