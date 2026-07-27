import numpy as np
from scipy.integrate import quad
from gsx import SNLP, grid_scheme, boundary_from

W=np.array([0.70,0.28,0.02]); BR=1.0/np.array([0.15,0.80,6.00]); LAM=120.0
def ref_at(md,d,rho,x,gam=1.0):
    """xi(x) to ~1e-12 by direct quadrature of the (singularity-subtracted)
    inverse Fourier integral -- independent of any grid."""
    J0,J1 = boundary_from(lambda z: md.Lxi(z,d,rho),K=2)
    a=J0; b=J1+gam*J0
    def integ(om):
        z=1j*om
        g=md.Lxi(np.array([z]),d,rho)[0]-(a/(gam+z)+b/(gam+z)**2)
        return float(np.real(g*np.exp(1j*om*x)))
    tot=0.0
    for lo,hi in [(0,10),(10,100),(100,1e3),(1e3,1e4),(1e4,1e5),(1e5,np.inf)]:
        tot+=quad(integ,lo,hi,limit=400,epsabs=1e-14,epsrel=1e-12)[0]
    return (a+b*x)*np.exp(-gam*x)+tot/np.pi

print("Kink at u = M from the retention atom: order study at a FIXED distance")
print("above the kink, against a grid-independent quadrature reference.\n")
for M in (8.0, 8.0+1.0/3.0):
    md=SNLP(70.0,LAM,W,BR,M=M); d=0.2; rho=md.phi(d)
    cor=tuple(md.boundary_data(d,rho,K=2))
    fn=lambda z: md.Lxi(z,d,rho)
    tag="on grid " if abs(M*32-round(M*32))<1e-12 else "off grid"
    errs=[]
    for Jl in (5,6,7,8,9):
        h=2.0**-Jl
        m=int(round((M+0.5)/h-0.5)); x=(m+0.5)*h     # nearest midpoint to M+0.5
        v=np.real(grid_scheme(fn,Jl,1<<22,"shannon",corr=cor))[m]
        errs.append(abs(v-ref_at(md,d,rho,x)))
    print(f"  M = {M:8.5f} ({tag}) : "+"  ".join(f"J={j}:{e:.2e}" for j,e in zip(range(5,10),errs)))
    print(f"      observed orders: "+" ".join(f"{np.log2(errs[i]/errs[i+1]):.2f}" for i in range(4)))
# and right next to the kink
print("\nfirst midpoint ABOVE the kink (worst case, distance h/2):")
for M in (8.0, 8.0+1.0/3.0):
    md=SNLP(70.0,LAM,W,BR,M=M); d=0.2; rho=md.phi(d)
    cor=tuple(md.boundary_data(d,rho,K=2)); fn=lambda z: md.Lxi(z,d,rho)
    errs=[]
    for Jl in (5,6,7,8,9):
        h=2.0**-Jl
        m=int(np.floor(M/h)); x=(m+0.5)*h
        v=np.real(grid_scheme(fn,Jl,1<<22,"shannon",corr=cor))[m]
        errs.append(abs(v-ref_at(md,d,rho,x)))
    tag="on grid " if abs(M*32-round(M*32))<1e-12 else "off grid"
    print(f"  M = {M:8.5f} ({tag}) : "+"  ".join(f"J={j}:{e:.2e}" for j,e in zip(range(5,10),errs)))
    print(f"      observed orders: "+" ".join(f"{np.log2(errs[i]/errs[i+1]):.2f}" for i in range(4)))
