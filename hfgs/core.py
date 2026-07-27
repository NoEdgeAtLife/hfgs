"""
core.py -- Gerber-Shiu / first-passage functionals of a spectrally negative Levy
process by a *filtered, half-shifted* Fourier scheme.

Accompanies the paper

    "Half-shifted Fourier inversion for finite-horizon Gerber-Shiu
     functionals: fourth-order accuracy at first-order cost, exact
     sensitivities, and applications to credit, capacity planning and
     differentiable design."

Model
-----
    X_t = u + c t + sigma B_t - L_t ,   L a subordinator, Levy measure nu
    psi(z) = c z + sigma^2 z^2 / 2 - A(z),   A(z) = int (1 - e^{-zx}) nu(dx)
    tau    = inf{t : X_t < 0}
    xi(u)  = E_u[ e^{-delta tau} kappa(X_{tau-}, |X_tau|) 1{tau < inf} ]

Master transform (Theorem 1 of the paper; kappa_0 = kappa(0,0)):

    L xi(z) = [ Lw(z) - Lw(rho) - kappa_0 (sigma^2/2)(z - rho) ] / (delta - psi(z))
    w(x)    = int_0^inf kappa(x,y) nu(x + dy),      rho = Phi(delta).

Discretisation
--------------
    A_m[s] = (1/2pi) int_{|omega| <= pi/h} xihat(omega) e^{i omega x_{m+1/2}} s(omega h) domega
    x_{m+1/2} = (m + 1/2) h,   h = 2^{-J}

evaluated by one N-point FFT.  s = 1/sinc(th/2) is the Haar scheme of Tse
(2020); s = sinc(th/2) returns cell averages; s = 1 ("shannon") is the
half-shifted band-limited projection and is the one the paper recommends.
With the two-term singularity correction (`corrected=True`) the scheme is
fourth order.
"""
from __future__ import annotations

import numpy as np
from math import comb, factorial

__all__ = ["SNLP", "grid_scheme", "midpoints", "euler_invert",
           "gaver_digits", "boundary_from", "finite_horizon", "threshold",
           "dLxi", "dLxi_dc", "dLxi_dlam"]


# ===========================================================================
#  Model:  jump part is a Poisson(lam) mixture of exponentials, optionally
#  capped at M (which puts an ATOM at M in the Levy measure), plus an optional
#  Gaussian component sigma.
# ===========================================================================
class SNLP:
    def __init__(self, c, lam, w, b, sigma=0.0, M=np.inf):
        self.c = float(c)
        self.lam = float(lam)
        self.w = np.asarray(w, float)
        self.b = np.asarray(b, float)
        self.sigma = float(sigma)
        self.M = float(M)
        assert abs(self.w.sum() - 1.0) < 1e-12, "severity weights must sum to 1"

    # ---------------- Levy exponent ----------------
    def _c(self, z):
        return np.asarray(z, dtype=complex)

    def G(self, z):
        """G(z) = A(z)/z = lam * int_0^M e^{-zy} Fbar(y) dy  (stable at z=0)."""
        z = self._c(z)
        out = np.zeros(np.shape(z), dtype=complex)
        for wi, bi in zip(self.w, self.b):
            s = z + bi
            if np.isfinite(self.M):
                out = out + wi * (1.0 - np.exp(-s * self.M)) / s
            else:
                out = out + wi / s
        return self.lam * out

    def Gp(self, z):
        z = self._c(z)
        out = np.zeros(np.shape(z), dtype=complex)
        for wi, bi in zip(self.w, self.b):
            s = z + bi
            if np.isfinite(self.M):
                e = np.exp(-s * self.M)
                out = out + wi * (self.M * e * s - (1.0 - e)) / s ** 2
            else:
                out = out - wi / s ** 2
        return self.lam * out

    def A(self, z):
        z = self._c(z)
        return z * self.G(z)

    def Ap(self, z):
        z = self._c(z)
        return self.G(z) + z * self.Gp(z)

    def psi(self, z):
        z = self._c(z)
        return self.c * z + 0.5 * self.sigma ** 2 * z ** 2 - self.A(z)

    def psip(self, z):
        z = self._c(z)
        return self.c + self.sigma ** 2 * z - self.Ap(z)

    def mu1(self):
        """lam E[X ^ M] -- retained expected loss rate."""
        return float(np.real(self.G(0.0)))

    def ceded_rate(self):
        """lam E[(X - M)^+]."""
        if not np.isfinite(self.M):
            return 0.0
        return self.lam * float(np.sum(self.w * np.exp(-self.b * self.M) / self.b))

    # ---------------- penalty transform  Lw ----------------
    def Lw(self, z, kappa=("ruin",)):
        """Laplace transform of w(x) = int kappa(x,y) nu(x+dy)."""
        z = self._c(z)
        kind = kappa[0]
        if kind == "ruin":                      # kappa == 1  ->  w = nubar
            return self.G(z)
        if not np.isfinite(self.M):
            if kind == "power":                 # kappa = y^n
                n = kappa[1]
                out = np.zeros(np.shape(z), dtype=complex)
                for wi, bi in zip(self.w, self.b):
                    out = out + wi / (bi ** n * (z + bi))
                return self.lam * factorial(n) * out
            if kind == "expo":                  # kappa = exp(-a y)
                a = kappa[1]
                out = np.zeros(np.shape(z), dtype=complex)
                for wi, bi in zip(self.w, self.b):
                    out = out + wi * bi / ((a + bi) * (z + bi))
                return self.lam * out
        raise ValueError(f"unsupported kappa {kappa} (with M={self.M})")

    @staticmethod
    def kappa0(kappa=("ruin",)):
        """kappa(0,0): the creeping penalty (only matters if sigma > 0)."""
        kind = kappa[0]
        if kind == "ruin":
            return 1.0
        if kind == "power":
            return 1.0 if kappa[1] == 0 else 0.0
        if kind == "expo":
            return 1.0
        raise ValueError(kappa)

    # ---------------- Lundberg root ----------------
    def phi(self, delta, rho0=None, tol=1e-14, itmax=300):
        """Unique root of psi(rho) = delta with Re rho > 0 (Re delta > 0).

        No safety loading is required: psi is convex on [0, inf) with
        psi(0) = 0 and psi(+inf) = +inf, so a positive real root exists for
        every delta > 0 whatever the sign of psi'(0+) = c - mu1.
        """
        from scipy.optimize import brentq

        delta = complex(delta)
        d = delta.real
        if rho0 is None:
            if d <= 0.0:
                # delta = 0: rho = 0 if c > mu1, else the positive root of psi = 0
                if self.c - self.mu1() > 0:
                    rho0 = 0.0 + 0.0j
                else:
                    hi = 1.0
                    while float(np.real(self.psi(hi))) < 0.0:
                        hi *= 2.0
                    rho0 = complex(brentq(lambda r: float(np.real(self.psi(r))),
                                          1e-12, hi, xtol=1e-16, rtol=8.9e-16))
            else:
                hi = 1.0
                while float(np.real(self.psi(hi))) < d:
                    hi *= 2.0
                rho0 = complex(brentq(lambda r: float(np.real(self.psi(r))) - d,
                                      1e-300, hi, xtol=1e-16, rtol=8.9e-16))
        rho = complex(rho0)
        if rho.real <= 0.0 and d > 0.0:
            rho = complex(1e-12, rho.imag)
        if d <= 0.0 and abs(rho) == 0.0:
            return 0.0 + 0.0j
        for _ in range(itmax):
            F = complex(self.psi(rho)) - delta
            Fp = complex(self.psip(rho))
            step = F / Fp
            nxt = rho - step
            while nxt.real <= 0.0:              # stay in the right half plane
                step *= 0.5
                nxt = rho - step
                if abs(step) < 1e-300:
                    break
            rho = nxt
            if abs(step) < tol * max(1.0, abs(rho)):
                break
        return rho

    # ---------------- master transform ----------------
    def Lxi(self, z, delta, rho=None, kappa=("ruin",)):
        """L xi(z) = [Lw(z) - Lw(rho) - k0 sig^2/2 (z-rho)] / (delta - psi(z))."""
        if rho is None:
            rho = self.phi(delta)
        z = self._c(z)
        k0 = self.kappa0(kappa)
        half = 0.5 * self.sigma ** 2
        num = self.Lw(z, kappa) - self.Lw(rho, kappa) - k0 * half * (z - rho)
        den = delta - self.psi(z)
        out = np.empty(np.shape(z), dtype=complex)
        scale = max(1.0, abs(complex(rho)))
        bad = np.abs(z - rho) < 1e-7 * scale
        ok = ~bad
        out[ok] = num[ok] / den[ok]
        if np.any(bad):                          # removable singularity
            lw_p = complex(self._dLw(rho, kappa))
            out[bad] = (k0 * half - lw_p) / complex(self.psip(rho))
        return out

    def _dLw(self, z, kappa=("ruin",), eps=None):
        """d/dz Lw(z) -- analytic for the mixture family."""
        z = self._c(z)
        kind = kappa[0]
        if kind == "ruin":
            return self.Gp(z)
        if kind == "power":
            n = kappa[1]
            out = np.zeros(np.shape(z), dtype=complex)
            for wi, bi in zip(self.w, self.b):
                out = out - wi / (bi ** n * (z + bi) ** 2)
            return self.lam * factorial(n) * out
        if kind == "expo":
            a = kappa[1]
            out = np.zeros(np.shape(z), dtype=complex)
            for wi, bi in zip(self.w, self.b):
                out = out - wi * bi / ((a + bi) * (z + bi) ** 2)
            return self.lam * out
        raise ValueError(kappa)

    # ---------------- boundary data xi(0+), xi'(0+) ----------------
    def boundary_data(self, delta, rho=None, kappa=("ruin",), K=2,
                      z0=None, nnode=8):
        """(J_0, ..., J_{K-1}) = (xi(0+), xi'(0+), ...) by Richardson
        extrapolation of z L xi(z) as z -> infinity along the real axis.

        Model-free: uses only evaluations of the transform.
        """
        if rho is None:
            rho = self.phi(delta)
        scale = float(max(np.max(self.b), abs(self.c) + 1.0, 1.0))
        if z0 is None:
            z0 = 1.0e3 * scale
        coef = boundary_from(lambda zz: self.Lxi(zz, delta, rho, kappa),
                             K=K, z0=z0, nnode=nnode)
        # xi(.;delta) is complex when delta is: keep the imaginary parts.
        return coef if complex(delta).imag else np.real(coef)


def boundary_from(Lfun, K=2, z0=1e3, nnode=8):
    """(f(0+), f'(0+), ...) from the large-z expansion  z Lf(z) ~ sum J_k z^{-k}.

    Works for any transform, in particular for a *derivative* transform
    d/dtheta L xi -- which is what Theorem 8 needs if the corrected scheme is
    to keep its order under differentiation.
    """
    zs = z0 * 2.0 ** np.arange(nnode)
    vals = zs * np.asarray(Lfun(zs.astype(complex)))
    V = np.vander(1.0 / zs, nnode, increasing=True)
    coef = np.linalg.solve(V, vals)
    return coef[:K]


# ===========================================================================
#  Filters
# ===========================================================================
def _filter(name, th):
    """Multiplier s(theta) on |theta| <= pi.  All satisfy s(0) = 1."""
    v = 0.5 * th
    if name == "shannon":                        # s = 1      (paper's choice)
        return np.ones_like(th, dtype=float)
    sn = np.where(np.abs(v) < 1e-12, 1.0, np.sin(np.where(np.abs(v) < 1e-12, 1.0, v)) / np.where(np.abs(v) < 1e-12, 1.0, v))
    if name == "cell":                           # s = sinc    (cell averages)
        return sn
    if name == "haar":                           # s = 1/sinc  (Tse 2020)
        return 1.0 / sn
    raise ValueError(name)


def midpoints(J, N):
    return (np.arange(N) + 0.5) * 2.0 ** -J


def grid_scheme(Lxi_fun, J, N, filt="shannon", corr=None, gamma=1.0):
    """One FFT -> values at the N midpoints (m+1/2) 2^{-J}, m = 0..N-1.

    Lxi_fun(z)  : vectorised, accepts complex z (we pass z = i*omega)
    corr        : None, or (J0, J1) = (xi(0+), xi'(0+)) to activate the
                  two-term singularity correction (Theorem 4).
    gamma       : decay rate of the corrector q(u) = (a + b u) e^{-gamma u}.
    """
    h = 2.0 ** -J
    th = -np.pi + 2.0 * np.pi * np.arange(N) / N
    om = th / h
    z = 1j * om
    g = Lxi_fun(z)

    if corr is not None:
        J0, J1 = corr
        a = J0
        b = J1 + gamma * J0
        g = g - (a / (gamma + z) + b / (gamma + z) ** 2)

    s = _filter(filt, th)
    F = (1.0 / h) * g * s * np.exp(0.5j * th)
    m = np.arange(N)
    out = ((-1.0) ** m) * np.fft.ifft(F)

    if corr is not None:
        x = midpoints(J, N)
        out = out + (a + b * x) * np.exp(-gamma * x)
    return out


# ===========================================================================
#  Abate-Whitt Euler inversion (double precision)
# ===========================================================================
def euler_invert(Fhat, t, n=20, m=13, A=18.4):
    """Invert L{f}(s) = Fhat(s) at time t.  Fhat may be array valued."""
    ea = np.exp(A / 2.0)
    terms = [ea / (2.0 * t) * np.real(Fhat(A / (2.0 * t) + 0j))]
    for k in range(1, n + m + 1):
        terms.append(((-1) ** k) * ea / t
                     * np.real(Fhat((A + 2j * k * np.pi) / (2.0 * t))))
    partial = np.cumsum(np.array(terms), axis=0)
    wts = np.array([comb(m, j) for j in range(m + 1)], float) / 2.0 ** m
    return np.tensordot(wts, partial[n:n + m + 1], axes=(0, 0))


def gaver_digits(N):
    """Decimal digits annihilated by cancellation in 2N-term Gaver-Stehfest."""
    tot = 0.0
    for n in range(1, 2 * N + 1):
        s = 0.0
        for j in range((n + 1) // 2, min(N, n) + 1):
            s += j ** (N + 1) * comb(N, j) * comb(2 * j, j) * comb(j, n - j)
        tot += s / factorial(N) / n
    return np.log10(tot)


# ===========================================================================
#  finite horizon:  L_t{ xi_t(.;delta) }(s) = xi(.; delta+s)/s
# ===========================================================================
def finite_horizon(model, t, J, N, delta=0.0, kappa=("ruin",), filt="shannon",
                   corrected=True, gamma=1.0, **kw):
    prev = [None]

    def Fhat(s):
        d = delta + s
        rho = model.phi(d, rho0=prev[0])
        prev[0] = rho
        corr = tuple(model.boundary_data(d, rho, kappa, K=2)) if corrected else None
        return grid_scheme(lambda z: model.Lxi(z, d, rho, kappa),
                           J, N, filt=filt, corr=corr, gamma=gamma) / s

    return np.real(euler_invert(Fhat, t, **kw))


# ===========================================================================
#  Exact parametric sensitivities (Theorem 7).
#
#      d/dtheta L xi(z) = [ dN(z) + L xi(z) dpsi(z) ] / (delta - psi(z)),
#      dN(z) = dLw(z) - dLw(rho) - Lw'(rho) rho_theta
#                     - k0 [ dhalf (z - rho) - (sigma^2/2) rho_theta ],
#      rho_theta = - dpsi(rho) / psi'(rho).
#
#  The singularity at z = rho stays removable, so the derivative transform is
#  as well conditioned as the transform itself.
# ===========================================================================
def dLxi(model, z, delta, rho, dpsi, dLw, dhalf=0.0, kappa=("ruin",)):
    z = np.asarray(z, dtype=complex)
    k0 = model.kappa0(kappa)
    half = 0.5 * model.sigma ** 2
    rho_t = -complex(dpsi(rho)) / complex(model.psip(rho))
    Lwp = complex(model._dLw(rho, kappa))

    def raw(zz):
        dN = (dLw(zz) - complex(dLw(rho)) - Lwp * rho_t
              - k0 * (dhalf * (zz - rho) - half * rho_t))
        return (dN + model.Lxi(zz, delta, rho, kappa) * dpsi(zz)) / (delta - model.psi(zz))

    out = np.empty(z.shape, dtype=complex)
    scale = max(1.0, abs(complex(rho)))
    bad = np.abs(z - rho) < 1e-6 * scale
    ok = ~bad
    out[ok] = raw(z[ok])
    if np.any(bad):                    # 4-point average on a small circle
        eps = 1e-5 * scale
        acc = 0.0
        for k in range(4):
            acc = acc + raw(np.array([rho + eps * np.exp(1j * np.pi * k / 2)]))[0]
        out[bad] = acc / 4.0
    return out


def dLxi_dc(model, z, delta, rho, kappa=("ruin",)):
    """theta = c.  Reduces to  [z Lxi(z) - rho Lxi(rho)] / (delta - psi(z))."""
    return dLxi(model, z, delta, rho, dpsi=lambda zz: np.asarray(zz, complex),
                dLw=lambda zz: np.zeros_like(np.asarray(zz, complex)), kappa=kappa)


def dLxi_dlam(model, z, delta, rho, kappa=("ruin",)):
    """theta = lam (claim/request frequency).  A and Lw are linear in lam."""
    return dLxi(model, z, delta, rho,
                dpsi=lambda zz: -model.A(zz) / model.lam,
                dLw=lambda zz: model.Lw(zz, kappa) / model.lam, kappa=kappa)


def threshold(values, xs, alpha):
    """Smallest x on the grid with values(x) <= alpha, by log-linear interp."""
    i = int(np.argmax(values < alpha))
    if i <= 0:
        return 0.0
    lo, hi = i - 1, i
    y = np.log(np.maximum(values[[lo, hi]], 1e-300))
    return float(np.interp(np.log(alpha), y[::-1], xs[[lo, hi]][::-1]))
