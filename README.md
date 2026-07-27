# Half-shifted Fourier inversion for finite-horizon Gerber–Shiu functionals

Reference implementation and reproduction scripts for the paper.

Everything runs in IEEE double precision on a single core. No GPU, no
multiprecision arithmetic, no compiled extensions.

## Requirements

```
python >= 3.10
numpy, scipy, matplotlib
```

## The library

`code/gsx.py` is the whole method.

```python
from gsx import SNLP, grid_scheme, midpoints, finite_horizon

# X_t = u + c t + sigma B_t - L_t,  L a Poisson(lam) mixture-of-exponentials
# subordinator, severities sum_i w_i Exp(b_i), optionally capped at M.
mod = SNLP(c=25.0, lam=20.0, w=[1.0], b=[1.0], sigma=0.0)

rho  = mod.phi(0.0)                          # Lundberg root  psi(rho) = delta
bd   = tuple(mod.boundary_data(0.0, rho, K=2))   # (xi(0+), xi'(0+))
xi   = grid_scheme(lambda z: mod.Lxi(z, 0.0, rho),
                   J=8, N=1 << 18,           # h = 2^-J, N grid cells
                   filt="shannon", corr=bd)  # fourth-order scheme
u    = midpoints(8, 1 << 18)                 # values live at (m+1/2) h

# finite horizon: P(ruin before t) on the whole grid, one call
psi_t = finite_horizon(mod, t=10.0, J=6, N=1 << 18)
```

Key entry points:

| name | what it does |
|---|---|
| `SNLP` | model; `psi`, `A`, `Lw`, `phi` (Lundberg root, complex-capable), `Lxi` (Theorem 3.1), `boundary_data` |
| `grid_scheme` | one FFT → the whole midpoint grid; `filt` ∈ `{"shannon","haar","cell"}`, `corr` activates the two-term correction |
| `finite_horizon` | Abate–Whitt Euler inversion in `t` of `Lxi(.; delta+s)/s` |
| `dLxi`, `dLxi_dc`, `dLxi_dlam` | exact parametric sensitivities (Theorem 7.1) |
| `boundary_from` | boundary data of *any* transform — needed to differentiate the corrector |
| `euler_invert`, `gaver_digits`, `threshold` | utilities |

**Penalty functions.** `kappa=("ruin",)` is κ ≡ 1; `("power", n)` is κ = yⁿ;
`("expo", a)` is κ = e^{−ay}. Gerber–Shiu is linear in κ, so scale the output
for κ = c·e^{−ay}.

**Two things that are easy to get wrong** (both cost two orders of convergence):

1. Values live at cell **midpoints** `(m+1/2)·2^-J`, not at `m·2^-J`.
2. At complex discount rates the boundary data is **complex**. Do not take real
   parts before passing it to `corr`.

## Reproducing the paper

Run from inside `code/`. Approximate single-core timings in brackets.

| script | produces | time |
|---|---|---|
| `e1_order.py` | Tables 6.1–6.2: convergence orders 1/2/2/4, error-formula ratios | ~4 min |
| `e2_validate.py` | δ>0 with κ=y³; Gaussian component vs scale functions; finite-horizon vs Prabhu; Gaver–Stehfest conditioning (Table 7.2) | ~3 min |
| `e3_credit.py` | §9: Monte Carlo validation (4×10⁶ exact paths), spread term structure (Table 9.1), leverage ladder | ~4 min |
| `e4c_final.py` | §10–11: buffer ladder, transient overload, Newton sizing, offload frontier (Table 10.1), gradient order test | ~5 min |
| `e5_greeks.py` | §8: sensitivities vs closed forms and central differences | ~1 min |
| `e6_atom.py` | Remark 5.5: kink from an atom in ν, on- vs off-grid | ~2 min |
| `figs.py` | Figures 1–3 | ~2 min |

`figs.py` reads `credit_ts.npy` / `credit_ladder.npy` (written by
`e3_credit.py`) and `frontier.npy` (written by `e4c_final.py`), so run those
two first.

```bash
cd code
python3 e3_credit.py && python3 e4c_final.py && python3 figs.py
```

## Note on scope

The severity model is a mixture of exponentials, optionally capped. Mixtures of
exponentials are dense in the positive severities and closed under capping
(a cap puts an atom at M, which the transform of Theorem 3.1 handles with no
special treatment). Nothing in the theory depends on that choice — a different
family needs only its own `A(z)` and `Lw(z)`.
