# BBG Benchmark Specification

Paper-to-code audit for Baldacci, Bergault & Gueant (2020), "Algorithmic market making for options", *Quantitative Finance* 21(1), 85-97.

PDF: `docs/refs/algo_mm_options_BBG.pdf`

## A. Exact paper parameters (Section 4.1)

### Heston dynamics

Under P:
```
dS_t = sqrt(nu_t) S_t dW^S_t        (zero drift for simplicity)
dnu_t = kappa_P (theta_P - nu_t) dt + xi sqrt(nu_t) dW^nu_t
```

Under Q:
```
dnu_t = kappa_Q (theta_Q - nu_t) dt + xi sqrt(nu_t) dW^nu_t
```

with `dW^S dW^nu = rho dt`.

| Parameter | Value | Unit |
|-----------|-------|------|
| S_0 | 10 | EUR |
| nu_0 | 0.0225 | year^{-1} |
| kappa_P | 2 | year^{-1} |
| theta_P | 0.04 | year^{-1} |
| kappa_Q | 3 | year^{-1} |
| theta_Q | 0.0225 | year^{-1} |
| xi | 0.2 | year^{-1/2} |
| rho | -0.5 | dimensionless |

### Option book

20 European calls on S. Strike x Maturity grid:

- K = {8, 9, 10, 11, 12} EUR
- T^maturity = {1, 1.5, 2, 3} years

Options priced under Q using Black-Scholes with variance = nu_t.

### Control horizon and risk aversion

- T = 0.0012 year (approximately 0.3 trading day)
- gamma = 1e-3 EUR^{-1}
- Vega risk limit: V_bar = 1e7 EUR * year^{1/2}

### Intensity functions (Section 4.1)

```
Lambda^{i,j}(delta) = lambda^i / (1 + exp(alpha + (beta / V^i) * delta))
```

for all i in {1,...,N}, j in {a, b}, where:

- lambda^i = 252 * 30 / (1 + 0.7 * |S_0 - K^i|) year^{-1}
  - ATM (K=10): lambda = 252*30 / 1 = 7560 year^{-1} (~30 per day)
  - K=8: lambda = 252*30 / (1 + 0.7*2) = 7560 / 2.4 = 3150 year^{-1} (~12.5 per day)
  - K=12: same as K=8 by symmetry
- alpha = 0.7 (dimensionless)
  - Probability to trade at mid: 1/(1+e^0.7) ~ 33%
- beta = 150 year^{1/2}
  - Probability to trade at delta = V^i (one vega-unit away): 1/(1+e^{0.7+150}) ~ 0%
  - Probability to trade at implied vol 1% better: ~69% (see paper remark)

### Trade sizes

Constant per option (Dirac mass):

```
z^i = notional / O^i_0   contracts of option i per transaction
```

where O^i_0 = O^i(0, S_0, nu_0) is the initial option price. The paper states
"approximately 500,000€ per transaction" (the PDF text reads "5·10^6" which
appears to be OCR garbling of "5·10^5" — using 5e6 produces trade sizes that
exceed the vega risk limit in a single fill, which is inconsistent with the
paper's figures). We use notional = 500,000 EUR.

The measures mu^{i,b} and mu^{i,a} are Dirac masses at z^i.

### Important: intensity is NOT exponential

The paper uses a **logistic** intensity, not the exponential A*exp(-k*delta)
from Avellaneda-Stoikov. The form is:

```
Lambda(delta) = lambda / (1 + exp(alpha + beta * delta / V))
```

This is a sigmoid in (delta/V), where V is the option's vega. The BBG
intensity saturates at lambda for very favorable quotes and decays to 0
for very unfavorable quotes, with the transition scale set by vega.

## B. Solver-state reduction

### General case (Section 3.1)

Under Assumptions 1 (constant vega) and 2 (vega risk limits):

- V^i_t = V^i_0 := V^i for all t (frozen per option)
- Portfolio vega: V^pi_t = sum_i q^i_t V^i
- Value function: u(t, S, nu, q) = v(t, nu, V^pi) for all q in Q

The reduced HJB (eq. 4) is:

```
0 = d_t v(t, nu, V^pi) + a_P(t,nu) d_nu v + (1/2) nu xi^2 d^2_{nu,nu} v
    + V^pi (a_P - a_Q) / (2 sqrt(nu))
    - (gamma xi^2 / 8) V^{pi,2}
    + sum_{i=1}^N sum_{j=a,b} int z 1_{|V^pi - psi(j) z V^i| <= V_bar}
      H^{i,j}( (v(t,nu,V^pi) - v(t,nu,V^pi - psi(j) z V^i)) / z ) mu^{i,j}(dz)
```

Terminal condition: v(T, nu, V^pi) = 0.

With Dirac trade sizes z^i, the integral over z collapses to evaluation at z^i.

### Hamiltonian

```
H^{i,j}(p) := sup_{delta >= delta_infty} Lambda^{i,j}(delta) (delta - p)
```

For the logistic intensity, this does NOT have a closed-form like 1/k.
It must be solved numerically (e.g. bisection or Newton on the FOC).

### Special case a_P = a_Q

When P and Q drift functions agree, the `(a_P - a_Q)/(2 sqrt(nu))` term
vanishes. Then v does not depend on nu:

```
v(t, nu, V^pi) = w(t, V^pi)
```

The HJB simplifies to an ODE in (t, V^pi) only, which is much cheaper
to solve. The paper's numerical example has a_P != a_Q (kappa_P=2 vs
kappa_Q=3), so the general 3D solver is needed for the paper benchmark.

### Why single-option is degenerate

With N=1, V^pi = q * V^1, and the problem becomes a 1D inventory problem
in q (after discretizing). This loses the multi-option structure that
makes BBG scientifically interesting: the key feature is that different
options contribute different amounts of vega per contract, creating a
nontrivial portfolio optimization in the inventory space.

## C. Mapping to codebase

### Current stylized env (`src/applications/option_mm/`)

The existing `OptionMarketMakingEnv` is a separate stylized env:
- single option
- daily scale (dt = 1/252)
- exponential Poisson intensity (A * exp(-k * delta))
- constant half-spread = 1/k from A-S

This will remain untouched. It is NOT a BBG-faithful benchmark.

### New BBG benchmark (`src/applications/option_mm_bbg/`)

The new package implements the paper's actual setup:
- N options (default 20)
- logistic intensity per option
- Dirac trade sizes
- portfolio-vega state reduction
- 3D HJB solver on (t, nu, V^pi) grid
- quote extraction via numerical Hamiltonian optimization

### Key differences from old env

| Feature | Old env | BBG benchmark |
|---------|---------|---------------|
| Options | 1 | 20 (5 strikes x 4 maturities) |
| Intensity | exponential A*exp(-k*delta) | logistic lambda/(1+exp(alpha+beta*delta/V)) |
| Trade size | 1 contract | z^i = 5e6/O^i_0 contracts |
| State | (q, h, V_hat, tau) | (t, nu, V^pi) |
| Solver | backward ODE on q-grid | 3D PDE on (t, nu, V^pi) grid |
| Horizon | ~20 trading days | ~0.3 trading day |
| Hedge | per-step -net_delta | continuous Delta-hedge assumed |
| Currency | USD-like (S_0=100) | EUR (S_0=10) |
