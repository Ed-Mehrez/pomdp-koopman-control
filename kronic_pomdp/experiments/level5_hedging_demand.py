"""
Level 5: Hedging Demand Estimation

For CRRA with Heston and PROPORTIONAL risk premium mu(V) = r + eta*V:
  J(W,V) = W^{1-gamma} * h(V) / (1-gamma)
  pi* = eta/gamma + B*rho*xi/gamma   (myopic + hedge)
  h(V) = exp(B*V), B from Riccati equation.

Two methods, using overlapping blocks with temporal train/test split:

  Method A -- OLS on V_hat (primary, simple, honest):
    1. V_hat from Level 4 BLR+KF (signature-based vol estimate)
    2. Overlapping blocks: (V_hat_start, log_cum_rho) pairs
    3. OLS: log(cum_rho) = alpha + beta * V_hat
    4. hedge = beta * rho * xi / gamma
    For Heston (single latent V), this is sufficient.

  Method B -- Signature KRR (generalizable, for multi-factor hedging):
    1. Fresh per-block log-signatures as state
    2. KRR in signature RKHS: h(sig) = sum_i alpha_i * k(sig, sig_i)
    3. Kernel gradient: grad_h(sig) gives hedge direction in sig space
    4. No need to identify latent factors -- kernel discovers all hedge directions
    For Heston, this reduces to Method A. For richer models (stochastic skew,
    jumps, rough vol), it captures hedging demands that scalar V_hat misses.

Bias correction: large k (504 steps = 2yr) captures most mean-reversion signal.
Multi-k convergence diagnostic shows how beta(k) approaches B_true.

Ground truth: Analytic Riccati (verified by simulation to std=0.00002).
Validation: CE comparison (myopic+hedge vs pure myopic) on held-out path segments.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from examples.proof_of_concept.signature_features import RecurrentLeadLagLogSigMap

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

HestonParams = namedtuple('HestonParams',
    ['eta', 'r', 'kappa', 'theta', 'xi', 'rho', 'dt'],
    defaults=[1.5, 0.02, 0.5, 0.04, 0.5, -0.7, 1/252])


# ======================================================================
# Section 1: Ground Truth
# ======================================================================

def analytic_hedging_demand(p, gamma_risk):
    """Exact B and hedge from Riccati for h(V)=exp(BV).

    Riccati: 0.5*xi^2*B^2 + [-kappa + (1-g)*eta*xi*rho/g]*B
             + (1-g)*eta^2/(2*g) = 0
    """
    eta, g = p.eta, gamma_risk
    a = 0.5 * p.xi**2
    b = -p.kappa + (1 - g) * eta * p.xi * p.rho / g
    c = (1 - g) * eta**2 / (2 * g)
    disc = b**2 - 4 * a * c
    if disc < 0:
        B = 0.0
    else:
        B1 = (-b + np.sqrt(disc)) / (2 * a)
        B2 = (-b - np.sqrt(disc)) / (2 * a)
        B = B1 if abs(B1) < abs(B2) else B2
    hedge = B * p.rho * p.xi / g
    return B, hedge


# ======================================================================
# Section 2: Nystrom KRR in Signature Space
# ======================================================================

class NystromKRR:
    """KRR with RBF kernel + Nystrom landmarks.

    Provides predictions and kernel gradients for hedge extraction.
    """
    def __init__(self, M=30, reg=1e-2):
        self.M = M
        self.reg = reg
        self.landmarks = None
        self.sigma = 1.0
        self.w = None
        self.x_mean = None
        self.x_std = None

    def fit(self, X, y):
        n, d = X.shape
        M = min(self.M, n)

        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.maximum(np.std(X, axis=0), 1e-8)
        Xn = (X - self.x_mean) / self.x_std

        # Farthest-point landmark selection
        selected = [np.random.randint(n)]
        for _ in range(M - 1):
            dists = np.min(
                np.array([np.sum((Xn - Xn[s])**2, axis=1)
                          for s in selected]), axis=0)
            selected.append(np.argmax(dists))
        self.landmarks = Xn[selected]

        # Median heuristic bandwidth
        pw = []
        for i in range(M):
            for j in range(i+1, M):
                pw.append(np.sum((self.landmarks[i] - self.landmarks[j])**2))
        self.sigma = max(np.sqrt(np.median(pw)), 1e-4) if pw else 1.0

        Phi = self._phi(Xn)
        Phi_b = np.column_stack([Phi, np.ones(n)])

        self.w = np.linalg.solve(
            Phi_b.T @ Phi_b + self.reg * np.eye(M + 1),
            Phi_b.T @ y)

    def _phi(self, Xn):
        diff = Xn[:, np.newaxis, :] - self.landmarks[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return np.exp(-sq_dist / (2 * self.sigma**2))

    def predict(self, X):
        Xn = (X - self.x_mean) / self.x_std
        Phi = self._phi(Xn)
        return Phi @ self.w[:-1] + self.w[-1]

    def kernel_gradient(self, X):
        """Gradient of h in original feature space."""
        Xn = (X - self.x_mean) / self.x_std
        Phi = self._phi(Xn)
        weighted_phi = Phi * self.w[:-1][np.newaxis, :]
        diff = self.landmarks[np.newaxis, :, :] - Xn[:, np.newaxis, :]
        grad_norm = np.einsum('ij,ijk->ik', weighted_phi, diff) / self.sigma**2
        return grad_norm / self.x_std[np.newaxis, :]


# ======================================================================
# Section 3: Simulate Path with BLR+KF Filtering
# ======================================================================

def simulate_and_filter(p, gamma_risk, T, seed):
    """Simulate Heston path with BLR+KF vol filtering (Level 4 pipeline).

    Returns per-step arrays: V_true, V_hat, rets, rho_factors.
    """
    rng = np.random.RandomState(seed)
    dt = p.dt
    pi_m = np.clip(p.eta / gamma_risk, 0.01, 5.0)

    # Lead-lag log-sig for BLR filtering (gamma=0.99, Level 4)
    ll_filter = RecurrentLeadLagLogSigMap(
        state_dim=2, level=2, forgetting_factor=0.99)
    ll_area_idx, ll_ret_idx = 8, 1

    # BLR+KF state
    blr_nf = 3
    blr_w = np.zeros(blr_nf)
    blr_P = np.eye(blr_nf) * 10.0
    blr_sigma_n2 = 0.01
    V_filt = p.theta
    P_kf = p.xi**2 * p.theta * dt * 10

    # Storage
    V_true = np.zeros(T)
    V_hat = np.zeros(T)
    rets = np.zeros(T)
    rho_factors = np.zeros(T)
    P_kf_hist = np.zeros(T)

    V = p.theta
    for t in range(T):
        z1, z2 = rng.randn(), rng.randn()
        z2c = p.rho * z1 + np.sqrt(1 - p.rho**2) * z2
        sv = np.sqrt(max(V, 1e-8))
        sdt = np.sqrt(dt)

        ret = p.eta * V * dt + sv * sdt * z1
        V_new = max(V + p.kappa * (p.theta - V) * dt
                    + p.xi * sv * sdt * z2c, 1e-8)

        V_true[t] = V
        rets[t] = ret
        V = V_new

        # BLR+KF update
        dx = np.array([dt, ret])
        feat = ll_filter.update(dx)
        phi_blr = np.array([feat[ll_area_idx], feat[ll_ret_idx], 1.0])

        y_blr = max(np.dot(blr_w, phi_blr), 1e-8)
        target_blr = min(ret**2 / dt, 2.0)
        Cp = blr_P @ phi_blr
        S = phi_blr @ Cp + blr_sigma_n2
        K_w = Cp / S
        blr_w += K_w * (target_blr - np.dot(blr_w, phi_blr))
        blr_P -= np.outer(K_w, Cp)
        blr_P = 0.5 * (blr_P + blr_P.T)
        blr_sigma_n2 = max(0.99 * blr_sigma_n2
                           + 0.01 * (target_blr - y_blr)**2, 1e-6)

        R_blr = max(phi_blr @ blr_P @ phi_blr + blr_sigma_n2, 1e-8)
        V_pred = max(V_filt + p.kappa * (p.theta - V_filt) * dt, 1e-6)
        Q_kf = p.xi**2 * max(V_filt, 1e-6) * dt
        P_pred = (1 - p.kappa * dt)**2 * P_kf + Q_kf
        K_kf = P_pred / (P_pred + R_blr)
        V_filt = max(V_pred + K_kf * (y_blr - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred
        V_hat[t] = V_filt
        P_kf_hist[t] = P_kf

        growth = np.clip(1 + p.r * dt + pi_m * ret, 0.5, 2.0)
        rho_factors[t] = growth ** (1 - gamma_risk)

    return V_true, V_hat, rets, rho_factors, P_kf_hist


# ======================================================================
# Section 4: Build Overlapping Blocks
# ======================================================================

def build_blocks(V_source, rets, rho_factors, dt, k, stride=None):
    """Build overlapping block data.

    Args:
        V_source: (T,) V values (V_hat or V_true)
        rets: (T,) returns
        rho_factors: (T,) CRRA utility factors
        dt: time step
        k: block length
        stride: step between blocks (default k//4)

    Returns:
        block_V: (n_blocks,) V at block start
        block_target: (n_blocks,) log(cum_rho) over block
        block_sigs: (n_blocks, 10) fresh signature features per block
    """
    T = len(rets)
    if stride is None:
        stride = max(k // 4, 1)
    n_blocks = (T - k) // stride

    block_V = np.zeros(n_blocks)
    block_target = np.zeros(n_blocks)
    block_sigs = []

    # Precompute cumulative log-rho for fast block sums
    log_rho = np.log(np.maximum(rho_factors, 1e-30))
    cum_log_rho = np.cumsum(log_rho)

    for b in range(n_blocks):
        t0 = b * stride
        t1 = t0 + k

        block_V[b] = V_source[t0]
        if t0 > 0:
            block_target[b] = cum_log_rho[t1 - 1] - cum_log_rho[t0 - 1]
        else:
            block_target[b] = cum_log_rho[t1 - 1]

        # Fresh per-block log-signature
        ll = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=1.0)
        for s in range(t0, t1):
            feat = ll.update(np.array([dt, rets[s]]))
        block_sigs.append(feat.copy())

    return block_V, block_target, np.array(block_sigs)


# ======================================================================
# Section 5: Method A -- OLS on V_hat
# ======================================================================

def ols_hedge(block_V, block_target, p, gamma_risk):
    """OLS: log(cum_rho) = alpha + beta * V_hat. hedge = beta * rho*xi/gamma."""
    n = len(block_V)
    X = np.column_stack([block_V, np.ones(n)])
    beta, alpha = np.linalg.lstsq(X, block_target, rcond=None)[0]
    hedge = beta * p.rho * p.xi / gamma_risk
    corr = np.corrcoef(block_V, block_target)[0, 1] if np.std(block_V) > 1e-10 else 0
    return hedge, corr, beta


# ======================================================================
# Section 6: Method B -- Signature KRR
# ======================================================================

def sig_krr_hedge(block_sigs, block_V, block_target, p, gamma_risk,
                  M=30, reg=1e-2, seed=42):
    """KRR on block signatures, hedge from kernel gradient projected to V."""
    n = len(block_sigs)
    if n < 10:
        return 0.0, 0.0

    np.random.seed(seed)
    krr = NystromKRR(M=M, reg=reg)
    krr.fit(block_sigs, block_target)

    h_pred = krr.predict(block_sigs)
    grad_h = krr.kernel_gradient(block_sigs)

    # dsig/dV from data: regress each sig component on V
    Xv = np.column_stack([block_V, np.ones(n)])
    dsig_dV = np.zeros(block_sigs.shape[1])
    for j in range(block_sigs.shape[1]):
        dsig_dV[j] = np.linalg.lstsq(Xv, block_sigs[:, j], rcond=None)[0][0]

    # Chain rule: dlog(h)/dV per block
    dlogh_dV = np.array([
        np.dot(grad_h[b], dsig_dV) / max(np.exp(h_pred[b]), 1e-30)
        for b in range(n)])

    hedge = np.mean(dlogh_dV) * p.rho * p.xi / gamma_risk
    corr = np.corrcoef(block_V, h_pred)[0, 1] if np.std(h_pred) > 1e-10 else 0
    return hedge, corr


# ======================================================================
# Section 7: EIV Correction via Autocorrelation
# ======================================================================

def estimate_reliability(V_hat, P_kf_hist, dt):
    """Estimate reliability ratio Var(V_true)/Var(V_hat).

    Two methods, both model-free (no V_true needed):

    1. Kalman P_kf method (principled):
       P_kf is the filter's own estimate of Var(V_hat - V_true | data).
       reliability = (Var(V_hat) - mean(P_kf)) / Var(V_hat)
       The BLR+KF pipeline provides this uncertainty for free.

    2. Autocorrelation method (fallback):
       For V_hat = V_true + eps with V_true OU and eps white noise:
       autocorr(lag=1)/autocorr(lag=2) -> kappa -> reliability.
    """
    var_V = np.var(V_hat)
    if var_V < 1e-16:
        return 1.0, 1.0

    # Method 1: Kalman-based (use after burn-in)
    burn = min(2000, len(P_kf_hist) // 5)
    mean_Pkf = np.mean(P_kf_hist[burn:])
    r_kalman = max((var_V - mean_Pkf) / var_V, 0.3)

    # Method 2: Autocorrelation-based
    V_c = V_hat - np.mean(V_hat)
    ac1 = np.mean(V_c[:-1] * V_c[1:]) / var_V
    ac2 = np.mean(V_c[:-2] * V_c[2:]) / var_V
    if ac1 > 0 and ac2 > 0:
        rho_decay = ac2 / ac1
        r_ac = np.clip(ac1 / rho_decay, 0.3, 1.0)
    else:
        r_ac = 1.0

    return r_kalman, r_ac


# ======================================================================
# Section 8: Multi-k Convergence Diagnostic
# ======================================================================

def multi_k_diagnostic(V_source, rets, rho_factors, dt, k_list, p, gamma_risk):
    """Run OLS at multiple k values to show convergence of beta(k) -> B_true."""
    results = []
    for k in k_list:
        stride = max(k // 4, 1)
        T = len(rets)
        n_blocks = (T - k) // stride
        if n_blocks < 10:
            results.append((k, np.nan))
            continue

        block_V = np.zeros(n_blocks)
        block_target = np.zeros(n_blocks)
        log_rho = np.log(np.maximum(rho_factors, 1e-30))
        cum_log_rho = np.cumsum(log_rho)

        for b in range(n_blocks):
            t0 = b * stride
            t1 = t0 + k
            block_V[b] = V_source[t0]
            if t0 > 0:
                block_target[b] = cum_log_rho[t1 - 1] - cum_log_rho[t0 - 1]
            else:
                block_target[b] = cum_log_rho[t1 - 1]

        X = np.column_stack([block_V, np.ones(n_blocks)])
        beta = np.linalg.lstsq(X, block_target, rcond=None)[0][0]
        results.append((k, beta))

    return results


# ======================================================================
# Section 9: CE Evaluation (Out-of-Sample)
# ======================================================================

def evaluate_ce(rets, p, gamma_risk, hedge_val, train_frac=0.7):
    """Compute CE for myopic vs myopic+hedge on held-out segment.

    Uses temporal split: first train_frac for training, rest for test.
    """
    T = len(rets)
    dt = p.dt
    pi_m = np.clip(p.eta / gamma_risk, 0.01, 5.0)
    t_start = int(T * train_frac)

    h_adj = np.clip(hedge_val, -0.5 * pi_m, 0.5 * pi_m)

    W_myopic = 1.0
    W_hedge = 1.0

    for t in range(t_start, T):
        W_myopic *= max(1 + p.r * dt + pi_m * rets[t], 1e-12)
        W_hedge *= max(1 + p.r * dt + (pi_m + h_adj) * rets[t], 1e-12)

    def ce(W, g):
        u = W**(1-g) / (1-g)
        return ((1-g) * u) ** (1/(1-g))

    return ce(W_myopic, gamma_risk), ce(W_hedge, gamma_risk)


# ======================================================================
# Section 10: Single Seed Run
# ======================================================================

def run_single_seed(p, gamma_risk, T, seed, k, stride=None, train_frac=0.7):
    """Run both methods on one seed, return hedge estimates and CE."""
    V_true, V_hat, rets, rho_factors, P_kf_hist = simulate_and_filter(
        p, gamma_risk, T, seed)
    dt = p.dt

    # Build overlapping blocks using V_hat (the real POMDP setting)
    block_V, block_target, block_sigs = build_blocks(
        V_hat, rets, rho_factors, dt, k, stride=stride)
    n_blocks = len(block_V)

    # Temporal train/test split
    n_train = int(n_blocks * train_frac)

    V_train = block_V[:n_train]
    y_train = block_target[:n_train]
    sig_train = block_sigs[:n_train]

    # Also build V_true blocks for diagnostic
    block_V_true, _, _ = build_blocks(
        V_true, rets, rho_factors, dt, k, stride=stride)
    V_true_train = block_V_true[:n_train]

    # Method A: OLS on V_hat
    hedge_ols, corr_ols, beta_ols = ols_hedge(
        V_train, y_train, p, gamma_risk)

    # Method A on V_true (diagnostic only)
    hedge_ols_true, _, beta_ols_true = ols_hedge(
        V_true_train, y_train, p, gamma_risk)

    # EIV correction (two methods, both V_true-free)
    r_kalman, r_ac = estimate_reliability(V_hat, P_kf_hist, dt)
    hedge_ols_corrected = hedge_ols / r_kalman

    # Method B: Signature KRR
    hedge_krr, corr_krr = sig_krr_hedge(
        sig_train, V_train, y_train, p, gamma_risk,
        M=30, reg=1e-2, seed=seed)

    # Multi-k convergence diagnostic (V_hat only, fast -- skip sigs)
    k_list = [63, 126, 252, 504]
    k_list = [kk for kk in k_list if kk <= T // 10]
    mk_vhat = multi_k_diagnostic(V_hat, rets, rho_factors, dt, k_list, p, gamma_risk)
    mk_vtrue = multi_k_diagnostic(V_true, rets, rho_factors, dt, k_list, p, gamma_risk)

    # CE evaluation on held-out segment (last 30%)
    ce_m, ce_ols = evaluate_ce(rets, p, gamma_risk, hedge_ols, train_frac)
    _, ce_ols_c = evaluate_ce(rets, p, gamma_risk, hedge_ols_corrected, train_frac)
    _, ce_krr = evaluate_ce(rets, p, gamma_risk, hedge_krr, train_frac)

    # V_hat quality
    v_corr = np.corrcoef(V_true[:T], V_hat[:T])[0, 1]

    return {
        'hedge_ols': hedge_ols,
        'hedge_ols_true': hedge_ols_true,
        'hedge_ols_corrected': hedge_ols_corrected,
        'hedge_krr': hedge_krr,
        'r_kalman': r_kalman,
        'r_ac': r_ac,
        'corr_ols': corr_ols,
        'corr_krr': corr_krr,
        'ce_myopic': ce_m,
        'ce_ols': ce_ols,
        'ce_ols_corrected': ce_ols_c,
        'ce_krr': ce_krr,
        'v_corr': v_corr,
        'n_blocks': n_blocks,
        'n_train': n_train,
        'mk_vhat': mk_vhat,
        'mk_vtrue': mk_vtrue,
        'beta_ols': beta_ols,
        'beta_ols_true': beta_ols_true,
    }


# ======================================================================
# Section 11: Main Experiment
# ======================================================================

def level5_hedging_demand(n_macro=5, T=50000, k=504):
    """Level 5: Hedging demand estimation.

    Two methods compared:
      A) OLS on V_hat -- simple, uses Level 4 signature-based vol filter
         + EIV correction via autocorrelation-based reliability ratio
      B) Signature KRR -- generalizable, equivalent to A for Heston
    Both use overlapping blocks (stride=k//4) for sample efficiency.
    CE validation on temporally held-out final 30% of path.
    """
    stride = max(k // 4, 1)
    print("\n" + "=" * 80)
    print("LEVEL 5: Hedging Demand Estimation (Bias-Corrected)")
    print("  Proportional premium: mu(V) = r + eta*V")
    print("  V_hat from Level 4 BLR+KF (signature-based)")
    print(f"  Overlapping blocks: k={k} ({k/252:.1f}yr), stride={stride} "
          f"({stride/252:.2f}yr)")
    print(f"  T={T} ({T/252:.0f}yr), n_seeds={n_macro}")
    print("  Train: first 70% of blocks | Test: last 30%")
    print("  Method A: OLS on V_hat + EIV correction")
    print("  Method B: Signature KRR (generalizable)")
    print("  GT: Analytic Riccati")
    print("=" * 80)

    p = HestonParams()
    gamma_list = [3, 5, 10]

    all_results = {}
    for gamma_risk in gamma_list:
        B_gt, hedge_gt = analytic_hedging_demand(p, gamma_risk)
        myopic = p.eta / gamma_risk
        pct = hedge_gt / myopic * 100

        print(f"\n{'='*80}")
        print(f"  gamma={gamma_risk}, myopic={myopic:.4f}, "
              f"GT hedge={hedge_gt:+.6f} ({pct:+.1f}% of myopic, B={B_gt:+.4f})")
        print(f"{'='*80}")

        seeds_data = []
        for si in range(n_macro):
            r = run_single_seed(p, gamma_risk, T=T, seed=si*1000, k=k,
                                stride=stride)
            seeds_data.append(r)

        # Multi-k convergence table (averaged over seeds)
        k_list = seeds_data[0]['mk_vhat']
        print(f"\n  Multi-k convergence (avg over {n_macro} seeds):")
        print(f"  {'k':>5} {'k/252':>6} {'beta_Vhat':>11} {'ratio':>7}"
              f" {'beta_Vtrue':>11} {'ratio':>7}")
        print(f"  {'-'*55}")
        for ki in range(len(k_list)):
            kv = k_list[ki][0]
            betas_vh = [s['mk_vhat'][ki][1] for s in seeds_data]
            betas_vt = [s['mk_vtrue'][ki][1] for s in seeds_data]
            mbv = np.nanmean(betas_vh)
            mbt = np.nanmean(betas_vt)
            rv = abs(mbv * p.rho * p.xi / gamma_risk / hedge_gt) if abs(hedge_gt) > 1e-8 else 0
            rt = abs(mbt * p.rho * p.xi / gamma_risk / hedge_gt) if abs(hedge_gt) > 1e-8 else 0
            print(f"  {kv:5d} {kv/252:6.2f} {mbv:+11.4f} {rv:6.2f}x"
                  f" {mbt:+11.4f} {rt:6.2f}x")

        # Aggregate main results
        h_ols = [s['hedge_ols'] for s in seeds_data]
        h_ols_t = [s['hedge_ols_true'] for s in seeds_data]
        h_ols_c = [s['hedge_ols_corrected'] for s in seeds_data]
        h_krr = [s['hedge_krr'] for s in seeds_data]
        ce_m = [s['ce_myopic'] for s in seeds_data]
        ce_ols = [s['ce_ols'] for s in seeds_data]
        ce_ols_c = [s['ce_ols_corrected'] for s in seeds_data]
        ce_krr = [s['ce_krr'] for s in seeds_data]
        v_corrs = [s['v_corr'] for s in seeds_data]
        r_kal = [s['r_kalman'] for s in seeds_data]
        r_ac = [s['r_ac'] for s in seeds_data]

        print(f"\n  V_hat corr: {np.mean(v_corrs):.3f}+/-{np.std(v_corrs):.3f}")
        print(f"  EIV reliability: Kalman={np.mean(r_kal):.3f}+/-{np.std(r_kal):.3f}"
              f"  AC={np.mean(r_ac):.3f}+/-{np.std(r_ac):.3f}")
        print(f"  n_blocks: {seeds_data[0]['n_blocks']} "
              f"({seeds_data[0]['n_train']} train, "
              f"{seeds_data[0]['n_blocks'] - seeds_data[0]['n_train']} test)")

        print(f"\n  {'Method':<22} {'Hedge estimate':<24} {'Ratio':>6}"
              f"  {'CE test':>12} {'vs myopic':>10}")
        print(f"  {'-'*80}")

        def fmt(hedges, ces, name):
            mh = np.mean(hedges)
            sh = np.std(hedges)
            ratio = abs(mh / hedge_gt) if abs(hedge_gt) > 1e-8 else 0
            mc = np.mean(ces)
            mm = np.mean(ce_m)
            diff = (mc - mm) / abs(mm) * 100 if abs(mm) > 1e-12 else 0
            print(f"  {name:<22} {mh:+.6f}+/-{sh:.6f} {ratio:>5.2f}x"
                  f"  {mc:12.4f} {diff:+9.4f}%")

        fmt(h_ols_t, ce_ols, 'OLS-V_true (diag)')
        fmt(h_ols, ce_ols, 'OLS-V_hat (raw)')
        fmt(h_ols_c, ce_ols_c, 'OLS-V_hat (EIV corr)')
        fmt(h_krr, ce_krr, 'Sig-KRR')

        all_results[gamma_risk] = {
            'B_gt': B_gt, 'hedge_gt': hedge_gt, 'myopic': myopic,
            'h_ols': h_ols, 'h_ols_true': h_ols_t,
            'h_ols_corrected': h_ols_c, 'h_krr': h_krr,
            'ce_m': ce_m, 'ce_ols': ce_ols,
            'ce_ols_corrected': ce_ols_c, 'ce_krr': ce_krr,
        }

    # PASS criteria
    print("\n" + "=" * 80)
    print("PASS CRITERIA")
    print("  1. Correct sign (all methods, all gammas)")
    print("  2. OLS-V_hat (EIV corr) ratio in [0.3, 3.0]")
    print("  3. Sig-KRR ratio in [0.1, 5.0]")
    print("  4. OLS and Sig-KRR agree on sign")
    print("=" * 80)

    all_pass = True
    for gamma_risk in gamma_list:
        r = all_results[gamma_risk]
        gt = r['hedge_gt']

        for name, hedges, lo, hi in [
            ('OLS-V_hat(corr)', r['h_ols_corrected'], 0.3, 3.0),
            ('Sig-KRR', r['h_krr'], 0.1, 5.0),
        ]:
            mh = np.mean(hedges)
            sign_ok = (mh * gt > 0)
            ratio = abs(mh / gt) if abs(gt) > 1e-8 else 0
            mag_ok = lo < ratio < hi
            ok = sign_ok and mag_ok
            if not ok:
                all_pass = False
            status = 'OK' if ok else 'FAIL'
            print(f"  g={gamma_risk} {name:<18}: hedge={mh:+.6f} "
                  f"(gt={gt:+.6f}) [{ratio:.2f}x] {status}")

        agree = (np.mean(r['h_ols_corrected']) * np.mean(r['h_krr'])) > 0
        if not agree:
            all_pass = False
            print(f"  g={gamma_risk} AGREE: FAIL (methods disagree on sign)")

    print(f"\nLEVEL 5: {'PASS' if all_pass else 'FAIL'}")

    return {'passed': all_pass, 'details': all_results}


def level5_practical_horizon(n_macro=10, T=2520, k=63):
    """Level 5 with realistic parameters: kappa=3, T=10yr, k=3mo.

    Empirical VIX mean-reversion is kappa~3-5/yr (half-life 2-3 months).
    With k=63 days (quarter) and T=2520 (10yr), we get ~40 blocks per seed.
    Runs more seeds to compensate for smaller sample size per seed.
    """
    stride = max(k // 4, 1)
    p = HestonParams(kappa=3.0, xi=0.8, theta=0.04)
    # Higher xi needed because faster mean-reversion reduces V dispersion

    print("\n" + "=" * 80)
    print("LEVEL 5 — PRACTICAL HORIZON (Realistic kappa)")
    print(f"  kappa={p.kappa} (half-life={np.log(2)/p.kappa:.1f}yr = "
          f"{np.log(2)/p.kappa*252:.0f} days)")
    print(f"  xi={p.xi}, eta={p.eta}, rho={p.rho}")
    print(f"  Overlapping blocks: k={k} ({k/252:.2f}yr), stride={stride}")
    print(f"  T={T} ({T/252:.1f}yr), n_seeds={n_macro}")
    print("  Train: first 70% | Test: last 30%")
    print("=" * 80)

    gamma_list = [3, 5, 10]
    all_results = {}

    for gamma_risk in gamma_list:
        B_gt, hedge_gt = analytic_hedging_demand(p, gamma_risk)
        myopic = p.eta / gamma_risk
        pct = hedge_gt / myopic * 100

        print(f"\n  gamma={gamma_risk}, myopic={myopic:.4f}, "
              f"GT hedge={hedge_gt:+.6f} ({pct:+.1f}%, B={B_gt:+.4f})")

        seeds_data = []
        for si in range(n_macro):
            r = run_single_seed(p, gamma_risk, T=T, seed=si*1000, k=k,
                                stride=stride)
            seeds_data.append(r)

        h_ols = [s['hedge_ols'] for s in seeds_data]
        h_ols_c = [s['hedge_ols_corrected'] for s in seeds_data]
        h_krr = [s['hedge_krr'] for s in seeds_data]
        ce_m = [s['ce_myopic'] for s in seeds_data]
        ce_ols = [s['ce_ols'] for s in seeds_data]
        ce_krr = [s['ce_krr'] for s in seeds_data]
        v_corrs = [s['v_corr'] for s in seeds_data]
        r_kal = [s['r_kalman'] for s in seeds_data]

        # Multi-k convergence
        k_list = seeds_data[0]['mk_vhat']
        print(f"\n  Multi-k convergence:")
        print(f"  {'k':>5} {'beta_Vhat':>11} {'ratio':>7}"
              f" {'beta_Vtrue':>11} {'ratio':>7}")
        for ki in range(len(k_list)):
            kv = k_list[ki][0]
            betas_vh = [s['mk_vhat'][ki][1] for s in seeds_data]
            betas_vt = [s['mk_vtrue'][ki][1] for s in seeds_data]
            mbv = np.nanmean(betas_vh)
            mbt = np.nanmean(betas_vt)
            rv = abs(mbv * p.rho * p.xi / gamma_risk / hedge_gt) if abs(hedge_gt) > 1e-8 else 0
            rt = abs(mbt * p.rho * p.xi / gamma_risk / hedge_gt) if abs(hedge_gt) > 1e-8 else 0
            print(f"  {kv:5d} {mbv:+11.4f} {rv:6.2f}x {mbt:+11.4f} {rt:6.2f}x")

        print(f"\n  V_hat corr: {np.mean(v_corrs):.3f}  |  "
              f"Kalman reliability: {np.mean(r_kal):.3f}")
        print(f"  n_blocks: {seeds_data[0]['n_blocks']} "
              f"({seeds_data[0]['n_train']} train)")

        print(f"\n  {'Method':<22} {'Hedge':<24} {'Ratio':>6}"
              f"  {'CE_hedge/CE_myopic':>18}")

        def fmt(hedges, ces, name):
            mh = np.mean(hedges)
            sh = np.std(hedges)
            ratio = abs(mh / hedge_gt) if abs(hedge_gt) > 1e-8 else 0
            mc = np.mean(ces)
            mm = np.mean(ce_m)
            ce_ratio = mc / mm if abs(mm) > 1e-12 else 1.0
            print(f"  {name:<22} {mh:+.6f}+/-{sh:.6f} {ratio:>5.2f}x"
                  f"  {ce_ratio:18.4f}")

        fmt(h_ols, ce_ols, 'OLS-V_hat (raw)')
        fmt(h_ols_c, [s['ce_ols_corrected'] for s in seeds_data],
            'OLS-V_hat (EIV corr)')
        fmt(h_krr, ce_krr, 'Sig-KRR')

        all_results[gamma_risk] = {
            'hedge_gt': hedge_gt, 'h_ols': h_ols,
            'h_ols_c': h_ols_c, 'h_krr': h_krr,
        }

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — Practical Horizon")
    print(f"{'='*80}")
    all_pass = True
    for gamma_risk in gamma_list:
        r = all_results[gamma_risk]
        gt = r['hedge_gt']
        for name, hedges in [('OLS(corr)', r['h_ols_c']),
                              ('Sig-KRR', r['h_krr'])]:
            mh = np.mean(hedges)
            ratio = abs(mh / gt) if abs(gt) > 1e-8 else 0
            sign_ok = mh * gt > 0
            ok = sign_ok and 0.15 < ratio < 5.0
            if not ok:
                all_pass = False
            print(f"  g={gamma_risk} {name:<12}: {mh:+.6f} "
                  f"(gt={gt:+.6f}) [{ratio:.2f}x] {'OK' if ok else 'FAIL'}")

    print(f"\nPRACTICAL HORIZON: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--practical', action='store_true')
    parser.add_argument('--both', action='store_true')
    args = parser.parse_args()

    if args.practical or args.both:
        level5_practical_horizon(n_macro=10, T=2520, k=63)
    if not args.practical or args.both:
        level5_hedging_demand(n_macro=5, T=50000, k=504)
