"""
Unified GP Framework for Bubble Detection and Eigenfunction Pricing.

Every computation — from σ² estimation through the Feller α test to
eigenfunction pricing — is a Gaussian process computation (§12 of
theory_eigenfunction_bubble_detection.md).

The GP-KRR equivalence (R&W §6.2):
    KRR:  f*(x) = k(x,X)(K + λI)⁻¹y
    GP:   f*(x) = k(x,X)(K + σ²ₙI)⁻¹y     [λ = σ²ₙ]

The GP additionally provides posterior variance (principled UQ) and
marginal likelihood (automatic model selection).

Classes:
    FellerGP          — GP with parametric mean for the α > 2 test
    SigKKFFellerGP    — Full KKF in signature RKHS with adaptive R_t
    EDMDSigFellerGP   — EDMD regression on squared returns + GP
    MarginalLikelihoodFellerGP — KF marginalizing V, grid on β
    MLKFellerGP       — Multilevel kernel Feller GP with ARD
    ScaleFunctionGP   — GP on the vol drift/diffusion ratio (L3)
    GPBubbleDetector  — Unified detector dispatching across tiers
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist


class FellerGP:
    """GP with parametric mean for the Feller tail exponent α.

    Model (R&W §2.7, eq. 2.42):
        log σ²(z) = α·log|z| + c + f(z) + ε
        f ~ GP(0, σ²_f · k_SE)
        ε ~ N(0, Σ_n)    [heteroskedastic, from block noise estimation]

    The posterior on β = (α, c) gives P(α > 2) = Φ((α̂ - 2)/σ̂_α).

    When σ_f = 0 (selected by blocked CV), this collapses to WLS on
    log|z| — the degenerate GP (L1). When σ_f > 0, the GP residual f(z)
    absorbs non-power-law structure, widening the α posterior — honest UQ.
    """

    def __init__(self, n_landmarks=80, n_blocks=None, use_bipower=False):
        self.n_landmarks = n_landmarks
        self.n_blocks = n_blocks
        self.use_bipower = use_bipower

    def fit(self, z, dz, dt):
        """Estimate α from 1D price series z and its increments dz.

        Args:
            z: (n,) price levels X_t
            dz: (n,) increments ΔX_t
            dt: time step

        Returns:
            self (call .alpha_mean, .alpha_sd, .p_bubble after fit)
        """
        n = len(z)
        if self.use_bipower:
            # BV-robust: (π/2)|ΔX_i||ΔX_{i-1}|/dt (BNS 2004)
            # Consistent for continuous QV, robust to finite-activity jumps
            abs_dz = np.abs(dz)
            sq_inc = np.zeros(n)
            sq_inc[1:] = (np.pi / 2.0) * abs_dz[1:] * abs_dz[:-1] / dt
            sq_inc[0] = dz[0] ** 2 / dt  # no previous increment
        else:
            sq_inc = dz ** 2 / dt

        n_blocks = self.n_blocks or min(10, max(5, n // 500))
        block_len = n // n_blocks
        m = min(self.n_landmarks, n // 5)

        # --- Stage 1: NW kernel estimates of σ²(z) at landmarks ---
        quantiles = np.linspace(0.01, 0.99, m)
        landmarks = np.quantile(z, quantiles)
        ldists = np.abs(np.diff(landmarks))
        bw = np.median(ldists) if len(ldists) > 0 else np.std(z)
        bw = max(bw, 1e-8)

        diff = (landmarks[:, None] - z[None, :]) / bw
        K_nw = np.exp(-0.5 * diff ** 2)
        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm = K_sum > 1e-10
        sigma2_nw = np.zeros(m)
        n_eff = np.zeros(m)
        sigma2_nw[valid_lm] = (K_nw[valid_lm] @ sq_inc) / K_sum[valid_lm]
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = valid_lm & (np.abs(landmarks) > 1e-4) & (sigma2_nw > 1e-8) & (n_eff > 2)
        if np.sum(valid) < 10:
            self.alpha_mean = np.nan
            self.alpha_sd = np.nan
            self.p_bubble = 0.0
            self.vol_p_bubble = 0.0
            return self

        x = np.log(np.abs(landmarks[valid]))
        y = np.log(sigma2_nw[valid])
        nv = len(x)
        valid_idx = np.where(valid)[0]

        # --- Block-based noise variance (Priestley correction) ---
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                K_b = K_nw[j, sl]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ sq_inc[sl]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = np.var(block_ests, ddof=1) / len(block_ests)
            else:
                noise_var[jj] = 2.0 / n_eff[j]
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # --- Per-block α estimates (blocked bootstrap SE) ---
        block_alphas = []
        for b in range(n_blocks):
            sl = slice(b * block_len, min((b + 1) * block_len, n))
            z_b = z[sl]
            sq_b = sq_inc[sl]
            diff_b = (landmarks[:, None] - z_b[None, :]) / bw
            K_b = np.exp(-0.5 * diff_b ** 2)
            K_b_sum = K_b.sum(axis=1)
            valid_b = valid & (K_b_sum > 1e-10)
            if np.sum(valid_b) < 5:
                continue
            s2_b = np.zeros(m)
            s2_b[K_b_sum > 1e-10] = (K_b[K_b_sum > 1e-10] @ sq_b) / K_b_sum[K_b_sum > 1e-10]
            mask_b = valid_b & (s2_b > 1e-10)
            if np.sum(mask_b) < 5:
                continue
            x_b = np.log(np.abs(landmarks[mask_b]))
            y_b = np.log(s2_b[mask_b])
            n_eff_b = np.zeros(m)
            K_b_sq = (K_b ** 2).sum(axis=1)
            n_eff_b[K_b_sum > 1e-10] = K_b_sum[K_b_sum > 1e-10] ** 2 / K_b_sq[K_b_sum > 1e-10]
            w_b = n_eff_b[mask_b]
            H_b = np.column_stack([x_b, np.ones(len(x_b))])
            WH = H_b * w_b[:, None]
            try:
                beta_b = np.linalg.solve(WH.T @ H_b, WH.T @ y_b)
                block_alphas.append(beta_b[0])
            except np.linalg.LinAlgError:
                continue

        block_alpha_sd = None
        if len(block_alphas) >= 3:
            block_alpha_sd = np.std(block_alphas, ddof=1) / np.sqrt(len(block_alphas))

        # --- GP: blocked time-series CV for σ_f ---
        H = np.column_stack([x, np.ones(nv)])
        Sigma_n = np.diag(noise_var)
        x_range = x.max() - x.min()
        ell = max(x_range / 4.0, 0.1)
        sq_dists = (x[:, None] - x[None, :]) ** 2
        K_base = np.exp(-sq_dists / (2 * ell ** 2))

        # Assign landmarks to temporal blocks
        lm_block_id = np.zeros(nv, dtype=int)
        for jj, j in enumerate(valid_idx):
            block_weights = np.zeros(n_blocks)
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                block_weights[b] = K_nw[j, sl].sum()
            lm_block_id[jj] = np.argmax(block_weights)

        def _blocked_cv_mse(log_sf):
            sf2 = np.exp(2 * log_sf)
            total_mse = 0.0
            n_test = 0
            for fold in np.unique(lm_block_id):
                test_mask = lm_block_id == fold
                train_mask = ~test_mask
                if np.sum(train_mask) < 3 or np.sum(test_mask) < 1:
                    continue
                x_tr, y_tr = x[train_mask], y[train_mask]
                x_te, y_te = x[test_mask], y[test_mask]
                H_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
                H_te = np.column_stack([x_te, np.ones(len(x_te))])
                Sigma_tr = np.diag(noise_var[train_mask])
                C_tr = sf2 * K_base[np.ix_(train_mask, train_mask)] + Sigma_tr
                try:
                    L_tr = np.linalg.cholesky(C_tr)
                except np.linalg.LinAlgError:
                    return 1e10
                Cinv_y = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, y_tr))
                Cinv_H = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, H_tr))
                A = H_tr.T @ Cinv_H
                try:
                    beta_hat = np.linalg.solve(A, H_tr.T @ Cinv_y)
                except np.linalg.LinAlgError:
                    return 1e10
                r_tr = y_tr - H_tr @ beta_hat
                Cinv_r = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, r_tr))
                K_te_tr = sf2 * K_base[np.ix_(test_mask, train_mask)]
                y_pred = H_te @ beta_hat + K_te_tr @ Cinv_r
                total_mse += np.sum((y_te - y_pred) ** 2)
                n_test += len(y_te)
            return total_mse / max(1, n_test)

        log_sf_grid = np.concatenate([[-20], np.linspace(-4, 2, 20)])
        cv_scores = np.array([_blocked_cv_mse(lsf) for lsf in log_sf_grid])
        best_log_sf = log_sf_grid[np.argmin(cv_scores)]
        sf2_opt = np.exp(2 * best_log_sf) if best_log_sf > -19 else 0.0

        # --- GP posterior on β = (α, c) ---
        C = sf2_opt * K_base + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            self.alpha_mean, self.alpha_sd, self.p_bubble, self.vol_p_bubble = np.nan, np.nan, 0.0, 0.0
            return self

        A = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.alpha_mean, self.alpha_sd, self.p_bubble, self.vol_p_bubble = np.nan, np.nan, 0.0, 0.0
            return self

        beta_hat = A_inv @ H.T @ C_inv @ y
        gp_alpha_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        # Final SD: max of GP posterior and blocked bootstrap
        if block_alpha_sd is not None:
            alpha_sd = max(gp_alpha_sd, block_alpha_sd)
        else:
            alpha_sd = gp_alpha_sd

        self.alpha_mean = float(beta_hat[0])
        self.alpha_sd = alpha_sd
        self.sf2_opt = sf2_opt
        self.ell = ell

        if np.isnan(self.alpha_mean) or alpha_sd <= 0:
            self.p_bubble = 0.0
            self.vol_p_bubble = 0.0
        else:
            z_score = (self.alpha_mean - 2.0) / alpha_sd
            self.p_bubble = float(stats.norm.cdf(z_score))
            # Delta method: Var(P) ≈ φ(z)² · Var(α) / σ_α² = φ(z)²
            # vol(P) = φ(z) — maximized at α̂=2 (boundary), drops off away
            self.vol_p_bubble = float(stats.norm.pdf(z_score))

        # Store internals for diagnostics
        self._x = x
        self._y = y
        self._H = H
        self._C_inv = C_inv
        self._beta_hat = beta_hat
        self._A_inv = A_inv
        self._landmarks = landmarks[valid]
        self._sigma2_nw = sigma2_nw[valid]
        self._noise_var = noise_var
        self._block_alphas = block_alphas

        return self

    def predict(self, z_new):
        """GP posterior mean of log σ²(z) at new points."""
        x_new = np.log(np.abs(z_new))
        H_new = np.column_stack([x_new, np.ones(len(x_new))])
        mean_part = H_new @ self._beta_hat
        # GP residual
        r = self._y - self._H @ self._beta_hat
        Cinv_r = self._C_inv @ r
        sq_dists = (x_new[:, None] - self._x[None, :]) ** 2
        K_new = self.sf2_opt * np.exp(-sq_dists / (2 * self.ell ** 2))
        return mean_part + K_new @ Cinv_r

    def predict_variance(self, z_new):
        """GP posterior variance of log σ²(z) at new points."""
        x_new = np.log(np.abs(z_new))
        H_new = np.column_stack([x_new, np.ones(len(x_new))])
        sq_dists_new = (x_new[:, None] - self._x[None, :]) ** 2
        K_new = self.sf2_opt * np.exp(-sq_dists_new / (2 * self.ell ** 2))
        # Prior variance
        k_star = self.sf2_opt * np.ones(len(x_new))
        # Posterior variance (R&W eq. 2.24 extended for parametric mean)
        v = K_new @ self._C_inv
        var_gp = k_star - np.sum(v * K_new, axis=1)
        # Parametric mean uncertainty
        R = H_new - v @ self._H
        var_mean = np.sum(R @ self._A_inv * R, axis=1)
        return np.maximum(var_gp + var_mean, 0.0)


class SigKernelFellerGP:
    """Nonparametric GP Feller test with signature kernel.

    Fully nonparametric: no parametric mean (no α·log|z| + c assumption).
    Instead, estimates log σ²(z) via GP with zero mean and signature kernel,
    then evaluates the Feller integral directly from GP posterior samples.

    P(bubble) = P(∫_c^∞ x/σ²(x) dx < ∞ | data)

    estimated by sampling GP posterior functions and checking integral
    convergence. The local elasticity ε(z) = d log σ²/d log z is extracted
    from the GP posterior gradient — no power-law assumption needed.

    This handles DGPs that the parametric α test misses:
      - Log-modified: σ²(S) = S² log(S)^p (α=2 but bubble for p>0)
      - Quadratic local vol: σ²(S) = a + bS + cS²
      - Regime-switching: α varies with price level
      - SABR-type: effective σ² is not a clean power law

    Kernel options:
        'exp_sig': exp(⟨S^N_i, S^N_j⟩ / c) — exponential on truncated sigs
        'pde':     Goursat PDE signature kernel (untruncated, slower)

    Signatures computed cumulatively via RecurrentLeadLagLogSigMap (BCH).
    """

    def __init__(self, n_landmarks=80, n_blocks=None, use_bipower=False,
                 kernel='exp_sig', sig_level=2, sig_gamma=0.99,
                 path_window=100, n_posterior_samples=200):
        self.n_landmarks = n_landmarks
        self.n_blocks = n_blocks
        self.use_bipower = use_bipower
        self.kernel_type = kernel
        self.sig_level = sig_level
        self.sig_gamma = sig_gamma
        self.path_window = path_window
        self.n_posterior_samples = n_posterior_samples

    def fit(self, z, dz, dt):
        """Nonparametric Feller test from price series z.

        Pipeline:
          1. NW kernel estimates of σ²(z) at landmarks
          2. Cumulative lead-lag log-signatures at landmark times
          3. Block noise estimation
          4. GP with signature kernel (zero mean) → posterior on log σ²
          5. Sample GP posterior → evaluate Feller integral
          6. Local elasticity from GP gradient
        """
        n = len(z)

        # --- Squared increments ---
        if self.use_bipower:
            abs_dz = np.abs(dz)
            sq_inc = np.zeros(n)
            sq_inc[1:] = (np.pi / 2.0) * abs_dz[1:] * abs_dz[:-1] / dt
            sq_inc[0] = dz[0] ** 2 / dt
        else:
            sq_inc = dz ** 2 / dt

        n_blocks = self.n_blocks or min(10, max(5, n // 500))
        block_len = n // n_blocks
        m = min(self.n_landmarks, n // 5)

        # --- Stage 1: NW at landmarks with vol conditioning ---
        # For stochastic vol processes (SABR, 3/2, Heston), the NW estimate
        # σ̂²(z) is biased: extreme prices are visited during extreme vol
        # episodes, inflating σ̂² at the boundary. Fix: product NW kernel
        # K(z, v) = K_z(z_j, z_i) · K_v(v_j, v_i) where v = rolling QV.
        # This conditions on vol state, isolating the structural S^β scaling.
        #
        # For CEV (no stochastic vol), K_v ≈ 1 everywhere (v is smooth),
        # so this reduces to standard NW — no harm.

        # Rolling QV as vol proxy (window = min(200, n/10))
        qv_window = min(200, n // 10)
        rolling_qv = np.zeros(n)
        cumsum_sq = np.cumsum(dz ** 2)
        cumsum_sq = np.insert(cumsum_sq, 0, 0.0)
        for i in range(n):
            lo = max(0, i - qv_window + 1)
            rolling_qv[i] = (cumsum_sq[min(i + 1, len(cumsum_sq) - 1)]
                             - cumsum_sq[lo]) / ((i - lo + 1) * dt)
        # Normalize to unit variance for bandwidth selection
        qv_std = np.std(rolling_qv)
        if qv_std < 1e-10:
            qv_std = 1.0
        rolling_qv_norm = rolling_qv / qv_std

        quantiles = np.linspace(0.01, 0.99, m)
        landmarks = np.quantile(z, quantiles)
        ldists = np.abs(np.diff(landmarks))
        bw = np.median(ldists) if len(ldists) > 0 else np.std(z)
        bw = max(bw, 1e-8)

        # Vol bandwidth: wide (range/2) — mild conditioning
        bw_v = max(np.std(rolling_qv_norm) * 0.5, 0.1)

        # Product NW at each landmark, conditioned on a FIXED reference
        # vol level (median). This removes the price-vol selection bias:
        # we ask "what is σ²(z) when vol is at its typical level?"
        diff_z = (landmarks[:, None] - z[None, :]) / bw
        K_z = np.exp(-0.5 * diff_z ** 2)

        # Reference vol: median rolling QV (same for ALL landmarks)
        ref_qv = np.median(rolling_qv_norm)
        diff_v = (ref_qv - rolling_qv_norm[None, :]) / bw_v  # (1, n)
        K_v = np.exp(-0.5 * diff_v ** 2)  # (1, n) — broadcast to all landmarks

        # Product kernel
        K_nw = K_z * K_v
        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm = K_sum > 1e-10
        sigma2_nw = np.zeros(m)
        n_eff = np.zeros(m)
        sigma2_nw[valid_lm] = (K_nw[valid_lm] @ sq_inc) / K_sum[valid_lm]
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = (valid_lm & (np.abs(landmarks) > 1e-4)
                 & (sigma2_nw > 1e-8) & (n_eff > 2))
        if np.sum(valid) < 10:
            self._set_nan()
            return self

        x = np.log(np.abs(landmarks[valid]))  # log|z|
        y = np.log(sigma2_nw[valid])           # log σ̂²
        nv = len(x)
        valid_idx = np.where(valid)[0]

        # --- Stage 2: Cumulative signatures at landmark times ---
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', '..', 'examples', 'proof_of_concept'))
        from signature_features import RecurrentLeadLagLogSigMap

        log_prices = np.log(np.abs(z) + 1e-10)
        log_returns = np.diff(log_prices)

        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=self.sig_level,
            forgetting_factor=self.sig_gamma)
        all_sigs = np.zeros((n, sig_map.feature_dim))
        for i in range(len(log_returns)):
            all_sigs[i + 1] = sig_map.update(np.array([log_returns[i]]))

        lm_time_idx = np.zeros(nv, dtype=int)
        for jj, j in enumerate(valid_idx):
            lm_time_idx[jj] = np.argmax(K_nw[j])

        sigs = all_sigs[lm_time_idx]

        # For PDE kernel: extract windowed path segments
        if self.kernel_type == 'pde':
            hw = self.path_window // 2
            paths = []
            for t_idx in lm_time_idx:
                start = max(0, t_idx - hw)
                end = min(n, t_idx + hw)
                paths.append(log_prices[start:end].copy())
        else:
            paths = None

        # --- Stage 3: Block noise ---
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                K_b = K_nw[j, sl]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ sq_inc[sl]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = np.var(block_ests, ddof=1) / len(block_ests)
            else:
                noise_var[jj] = 2.0 / n_eff[j]
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # --- Stage 4: Build sum kernel (SE on price + sig kernel) ---
        # Sum kernel: k = σ²_z · k_SE(log|z|) + σ²_sig · k_exp_sig(sig)
        # SE component captures price-level dependence (nonparametric)
        # Sig component captures path-dependent deviations
        # No parametric mean — the SE kernel learns ε(z) freely
        Sigma_n = np.diag(noise_var)

        # SE kernel on log|z|
        x_range = x.max() - x.min()
        sq_dists = (x[:, None] - x[None, :]) ** 2

        # Sig kernel
        if self.kernel_type == 'exp_sig':
            inner_prods = sigs @ sigs.T
        elif self.kernel_type == 'pde':
            K_pde_raw = self._pde_kernel_matrix(paths)

        def _neg_log_marginal_likelihood(K_total):
            """Negative log marginal likelihood for zero-mean GP.

            log p(y|θ) = -½ y^T C⁻¹ y - ½ log|C| - n/2 log(2π)

            The complexity penalty (-½ log|C|) naturally penalizes short
            length scales, preventing overfitting to NW noise from
            stochastic vol averaging.
            """
            C = K_total + Sigma_n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            # Data fit: -½ y^T C⁻¹ y
            data_fit = -0.5 * y @ alpha
            # Complexity: -½ log|C| = -Σ log(diag(L))
            complexity = -np.sum(np.log(np.diag(L)))
            return -(data_fit + complexity)  # negative because we minimize

        # Grid search: ℓ_z × σ²_z × (c, σ²_sig for exp_sig)
        # Minimum ℓ = range/3: σ²(z) is smooth on >1/3 of the log-price
        # range. This prevents overfitting to vol-state wiggles in the NW
        # estimates (e.g. SABR, 3/2 model) while remaining much weaker
        # than assuming a parametric power law.
        ell_min = max(x_range / 3, 0.3)
        ell_candidates = [max(x_range / d, ell_min) for d in [3, 2, 1, 0.5]]
        log_sf_z_grid = np.linspace(-2, 3, 8)

        if self.kernel_type == 'exp_sig':
            triu_idx = np.triu_indices(nv, 1)
            median_ip = np.median(np.abs(inner_prods[triu_idx]))
            c_candidates = [max(median_ip * f, 1e-6)
                            for f in [0.25, 0.5, 1.0, 2.0, 4.0]]
            log_sf_sig_grid = np.concatenate([[-20], np.linspace(-2, 2, 6)])
        else:
            c_candidates = [1.0]
            log_sf_sig_grid = np.concatenate([[-20], np.linspace(-2, 2, 6)])

        best_cv = 1e10
        best_params = {}
        for ell in ell_candidates:
            K_se = np.exp(-sq_dists / (2 * ell ** 2))
            for lsf_z in log_sf_z_grid:
                sf2_z = np.exp(2 * lsf_z)
                for c_val in c_candidates:
                    if self.kernel_type == 'exp_sig':
                        K_sig_raw = np.exp(
                            np.clip(inner_prods / c_val, -50, 50))
                    else:
                        K_sig_raw = K_pde_raw
                    for lsf_sig in log_sf_sig_grid:
                        sf2_sig = (np.exp(2 * lsf_sig)
                                   if lsf_sig > -19 else 0.0)
                        K_total = sf2_z * K_se + sf2_sig * K_sig_raw
                        nlml = _neg_log_marginal_likelihood(K_total)
                        if nlml < best_cv:
                            best_cv = nlml
                            best_params = {
                                'ell': ell, 'sf2_z': sf2_z,
                                'c': c_val, 'sf2_sig': sf2_sig}

        ell_opt = best_params.get('ell', ell_candidates[1])
        sf2_z_opt = best_params.get('sf2_z', 1.0)
        c_opt = best_params.get('c', c_candidates[0])
        sf2_sig_opt = best_params.get('sf2_sig', 0.0)

        K_se_opt = sf2_z_opt * np.exp(-sq_dists / (2 * ell_opt ** 2))
        if self.kernel_type == 'exp_sig':
            K_sig_opt = sf2_sig_opt * np.exp(
                np.clip(inner_prods / c_opt, -50, 50))
        else:
            K_sig_opt = sf2_sig_opt * K_pde_raw

        K_opt = K_se_opt + K_sig_opt

        self.sf2_opt = sf2_z_opt
        self._ell_opt = ell_opt
        self._c_opt = c_opt
        self._sf2_sig_opt = sf2_sig_opt
        self._K_se = K_se_opt
        self._K_sig = K_sig_opt

        # --- Stage 5: GP posterior (zero mean) ---
        C = K_opt + Sigma_n
        try:
            L_C = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        alpha_gp = np.linalg.solve(L_C.T, np.linalg.solve(L_C, y))
        # Posterior mean at landmarks: μ* = K* C⁻¹ y = K_opt @ alpha_gp
        post_mean = K_opt @ alpha_gp
        # Posterior covariance: K* - K* C⁻¹ K*
        V = np.linalg.solve(L_C, K_opt)
        post_cov = K_opt - V.T @ V
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(post_cov)
        if eigvals.min() < 0:
            post_cov += (-eigvals.min() + 1e-8) * np.eye(nv)

        # --- Stage 6: Local elasticity from GP gradient ---
        # ε(z_j) = d(log σ²)/d(log z) — finite difference on sorted data
        sort_by_x = np.argsort(x)
        x_s = x[sort_by_x]
        pm_s = post_mean[sort_by_x]
        dx_s = np.diff(x_s)
        dy_s = np.diff(pm_s)
        # Protect against duplicate x values
        dx_s = np.maximum(dx_s, 1e-10)
        elasticity_mid = dy_s / dx_s
        elasticity_sorted = np.zeros(nv)
        elasticity_sorted[0] = elasticity_mid[0]
        elasticity_sorted[-1] = elasticity_mid[-1]
        elasticity_sorted[1:-1] = 0.5 * (elasticity_mid[:-1]
                                          + elasticity_mid[1:])
        # Map back to original order
        elasticity = np.zeros(nv)
        elasticity[sort_by_x] = elasticity_sorted

        # --- Stage 7: GP gradient-based P(bubble) ---
        # The Feller test requires σ²(z) for ALL z → ∞, which is impossible
        # nonparametrically. Instead, compute the LOCAL elasticity
        #   ε(z) = d log σ² / d log z
        # at the upper boundary from the GP posterior derivative.
        #
        # For SE kernel k(x,x') = sf² exp(-(x-x')²/(2ℓ²)):
        #   dk/dx₀ = -(x₀-x')/ℓ² · k(x₀,x')
        #   d²k/dx₀² = (1/ℓ² - (x₀-x')²/ℓ⁴) · k(x₀,x₀)  [at x₀=x']
        # The GP derivative at x₀ is Gaussian (R&W §9.4):
        #   E[df/dx|x₀] = dk(x₀,·)/dx₀ · C⁻¹ · y
        #   Var[df/dx|x₀] = d²k(x₀,x₀)/dx₀² - dk(x₀,·)/dx₀ · C⁻¹ · dk(·,x₀)/dx₀
        # Since f = log σ² and x = log|z|, df/dx = ε(z) directly.

        # Evaluate at multiple points in the upper part of data range
        # to get a robust boundary elasticity estimate
        x_upper_pctile = np.percentile(x, [70, 80, 90, 95])
        gradient_means = []
        gradient_vars = []

        C_inv_y = alpha_gp  # already computed: C⁻¹ y
        # C⁻¹ = L_C⁻ᵀ L_C⁻¹
        # dk(x0, x_j)/dx0 for SE kernel only (sig kernel has no x0 dependence)
        for x0 in x_upper_pctile:
            diff = x0 - x  # (nv,)
            k_x0 = sf2_z_opt * np.exp(-diff ** 2 / (2 * ell_opt ** 2))
            # Gradient of SE kernel w.r.t. x₀
            dk_dx0 = -diff / (ell_opt ** 2) * k_x0  # (nv,)

            # Posterior gradient mean: E[df/dx₀] = dk/dx₀ · C⁻¹ y
            grad_mean = dk_dx0 @ C_inv_y

            # Posterior gradient variance
            # Prior: d²k(x₀,x₀)/dx₀² = sf² / ℓ² (for SE at x₀=x₀)
            prior_grad_var = sf2_z_opt / (ell_opt ** 2)
            # Posterior reduction: dk/dx₀ · C⁻¹ · dk/dx₀
            v = np.linalg.solve(L_C, dk_dx0)
            reduction = v @ v
            grad_var = max(prior_grad_var - reduction, 1e-10)

            gradient_means.append(float(grad_mean))
            gradient_vars.append(float(grad_var))

        # Aggregate: weighted average of boundary elasticity estimates
        # Weight by inverse variance (more certain estimates count more)
        weights = 1.0 / np.array(gradient_vars)
        weights /= weights.sum()
        eps_gp_mean = float(np.sum(weights * np.array(gradient_means)))
        # Combined variance via inverse-variance weighting
        eps_gp_var = float(1.0 / np.sum(1.0 / np.array(gradient_vars)))
        eps_gp_sd = np.sqrt(eps_gp_var)

        # P(ε > 2) from GP posterior gradient
        z_score_gp = (eps_gp_mean - 2.0) / max(eps_gp_sd, 1e-8)
        p_bubble_gradient = float(stats.norm.cdf(z_score_gp))

        # Also keep Feller integral estimate via posterior sampling at landmarks
        # (less reliable but complementary — uses posterior covariance structure)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        post_mean_sorted = post_mean[sort_idx]
        post_cov_sorted = post_cov[np.ix_(sort_idx, sort_idx)]
        try:
            L_post = np.linalg.cholesky(
                post_cov_sorted + 1e-8 * np.eye(nv))
        except np.linalg.LinAlgError:
            L_post = np.zeros((nv, nv))

        z_grid = np.exp(x_sorted)
        n_converged = 0
        feller_integrals = []
        rng = np.random.RandomState(42)
        for _ in range(self.n_posterior_samples):
            eps = rng.randn(nv)
            log_s2 = post_mean_sorted + L_post @ eps
            s2 = np.exp(np.clip(log_s2, -50, 50))
            integrand = z_grid / np.maximum(s2, 1e-30)
            q75 = int(0.75 * nv)
            if q75 > 2 and nv - q75 > 2:
                log_z_u = np.log(z_grid[q75:] + 1e-10)
                log_int_u = np.log(integrand[q75:] + 1e-30)
                ok = np.isfinite(log_int_u)
                if np.sum(ok) >= 2:
                    H = np.column_stack([log_z_u[ok], np.ones(np.sum(ok))])
                    slope = np.linalg.lstsq(H, log_int_u[ok], rcond=None)[0][0]
                    if slope < -1.0:
                        n_converged += 1
            feller_integrals.append(np.trapz(integrand, z_grid))
        feller_integrals = np.array(feller_integrals)

        # --- Stage 8: P(bubble) and diagnostics ---
        self.p_bubble_feller = float(n_converged / self.n_posterior_samples)

        # Local elasticity at highest price (boundary) — from finite diffs
        eps_boundary = float(elasticity[-1])

        # Block bootstrap elasticity (for comparison / robustness)
        block_elasticities = []
        for b in range(n_blocks):
            sl = slice(b * block_len, min((b + 1) * block_len, n))
            z_b = z[sl]
            sq_b = sq_inc[sl]
            diff_b = (landmarks[:, None] - z_b[None, :]) / bw
            K_b = np.exp(-0.5 * diff_b ** 2)
            K_b_sum = K_b.sum(axis=1)
            valid_b = valid & (K_b_sum > 1e-10)
            if np.sum(valid_b) < 5:
                continue
            s2_b = np.zeros(m)
            s2_b[K_b_sum > 1e-10] = (
                (K_b[K_b_sum > 1e-10] @ sq_b) / K_b_sum[K_b_sum > 1e-10])
            mask_b = valid_b & (s2_b > 1e-10)
            if np.sum(mask_b) < 5:
                continue
            x_b = np.log(np.abs(landmarks[mask_b]))
            y_b = np.log(s2_b[mask_b])
            upper_q = np.percentile(x_b, 75)
            upper_mask = x_b >= upper_q
            if np.sum(upper_mask) >= 3:
                x_u = x_b[upper_mask]
                y_u = y_b[upper_mask]
                H_u = np.column_stack([x_u, np.ones(len(x_u))])
                try:
                    beta_u = np.linalg.lstsq(H_u, y_u, rcond=None)[0]
                    block_elasticities.append(beta_u[0])
                except np.linalg.LinAlgError:
                    continue

        # Primary P(bubble): GP posterior gradient (principled, R&W §9.4)
        # This gives P(ε > 2) where ε is the local volatility elasticity
        # at the upper boundary, with uncertainty from the GP posterior.
        self.p_bubble = p_bubble_gradient
        self.vol_p_bubble = 0.0

        # Report: effective α = GP gradient mean, SD = GP gradient SD
        self.alpha_mean = eps_gp_mean
        self.alpha_sd = eps_gp_sd

        # Store internals
        self._x = x
        self._y = y
        self._post_mean = post_mean
        self._post_cov = post_cov
        self._elasticity = elasticity
        self._landmarks = landmarks[valid]
        self._sigma2_nw = sigma2_nw[valid]
        self._noise_var = noise_var
        self._sigs = sigs
        self._all_sigs = all_sigs
        self._K = K_opt
        self._feller_integrals = feller_integrals
        self._block_elasticities = block_elasticities
        self._eps_boundary = eps_boundary
        # GP gradient diagnostics
        self._eps_gp_mean = eps_gp_mean
        self._eps_gp_sd = eps_gp_sd
        self._gradient_means = gradient_means
        self._gradient_vars = gradient_vars
        self._x_upper_pctile = x_upper_pctile

        return self

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0

    @staticmethod
    def _pde_kernel_matrix(paths):
        """PDE signature kernel via Goursat PDE on lead-lag embedded paths.

        Solves ∂²u/∂s∂t = ⟨ẋ_s, ẏ_t⟩ · u(s,t), u(0,·) = u(·,0) = 1.
        k(x,y) = u(L_x, L_y).
        """
        m = len(paths)
        K = np.zeros((m, m))

        # Precompute lead-lag embedded increments
        ll_incs = []
        for path in paths:
            inc = np.diff(path)
            ll = np.zeros((2 * len(inc), 2))
            ll[0::2, 0] = inc
            ll[1::2, 1] = inc
            ll_incs.append(ll)

        for i in range(m):
            for j in range(i, m):
                dx, dy = ll_incs[i], ll_incs[j]
                L1, L2 = len(dx), len(dy)
                u = np.ones((L1 + 1, L2 + 1))
                for ii in range(1, L1 + 1):
                    for jj in range(1, L2 + 1):
                        inc = dx[ii - 1] @ dy[jj - 1]
                        u[ii, jj] = (u[ii - 1, jj] + u[ii, jj - 1]
                                     - u[ii - 1, jj - 1]
                                     + inc * u[ii - 1, jj - 1])
                K[i, j] = u[L1, L2]
                K[j, i] = K[i, j]

        eigmin = np.linalg.eigvalsh(K)[0]
        if eigmin < 0:
            K += (-eigmin + 1e-6) * np.eye(m)

        return K

    def local_elasticity(self, z_query=None):
        """Local volatility elasticity ε(z) = d log σ²/d log z.

        If z_query is None, returns elasticity at all landmarks.
        """
        if z_query is None:
            return self._x, self._elasticity
        # Interpolate
        return np.interp(np.log(np.abs(z_query)), self._x, self._elasticity)


class LocalVolElasticityGP:
    """2D GP structural elasticity test for stochastic vol processes.

    For dS = μ(S,V)dt + σ(S,V)dW₁, dV = a(V)dt + b(V)dW₂ (correlated),
    the 1D Feller test on marginal E[σ²|S] is WRONG — the S-V correlation
    inflates the marginal elasticity (e.g., SABR β=1.5 gives marginal α≈6).

    This class estimates σ²(S, V̂) with a 2D GP on (log|S|, log V̂), then
    uses the GP posterior gradient in the S-direction (R&W §9.4) to extract
    the STRUCTURAL elasticity ε = ∂ log σ² / ∂ log S, conditioning on V̂.

    The key insight: by conditioning on a fixed V̂ level, we break the
    price-vol selection bias. The GP gradient gives ε with uncertainty,
    so P(bubble) = P(ε > 2) from the Gaussian posterior on the derivative.

    Collinearity check: when V̂ ≈ f(S) (CEV, no stochastic vol), the
    data lies on a 1D manifold in (log S, log V̂) and the GP partial
    derivative ∂/∂s at fixed v̂ is ill-defined. We detect this via R²
    of log V̂ ~ log|S| and orthogonalize: v̂_⊥ = log V̂ - α̂·log|S|.
    When R² ≈ 1, v̂_⊥ is pure noise, ℓ_v → ∞ via ML, and the test
    reduces to 1D — correctly recovering the total elasticity β.

    Pipeline:
      1. Estimate V̂ from rolling QV (or use provided vol_proxy)
      1b. Orthogonalize: regress log V̂ on log|S|, use residual
      2. 2D NW: σ̂²(S_j, V̂_j) at landmarks in (S, V̂_⊥) space
      3. Block noise estimation on log σ̂²
      4. 2D GP with product SE kernel on (log|S|, V̂_⊥), selected by ML
      5. GP posterior gradient in S-direction at upper boundary
      6. P(bubble) = P(ε > 2) from gradient posterior
    """

    def __init__(self, n_landmarks=80, n_blocks=None, qv_window=200):
        self.n_landmarks = n_landmarks
        self.n_blocks = n_blocks
        self.qv_window = qv_window

    def _rolling_qv(self, z, dz, dt):
        """Estimate instantaneous vol from rolling quadratic variation."""
        n = len(z)
        w = min(self.qv_window, n // 5)
        qv = np.zeros(n)
        cs = np.cumsum(dz ** 2)
        cs = np.insert(cs, 0, 0.0)
        for i in range(n):
            lo = max(0, i - w + 1)
            hi = min(i + 1, len(cs) - 1)
            qv[i] = (cs[hi] - cs[lo]) / ((hi - lo) * dt + 1e-30)
        return np.maximum(qv, 1e-10)

    def fit(self, z, dz, dt, vol_proxy=None):
        """Estimate structural elasticity via 2D GP.

        Args:
            z: (n,) price levels S_t
            dz: (n,) increments ΔS_t
            dt: time step
            vol_proxy: (n,) optional vol proxy. If None, uses rolling QV.

        Returns:
            self (call .alpha_mean, .alpha_sd, .p_bubble after fit)
        """
        n = len(z)
        sq_inc = dz ** 2 / dt

        n_blocks = self.n_blocks or min(10, max(5, n // 500))
        block_len = n // n_blocks
        m = min(self.n_landmarks, n // 5)

        # --- Stage 1: Vol proxy ---
        if vol_proxy is not None:
            V = vol_proxy.copy()
        else:
            V = self._rolling_qv(z, dz, dt)

        # Work in log space for both coordinates
        log_z = np.log(np.maximum(np.abs(z), 1e-10))
        log_v = np.log(np.maximum(V, 1e-10))

        # --- Stage 1b: Orthogonalize V̂ ---
        # For CEV, V̂ = σ₀²S^β so log V̂ ≈ β·log|S| + const (R²≈1).
        # The GP partial derivative ∂/∂s at fixed v̂ is ill-defined when
        # data lies on a 1D manifold. Fix: project out the S-component.
        # v_⊥ = log V̂ - (α̂·log|S| + ĉ) captures independent vol state.
        # For SABR, v_⊥ captures the stochastic V dynamics.
        H_orth = np.column_stack([log_z, np.ones(n)])
        beta_orth = np.linalg.lstsq(H_orth, log_v, rcond=None)[0]
        log_v_pred = H_orth @ beta_orth
        log_v_resid = log_v - log_v_pred
        r2_sv = 1.0 - np.var(log_v_resid) / max(np.var(log_v), 1e-30)
        self._r2_sv = float(r2_sv)
        self._sv_slope = float(beta_orth[0])

        # Use orthogonalized residual as 2nd GP dimension
        log_v_orth = log_v_resid

        # --- Stage 2: NW at landmarks ---
        # 1D NW in z-space (like FellerGP) — avoids 2D curse of dimensionality.
        # The V̂_⊥ dimension enters ONLY through the GP kernel, not through NW.
        quantiles = np.linspace(0.01, 0.99, m)
        landmarks_z = np.quantile(log_z, quantiles)
        ldists = np.abs(np.diff(landmarks_z))
        bw_s = np.median(ldists) if len(ldists) > 0 else np.std(log_z)
        bw_s = max(bw_s, 1e-8)

        diff_s = (landmarks_z[:, None] - log_z[None, :]) / bw_s
        K_nw = np.exp(-0.5 * diff_s ** 2)
        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)

        sigma2_nw = np.zeros(m)
        n_eff = np.zeros(m)
        valid_lm = K_sum > 1e-10
        sigma2_nw[valid_lm] = (K_nw[valid_lm] @ sq_inc) / K_sum[valid_lm]
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = (valid_lm & (sigma2_nw > 1e-8) & (n_eff > 2)
                 & (np.abs(np.exp(landmarks_z)) > 1e-4))

        if np.sum(valid) < 10:
            self._set_nan()
            return self

        x_s = landmarks_z[valid]  # log|S| at valid landmarks
        y = np.log(sigma2_nw[valid])
        nv = len(x_s)
        valid_idx = np.where(valid)[0]

        # V̂_⊥ at each landmark: weighted average of orthogonalized vol
        # at the data points contributing to each landmark
        x_v = np.zeros(nv)
        for jj, j in enumerate(valid_idx):
            w = K_nw[j]
            w_sum = w.sum()
            if w_sum > 1e-10:
                x_v[jj] = (w @ log_v_orth) / w_sum

        # --- Stage 3: Block noise estimation ---
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                K_b = K_nw[j, sl]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ sq_inc[sl]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = np.var(block_ests, ddof=1) / len(block_ests)
            else:
                noise_var[jj] = 2.0 / n_eff[valid_idx[jj]]
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # --- Stage 4: GP with parametric mean + 2D product SE kernel ---
        # Model: log σ̂²(s, v⊥) = α·s + c + f(s, v⊥) + ε
        # where s = log|S|, v⊥ = orthogonalized log V̂
        # The parametric mean α·s + c captures the power-law trend.
        # The GP residual f captures non-power-law structure and V̂ effects.
        # P(bubble) from posterior on α + GP gradient correction.
        H = np.column_stack([x_s, np.ones(nv)])
        Sigma_n = np.diag(noise_var)
        s_range = x_s.max() - x_s.min()
        v_range = x_v.max() - x_v.min()

        # Squared distances in each dimension
        sq_dists_s = (x_s[:, None] - x_s[None, :]) ** 2
        sq_dists_v = (x_v[:, None] - x_v[None, :]) ** 2

        def _neg_log_ml(log_sf, ell_s, ell_v):
            """Neg log marginal likelihood with parametric mean (R&W §2.7)."""
            sf2 = np.exp(2 * log_sf)
            K = sf2 * np.exp(-sq_dists_s / (2 * ell_s ** 2)
                             - sq_dists_v / (2 * ell_v ** 2))
            C = K + Sigma_n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e10
            # Profile out β = (α, c): R&W eq 2.45
            Cinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Cinv_H = np.linalg.solve(L.T, np.linalg.solve(L, H))
            A = H.T @ Cinv_H
            try:
                beta_hat = np.linalg.solve(A, H.T @ Cinv_y)
            except np.linalg.LinAlgError:
                return 1e10
            r = y - H @ beta_hat
            Cinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
            data_fit = -0.5 * r @ Cinv_r
            complexity = -np.sum(np.log(np.diag(L)))
            return -(data_fit + complexity)

        # Grid search over (sf, ℓ_s, ℓ_v)
        ell_s_min = max(s_range / 3, 0.3)
        ell_s_candidates = [max(s_range / d, ell_s_min)
                            for d in [3, 2, 1, 0.5]]
        # ℓ_v: include ∞ (V dimension off) and finite values
        ell_v_candidates = [1e6]  # always include "off"
        if v_range > 0.01:
            ell_v_candidates += [max(v_range, 0.3),
                                 max(v_range / 2, 0.2)]
        # Also include σ_f = 0 (degenerate: pure WLS, no GP residual)
        log_sf_grid = np.concatenate([[-20], np.linspace(-2, 2, 6)])

        best_nlml = 1e10
        best_params = {}
        for ell_s in ell_s_candidates:
            for ell_v in ell_v_candidates:
                for lsf in log_sf_grid:
                    nlml = _neg_log_ml(lsf, ell_s, ell_v)
                    if nlml < best_nlml:
                        best_nlml = nlml
                        best_params = {'ell_s': ell_s, 'ell_v': ell_v,
                                       'log_sf': lsf}

        ell_s_opt = best_params.get('ell_s', ell_s_candidates[1])
        ell_v_opt = best_params.get('ell_v', 1e6)
        log_sf_best = best_params.get('log_sf', -20)
        sf2_opt = np.exp(2 * log_sf_best) if log_sf_best > -19 else 0.0

        K_opt = sf2_opt * np.exp(-sq_dists_s / (2 * ell_s_opt ** 2)
                                 - sq_dists_v / (2 * ell_v_opt ** 2))

        # --- Stage 5: GP posterior on β = (α, c) ---
        C = K_opt + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        A = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        beta_hat = A_inv @ H.T @ C_inv @ y
        gp_alpha_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        # Block bootstrap α for robustness
        block_alphas = []
        for b in range(n_blocks):
            sl = slice(b * block_len, min((b + 1) * block_len, n))
            K_b = K_nw[:, sl]
            K_b_sum = K_b.sum(axis=1)
            valid_b = valid & (K_b_sum > 1e-10)
            if np.sum(valid_b) < 5:
                continue
            s2_b = np.zeros(m)
            mask_pos = K_b_sum > 1e-10
            s2_b[mask_pos] = (K_b[mask_pos] @ sq_inc[sl]) / K_b_sum[mask_pos]
            mask_b = valid_b & (s2_b > 1e-10)
            if np.sum(mask_b) < 5:
                continue
            x_b = landmarks_z[mask_b]
            y_b = np.log(s2_b[mask_b])
            n_eff_b = np.zeros(m)
            K_b_sq = (K_b ** 2).sum(axis=1)
            n_eff_b[mask_pos] = K_b_sum[mask_pos] ** 2 / K_b_sq[mask_pos]
            w_b = n_eff_b[mask_b]
            H_b = np.column_stack([x_b, np.ones(len(x_b))])
            WH = H_b * w_b[:, None]
            try:
                beta_b = np.linalg.solve(WH.T @ H_b, WH.T @ y_b)
                block_alphas.append(beta_b[0])
            except np.linalg.LinAlgError:
                continue

        block_alpha_sd = None
        if len(block_alphas) >= 3:
            block_alpha_sd = (np.std(block_alphas, ddof=1)
                              / np.sqrt(len(block_alphas)))

        # Combined α estimate: parametric mean + GP gradient correction
        alpha_param = float(beta_hat[0])

        # If GP residual is active, add the GP gradient correction
        # at the upper boundary (the GP can shift the local elasticity)
        eps_gp_correction = 0.0
        eps_gp_var = 0.0
        if sf2_opt > 1e-8:
            r = y - H @ beta_hat
            C_inv_r = C_inv @ r
            v_ref = np.median(x_v)
            s_upper = np.percentile(x_s, [70, 80, 90, 95])

            gradient_means = []
            gradient_vars = []
            for s0 in s_upper:
                diff_s_0 = s0 - x_s
                diff_v_0 = v_ref - x_v
                k_x0 = sf2_opt * np.exp(
                    -diff_s_0 ** 2 / (2 * ell_s_opt ** 2)
                    - diff_v_0 ** 2 / (2 * ell_v_opt ** 2))
                dk_ds0 = -diff_s_0 / (ell_s_opt ** 2) * k_x0

                grad_mean = dk_ds0 @ C_inv_r
                prior_grad_var = sf2_opt / (ell_s_opt ** 2)
                try:
                    L_C = np.linalg.cholesky(C)
                    v_solve = np.linalg.solve(L_C, dk_ds0)
                    reduction = v_solve @ v_solve
                except np.linalg.LinAlgError:
                    reduction = 0.0
                grad_var = max(prior_grad_var - reduction, 1e-10)

                gradient_means.append(float(grad_mean))
                gradient_vars.append(float(grad_var))

            weights = 1.0 / np.array(gradient_vars)
            weights /= weights.sum()
            eps_gp_correction = float(
                np.sum(weights * np.array(gradient_means)))
            eps_gp_var = float(
                1.0 / np.sum(1.0 / np.array(gradient_vars)))
            self._gradient_means = gradient_means
            self._gradient_vars = gradient_vars
        else:
            self._gradient_means = []
            self._gradient_vars = []

        # Total elasticity: parametric α + GP correction
        eps_mean = alpha_param + eps_gp_correction
        eps_sd_gp = np.sqrt(gp_alpha_sd ** 2 + eps_gp_var)
        if block_alpha_sd is not None:
            eps_sd = max(eps_sd_gp, block_alpha_sd)
        else:
            eps_sd = eps_sd_gp

        # --- Stage 7: P(bubble) = P(ε > 2) ---
        z_score = (eps_mean - 2.0) / max(eps_sd, 1e-8)
        self.p_bubble = float(stats.norm.cdf(z_score))

        self.alpha_mean = eps_mean
        self.alpha_sd = eps_sd
        self.vol_p_bubble = float(stats.norm.pdf(z_score))
        self.p_bubble_feller = self.p_bubble

        # Diagnostics
        self._x_s = x_s
        self._x_v = x_v
        self._y = y
        self._sigma2_nw = sigma2_nw[valid]
        self._noise_var = noise_var
        self._ell_s_opt = ell_s_opt
        self._ell_v_opt = ell_v_opt
        self._sf2_opt = sf2_opt
        self._v_active = ell_v_opt < 1e5
        self._alpha_param = alpha_param
        self._eps_gp_correction = eps_gp_correction
        self._block_alphas = block_alphas

        return self

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0


class ResolventFellerGP:
    """Resolvent-based structural σ² estimation for bubble detection.

    For non-separable stochastic vol (SABR, 3/2 model), the marginal
    E[σ²|S] has inflated elasticity due to price-vol correlation.
    This class estimates σ² via the resolvent R_λ = (λI - L)⁻¹,
    conditioning on the 2D state (S, V̂) to break the selection bias.

    Pipeline:
      1. Estimate V̂ from rolling QV, orthogonalize against log|S|
      2. Precompute discounted forward sums (resolvent targets) in O(N)
      3. 2D product-kernel NW at landmarks → R̂_λ(S²), R̂_λ(S)
      4. CdC: σ̂² = λ R_λ(S²) - 2S λ R_λ(S) + S²
         (also direct NW on (ΔS)²/Δt for comparison)
      5. Block noise → GP with parametric mean → gradient → P(bubble)

    The resolvent advantage: each R_λf(x) averages over exponentially-
    weighted future observations, giving much lower variance per landmark
    than single-step squared increments. The CdC cancellation (both terms
    ~S² but difference ~σ²) is tolerable because the resolvent terms
    have low variance.

    Choice of λ: controls the memory horizon 1/λ. Too small → resolvent
    diverges for explosive processes. Too large → R_λf ≈ f/λ, loses
    dynamics. Default: λ = 5/(T_char·Δt) with T_char ~ 200 steps.
    """

    def __init__(self, n_landmarks=80, n_blocks=None, qv_window=200,
                 lam='auto', method='resolvent'):
        """
        Args:
            n_landmarks: Number of NW estimation points
            n_blocks: Block count for noise estimation (auto if None)
            qv_window: Rolling QV window for vol proxy
            lam: Resolvent parameter λ ('auto' or float)
            method: 'resolvent' (CdC via resolvent) or 'direct' (NW on Δz²)
        """
        self.n_landmarks = n_landmarks
        self.n_blocks = n_blocks
        self.qv_window = qv_window
        self.lam = lam
        self.method = method

    def _rolling_qv(self, z, dz, dt):
        """Estimate instantaneous vol from rolling quadratic variation."""
        n = len(z)
        w = min(self.qv_window, n // 5)
        qv = np.zeros(n)
        cs = np.cumsum(dz ** 2)
        cs = np.insert(cs, 0, 0.0)
        for i in range(n):
            lo = max(0, i - w + 1)
            hi = min(i + 1, len(cs) - 1)
            qv[i] = (cs[hi] - cs[lo]) / ((hi - lo) * dt + 1e-30)
        return np.maximum(qv, 1e-10)

    def fit(self, z, dz, dt, vol_proxy=None):
        """Estimate structural elasticity via resolvent + 2D GP.

        Args:
            z: (n,) price levels S_t
            dz: (n,) increments ΔS_t
            dt: time step
            vol_proxy: (n,) optional vol proxy. If None, uses rolling QV.

        Returns:
            self
        """
        n = len(z)
        S = z.copy()
        sq_inc = dz ** 2 / dt

        # --- Stage 1: Vol proxy + orthogonalize ---
        if vol_proxy is not None:
            V = vol_proxy.copy()
        else:
            V = self._rolling_qv(z, dz, dt)

        log_z = np.log(np.maximum(np.abs(S), 1e-10))
        log_v = np.log(np.maximum(V, 1e-10))

        # Orthogonalize V̂ against S to break collinearity
        H_orth = np.column_stack([log_z, np.ones(n)])
        beta_orth = np.linalg.lstsq(H_orth, log_v, rcond=None)[0]
        log_v_resid = log_v - H_orth @ beta_orth
        r2_sv = 1.0 - np.var(log_v_resid) / max(np.var(log_v), 1e-30)
        self._r2_sv = float(r2_sv)

        # --- Stage 2: Resolvent — precompute discounted forward sums ---
        if self.lam == 'auto':
            T_char = min(200, n // 10)
            lam = 5.0 / max(T_char * dt, 1e-6)
        else:
            lam = float(self.lam)
        gamma_disc = np.exp(-lam * dt)
        self._lam = lam

        # disc_S2[i] = Σ_{k≥0} γ^k · S_{i+k}² · Δt
        # disc_S1[i] = Σ_{k≥0} γ^k · S_{i+k} · Δt
        # Computed backward in O(N)
        disc_S2 = np.zeros(n)
        disc_S1 = np.zeros(n)
        disc_S2[n - 1] = S[n - 1] ** 2 * dt
        disc_S1[n - 1] = S[n - 1] * dt
        for i in range(n - 2, -1, -1):
            disc_S2[i] = S[i] ** 2 * dt + gamma_disc * disc_S2[i + 1]
            disc_S1[i] = S[i] * dt + gamma_disc * disc_S1[i + 1]

        # Truncate: only use anchor points with enough future data
        # (need ≥ 5 e-folding times for resolvent to converge)
        min_future = int(5.0 / (lam * dt + 1e-10))
        usable = n - min_future
        if usable < 100:
            # Not enough data for resolvent; fall back to shorter horizon
            min_future = n // 2
            usable = n - min_future

        # --- Stage 3: 2D NW at landmarks ---
        n_blocks = self.n_blocks or min(10, max(5, n // 500))
        block_len = n // n_blocks
        m = min(self.n_landmarks, usable // 5)

        # Landmarks in log|S| space (quantile-based, from usable range)
        quantiles = np.linspace(0.01, 0.99, m)
        lm_log_z = np.quantile(log_z[:usable], quantiles)

        # Bandwidths
        ldists = np.abs(np.diff(lm_log_z))
        bw_s = np.median(ldists) if len(ldists) > 0 else np.std(log_z)
        bw_s = max(bw_s, 1e-8)

        # V̂_⊥ bandwidth: wide (mild conditioning)
        std_v_resid = np.std(log_v_resid[:usable])
        has_sv = std_v_resid > 0.01
        bw_v = max(std_v_resid * 0.5, 0.1) if has_sv else 1e6

        # S-kernel (all landmarks × usable data points)
        diff_s = ((lm_log_z[:, None] - log_z[None, :usable]) / bw_s)
        K_s = np.exp(-0.5 * diff_s ** 2)

        # V̂_⊥ at landmarks (weighted average)
        lm_v_resid = np.zeros(m)
        for j in range(m):
            ws = K_s[j].sum()
            if ws > 1e-10:
                lm_v_resid[j] = (K_s[j] @ log_v_resid[:usable]) / ws

        # Product kernel
        diff_v = ((lm_v_resid[:, None] - log_v_resid[None, :usable])
                  / bw_v)
        K_nw = K_s * np.exp(-0.5 * diff_v ** 2)
        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm = K_sum > 1e-10

        n_eff = np.zeros(m)
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        # --- Stage 4: σ̂² from resolvent CdC AND direct NW ---
        S_lm = np.exp(lm_log_z)

        # Resolvent estimates
        R_S2 = np.zeros(m)
        R_S1 = np.zeros(m)
        R_S2[valid_lm] = ((K_nw[valid_lm] @ disc_S2[:usable])
                          / K_sum[valid_lm])
        R_S1[valid_lm] = ((K_nw[valid_lm] @ disc_S1[:usable])
                          / K_sum[valid_lm])

        # CdC: σ²(x) = λR_λ(S²) - 2S·λR_λ(S) + S²
        sigma2_cdc = (lam * R_S2 - 2 * S_lm * lam * R_S1
                      + S_lm ** 2)

        # Direct NW on (ΔS)²/Δt
        sigma2_direct = np.zeros(m)
        sigma2_direct[valid_lm] = ((K_nw[valid_lm] @ sq_inc[:usable])
                                   / K_sum[valid_lm])

        # Choose method
        if self.method == 'resolvent':
            sigma2_est = sigma2_cdc
        else:
            sigma2_est = sigma2_direct

        # Filter valid landmarks
        valid = (valid_lm & (sigma2_est > 1e-8) & (n_eff > 2)
                 & (S_lm > 1e-4))

        if np.sum(valid) < 10:
            self._set_nan()
            return self

        x_s = lm_log_z[valid]
        x_v = lm_v_resid[valid]
        y = np.log(sigma2_est[valid])
        nv = len(x_s)
        valid_idx = np.where(valid)[0]

        # --- Stage 5: Block noise estimation ---
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                sl = slice(b * block_len,
                           min((b + 1) * block_len, usable))
                K_b = K_nw[j, sl]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    if self.method == 'resolvent':
                        r_s2_b = (K_b @ disc_S2[sl]) / K_b_sum
                        r_s1_b = (K_b @ disc_S1[sl]) / K_b_sum
                        s_j = S_lm[j]
                        est_b = (lam * r_s2_b - 2 * s_j * lam * r_s1_b
                                 + s_j ** 2)
                    else:
                        est_b = (K_b @ sq_inc[sl]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = (np.var(block_ests, ddof=1)
                                 / len(block_ests))
            else:
                noise_var[jj] = 2.0 / n_eff[valid_idx[jj]]
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # --- Stage 6: GP with parametric mean + 2D product kernel ---
        H = np.column_stack([x_s, np.ones(nv)])
        Sigma_n = np.diag(noise_var)
        s_range = x_s.max() - x_s.min()
        v_range = x_v.max() - x_v.min()

        sq_dists_s = (x_s[:, None] - x_s[None, :]) ** 2
        sq_dists_v = (x_v[:, None] - x_v[None, :]) ** 2

        def _neg_log_ml(log_sf, ell_s, ell_v):
            sf2 = np.exp(2 * log_sf) if log_sf > -19 else 0.0
            if sf2 < 1e-15:
                # Degenerate GP: pure WLS
                C = Sigma_n.copy()
            else:
                K = sf2 * np.exp(-sq_dists_s / (2 * ell_s ** 2)
                                 - sq_dists_v / (2 * ell_v ** 2))
                C = K + Sigma_n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e10
            Cinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Cinv_H = np.linalg.solve(L.T, np.linalg.solve(L, H))
            A = H.T @ Cinv_H
            try:
                beta_hat = np.linalg.solve(A, H.T @ Cinv_y)
            except np.linalg.LinAlgError:
                return 1e10
            r = y - H @ beta_hat
            Cinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
            return 0.5 * r @ Cinv_r + np.sum(np.log(np.diag(L)))

        ell_s_min = max(s_range / 3, 0.3)
        ell_s_cands = list(set([max(s_range / d, ell_s_min)
                                for d in [3, 2, 1, 0.5]]))
        ell_v_cands = [1e6]
        if has_sv and v_range > 0.01:
            ell_v_cands += [max(v_range, 0.3), max(v_range / 2, 0.2)]
        log_sf_grid = np.concatenate([[-20], np.linspace(-2, 2, 6)])

        best_nlml = 1e10
        best_params = {}
        for ell_s in ell_s_cands:
            for ell_v in ell_v_cands:
                for lsf in log_sf_grid:
                    nlml = _neg_log_ml(lsf, ell_s, ell_v)
                    if nlml < best_nlml:
                        best_nlml = nlml
                        best_params = {'ell_s': ell_s, 'ell_v': ell_v,
                                       'log_sf': lsf}

        ell_s_opt = best_params.get('ell_s', ell_s_cands[0])
        ell_v_opt = best_params.get('ell_v', 1e6)
        log_sf_best = best_params.get('log_sf', -20)
        sf2_opt = np.exp(2 * log_sf_best) if log_sf_best > -19 else 0.0

        if sf2_opt < 1e-15:
            K_opt = np.zeros((nv, nv))
        else:
            K_opt = sf2_opt * np.exp(
                -sq_dists_s / (2 * ell_s_opt ** 2)
                - sq_dists_v / (2 * ell_v_opt ** 2))

        C = K_opt + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        A = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        beta_hat = A_inv @ H.T @ C_inv @ y
        gp_alpha_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        # Block bootstrap α
        block_alphas = []
        for b in range(n_blocks):
            sl = slice(b * block_len,
                       min((b + 1) * block_len, usable))
            K_b = K_nw[:, sl]
            K_b_sum = K_b.sum(axis=1)
            valid_b = valid & (K_b_sum > 1e-10)
            if np.sum(valid_b) < 5:
                continue
            s2_b = np.zeros(m)
            mask_pos = K_b_sum > 1e-10
            if self.method == 'resolvent':
                r_s2_b = np.zeros(m)
                r_s1_b = np.zeros(m)
                r_s2_b[mask_pos] = ((K_b[mask_pos] @ disc_S2[sl])
                                    / K_b_sum[mask_pos])
                r_s1_b[mask_pos] = ((K_b[mask_pos] @ disc_S1[sl])
                                    / K_b_sum[mask_pos])
                s2_b = (lam * r_s2_b - 2 * S_lm * lam * r_s1_b
                        + S_lm ** 2)
            else:
                s2_b[mask_pos] = ((K_b[mask_pos] @ sq_inc[sl])
                                  / K_b_sum[mask_pos])
            mask_b = valid_b & (s2_b > 1e-10)
            if np.sum(mask_b) < 5:
                continue
            x_b = lm_log_z[mask_b]
            y_b = np.log(s2_b[mask_b])
            n_eff_b = np.zeros(m)
            K_b_sq = (K_b ** 2).sum(axis=1)
            n_eff_b[mask_pos] = (K_b_sum[mask_pos] ** 2
                                 / K_b_sq[mask_pos])
            w_b = n_eff_b[mask_b]
            H_b = np.column_stack([x_b, np.ones(len(x_b))])
            WH = H_b * w_b[:, None]
            try:
                b_hat = np.linalg.solve(WH.T @ H_b, WH.T @ y_b)
                block_alphas.append(b_hat[0])
            except np.linalg.LinAlgError:
                continue

        block_alpha_sd = None
        if len(block_alphas) >= 3:
            block_alpha_sd = (np.std(block_alphas, ddof=1)
                              / np.sqrt(len(block_alphas)))

        # --- Stage 7: Combined α + GP gradient → P(bubble) ---
        alpha_param = float(beta_hat[0])
        eps_gp_correction = 0.0
        if sf2_opt > 1e-8:
            r = y - H @ beta_hat
            C_inv_r = C_inv @ r
            v_ref = np.median(x_v)
            s_upper = np.percentile(x_s, [70, 80, 90, 95])

            grad_means = []
            grad_vars = []
            try:
                L_C = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L_C = None

            for s0 in s_upper:
                d_s = s0 - x_s
                d_v = v_ref - x_v
                k_x0 = sf2_opt * np.exp(
                    -d_s ** 2 / (2 * ell_s_opt ** 2)
                    - d_v ** 2 / (2 * ell_v_opt ** 2))
                dk = -d_s / (ell_s_opt ** 2) * k_x0
                gm = dk @ C_inv_r
                pgv = sf2_opt / (ell_s_opt ** 2)
                if L_C is not None:
                    v_sol = np.linalg.solve(L_C, dk)
                    red = v_sol @ v_sol
                else:
                    red = 0.0
                gv = max(pgv - red, 1e-10)
                grad_means.append(float(gm))
                grad_vars.append(float(gv))

            w_inv = 1.0 / np.array(grad_vars)
            w_inv /= w_inv.sum()
            eps_gp_correction = float(
                np.sum(w_inv * np.array(grad_means)))

        eps_mean = alpha_param + eps_gp_correction
        eps_sd = gp_alpha_sd
        if block_alpha_sd is not None:
            eps_sd = max(eps_sd, block_alpha_sd)

        z_score = (eps_mean - 2.0) / max(eps_sd, 1e-8)
        self.p_bubble = float(stats.norm.cdf(z_score))
        self.alpha_mean = eps_mean
        self.alpha_sd = eps_sd
        self.vol_p_bubble = float(stats.norm.pdf(z_score))
        self.p_bubble_feller = self.p_bubble

        # Diagnostics
        self._x_s = x_s
        self._x_v = x_v
        self._y = y
        self._sigma2_cdc = sigma2_cdc[valid]
        self._sigma2_direct = sigma2_direct[valid]
        self._ell_s_opt = ell_s_opt
        self._ell_v_opt = ell_v_opt
        self._sf2_opt = sf2_opt
        self._v_active = ell_v_opt < 1e5
        self._alpha_param = alpha_param
        self._eps_gp_correction = eps_gp_correction
        self._block_alphas = block_alphas
        self._n_eff = n_eff[valid]
        self._has_sv = has_sv

        return self

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0


class SigBLRFellerGP:
    """Signature BLR-based Feller test — path-conditioned σ² estimation.

    Bypasses CdC entirely. Instead of extracting σ² from the generator,
    directly observes y_t = (ΔS_t)²/Δt (noisy measurement of σ²) and
    filters it via Bayesian Linear Regression on signature features:

        y_t = w^T Φ(sig_t) + ε_t,    w ~ N(μ_w, Σ_w)

    The signature encodes both price level S and vol state V through
    the Lévy area (QV), so the filtered σ̂²_t is conditioned on the
    full 2D state — breaking the marginal price-vol bias WITHOUT
    explicit 2D NW or CdC subtraction.

    Pipeline:
      1. Compute cumulative lead-lag log-signatures (RecurrentLeadLagLogSig)
      2. Online BLR: Kalman filter on regression weights w
      3. σ̂²_t = μ_w^T Φ_t (filtered vol, path-conditioned)
      4. GP on log σ̂² vs log|S| with parametric mean → gradient → P(bubble)

    The BLR loss function is the Kalman filter marginal likelihood
    (prediction error decomposition), which naturally selects the
    regularization strength and handles the chi-squared noise of (ΔS)².

    Connection to Level 4/5 of graduated_sanity_checks.py: same BLR
    machinery, but applied to σ² estimation instead of V̂ filtering.
    """

    def __init__(self, n_landmarks=80, n_blocks=None,
                 sig_gamma=0.99, sig_level=2,
                 blr_obs_var='auto', blr_process_var=1e-4):
        """
        Args:
            n_landmarks: Number of price-level bins for elasticity test
            n_blocks: Blocks for noise estimation (auto if None)
            sig_gamma: Signature forgetting factor (0.99 = 100-step window)
            sig_level: Signature truncation level
            blr_obs_var: Observation noise variance ('auto' or float)
            blr_process_var: Process noise on BLR weights (random walk)
        """
        self.n_landmarks = n_landmarks
        self.n_blocks = n_blocks
        self.sig_gamma = sig_gamma
        self.sig_level = sig_level
        self.blr_obs_var = blr_obs_var
        self.blr_process_var = blr_process_var

    def fit(self, z, dz, dt):
        """Estimate structural elasticity via signature BLR.

        Args:
            z: (n,) price levels S_t
            dz: (n,) increments ΔS_t
            dt: time step

        Returns:
            self
        """
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', '..', 'examples', 'proof_of_concept'))
        from signature_features import RecurrentLeadLagLogSigMap

        n = len(z)
        sq_inc = dz ** 2 / dt  # noisy observations of σ²

        # --- Stage 1: Compute signature features ---
        log_prices = np.log(np.maximum(np.abs(z), 1e-10))
        log_returns = np.diff(log_prices)

        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=self.sig_level,
            forgetting_factor=self.sig_gamma)
        d_sig = sig_map.feature_dim
        all_sigs = np.zeros((n, d_sig))
        for i in range(len(log_returns)):
            all_sigs[i + 1] = sig_map.update(np.array([log_returns[i]]))

        # Augment with log|S| and 1 (bias) for interpretability
        # Features: [sig_features, log|S|, 1]
        d_feat = d_sig + 2
        Phi = np.column_stack([all_sigs, log_prices, np.ones(n)])

        # --- Stage 2: Online BLR (Kalman filter on weights) ---
        # State: w ∈ R^d_feat
        # Dynamics: w_{t+1} = w_t + η_t, η ~ N(0, q·I)
        # Observation: y_t = Φ_t^T w + ε_t, ε ~ N(0, r)
        #
        # The BLR gives σ̂²_t = Φ_t^T μ_w with posterior uncertainty.

        if self.blr_obs_var == 'auto':
            # Initial R from empirical variance of log(sq_inc)
            # (ΔS)²/Δt has Var = 2σ⁴ for Gaussian increments
            median_sq = np.median(sq_inc[sq_inc > 0])
            r = max(2.0 * median_sq ** 2, 1e-4)
        else:
            r = float(self.blr_obs_var)

        q = float(self.blr_process_var)

        # Initialize BLR
        mu_w = np.zeros(d_feat)
        Sigma_w = np.eye(d_feat) * 1.0  # diffuse prior

        sigma2_filtered = np.zeros(n)
        sigma2_var = np.zeros(n)  # posterior variance of σ̂²

        # Burn-in: skip first few steps where signatures are zero
        burn_in = max(10, int(1.0 / (1.0 - self.sig_gamma + 1e-10)))

        for t in range(n):
            phi_t = Phi[t]

            # Predict
            mu_pred = mu_w  # random walk: no change
            Sigma_pred = Sigma_w + q * np.eye(d_feat)

            # Innovation
            y_pred = phi_t @ mu_pred
            sigma2_filtered[t] = max(y_pred, 1e-10)

            # Innovation variance: Φ^T Σ_pred Φ + R
            S_t = phi_t @ Sigma_pred @ phi_t + r

            # Posterior variance of prediction
            sigma2_var[t] = phi_t @ Sigma_pred @ phi_t

            if t < burn_in:
                # Don't update during burn-in (signatures not ready)
                mu_w = mu_pred
                Sigma_w = Sigma_pred
                continue

            # Kalman gain
            K_gain = Sigma_pred @ phi_t / S_t

            # Update
            innovation = sq_inc[t] - y_pred
            mu_w = mu_pred + K_gain * innovation
            Sigma_w = Sigma_pred - np.outer(K_gain, K_gain) * S_t

            # Ensure symmetry
            Sigma_w = 0.5 * (Sigma_w + Sigma_w.T)

        # --- Stage 3: Bin filtered σ̂² by price level ---
        # Use data after burn-in
        valid_range = slice(burn_in, n)
        S_valid = z[valid_range]
        s2_valid = sigma2_filtered[valid_range]
        s2_var_valid = sigma2_var[valid_range]
        log_z_valid = log_prices[valid_range]

        # Filter out negative/tiny σ̂² predictions
        pos_mask = s2_valid > 1e-8
        if np.sum(pos_mask) < 50:
            self._set_nan()
            return self

        S_pos = S_valid[pos_mask]
        s2_pos = s2_valid[pos_mask]
        log_z_pos = log_z_valid[pos_mask]

        # Bin by price level (NW-style smoothing)
        m = min(self.n_landmarks, len(S_pos) // 10)
        n_blocks = self.n_blocks or min(10, max(5, n // 500))
        block_len = n // n_blocks

        quantiles = np.linspace(0.02, 0.98, m)
        lm_log_z = np.quantile(log_z_pos, quantiles)
        ldists = np.abs(np.diff(lm_log_z))
        bw = np.median(ldists) if len(ldists) > 0 else np.std(log_z_pos)
        bw = max(bw, 1e-8)

        diff = (lm_log_z[:, None] - log_z_pos[None, :]) / bw
        K_nw = np.exp(-0.5 * diff ** 2)
        K_sum = K_nw.sum(axis=1)
        valid_lm = K_sum > 1e-10

        # Weighted average of filtered σ̂² at each price bin
        sigma2_binned = np.zeros(m)
        n_eff = np.zeros(m)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        sigma2_binned[valid_lm] = ((K_nw[valid_lm] @ s2_pos)
                                   / K_sum[valid_lm])
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = (valid_lm & (sigma2_binned > 1e-8) & (n_eff > 2)
                 & (np.exp(lm_log_z) > 1e-4))

        if np.sum(valid) < 10:
            self._set_nan()
            return self

        x = lm_log_z[valid]
        y = np.log(sigma2_binned[valid])
        nv = len(x)
        valid_idx = np.where(valid)[0]

        # --- Stage 4: Block noise estimation ---
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                b_start = b * block_len - burn_in
                b_end = min((b + 1) * block_len - burn_in,
                            len(s2_pos))
                if b_start < 0:
                    b_start = 0
                if b_end <= b_start:
                    continue
                # Which pos_mask indices fall in this block?
                # Approximate: use the NW kernel on the block
                K_b = K_nw[j, b_start:b_end]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ s2_pos[b_start:b_end]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = (np.var(block_ests, ddof=1)
                                 / len(block_ests))
            else:
                noise_var[jj] = 2.0 / max(n_eff[valid_idx[jj]], 1.0)
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # --- Stage 5: GP with parametric mean on log σ̂² ---
        H = np.column_stack([x, np.ones(nv)])
        Sigma_n = np.diag(noise_var)
        x_range = x.max() - x.min()
        sq_dists = (x[:, None] - x[None, :]) ** 2

        def _neg_log_ml(log_sf, ell):
            sf2 = np.exp(2 * log_sf) if log_sf > -19 else 0.0
            if sf2 < 1e-15:
                C = Sigma_n.copy()
            else:
                K = sf2 * np.exp(-sq_dists / (2 * ell ** 2))
                C = K + Sigma_n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e10
            Cinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Cinv_H = np.linalg.solve(L.T, np.linalg.solve(L, H))
            A = H.T @ Cinv_H
            try:
                beta_hat = np.linalg.solve(A, H.T @ Cinv_y)
            except np.linalg.LinAlgError:
                return 1e10
            r = y - H @ beta_hat
            Cinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
            return 0.5 * r @ Cinv_r + np.sum(np.log(np.diag(L)))

        ell_min = max(x_range / 3, 0.3)
        ell_cands = list(set([max(x_range / d, ell_min)
                              for d in [3, 2, 1, 0.5]]))
        log_sf_grid = np.concatenate([[-20], np.linspace(-2, 2, 6)])

        best_nlml = 1e10
        best_ell = ell_cands[0]
        best_lsf = -20
        for ell in ell_cands:
            for lsf in log_sf_grid:
                nlml = _neg_log_ml(lsf, ell)
                if nlml < best_nlml:
                    best_nlml = nlml
                    best_ell = ell
                    best_lsf = lsf

        sf2_opt = np.exp(2 * best_lsf) if best_lsf > -19 else 0.0
        if sf2_opt < 1e-15:
            K_opt = np.zeros((nv, nv))
        else:
            K_opt = sf2_opt * np.exp(-sq_dists / (2 * best_ell ** 2))

        C = K_opt + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        A = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._set_nan()
            return self

        beta_hat = A_inv @ H.T @ C_inv @ y
        gp_alpha_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        # Block bootstrap α
        block_alphas = []
        for b in range(n_blocks):
            b_start = b * block_len - burn_in
            b_end = min((b + 1) * block_len - burn_in, len(s2_pos))
            if b_start < 0:
                b_start = 0
            if b_end - b_start < 10:
                continue
            K_b = K_nw[:, b_start:b_end]
            K_b_sum = K_b.sum(axis=1)
            valid_b = valid & (K_b_sum > 1e-10)
            if np.sum(valid_b) < 5:
                continue
            s2_b = np.zeros(m)
            mask_pos = K_b_sum > 1e-10
            s2_b[mask_pos] = ((K_b[mask_pos] @ s2_pos[b_start:b_end])
                              / K_b_sum[mask_pos])
            mask_b = valid_b & (s2_b > 1e-10)
            if np.sum(mask_b) < 5:
                continue
            x_b = lm_log_z[mask_b]
            y_b = np.log(s2_b[mask_b])
            n_eff_b = np.zeros(m)
            K_b_sq = (K_b ** 2).sum(axis=1)
            n_eff_b[mask_pos] = (K_b_sum[mask_pos] ** 2
                                 / K_b_sq[mask_pos])
            w_b = n_eff_b[mask_b]
            H_b = np.column_stack([x_b, np.ones(len(x_b))])
            WH = H_b * w_b[:, None]
            try:
                b_hat = np.linalg.solve(WH.T @ H_b, WH.T @ y_b)
                block_alphas.append(b_hat[0])
            except np.linalg.LinAlgError:
                continue

        block_alpha_sd = None
        if len(block_alphas) >= 3:
            block_alpha_sd = (np.std(block_alphas, ddof=1)
                              / np.sqrt(len(block_alphas)))

        alpha_param = float(beta_hat[0])
        alpha_sd = gp_alpha_sd
        if block_alpha_sd is not None:
            alpha_sd = max(alpha_sd, block_alpha_sd)

        z_score = (alpha_param - 2.0) / max(alpha_sd, 1e-8)
        self.p_bubble = float(stats.norm.cdf(z_score))
        self.alpha_mean = alpha_param
        self.alpha_sd = alpha_sd
        self.vol_p_bubble = float(stats.norm.pdf(z_score))
        self.p_bubble_feller = self.p_bubble

        # Diagnostics
        self._x = x
        self._y = y
        self._sigma2_binned = sigma2_binned[valid]
        self._sigma2_filtered = sigma2_filtered
        self._sf2_opt = sf2_opt
        self._ell_opt = best_ell
        self._block_alphas = block_alphas
        self._mu_w = mu_w
        self._burn_in = burn_in
        self._d_sig = d_sig

        return self

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0


class SigKKFFellerGP:
    """Signature KKF Feller test — online BLR in Nyström RKHS.

    Estimates σ̂²(S) via online Bayesian linear regression (BLR) on
    Nyström features of lead-lag log-signatures. The BLR Kalman filter
    tracks the regression coefficients C_t, making them time-varying
    for non-stationary processes (regime switching, structural breaks).

    The observation model is:
        y_t = log((Δlog S_t)²) = C_t^T φ(sig_t) + ε_t
    where ε_t has known variance ≈ 4.93 (from log(χ²_1) distribution)
    and φ(sig_t) are Nyström features of the signature path.

    Pipeline:
      1. 1D lead-lag log-signatures (γ≥0.999) → BCH level-2 features
      2. RBF kernel → Nyström features φ_t (m-dimensional)
      3. Initialize C from EDMD regression (global least-squares)
      4. BLR-KKF: filter C_t via Kalman on regression coefficients
         - State: C_t (m-dim), dynamics C_{t+1} = C_t + w
         - Observation: y_t = log((Δlog S)²), H_t = φ(sig_t)
         - Known noise: R = Var[log χ²_1] ≈ 4.93
      5. σ̂²_t = exp(C_t^T φ_t) · S²_t / dt (adaptive, always > 0)
      6. V̂ from signature QV area (Lévy area of lead-lag)
      7. GP: log σ̂² ~ α·log S + β_v·log V̂ + c + f(log S) → P(α > 2)

    For stationary processes (q_scale=0), reduces to EDMD regression.
    For non-stationary processes (q_scale>0), C_t adapts over time.

    Note: the key insight is that RKHS features φ_t are OBSERVED
    (computed from signatures), so we filter the regression
    coefficients C, not the features themselves.
    """

    def __init__(self, n_nystrom=60, n_gp_landmarks=80, n_blocks=None,
                 sig_gamma=0.99, sig_level=2, kernel_sigma='auto',
                 edmd_reg=1e-4, q_scale=0.0):
        self.n_nystrom = n_nystrom
        self.n_gp_landmarks = n_gp_landmarks
        self.n_blocks = n_blocks
        self.sig_gamma = sig_gamma
        self.sig_level = sig_level
        self.kernel_sigma = kernel_sigma
        self.edmd_reg = edmd_reg
        self.q_scale = q_scale

    def fit(self, z, dz, dt):
        """Full pipeline: sigs → Nyström → BLR-KKF → GP Feller test.

        Args:
            z: (n,) price levels S_t
            dz: (n,) increments ΔS_t
            dt: time step

        Returns:
            self
        """
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', '..', 'examples', 'proof_of_concept'))
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', 'utils'))
        from signature_features import RecurrentLeadLagLogSigMap
        from nystrom_utils import (farthest_point_sampling_euclidean,
                                    nystrom_approximation, rbf_kernel)

        n = len(z)
        log_z = np.log(np.maximum(np.abs(z), 1e-10))
        log_returns = np.diff(log_z)

        # --- Stage 0: Compute 1D lead-lag log-signatures ---
        # Use γ≥0.999 (1000-step effective window) so that the
        # level-1 components l1 ≈ log(S_t/S_{t-1000}) carry price
        # level information.
        #
        # BCH at level 2 (exact, no truncation error):
        #   l1 ← γ·l1 + dx
        #   l2 ← γ²·l2 + 0.5·[γ·l1, dx]
        # Features: l1 = (lead, lag), l2 = (Lévy area ≈ QV)
        sig_gamma = max(self.sig_gamma, 0.999)
        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=self.sig_level,
            forgetting_factor=sig_gamma)
        d_sig = sig_map.feature_dim
        all_sigs = np.zeros((n, d_sig))
        for i in range(len(log_returns)):
            all_sigs[i + 1] = sig_map.update(
                np.array([log_returns[i]]))

        burn_in = max(20, int(2.0 / (1.0 - sig_gamma + 1e-10)))
        idx_start = burn_in
        sigs = all_sigs[idx_start:]
        n_valid = len(sigs)

        if n_valid < 200:
            self._set_nan()
            return self

        log_z_valid = log_z[idx_start:idx_start + n_valid]
        z_valid = z[idx_start:idx_start + n_valid]

        # --- Stage 1: RBF kernel + Nyström ---
        if self.kernel_sigma == 'auto':
            sub = sigs[::max(1, n_valid // 300)]
            if len(sub) > 5:
                dists = pdist(sub)
                sigma_k = max(np.median(dists), 1e-6)
            else:
                sigma_k = 1.0
        else:
            sigma_k = float(self.kernel_sigma)

        m = min(self.n_nystrom, n_valid // 5)
        if m < 10:
            self._set_nan()
            return self

        def _kernel(X, Y):
            return rbf_kernel(X, Y, sigma_k)

        landmarks, _ = farthest_point_sampling_euclidean(sigs, m)
        Phi, _ = nystrom_approximation(
            sigs, landmarks, _kernel, self.edmd_reg)

        # --- Stage 2: Initialize C from EDMD regression ---
        Phi_t = Phi[:-1]
        lr = log_returns[idx_start:idx_start + n_valid - 1]
        log_sq_lr = np.log(np.maximum(lr ** 2, 1e-20))

        reg_I = self.edmd_reg * np.eye(m)
        PhiTPhi = Phi_t.T @ Phi_t + reg_I
        C_init = np.linalg.solve(PhiTPhi, Phi_t.T @ log_sq_lr)

        # --- Stage 3: BLR-KKF on regression coefficients ---
        # State: C_t (m-dim), observed: y_t = log((Δlog S)²)
        # Observation matrix: H_t = φ(sig_t) (row vector)
        # Noise: R = Var[log χ²_1] ≈ 4.93 (known)
        median_log_sq = np.median(log_sq_lr)

        if self.q_scale <= 0:
            # Stationary: C is fixed at EDMD solution (vectorized)
            C_hat = C_init
            log_pred = Phi @ C_init
            log_pred = np.clip(log_pred, median_log_sq - 5,
                               median_log_sq + 5)
            sigma2_log = np.exp(log_pred) / dt
            P_c = None
        else:
            # Non-stationary: BLR-KKF filters C_t online
            C_hat = C_init.copy()
            P_c = np.eye(m) * 0.01
            Q_c = np.eye(m) * self.q_scale
            R_obs = 4.93  # Var[log(chi^2_1)]

            sigma2_log = np.zeros(n_valid)

            for t in range(n_valid - 1):
                phi_t = Phi[t]

                # Predict (identity dynamics for C)
                P_pred = P_c + Q_c

                # Observe y_t = log((Δlog S)²)
                y_t = log_sq_lr[t] if t < len(log_sq_lr) \
                    else median_log_sq
                y_pred = float(C_hat @ phi_t)

                # Kalman update on C
                S_innov = float(phi_t @ P_pred @ phi_t) + R_obs
                if S_innov > 1e-15:
                    K_gain = P_pred @ phi_t / S_innov
                    C_hat = C_hat + K_gain * (y_t - y_pred)
                    P_c = (P_pred
                           - np.outer(K_gain, K_gain) * S_innov)
                    P_c = 0.5 * (P_c + P_c.T)
                else:
                    P_c = P_pred

                # σ̂² from filtered coefficients × observed features
                log_s2 = float(C_hat @ phi_t)
                log_s2 = np.clip(log_s2, median_log_sq - 5,
                                 median_log_sq + 5)
                sigma2_log[t] = np.exp(log_s2) / dt

            sigma2_log[-1] = sigma2_log[-2] if n_valid > 1 \
                else np.exp(median_log_sq) / dt

        # Convert to level-space: σ²(S,V) = S² · σ²_log
        sigma2_hat = sigma2_log * z_valid ** 2

        # --- Stage 4: V proxy from signature QV area ---
        qv = np.abs(all_sigs[idx_start:idx_start + n_valid, 2])
        log_v_hat = np.log(np.maximum(qv, 1e-15))

        # --- Stage 5: GP elasticity with 2D conditioning ---
        self._gp_feller_2d(
            log_z_valid, log_v_hat, sigma2_hat, n, dt)

        # Diagnostics
        self._sigma2_log = sigma2_log
        self._sigma2_hat = sigma2_hat
        self._sigma_k = sigma_k
        self._m = m
        self._C_hat = C_hat
        self._P_c = P_c

        return self

    def _gp_feller_2d(self, log_z, log_v, sigma2, n, dt):
        """2D GP Feller test with V̂ as parametric covariate.

        Uses H = [log S, log V̂, 1] when V̂ provides info beyond S
        (stochastic vol). Falls back to H = [log S, 1] when collinear.
        The α coefficient is the PARTIAL effect — structural elasticity.
        """
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', 'utils'))
        from nystrom_utils import farthest_point_sampling_euclidean

        pos = sigma2 > 1e-10
        n_pos = int(np.sum(pos))
        if n_pos < 50:
            self._set_nan()
            return

        x_s = log_z[pos]
        x_v = log_v[pos]
        s2 = sigma2[pos]
        n_blocks = self.n_blocks or min(10, max(5, n // 500))

        # Collinearity check
        r2_sv = np.corrcoef(x_s, x_v)[0, 1] ** 2
        has_sv = r2_sv < 0.8

        # 2D FPS landmarks in normalized (log S, log V̂) space
        m_gp = min(self.n_gp_landmarks, n_pos // 5)
        if m_gp < 10:
            self._set_nan()
            return

        s_std = max(np.std(x_s), 1e-6)
        v_std = max(np.std(x_v), 1e-6)

        if has_sv:
            X_norm = np.column_stack([x_s / s_std, x_v / v_std])
        else:
            X_norm = (x_s / s_std)[:, None]

        _, lm_idx = farthest_point_sampling_euclidean(X_norm, m_gp)
        lm_s = x_s[lm_idx]
        lm_v = x_v[lm_idx]

        # 2D product NW at landmarks
        dists_s = np.sort(np.abs(np.diff(np.sort(lm_s))))
        bw_s = max(np.median(dists_s[dists_s > 1e-8]) if np.any(
            dists_s > 1e-8) else np.std(x_s), 1e-8)

        diff_s = (lm_s[:, None] - x_s[None, :]) / bw_s

        if has_sv:
            dists_v = np.sort(np.abs(np.diff(np.sort(lm_v))))
            bw_v = max(np.median(dists_v[dists_v > 1e-8]) if np.any(
                dists_v > 1e-8) else np.std(x_v), 1e-8)
            diff_v = (lm_v[:, None] - x_v[None, :]) / bw_v
            K_nw = np.exp(-0.5 * diff_s ** 2 - 0.5 * diff_v ** 2)
        else:
            K_nw = np.exp(-0.5 * diff_s ** 2)

        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm = K_sum > 1e-10

        sigma2_binned = np.zeros(m_gp)
        n_eff = np.zeros(m_gp)
        v_binned = np.zeros(m_gp)

        sigma2_binned[valid_lm] = (
            K_nw[valid_lm] @ s2) / K_sum[valid_lm]
        v_binned[valid_lm] = (
            K_nw[valid_lm] @ x_v) / K_sum[valid_lm]
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = (valid_lm & (sigma2_binned > 1e-8) & (n_eff > 2)
                 & (np.exp(lm_s) > 1e-4))

        if np.sum(valid) < 10:
            self._set_nan()
            return

        x = lm_s[valid]
        v = v_binned[valid]
        y = np.log(sigma2_binned[valid])
        nv = len(x)
        valid_idx = np.where(valid)[0]

        # Parametric mean: structural elasticity via partial coefficient
        if has_sv:
            H = np.column_stack([x, v, np.ones(nv)])
        else:
            H = np.column_stack([x, np.ones(nv)])

        # Block noise estimation
        block_len = n_pos // n_blocks
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                b_s = b * block_len
                b_e = min((b + 1) * block_len, n_pos)
                if b_e <= b_s:
                    continue
                K_b = K_nw[j, b_s:b_e]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ s2[b_s:b_e]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = (np.var(block_ests, ddof=1)
                                 / len(block_ests))
            else:
                noise_var[jj] = 2.0 / max(n_eff[valid_idx[jj]], 1)
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # GP with SE kernel on log S + parametric mean
        Sigma_n = np.diag(noise_var)
        x_range = x.max() - x.min()
        sq_dists = (x[:, None] - x[None, :]) ** 2

        def _neg_log_ml(log_sf, ell):
            sf2 = np.exp(2 * log_sf) if log_sf > -19 else 0.0
            if sf2 < 1e-15:
                C = Sigma_n.copy()
            else:
                K = sf2 * np.exp(-sq_dists / (2 * ell ** 2))
                C = K + Sigma_n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e10
            Cinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Cinv_H = np.linalg.solve(L.T, np.linalg.solve(L, H))
            A_mat = H.T @ Cinv_H
            try:
                beta_hat = np.linalg.solve(A_mat, H.T @ Cinv_y)
            except np.linalg.LinAlgError:
                return 1e10
            r = y - H @ beta_hat
            Cinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
            return 0.5 * r @ Cinv_r + np.sum(np.log(np.diag(L)))

        ell_min = max(x_range / 3, 0.3)
        ell_cands = list(set([max(x_range / d, ell_min)
                              for d in [4, 2, 1, 0.5]]))
        log_sf_grid = [-20, -1, 0, 1, 2]

        best_nlml = 1e10
        best_ell = ell_cands[0]
        best_lsf = -20
        for ell in ell_cands:
            for lsf in log_sf_grid:
                nlml = _neg_log_ml(lsf, ell)
                if nlml < best_nlml:
                    best_nlml = nlml
                    best_ell = ell
                    best_lsf = lsf

        sf2_opt = np.exp(2 * best_lsf) if best_lsf > -19 else 0.0
        if sf2_opt < 1e-15:
            K_opt = np.zeros((nv, nv))
        else:
            K_opt = sf2_opt * np.exp(
                -sq_dists / (2 * best_ell ** 2))

        C = K_opt + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            self._set_nan()
            return

        A_mat = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A_mat)
        except np.linalg.LinAlgError:
            self._set_nan()
            return

        # GP posterior on parametric coefficients β = (α, [β_v], c)
        # R&W §2.7 eq 2.42: β̂ = (H^T C⁻¹ H)⁻¹ H^T C⁻¹ y
        # Posterior covariance: A⁻¹ = (H^T C⁻¹ H)⁻¹
        # Block noise is already in Sigma_n → C → A_inv, so
        # A_inv[0,0] is the correct posterior variance for α.
        beta_hat = A_inv @ H.T @ C_inv @ y
        alpha_post_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        alpha_param = float(beta_hat[0])

        # P(bubble) = P(α > 2 | data) from Gaussian posterior
        z_score = (alpha_param - 2.0) / max(alpha_post_sd, 1e-8)
        self.p_bubble = float(stats.norm.cdf(z_score))
        self.alpha_mean = alpha_param
        self.alpha_sd = alpha_post_sd
        self.vol_p_bubble = float(stats.norm.pdf(z_score))
        self.p_bubble_feller = self.p_bubble

        # Diagnostics
        self._x_s = x
        self._x_v = v
        self._y = y
        self._has_sv = has_sv
        self._r2_sv = r2_sv
        self._sf2_opt = sf2_opt
        self._ell_opt = best_ell
        self._A_inv = A_inv
        self._beta_v = float(beta_hat[1]) if has_sv else None

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0


class EDMDSigFellerGP:
    """EDMD regression on squared returns + GP Feller test.

    Simpler than SigKKFFellerGP: no Kalman filter. Instead, directly
    regresses Nyström features of signature onto (Δlog S)² via EDMD,
    giving σ̂²_t = C_σ^T φ(sig_t). The GP elasticity step is identical.

    Discrete EDMD sidesteps the generator's V-component issue: K^Δt
    just needs (φ_t, φ_{t+1}) pairs regardless of hidden state dimension.
    The generator version (gEDMD) would need L = μ_S∂_S + μ_V∂_V + ...
    with unobserved V terms — the projected generator in signature space
    captures this implicitly but with more approximation error.

    Pipeline:
      1. Lead-lag log-signatures → RBF kernel → Nyström features φ_t
      2. EDMD regression: C_σ = argmin ||Φ C_σ - (Δlog S)²||²
      3. σ̂²_t = S²_t · C_σ^T φ_t / dt (direct, no KF smoothing)
      4. V̂ from signature QV area, 2D landmarks, 2D NW
      5. GP: log σ̂² ~ α·log S + β_v·log V̂ + c + f(log S) → P(α > 2)
    """

    def __init__(self, n_nystrom=60, n_gp_landmarks=80, n_blocks=None,
                 sig_gamma=0.99, sig_level=2, kernel_sigma='auto',
                 edmd_reg=1e-4):
        self.n_nystrom = n_nystrom
        self.n_gp_landmarks = n_gp_landmarks
        self.n_blocks = n_blocks
        self.sig_gamma = sig_gamma
        self.sig_level = sig_level
        self.kernel_sigma = kernel_sigma
        self.edmd_reg = edmd_reg

    def fit(self, z, dz, dt):
        """EDMD regression pipeline: sigs → Nyström → regression → GP.

        Args:
            z: (n,) price levels S_t
            dz: (n,) increments ΔS_t
            dt: time step

        Returns:
            self
        """
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', '..', 'examples', 'proof_of_concept'))
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', 'utils'))
        from signature_features import RecurrentLeadLagLogSigMap
        from nystrom_utils import (farthest_point_sampling_euclidean,
                                    nystrom_approximation, rbf_kernel)

        n = len(z)
        log_z = np.log(np.maximum(np.abs(z), 1e-10))
        log_returns = np.diff(log_z)

        # --- Stage 0: Compute 1D lead-lag log-signatures ---
        # Use γ=0.999 for long-memory price level discrimination.
        sig_gamma = max(self.sig_gamma, 0.999)
        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=self.sig_level,
            forgetting_factor=sig_gamma)
        d_sig = sig_map.feature_dim
        all_sigs = np.zeros((n, d_sig))

        for i in range(len(log_returns)):
            all_sigs[i + 1] = sig_map.update(
                np.array([log_returns[i]]))

        burn_in = max(20, int(2.0 / (1.0 - sig_gamma + 1e-10)))
        idx_start = burn_in
        sigs = all_sigs[idx_start:]
        n_valid = len(sigs)

        if n_valid < 200:
            self._set_nan()
            return self

        # --- Stage 1: RBF kernel + Nyström ---
        if self.kernel_sigma == 'auto':
            sub = sigs[::max(1, n_valid // 300)]
            if len(sub) > 5:
                dists = pdist(sub)
                sigma_k = max(np.median(dists), 1e-6)
            else:
                sigma_k = 1.0
        else:
            sigma_k = float(self.kernel_sigma)

        m = min(self.n_nystrom, n_valid // 5)
        if m < 10:
            self._set_nan()
            return self

        def _kernel(X, Y):
            return rbf_kernel(X, Y, sigma_k)

        landmarks, _ = farthest_point_sampling_euclidean(sigs, m)
        Phi, _ = nystrom_approximation(
            sigs, landmarks, _kernel, self.edmd_reg)

        # --- Stage 2: EDMD regression on log((Δlog S)²) ---
        Phi_t = Phi[:-1]
        lr = log_returns[idx_start:idx_start + n_valid - 1]
        sq_lr = lr ** 2

        reg_I = self.edmd_reg * np.eye(m)
        PhiTPhi = Phi_t.T @ Phi_t + reg_I

        # Log-space regression for positivity
        log_sq_lr = np.log(np.maximum(sq_lr, 1e-20))
        C_log_sigma = np.linalg.solve(PhiTPhi, Phi_t.T @ log_sq_lr)

        # σ̂²_log,t = exp(C_log^T φ_t) / dt (always positive)
        log_pred = Phi @ C_log_sigma
        median_log_sq = np.median(log_sq_lr)
        log_pred = np.clip(log_pred, median_log_sq - 5,
                           median_log_sq + 5)
        sigma2_log = np.exp(log_pred) / dt

        # Convert to level-space
        z_valid = z[idx_start:idx_start + n_valid]
        log_z_valid = log_z[idx_start:idx_start + n_valid]
        sigma2_hat = sigma2_log * z_valid ** 2

        # --- Stage 3: V proxy from signature QV area ---
        # 1D lead-lag: QV = Lévy area at index 2
        qv = np.abs(all_sigs[idx_start:idx_start + n_valid, 2])
        log_v_hat = np.log(np.maximum(qv, 1e-15))

        # --- Stage 4: GP elasticity with 2D conditioning ---
        # Reuse the same 2D GP method
        self._gp_feller_2d(
            log_z_valid, log_v_hat, sigma2_hat, n, dt)

        # Diagnostics
        self._sigma2_hat = sigma2_hat
        self._sigma_k = sigma_k
        self._m = m
        self._C_log_sigma = C_log_sigma

        return self

    def _gp_feller_2d(self, log_z, log_v, sigma2, n, dt):
        """2D GP Feller test — identical to SigKKFFellerGP version."""
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            '..', 'utils'))
        from nystrom_utils import farthest_point_sampling_euclidean

        pos = sigma2 > 1e-10
        n_pos = int(np.sum(pos))
        if n_pos < 50:
            self._set_nan()
            return

        x_s = log_z[pos]
        x_v = log_v[pos]
        s2 = sigma2[pos]
        n_blocks = self.n_blocks or min(10, max(5, n // 500))

        r2_sv = np.corrcoef(x_s, x_v)[0, 1] ** 2
        has_sv = r2_sv < 0.8

        m_gp = min(self.n_gp_landmarks, n_pos // 5)
        if m_gp < 10:
            self._set_nan()
            return

        s_std = max(np.std(x_s), 1e-6)
        v_std = max(np.std(x_v), 1e-6)

        if has_sv:
            X_norm = np.column_stack([x_s / s_std, x_v / v_std])
        else:
            X_norm = (x_s / s_std)[:, None]

        _, lm_idx = farthest_point_sampling_euclidean(X_norm, m_gp)
        lm_s = x_s[lm_idx]
        lm_v = x_v[lm_idx]

        dists_s = np.sort(np.abs(np.diff(np.sort(lm_s))))
        bw_s = max(np.median(dists_s[dists_s > 1e-8]) if np.any(
            dists_s > 1e-8) else np.std(x_s), 1e-8)

        diff_s = (lm_s[:, None] - x_s[None, :]) / bw_s

        if has_sv:
            dists_v = np.sort(np.abs(np.diff(np.sort(lm_v))))
            bw_v = max(np.median(dists_v[dists_v > 1e-8]) if np.any(
                dists_v > 1e-8) else np.std(x_v), 1e-8)
            diff_v = (lm_v[:, None] - x_v[None, :]) / bw_v
            K_nw = np.exp(-0.5 * diff_s ** 2 - 0.5 * diff_v ** 2)
        else:
            K_nw = np.exp(-0.5 * diff_s ** 2)

        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm = K_sum > 1e-10

        sigma2_binned = np.zeros(m_gp)
        n_eff = np.zeros(m_gp)
        v_binned = np.zeros(m_gp)

        sigma2_binned[valid_lm] = (
            K_nw[valid_lm] @ s2) / K_sum[valid_lm]
        v_binned[valid_lm] = (
            K_nw[valid_lm] @ x_v) / K_sum[valid_lm]
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = (valid_lm & (sigma2_binned > 1e-8) & (n_eff > 2)
                 & (np.exp(lm_s) > 1e-4))

        if np.sum(valid) < 10:
            self._set_nan()
            return

        x = lm_s[valid]
        v = v_binned[valid]
        y = np.log(sigma2_binned[valid])
        nv = len(x)
        valid_idx = np.where(valid)[0]

        if has_sv:
            H = np.column_stack([x, v, np.ones(nv)])
        else:
            H = np.column_stack([x, np.ones(nv)])

        block_len = n_pos // n_blocks
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                b_s = b * block_len
                b_e = min((b + 1) * block_len, n_pos)
                if b_e <= b_s:
                    continue
                K_b = K_nw[j, b_s:b_e]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ s2[b_s:b_e]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = (np.var(block_ests, ddof=1)
                                 / len(block_ests))
            else:
                noise_var[jj] = 2.0 / max(n_eff[valid_idx[jj]], 1)
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        Sigma_n = np.diag(noise_var)
        x_range = x.max() - x.min()
        sq_dists = (x[:, None] - x[None, :]) ** 2

        def _neg_log_ml(log_sf, ell):
            sf2 = np.exp(2 * log_sf) if log_sf > -19 else 0.0
            if sf2 < 1e-15:
                C = Sigma_n.copy()
            else:
                K = sf2 * np.exp(-sq_dists / (2 * ell ** 2))
                C = K + Sigma_n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e10
            Cinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Cinv_H = np.linalg.solve(L.T, np.linalg.solve(L, H))
            A_mat = H.T @ Cinv_H
            try:
                beta_hat = np.linalg.solve(A_mat, H.T @ Cinv_y)
            except np.linalg.LinAlgError:
                return 1e10
            r = y - H @ beta_hat
            Cinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
            return 0.5 * r @ Cinv_r + np.sum(np.log(np.diag(L)))

        ell_min = max(x_range / 3, 0.3)
        ell_cands = list(set([max(x_range / d, ell_min)
                              for d in [4, 2, 1, 0.5]]))
        log_sf_grid = [-20, -1, 0, 1, 2]

        best_nlml = 1e10
        best_ell = ell_cands[0]
        best_lsf = -20
        for ell in ell_cands:
            for lsf in log_sf_grid:
                nlml = _neg_log_ml(lsf, ell)
                if nlml < best_nlml:
                    best_nlml = nlml
                    best_ell = ell
                    best_lsf = lsf

        sf2_opt = np.exp(2 * best_lsf) if best_lsf > -19 else 0.0
        if sf2_opt < 1e-15:
            K_opt = np.zeros((nv, nv))
        else:
            K_opt = sf2_opt * np.exp(
                -sq_dists / (2 * best_ell ** 2))

        C = K_opt + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            self._set_nan()
            return

        A_mat = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A_mat)
        except np.linalg.LinAlgError:
            self._set_nan()
            return

        # GP posterior on parametric coefficients β = (α, [β_v], c)
        # R&W §2.7 eq 2.42: β̂ = (H^T C⁻¹ H)⁻¹ H^T C⁻¹ y
        # Posterior covariance: A⁻¹ = (H^T C⁻¹ H)⁻¹
        # Block noise already in Sigma_n → C → A_inv.
        beta_hat = A_inv @ H.T @ C_inv @ y
        alpha_post_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        alpha_param = float(beta_hat[0])

        # P(bubble) = P(α > 2 | data) from Gaussian posterior
        z_score = (alpha_param - 2.0) / max(alpha_post_sd, 1e-8)
        self.p_bubble = float(stats.norm.cdf(z_score))
        self.alpha_mean = alpha_param
        self.alpha_sd = alpha_post_sd
        self.vol_p_bubble = float(stats.norm.pdf(z_score))
        self.p_bubble_feller = self.p_bubble

        self._x_s = x
        self._x_v = v
        self._y = y
        self._has_sv = has_sv
        self._r2_sv = r2_sv
        self._sf2_opt = sf2_opt
        self._ell_opt = best_ell
        self._A_inv = A_inv
        self._beta_v = float(beta_hat[1]) if has_sv else None

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0


class MarginalLikelihoodFellerGP:
    """Parametric Feller test via 1D Kalman filter on log V.

    SPECIALIZED: assumes σ(S,V) = V·S^β (separable SV). For general
    processes, use FellerGP or MLKFellerGP instead.

    For separable SV processes dS = V·S^β·dW₁, the structural
    elasticity α = 2β determines whether S is a strict local martingale
    (bubble ⟺ α > 2, i.e. β > 1).

    Marginalizes V via a 1D Kalman filter:

        State:       x_t = log V_t  (random walk prior)
        Observation: y_t = log((Δlog S_t)²)
                         = 2·x_t + 2(β-1)·log S_t + log(dt) + log(χ²₁)

    For each candidate β on a grid, the KF computes the marginal
    likelihood p(y₁,...,y_T | β). The posterior on β is:

        p(β | y) ∝ p(y | β) · p(β)

    with flat prior. P(bubble) = P(β > 1 | data) = P(α > 2 | data).

    The observation noise is log(χ²₁) with known moments:
        mean = ψ(1/2) + log(2) ≈ -1.2704
        var  = ψ'(1/2)         = π²/2 ≈ 4.9348

    Limitations:
    - Assumes σ(S,V) = V·S^β (separable). Non-separable models
      (e.g. σ(S,V) = V·g(S)) need FellerGP/MLKFellerGP.
    - β-q identifiability: when V covaries with S (ρ≠0), q and β
      trade off. The q prior matters. Default: empirical Bayes from
      lag-1 autocorrelation of Δy (β-independent).
    - For CEV (no stochastic vol), gives extremely tight estimates.
      For SABR with |ρ| > 0.3, prefer FellerGP (nonparametric).
    """

    def __init__(self, beta_grid=None, q_logvol=None, burn_in=200):
        """
        Args:
            beta_grid: array of β candidates. Default: linspace(0.5, 3.5, 61).
            q_logvol: process noise variance for log V random walk.
                      Default: auto-calibrated from data.
            burn_in: number of initial steps to skip for signature warm-up.
        """
        if beta_grid is None:
            self.beta_grid = np.linspace(0.5, 3.5, 61)
        else:
            self.beta_grid = np.asarray(beta_grid)
        self.q_logvol = q_logvol
        self.burn_in = burn_in

    def fit(self, z, dz, dt):
        """Run marginal likelihood Feller test.

        Args:
            z: (n,) price levels S_t
            dz: (n,) increments ΔS_t (unused, kept for API compat)
            dt: time step

        Returns:
            self
        """
        n = len(z)
        log_z = np.log(np.maximum(np.abs(z), 1e-10))
        log_returns = np.diff(log_z)

        # Observation: y_t = log((Δlog S)²)
        log_sq_lr = np.log(np.maximum(log_returns ** 2, 1e-30))

        # Skip burn-in
        idx_start = self.burn_in
        if idx_start >= len(log_sq_lr) - 100:
            self._set_nan()
            return self

        y = log_sq_lr[idx_start:]
        log_s = log_z[idx_start + 1:]  # S at time of return
        n_obs = len(y)

        # log(χ²₁) moments
        obs_bias = -1.2703628454614782   # ψ(1/2) + log(2)
        obs_var = 4.934802200544679      # π²/2

        # Estimate q (process noise for log V) from data, then
        # marginalize β only. We estimate q from the lag-1
        # autocorrelation of residuals, which is β-independent.
        #
        # For random walk + noise: y_t = 2x_t + h_t + ε_t
        # Δy_t - Δh_t = 2·w_t + Δε_t  (w_t = state noise, ε_t = obs noise)
        # Var(Δy-Δh) = 4q + 2·obs_var + Var(Δh)
        # Cov(Δ_t, Δ_{t+1}) = -obs_var  (from differencing iid noise)
        # → lag-1 autocorr ρ₁ = -obs_var / Var(Δy-Δh)
        # → 4q = -obs_var/ρ₁ - 2·obs_var - Var(Δh)   [approx]
        #
        # Simpler: use residuals after removing a rough β·logS trend.
        # The lag-1 autocorrelation of Δy gives q without needing β.
        if self.q_logvol is not None:
            q = float(self.q_logvol)
        else:
            dy = np.diff(y)
            var_dy = np.var(dy)
            # Lag-1 autocovariance of Δy
            cov_lag1 = np.mean(dy[:-1] * dy[1:]) - np.mean(dy) ** 2
            # For pure noise (q=0): cov_lag1 ≈ -obs_var, var_dy ≈ 2·obs_var
            # For q>0: cov_lag1 ≈ -obs_var (unchanged), var_dy increases
            # → q ≈ (var_dy + 2·cov_lag1 - Var(Δh)) / 4
            # Approximate Var(Δh) ≈ 0 (logS changes slowly vs obs noise)
            # Then: 4q ≈ var_dy + 2·cov_lag1
            # (since var_dy = 4q + 2·obs_var + Var(Δh) and
            #  cov_lag1 ≈ -obs_var)
            q_est = (var_dy + 2 * cov_lag1) / 4
            q = float(np.clip(q_est, 1e-6, 1.0))

        # Run KF for each β → marginal log-likelihood p(y|β, q)
        # The KF integrates out the full log V trajectory.
        log_ml = np.zeros(len(self.beta_grid))

        for ib, beta in enumerate(self.beta_grid):
            # For this β: y_t = 2·x_t + h_t + ε_t
            # where h_t = 2(β-1)·log S_t + log(dt) + obs_bias
            # and ε_t ~ N(0, obs_var)
            h = 2 * (beta - 1) * log_s + np.log(dt) + obs_bias

            # Initialize state from first few observations
            init_obs = y[:min(20, n_obs)] - h[:min(20, n_obs)]
            x_hat = np.mean(init_obs) / 2
            P = 1.0  # wide initial uncertainty

            H = 2.0  # observation matrix (scalar)
            ll = 0.0  # accumulated log-likelihood

            for t in range(n_obs):
                # Predict
                x_pred = x_hat  # random walk
                P_pred = P + q

                # Innovation
                y_pred = H * x_pred + h[t]
                innov = y[t] - y_pred
                S_innov = H ** 2 * P_pred + obs_var

                # Log-likelihood contribution (Bayesian marginal
                # likelihood — each term integrates out x_t)
                ll += -0.5 * (np.log(2 * np.pi * S_innov)
                              + innov ** 2 / S_innov)

                # Update
                K = H * P_pred / S_innov
                x_hat = x_pred + K * innov
                P = (1 - K * H) * P_pred

            log_ml[ib] = ll

        # Posterior on β: p(β|y) ∝ p(y|β)·p(β) with flat p(β)
        log_ml -= np.max(log_ml)  # numerical stability
        posterior = np.exp(log_ml)
        posterior /= np.sum(posterior) * (self.beta_grid[1] - self.beta_grid[0])

        # Posterior mean and posterior SD for β
        weights = np.exp(log_ml)
        weights /= weights.sum()
        beta_mean = np.sum(weights * self.beta_grid)
        beta_var = np.sum(weights * (self.beta_grid - beta_mean) ** 2)
        beta_sd = np.sqrt(max(beta_var, 1e-12))

        self._q_logvol = q

        # Convert to α = 2β
        self.alpha_mean = float(2 * beta_mean)
        self.alpha_sd = float(2 * beta_sd)

        # P(bubble) = P(β > 1) = P(α > 2) — direct from posterior grid
        mask_bubble = self.beta_grid > 1.0
        self.p_bubble = float(np.sum(weights[mask_bubble]))
        self.p_bubble_feller = self.p_bubble

        # vol(P): posterior density at β=1 (boundary) × dβ
        # This is ∂P/∂threshold evaluated at threshold=1, i.e. how
        # sensitive P(bubble) is to the threshold choice.
        # = p(β=1 | data) from the posterior density
        idx_boundary = np.argmin(np.abs(self.beta_grid - 1.0))
        self.vol_p_bubble = float(posterior[idx_boundary]
                                  * (self.beta_grid[1] - self.beta_grid[0]))

        # Diagnostics
        self._beta_grid_vals = self.beta_grid
        self._log_ml = log_ml  # already shifted
        self._posterior = posterior
        self._weights = weights
        self._n_obs = n_obs

        return self

    def _set_nan(self):
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.p_bubble_feller = 0.0
        self.vol_p_bubble = 0.0


class MLKFellerGP:
    """Multilevel Kernel Feller GP with Bayesian Model Averaging.

    Compares a finite set of candidate GP models via log marginal
    likelihood (R&W §5.2) and averages P(bubble) across models
    weighted by posterior model probability (BMA).

    Candidate models (each has parametric mean H and SE kernel):

        M₁: H=[log|z|, 1], k=k_SE(z)              — stationary Feller
        M₂: H=[log|z|, 1], k=k_SE(z)·k_SE(t)      — non-stationary
        M₃: H=[log|z|, 1], k=k_SE(z)·k_SE(sig)    — path-dependent
        M₄: H=[log|z|, log V̂, 1], k=k_SE(z)       — conditional Feller
        M₅: H=[log|z|, 1], k=k_SE(z)·k_SE(t)·k_SE(sig) — full aux

    Model priors encode structural knowledge:
        - M₁ gets uniform prior (always plausible)
        - M₄ (vol conditioning) gets a SKEPTICAL prior (log P(M₄) =
          -log(BF_required)) because the V̂ proxy estimated from QV
          is confounded with S through QV ∝ V²·S^{2β}. The partial
          coefficient α in H=[logS, logV̂, 1] is unreliable when
          corr(logS, logV̂) is high. The prior penalizes M₄ so it
          is only selected when the data strongly support it.

    For each model Mᵢ:
        1. Optimize kernel hyperparameters via marginal likelihood grid
        2. Compute log evidence p(y|Mᵢ) (Laplace-approximated)
        3. GP posterior: β̂=(α,c), posterior SD from A⁻¹=(H^T C⁻¹ H)⁻¹
        4. P(bubble|Mᵢ) = P(α > 2 | data, Mᵢ)

    BMA output:
        P(bubble) = Σᵢ P(bubble|Mᵢ) · P(Mᵢ|y)
        α_mean = Σᵢ α̂ᵢ · P(Mᵢ|y)
        α_sd = sqrt(Σᵢ [σ²ᵢ + (αᵢ - α_mean)²] · P(Mᵢ|y))

    References:
        Rasmussen & Williams (2006) Ch. 5 — model selection
        Hoeting et al. (1999) — Bayesian Model Averaging
        Duvenaud et al. (2013) — compositional kernel search
    """

    def __init__(self, n_landmarks=100, n_blocks=None,
                 sig_gamma=0.99, use_signatures='auto',
                 use_vol_proxy='auto', use_time=True,
                 vol_confounding_penalty=10.0):
        """
        Args:
            n_landmarks: number of NW landmarks
            n_blocks: temporal blocks for noise estimation
            sig_gamma: forgetting factor for lead-lag log-sig
            use_signatures: 'auto', True, or False
            use_vol_proxy: 'auto', True, or False
            use_time: whether to include time models
            vol_confounding_penalty: Bayes factor penalty for M₄.
                QV-based V̂ is confounded with S (QV ∝ V²·S^{2β}).
                Set higher when confounding is expected (e.g. SABR,
                3/2 model). Default 10 = require BF > 10 to activate.
        """
        self.n_landmarks = n_landmarks
        self.n_blocks = n_blocks
        self.sig_gamma = sig_gamma
        self.use_signatures = use_signatures
        self.use_vol_proxy = use_vol_proxy
        self.use_time = use_time
        self.vol_confounding_penalty = vol_confounding_penalty

    def fit(self, z, dz, dt, vol_proxy=None):
        """Full pipeline: NW → block noise → BMA across models.

        Args:
            z: (n,) price levels
            dz: (n,) increments
            dt: time step
            vol_proxy: (n,) optional vol proxy (e.g. realized vol)

        Returns:
            self
        """
        n = len(z)
        sq_inc = dz ** 2 / dt
        self._qv_based_vol = False

        n_blocks = self.n_blocks or min(10, max(5, n // 500))
        block_len = n // n_blocks
        m = min(self.n_landmarks, n // 5)

        # --- Stage 0: Feature preparation ---
        log_z = np.log(np.maximum(np.abs(z), 1e-10))
        t_raw = np.arange(n, dtype=float) * dt
        T_obs = t_raw[-1] if t_raw[-1] > 0 else 1.0
        t_norm = t_raw / T_obs

        # Signature features (QV area from lead-lag log-sig)
        raw_qv = None
        sig_features = None
        if self.use_signatures != False:
            raw_qv = self._compute_signatures(z, dz)
            if raw_qv is not None:
                sig_mean = np.mean(raw_qv)
                sig_std = max(np.std(raw_qv), 1e-8)
                sig_features = (raw_qv - sig_mean) / sig_std

        # Vol proxy: external if provided, else signature QV area.
        # QV ∝ V²·S^{2β} so log(QV) is confounded with logS — this is
        # why M₄ has a skeptical prior (vol_confounding_penalty).
        vol_norm = None
        if vol_proxy is not None and self.use_vol_proxy != False:
            vol_mean = np.mean(vol_proxy)
            vol_std = max(np.std(vol_proxy), 1e-8)
            vol_norm = (vol_proxy - vol_mean) / vol_std
        elif self.use_vol_proxy == 'auto' and raw_qv is not None:
            log_qv = np.log(np.maximum(np.abs(raw_qv), 1e-15))
            finite_mask = log_qv > -30
            if np.sum(finite_mask) > 100:
                qv_mean = np.mean(log_qv[finite_mask])
                qv_std = max(np.std(log_qv[finite_mask]), 1e-8)
                vol_norm = (log_qv - qv_mean) / qv_std
            # QV-based proxy has structural confounding:
            # log(QV) = 2·logV + 2β·logS + const
            # Apply extra penalty on top of vol_confounding_penalty.
            # External vol proxies (implied vol, BPV) don't need this.
            self._qv_based_vol = True

        # --- Stage 1: NW at landmarks (z-only, like FellerGP) ---
        quantiles = np.linspace(0.01, 0.99, m)
        landmarks_z = np.quantile(z, quantiles)
        ldists = np.abs(np.diff(landmarks_z))
        bw = max(np.median(ldists) if len(ldists) > 0 else np.std(z), 1e-8)

        diff = (landmarks_z[:, None] - z[None, :]) / bw
        K_nw = np.exp(-0.5 * diff ** 2)
        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm_mask = K_sum > 1e-10
        sigma2_nw = np.zeros(m)
        n_eff = np.zeros(m)
        sigma2_nw[valid_lm_mask] = (
            K_nw[valid_lm_mask] @ sq_inc) / K_sum[valid_lm_mask]
        n_eff[valid_lm_mask] = (
            K_sum[valid_lm_mask] ** 2 / K_sq_sum[valid_lm_mask])

        valid = (valid_lm_mask & (np.abs(landmarks_z) > 1e-4)
                 & (sigma2_nw > 1e-8) & (n_eff > 2))
        if np.sum(valid) < 10:
            self._set_degenerate()
            return self

        nv = int(np.sum(valid))
        valid_idx = np.where(valid)[0]
        x_logz = np.log(np.abs(landmarks_z[valid]))
        y = np.log(sigma2_nw[valid])

        lm_data_idx = np.array([
            np.argmin(np.abs(z - landmarks_z[j])) for j in valid_idx])
        x_t = t_norm[lm_data_idx]
        x_sig = (sig_features[lm_data_idx]
                 if sig_features is not None else None)
        x_vol = vol_norm[lm_data_idx] if vol_norm is not None else None

        # --- Stage 2: Block noise estimation ---
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                K_b = K_nw[j, sl]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ sq_inc[sl]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = (np.var(block_ests, ddof=1)
                                 / len(block_ests))
            else:
                noise_var[jj] = 2.0 / n_eff[valid][jj]
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])
        Sigma_n = np.diag(noise_var)

        # --- Stage 3: Precompute kernel ingredients ---
        sq_dist_z = (x_logz[:, None] - x_logz[None, :]) ** 2
        z_range = max(x_logz.max() - x_logz.min(), 0.1)

        sq_dist_t = (x_t[:, None] - x_t[None, :]) ** 2

        sq_dist_sig = None
        if x_sig is not None:
            sq_dist_sig = (x_sig[:, None] - x_sig[None, :]) ** 2

        sq_dist_vol = None
        if x_vol is not None:
            sq_dist_vol = (x_vol[:, None] - x_vol[None, :]) ** 2

        # --- Stage 4: Enumerate and score candidate models ---
        # Each model = (name, H matrix, kernel builder, log prior)
        H_base = np.column_stack([x_logz, np.ones(nv)])

        # V̂ confounding check: R² between logS and logV̂
        r2_sv = 0.0
        if x_vol is not None:
            r2_sv = float(np.corrcoef(x_logz, x_vol)[0, 1] ** 2)

        # Build H for M₄ (conditional Feller with V̂ covariate)
        has_vol_covariate = (x_vol is not None and r2_sv < 0.8)
        if has_vol_covariate:
            H_vol = np.column_stack([x_logz, x_vol, np.ones(nv)])
        else:
            H_vol = H_base  # fall back to base if collinear

        # Model definitions: (name, H, kernel_dims, log_prior)
        # log_prior encodes structural knowledge about confounding
        log_prior_base = 0.0  # uniform
        # QV-based V̂ has structural confounding (logQV ∝ 2β·logS + 2logV)
        # so it gets a much stronger penalty than external vol proxies.
        base_penalty = self.vol_confounding_penalty
        if getattr(self, '_qv_based_vol', False):
            base_penalty = max(base_penalty, 100.0)  # BF > 100 required
        log_prior_vol = -np.log(base_penalty)

        candidates = [
            ('M1_stationary', H_base, [], log_prior_base),
        ]

        if self.use_time:
            candidates.append(
                ('M2_time', H_base, ['t'], log_prior_base))

        if x_sig is not None and self.use_signatures != False:
            candidates.append(
                ('M3_signature', H_base, ['sig'], log_prior_base))

        if has_vol_covariate:
            candidates.append(
                ('M4_conditional', H_vol, [], log_prior_vol))

        if (self.use_time and x_sig is not None
                and self.use_signatures != False):
            candidates.append(
                ('M5_time_sig', H_base, ['t', 'sig'], log_prior_base))

        # Score each model: optimize σ_f and length scales, compute LML
        model_results = []

        for name, H_m, kernel_dims, log_prior in candidates:
            best_lml = -np.inf
            best_params = None

            # Length scale grids for each dimension
            ell_z_cands = [max(z_range / d, 0.1) for d in [4, 2, 1]]
            ell_t_cands = [0.5, 0.25, 0.15] if 't' in kernel_dims else [None]
            ell_sig_cands = ([max((x_sig.max() - x_sig.min()) / d, 0.1)
                              for d in [2, 1]]
                             if 'sig' in kernel_dims else [None])

            sf_grid = [-20, -2, -1, 0, 1, 2]

            for ell_z in ell_z_cands:
                K_z = np.exp(-sq_dist_z / (2 * ell_z ** 2))
                for ell_t in ell_t_cands:
                    K_t = (np.exp(-sq_dist_t / (2 * ell_t ** 2))
                           if ell_t is not None else None)
                    for ell_sig in ell_sig_cands:
                        K_sig = (np.exp(-sq_dist_sig / (2 * ell_sig ** 2))
                                 if ell_sig is not None
                                 and sq_dist_sig is not None else None)

                        for log_sf in sf_grid:
                            sf2 = (np.exp(2 * log_sf)
                                   if log_sf > -19 else 0.0)
                            K = sf2 * K_z
                            if K_t is not None:
                                K = K * K_t
                            if K_sig is not None:
                                K = K * K_sig
                            lml = self._log_marginal_likelihood(
                                y, H_m, K, Sigma_n)
                            if lml > best_lml:
                                best_lml = lml
                                best_params = {
                                    'log_sf': log_sf, 'ell_z': ell_z,
                                    'ell_t': ell_t, 'ell_sig': ell_sig}

            if best_lml == -np.inf or best_params is None:
                continue

            # GP posterior for this model
            sf2 = (np.exp(2 * best_params['log_sf'])
                   if best_params['log_sf'] > -19 else 0.0)
            K_opt = sf2 * np.exp(
                -sq_dist_z / (2 * best_params['ell_z'] ** 2))
            if best_params.get('ell_t') is not None:
                K_opt = K_opt * np.exp(
                    -sq_dist_t / (2 * best_params['ell_t'] ** 2))
            if best_params.get('ell_sig') is not None and sq_dist_sig is not None:
                K_opt = K_opt * np.exp(
                    -sq_dist_sig / (2 * best_params['ell_sig'] ** 2))

            C = K_opt + Sigma_n
            try:
                C_inv = np.linalg.inv(C)
                A_mat = H_m.T @ C_inv @ H_m
                A_inv = np.linalg.inv(A_mat)
            except np.linalg.LinAlgError:
                continue

            beta_hat = A_inv @ H_m.T @ C_inv @ y
            alpha_m = float(beta_hat[0])
            alpha_sd_m = float(np.sqrt(max(0, A_inv[0, 0])))

            if alpha_sd_m <= 0 or np.isnan(alpha_m):
                continue

            # P(bubble | Mᵢ) from GP posterior
            z_score = (alpha_m - 2.0) / alpha_sd_m
            p_bubble_m = float(stats.norm.cdf(z_score))

            model_results.append({
                'name': name,
                'log_evidence': best_lml + log_prior,
                'log_ml': best_lml,
                'log_prior': log_prior,
                'alpha': alpha_m,
                'alpha_sd': alpha_sd_m,
                'p_bubble': p_bubble_m,
                'params': best_params,
                'H': H_m,
                'C_inv': C_inv,
                'A_inv': A_inv,
                'beta_hat': beta_hat,
            })

        if not model_results:
            self._set_degenerate()
            return self

        # --- Stage 5: BMA across models ---
        log_evidences = np.array([r['log_evidence'] for r in model_results])
        log_evidences -= np.max(log_evidences)  # numerical stability
        model_weights = np.exp(log_evidences)
        model_weights /= model_weights.sum()

        # BMA P(bubble)
        p_bubbles = np.array([r['p_bubble'] for r in model_results])
        self.p_bubble = float(np.sum(model_weights * p_bubbles))

        # BMA α (law of total expectation + variance)
        alphas = np.array([r['alpha'] for r in model_results])
        alpha_sds = np.array([r['alpha_sd'] for r in model_results])
        self.alpha_mean = float(np.sum(model_weights * alphas))
        # Total variance = E[Var] + Var[E]  (law of total variance)
        e_var = float(np.sum(model_weights * alpha_sds ** 2))
        var_e = float(np.sum(
            model_weights * (alphas - self.alpha_mean) ** 2))
        self.alpha_sd = float(np.sqrt(e_var + var_e))

        # vol(P) from delta method on BMA
        z_score_bma = ((self.alpha_mean - 2.0)
                       / max(self.alpha_sd, 1e-8))
        self.vol_p_bubble = float(stats.norm.pdf(z_score_bma))

        # Store internals for diagnostics and p_bubble_local
        self._model_results = model_results
        self._model_weights = model_weights
        self._x_logz = x_logz
        self._x_t = x_t
        self._x_sig = x_sig
        self._x_vol = x_vol
        self._y = y
        self._noise_var = noise_var
        self._lm_data_idx = lm_data_idx
        self._log_z = log_z
        self._t_norm = t_norm
        self._sig_features = sig_features
        self._vol_norm = vol_norm
        self._dt = dt
        self._T_obs = T_obs
        self._z = z
        self._r2_sv = r2_sv
        self._Sigma_n = Sigma_n
        self._sq_dist_z = sq_dist_z
        self._sq_dist_t = sq_dist_t
        self._sq_dist_sig = sq_dist_sig

        # For backward compat: use MAP model for predict/p_bubble_local
        map_idx = int(np.argmax(
            [r['log_evidence'] for r in model_results]))
        self._map_model = model_results[map_idx]
        self._H = model_results[map_idx]['H']
        self._C_inv = model_results[map_idx]['C_inv']
        self._A_inv = model_results[map_idx]['A_inv']
        self._beta_hat = model_results[map_idx]['beta_hat']
        self._params = model_results[map_idx]['params']

        return self

    def p_bubble_local(self, t_query=None, v_query=None, sig_query=None):
        """Time-local P(α>2). Global if no args given.

        Constructs a test grid in log|z| at the query point in auxiliary
        dimensions, gets GP posterior from MAP model, fits local α via
        WLS on the conditioned posterior.
        """
        if not hasattr(self, '_x_logz'):
            return self.p_bubble if hasattr(self, 'p_bubble') else 0.0

        if t_query is None and v_query is None and sig_query is None:
            return self.p_bubble

        # Build test grid in log|z|
        logz_grid = np.linspace(
            self._x_logz.min(), self._x_logz.max(), 50)
        n_grid = len(logz_grid)

        # Auxiliary dims at query values
        t_grid = np.full(n_grid, (t_query / self._T_obs
                                   if t_query is not None
                                   else self._x_t[-1]))

        sig_grid = None
        if self._x_sig is not None:
            sig_grid = np.full(n_grid, (sig_query if sig_query is not None
                                         else self._x_sig[-1]))

        vol_grid = None
        if self._x_vol is not None:
            vol_grid = np.full(n_grid, (v_query if v_query is not None
                                         else self._x_vol[-1]))

        # GP posterior at test grid (MAP model)
        K_star = self._build_kernel_cross(
            logz_grid, t_grid, sig_grid, vol_grid,
            self._x_logz, self._x_t, self._x_sig, self._x_vol,
            self._params)

        n_h = self._H.shape[1]
        if n_h == 3 and vol_grid is not None:
            H_star = np.column_stack([logz_grid, vol_grid, np.ones(n_grid)])
        else:
            H_star = np.column_stack([logz_grid, np.ones(n_grid)])

        r = self._y - self._H @ self._beta_hat
        Cinv_r = self._C_inv @ r
        y_pred = H_star @ self._beta_hat + K_star @ Cinv_r

        # GP posterior variance at test grid
        sf2 = (np.exp(2 * self._params['log_sf'])
               if self._params['log_sf'] > -19 else 0.0)
        v = K_star @ self._C_inv
        var_gp = sf2 - np.sum(v * K_star, axis=1)
        R = H_star - v @ self._H
        var_mean = np.sum(R @ self._A_inv * R, axis=1)
        post_var = np.maximum(var_gp + var_mean, 1e-10)

        # WLS fit: y_pred = α_local·log|z| + c_local
        w = 1.0 / post_var
        H_local = np.column_stack([logz_grid, np.ones(n_grid)])
        WH = H_local * w[:, None]
        try:
            A_local = WH.T @ H_local
            A_local_inv = np.linalg.inv(A_local)
            beta_local = A_local_inv @ (WH.T @ y_pred)
            alpha_local = float(beta_local[0])
            alpha_local_sd = float(np.sqrt(max(0, A_local_inv[0, 0])))
            if alpha_local_sd <= 0:
                return 0.0
            z_score = (alpha_local - 2.0) / alpha_local_sd
            return float(stats.norm.cdf(z_score))
        except np.linalg.LinAlgError:
            return self.p_bubble

    def predict(self, z_new, t_new=None, v_new=None, sig_new=None):
        """GP posterior mean of log σ² at new points (MAP model)."""
        x_logz_new = np.log(np.maximum(np.abs(z_new), 1e-10))
        n_new = len(x_logz_new)
        t_arr = (np.full(n_new, self._x_t[-1]) if t_new is None
                 else t_new / self._T_obs)

        n_h = self._H.shape[1]
        if n_h == 3 and v_new is not None:
            H_new = np.column_stack([x_logz_new, v_new, np.ones(n_new)])
        else:
            H_new = np.column_stack([x_logz_new, np.ones(n_new)])

        K_star = self._build_kernel_cross(
            x_logz_new, t_arr, sig_new, v_new,
            self._x_logz, self._x_t, self._x_sig, self._x_vol,
            self._params)
        r = self._y - self._H @ self._beta_hat
        return H_new @ self._beta_hat + K_star @ (self._C_inv @ r)

    @property
    def active_dimensions(self):
        """Which auxiliary dimensions the MAP model uses."""
        if not hasattr(self, '_map_model'):
            return []
        name = self._map_model['name']
        dims = []
        if 'time' in name or self._params.get('ell_t') is not None:
            dims.append('time')
        if 'sig' in name or self._params.get('ell_sig') is not None:
            dims.append('signature')
        if 'conditional' in name:
            dims.append('vol_proxy')
        return dims

    @property
    def model_comparison(self):
        """Model comparison table: name, weight, α, P(bubble)."""
        if not hasattr(self, '_model_results'):
            return {}
        return {
            r['name']: {
                'weight': float(w),
                'log_ml': r['log_ml'],
                'log_prior': r['log_prior'],
                'alpha': r['alpha'],
                'alpha_sd': r['alpha_sd'],
                'p_bubble': r['p_bubble'],
            }
            for r, w in zip(self._model_results, self._model_weights)
        }

    # --- Private helpers ---

    def _compute_signatures(self, z, dz):
        """Compute lead-lag log-sig QV area features."""
        try:
            import sys
            import os
            sig_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                     'examples', 'proof_of_concept')
            if sig_path not in sys.path:
                sys.path.insert(0, sig_path)
            from signature_features import RecurrentLeadLagLogSigMap
        except ImportError:
            return None

        n = len(z)
        log_returns = np.diff(np.log(np.maximum(np.abs(z), 1e-10)))
        log_returns = np.concatenate([[0.0], log_returns])

        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=2, forgetting_factor=self.sig_gamma)

        qv_area = np.zeros(n)
        for i in range(n):
            feat = sig_map.update(np.array([log_returns[i]]))
            if len(feat) > 2:
                qv_area[i] = feat[2]
            else:
                qv_area[i] = feat[-1]
        return qv_area

    def _log_marginal_likelihood(self, y, H, K, Sigma_n):
        """R&W §2.7 eq 2.45: log marginal likelihood with parametric mean."""
        nv = len(y)
        C = K + Sigma_n
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return -np.inf
        Cinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
        Cinv_H = np.linalg.solve(L.T, np.linalg.solve(L, H))
        A = H.T @ Cinv_H
        try:
            L_A = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return -np.inf
        beta_hat = np.linalg.solve(
            L_A.T, np.linalg.solve(L_A, H.T @ Cinv_y))
        r = y - H @ beta_hat
        Cinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
        lml = (-0.5 * r @ Cinv_r
               - np.sum(np.log(np.diag(L)))
               - np.sum(np.log(np.diag(L_A)))
               - 0.5 * (nv - H.shape[1]) * np.log(2 * np.pi))
        return float(lml)

    def _build_kernel_cross(self, x_logz_1, x_t_1, x_sig_1, x_vol_1,
                             x_logz_2, x_t_2, x_sig_2, x_vol_2, params):
        """Cross-kernel between two sets of points."""
        sf2 = (np.exp(2 * params['log_sf'])
               if params['log_sf'] > -19 else 0.0)
        ell_z = params['ell_z']
        K = sf2 * np.exp(-(x_logz_1[:, None] - x_logz_2[None, :]) ** 2
                          / (2 * ell_z ** 2))
        if params.get('ell_t') is not None:
            K = K * np.exp(-(x_t_1[:, None] - x_t_2[None, :]) ** 2
                            / (2 * params['ell_t'] ** 2))
        if (params.get('ell_sig') is not None
                and x_sig_1 is not None and x_sig_2 is not None):
            K = K * np.exp(-(x_sig_1[:, None] - x_sig_2[None, :]) ** 2
                            / (2 * params['ell_sig'] ** 2))
        return K

    def _set_degenerate(self):
        """Set degenerate results when fit fails."""
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.vol_p_bubble = 0.0
        self._model_results = []
        self._model_weights = np.array([])


class ScaleFunctionGP:
    """GP on the scale function integrand for vol explosion detection (L3).

    For a companion vol process Y with drift μ(y) and diffusion σ²(y),
    the scale function is:
        s(y) = ∫ exp(-∫ 2μ(v)/σ²(v) dv) dy

    Bubble (vol explosion) ⟺ s(∞) < ∞.

    We put a GP prior on the integrand g(y, t) = 2μ(y)/σ²(y) using a
    product kernel:

        k((y,t), (y',t')) = σ²_y · k_SE(y,y'; ℓ_y) · [1 + σ²_t · k_SE(t,t'; ℓ_t)]

    The time component k_SE(t,t') allows the GP to detect non-stationarity
    (trending g → explosive Y → bubble). When Y is ergodic, marginal
    likelihood drives σ²_t → 0 and the kernel collapses to k_SE(y,y') —
    the stationary scale function test. When Y is explosive, σ²_t > 0
    picks up the temporal drift within the same unified kernel framework.

    When data is uninformative, the posterior reverts to the prior
    (mean 0) → P(s(∞) < ∞) ≈ 0.5.
    """

    def __init__(self, n_landmarks=40):
        self.n_landmarks = n_landmarks

    def fit(self, y, dy, dt):
        """Estimate the scale function integrand g(y, t).

        Uses NW kernel regression localized in (y, t) to estimate g at
        landmarks, then fits a GP with product kernel for posterior
        inference on the scale function.

        Args:
            y: (n,) vol process levels Y_t
            dy: (n,) vol increments ΔY_t
            dt: time step

        Returns:
            self
        """
        n = len(y)
        t = np.arange(n, dtype=float) * dt
        T_obs = t[-1]
        m_y = min(self.n_landmarks, n // 10)

        # --- Landmarks: grid in y-space, subsampled in time ---
        pcts = np.linspace(5, 95, m_y)
        y_grid = np.unique(np.percentile(y, pcts))
        m_y = len(y_grid)

        # Time blocks for localized estimation
        n_time_blocks = min(5, max(2, n // 2000))
        t_edges = np.linspace(0, T_obs, n_time_blocks + 1)
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

        if m_y < 5:
            self._set_uninformative()
            return self

        # NW bandwidth for y
        bw_y = np.median(np.diff(y_grid)) * 2.0
        bw_y = max(bw_y, np.std(y) * 0.1, 1e-8)
        # NW bandwidth for time (half block width)
        bw_t = (T_obs / n_time_blocks) / 2.0

        # --- NW estimates of g(y, t) at (y_grid × t_centers) landmarks ---
        lm_y_list = []
        lm_t_list = []
        g_list = []
        noise_list = []
        neff_list = []

        for tb in range(n_time_blocks):
            t_lo, t_hi = t_edges[tb], t_edges[tb + 1]
            t_c = t_centers[tb]
            # Time window mask (soft via kernel, hard for block noise)
            time_weights = np.exp(-0.5 * ((t - t_c) / bw_t) ** 2)

            for iy in range(m_y):
                yc = y_grid[iy]
                space_weights = np.exp(-0.5 * ((y - yc) / bw_y) ** 2)
                w = space_weights * time_weights
                w_sum = w.sum()
                if w_sum < 1e-10:
                    continue
                w_sq_sum = (w ** 2).sum()
                n_eff = w_sum ** 2 / w_sq_sum
                if n_eff < 5:
                    continue

                mu_hat = (w @ dy) / (w_sum * dt)
                resid = dy - mu_hat * dt
                sig2_hat = (w @ (resid ** 2)) / (w_sum * dt)
                sig2_hat = max(sig2_hat, 1e-10)
                g_hat = 2.0 * mu_hat / sig2_hat

                # Block-based noise estimate within this time window
                hard_mask = (t >= t_lo) & (t < t_hi)
                n_hard = hard_mask.sum()
                if n_hard < 20:
                    noise_var = max(4.0 / n_eff, 1.0)
                else:
                    sub_blocks = min(4, n_hard // 20)
                    hard_idx = np.where(hard_mask)[0]
                    block_g = []
                    sub_len = len(hard_idx) // sub_blocks
                    for sb in range(sub_blocks):
                        sb_idx = hard_idx[sb * sub_len:(sb + 1) * sub_len]
                        if len(sb_idx) < 10:
                            continue
                        w_sb = space_weights[sb_idx]
                        w_sb_sum = w_sb.sum()
                        if w_sb_sum < 1e-10:
                            continue
                        mu_sb = (w_sb @ dy[sb_idx]) / (w_sb_sum * dt)
                        resid_sb = dy[sb_idx] - mu_sb * dt
                        sig2_sb = (w_sb @ (resid_sb ** 2)) / (w_sb_sum * dt)
                        sig2_sb = max(sig2_sb, 1e-10)
                        block_g.append(2.0 * mu_sb / sig2_sb)
                    if len(block_g) >= 2:
                        noise_var = np.var(block_g, ddof=1) / len(block_g)
                    else:
                        noise_var = max(4.0 / n_eff, 1.0)

                lm_y_list.append(yc)
                lm_t_list.append(t_c)
                g_list.append(g_hat)
                noise_list.append(max(noise_var, 1e-4))
                neff_list.append(n_eff)

        lm_y = np.array(lm_y_list)
        lm_t = np.array(lm_t_list)
        g_obs = np.array(g_list)
        noise_var = np.array(noise_list)
        nv = len(g_obs)

        if nv < 5:
            self._set_uninformative()
            return self

        # --- Product kernel: k_y(y,y') * [1 + σ²_t * k_t(t,t')] ---
        # Length scales
        ell_y = max((lm_y.max() - lm_y.min()) / 4.0, 1e-8)
        ell_t = T_obs / 4.0
        sq_dist_y = (lm_y[:, None] - lm_y[None, :]) ** 2
        sq_dist_t = (lm_t[:, None] - lm_t[None, :]) ** 2
        K_y_base = np.exp(-sq_dist_y / (2 * ell_y ** 2))
        K_t_base = np.exp(-sq_dist_t / (2 * ell_t ** 2))

        # Grid search over (σ²_y, σ²_t) via marginal likelihood
        best_params = (1.0, 0.0)
        best_lml = -np.inf
        log_sf_y_grid = np.linspace(-1, 3, 12)
        log_sf_t_grid = np.concatenate([[-20], np.linspace(-2, 2, 8)])

        for log_sf_y in log_sf_y_grid:
            sf2_y = np.exp(2 * log_sf_y)
            for log_sf_t in log_sf_t_grid:
                sf2_t = np.exp(2 * log_sf_t) if log_sf_t > -19 else 0.0
                K = sf2_y * K_y_base * (1.0 + sf2_t * K_t_base) + np.diag(noise_var)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, g_obs))
                    lml = (-0.5 * g_obs @ alpha
                           - np.sum(np.log(np.diag(L)))
                           - 0.5 * nv * np.log(2 * np.pi))
                    if lml > best_lml:
                        best_lml = lml
                        best_params = (sf2_y, sf2_t)
                except np.linalg.LinAlgError:
                    continue

        sf2_y_opt, sf2_t_opt = best_params

        # --- GP posterior ---
        K_opt = (sf2_y_opt * K_y_base * (1.0 + sf2_t_opt * K_t_base)
                 + np.diag(noise_var))
        try:
            C_inv = np.linalg.inv(K_opt)
        except np.linalg.LinAlgError:
            self._set_uninformative()
            return self

        # --- P(bubble) via Monte Carlo on scale function convergence ---
        # Evaluate GP posterior at fine y-grid at the FINAL time point
        # (we care about whether s(∞) < ∞ at the current time)
        n_fine = 100
        y_fine = np.linspace(lm_y.min(), lm_y.max() * 1.5, n_fine)
        t_final = T_obs  # evaluate at final observation time

        sq_dy_fine = (y_fine[:, None] - lm_y[None, :]) ** 2
        sq_dt_fine = (t_final - lm_t[None, :]) ** 2
        K_fine_y = np.exp(-sq_dy_fine / (2 * ell_y ** 2))
        K_fine_t = np.exp(-sq_dt_fine / (2 * ell_t ** 2))
        K_fine = sf2_y_opt * K_fine_y * (1.0 + sf2_t_opt * K_fine_t)

        g_mean_fine = K_fine @ (C_inv @ g_obs)

        # Posterior covariance at fine grid
        sq_dy_ff = (y_fine[:, None] - y_fine[None, :]) ** 2
        K_ff_y = np.exp(-sq_dy_ff / (2 * ell_y ** 2))
        # At same time point, k_t(t_final, t_final) = 1
        K_ff = sf2_y_opt * K_ff_y * (1.0 + sf2_t_opt)
        g_cov_fine = K_ff - K_fine @ C_inv @ K_fine.T
        g_cov_fine += 1e-6 * np.eye(n_fine)

        try:
            L_cov = np.linalg.cholesky(g_cov_fine)
        except np.linalg.LinAlgError:
            L_cov = np.diag(np.sqrt(np.maximum(np.diag(g_cov_fine), 1e-10)))

        rng = np.random.RandomState(42)
        n_samples = 500
        n_converge = 0
        dy_fine = np.diff(y_fine)

        for _ in range(n_samples):
            g_sample = g_mean_fine + L_cov @ rng.randn(n_fine)
            # Scale function: s'(y) = exp(-∫ g(v) dv)
            cumul = np.cumsum(
                0.5 * (g_sample[:-1] + g_sample[1:]) * dy_fine)
            cumul = np.concatenate([[0.0], cumul])
            s_prime = np.exp(-cumul)
            # s(∞) ≈ ∫ s'(y) dy
            s_val = np.sum(0.5 * (s_prime[:-1] + s_prime[1:]) * dy_fine)
            # Extrapolate: if g is positive at boundary, s' decays
            g_tail = g_sample[-1]
            if g_tail > 0.1:
                s_extrap = s_val + s_prime[-1] / g_tail
                if np.isfinite(s_extrap):
                    n_converge += 1

        self.p_bubble = float(n_converge / n_samples)
        self.g_mean = g_mean_fine
        self.g_var = np.diag(g_cov_fine)
        self.y_grid = y_fine
        self.y_landmarks = lm_y
        self.t_landmarks = lm_t
        self.g_at_landmarks = g_obs
        self.sf2_y = sf2_y_opt
        self.sf2_t = sf2_t_opt
        self.ell_y = ell_y
        self.ell_t = ell_t

        return self

    def _set_uninformative(self):
        """Set uninformative results (P ≈ 0.5)."""
        self.p_bubble = 0.5
        self.g_mean = None
        self.g_var = None
        self.y_grid = None


class GPBubbleDetector:
    """Unified GP bubble detector dispatching across tiers.

    Uses MLKFellerGP (multilevel kernel) as the primary Feller test.
    ARD length scales automatically select which dimensions matter —
    tiers emerge as limiting cases, not separate code paths.

    Tiers (via MLKFellerGP ARD):
        L1/L2:   ℓ_t, ℓ_sig, ℓ_v → ∞  (stationary, price-only)
        L2-SV:   ℓ_v finite            (vol-conditioned α)
        Regime:  ℓ_t finite            (time-varying α)
        L2-dir:  MLKFellerGP per direction w^T X
        L3:      ScaleFunctionGP on vol companion (separate test)

    test_conditional_feller() is removed — subsumed by ℓ_v finite in MLK.
    """

    def __init__(self, n_landmarks=80, use_mlk=True):
        self.n_landmarks = n_landmarks
        self.use_mlk = use_mlk

    def fit(self, X, dt):
        """Fit detector to multivariate time series.

        Args:
            X: (T, d) multivariate price/vol time series
            dt: time step
        """
        self.dt = dt
        if X.ndim == 1:
            X = X[:, None]
        self.X = X
        self.d = X.shape[1]
        self.dX = np.diff(X, axis=0)
        self.X_t = X[:-1]
        return self

    def _get_vol_idx(self, price_idx):
        """Find column most correlated with |ΔX_price|."""
        if self.d < 2:
            return None
        abs_dp = np.abs(self.dX[:, price_idx])
        corrs = [np.corrcoef(abs_dp, self.X_t[:, j])[0, 1]
                 if j != price_idx else -1.0 for j in range(self.d)]
        return int(np.argmax(corrs))

    def test_feller(self, asset_idx=0, vol_idx=None):
        """Feller α test on a single asset using MLKFellerGP.

        When use_mlk=True (default), uses the multilevel kernel GP which
        automatically conditions on time and vol proxy via ARD.

        Returns:
            dict with alpha_mean, alpha_sd, p_bubble, active_dimensions
        """
        z = self.X_t[:, asset_idx]
        dz = self.dX[:, asset_idx]

        vol_proxy = None
        if self.d >= 2:
            vi = vol_idx if vol_idx is not None else self._get_vol_idx(asset_idx)
            if vi is not None:
                vol_proxy = self.X_t[:, vi]

        if self.use_mlk:
            gp = MLKFellerGP(n_landmarks=self.n_landmarks)
            gp.fit(z, dz, self.dt, vol_proxy=vol_proxy)
            return {
                'alpha_mean': gp.alpha_mean,
                'alpha_sd': gp.alpha_sd,
                'p_bubble': gp.p_bubble,
                'active_dimensions': gp.active_dimensions,
                'gp': gp,
            }
        else:
            gp = FellerGP(n_landmarks=self.n_landmarks)
            gp.fit(z, dz, self.dt)
            return {
                'alpha_mean': gp.alpha_mean,
                'alpha_sd': gp.alpha_sd,
                'p_bubble': gp.p_bubble,
                'sf2': getattr(gp, 'sf2_opt', None),
                'gp': gp,
            }

    def test_directional(self, n_directions=36, price_indices=None,
                          vol_idx=None):
        """L2-dir: Directional Feller test scanning portfolio directions.

        Each direction now uses MLKFellerGP, conditioning on (t, sig, v).

        Returns:
            dict with alpha_max, w_star, p_bubble (Šidák-corrected)
        """
        if self.d < 2:
            return self.test_feller(0)

        if price_indices is None:
            price_indices = list(range(self.d))

        if len(price_indices) == 2:
            angles = np.linspace(0, np.pi, n_directions, endpoint=False)
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            rng = np.random.RandomState(42)
            directions = rng.randn(n_directions, len(price_indices))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        X_sub = self.X_t[:, price_indices]
        dX_sub = self.dX[:, price_indices]

        # Vol proxy for directional test
        vol_proxy = None
        if vol_idx is not None and vol_idx not in price_indices:
            vol_proxy = self.X_t[:, vol_idx]

        all_alphas = []
        all_sds = []
        all_valid = []

        for w in directions:
            z = X_sub @ w
            dz = dX_sub @ w
            if self.use_mlk:
                gp = MLKFellerGP(n_landmarks=self.n_landmarks,
                                  use_signatures=False)  # fast for scan
                gp.fit(z, dz, self.dt, vol_proxy=vol_proxy)
            else:
                gp = FellerGP(n_landmarks=self.n_landmarks)
                gp.fit(z, dz, self.dt)
            all_alphas.append(gp.alpha_mean)
            all_sds.append(gp.alpha_sd)
            all_valid.append(not np.isnan(gp.alpha_mean) and gp.alpha_sd > 0)

        all_alphas = np.array(all_alphas)
        all_sds = np.array(all_sds)
        valid_mask = np.array(all_valid)

        if not np.any(valid_mask):
            return {'alpha_max': np.nan, 'p_bubble': 0.0, 'w_star': None}

        z_scores = np.full(len(directions), -np.inf)
        for i in range(len(directions)):
            if valid_mask[i] and all_sds[i] > 0:
                z_scores[i] = (all_alphas[i] - 2.0) / all_sds[i]

        z_max_idx = np.argmax(z_scores)
        z_max = z_scores[z_max_idx]
        n_eff = max(self.d, np.sum(valid_mask) // 4)

        if np.isfinite(z_max):
            p_bubble = float(stats.norm.cdf(z_max) ** n_eff)
        else:
            p_bubble = 0.0

        return {
            'alpha_max': float(all_alphas[z_max_idx]),
            'alpha_max_sd': float(all_sds[z_max_idx]),
            'w_star': directions[z_max_idx],
            'p_bubble': p_bubble,
            'z_max': float(z_max),
            'n_directions': len(directions),
        }

    def test_conditional_feller(self, price_idx=0, vol_idx=None,
                                 n_vol_bins=5):
        """L2-SV: Conditional Feller test binning by vol proxy.

        Note: With MLKFellerGP, vol-conditioning is subsumed by ℓ_v in the
        ARD kernel. This method is kept for backward compatibility and uses
        the per-bin FellerGP approach regardless of use_mlk setting.
        """
        if vol_idx is None:
            if self.d < 2:
                return self.test_feller(price_idx)
            vol_idx = self._get_vol_idx(price_idx)

        vol_proxy = self.X_t[:, vol_idx]
        z_all = self.X_t[:, price_idx]
        dz_all = self.dX[:, price_idx]

        bin_edges = np.unique(np.quantile(vol_proxy, np.linspace(0, 1, n_vol_bins + 1)))
        actual_bins = len(bin_edges) - 1

        alpha_per_bin = np.full(actual_bins, np.nan)
        sd_per_bin = np.full(actual_bins, np.nan)

        for b in range(actual_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (vol_proxy >= lo) & (vol_proxy < hi) if b < actual_bins - 1 else (vol_proxy >= lo)
            if np.sum(mask) < 200:
                continue
            gp = FellerGP(n_landmarks=min(60, np.sum(mask) // 5))
            gp.fit(z_all[mask], dz_all[mask], self.dt)
            alpha_per_bin[b] = gp.alpha_mean
            sd_per_bin[b] = gp.alpha_sd

        valid = ~np.isnan(alpha_per_bin) & ~np.isnan(sd_per_bin) & (sd_per_bin > 0)
        z_scores = np.full(actual_bins, -np.inf)
        for b in range(actual_bins):
            if valid[b]:
                z_scores[b] = (alpha_per_bin[b] - 2.0) / sd_per_bin[b]

        z_max = np.max(z_scores)
        n_valid = max(1, int(np.sum(valid)))
        p_bubble = float(stats.norm.cdf(z_max) ** n_valid) if np.isfinite(z_max) else 0.0

        return {
            'alpha_per_bin': alpha_per_bin,
            'sd_per_bin': sd_per_bin,
            'p_bubble': p_bubble,
            'vol_bin_edges': bin_edges,
            'n_vol_bins': actual_bins,
        }

    def test_vol_explosion(self, price_idx=0, vol_idx=None):
        """L3: Scale function GP test for vol companion explosion.

        Returns:
            dict with P(bubble) from GP posterior on scale function convergence
        """
        if self.d < 2:
            return {'p_bubble': 0.5, 'method': 'vol_explosion',
                    'note': 'Requires observed vol companion (d >= 2)'}

        if vol_idx is None:
            vol_idx = self._get_vol_idx(price_idx)

        y = self.X_t[:, vol_idx]
        dy = self.dX[:, vol_idx]

        gp = ScaleFunctionGP(n_landmarks=40)
        gp.fit(y, dy, self.dt)

        return {
            'p_bubble': gp.p_bubble,
            'method': 'vol_explosion',
            'vol_idx': vol_idx,
            'gp': gp,
        }

    def test_hidden_vol_explosion(self, price_idx=0, vol_method='signature',
                                   block_size=50, sig_gamma=0.99):
        """L-HV: Detect vol explosion from price path alone (hidden vol).

        When the companion vol process is unobserved, reconstruct it from
        the price path and test for explosion via ScaleFunctionGP.

        Under Q, dX = rX dt + σ(X,Y)X dW. Block RV or signature QV
        consistently estimates Y² (for multiplicative vol). If Y is
        explosive → estimated vol grows → ScaleFunctionGP detects
        convergent scale function → P(bubble) > 0.5.

        Theoretical basis: conditional EVT/LDT — the tail index of the
        return distribution conditional on recent history encodes the vol
        dynamics. Growing conditional tail heaviness ↔ vol explosion.

        Args:
            price_idx: which column is the price
            vol_method: 'signature' (lead-lag log-sig QV area) or
                        'block_rv' (non-overlapping block realized vol)
            block_size: block length for block_rv method
            sig_gamma: forgetting factor for signature method (0.99 = ~100 step window)

        Returns:
            dict with p_bubble, reconstructed vol series, method details
        """
        z = self.X_t[:, price_idx]
        dz = self.dX[:, price_idx]
        n = len(z)

        if vol_method == 'signature':
            # Lead-lag log-sig: QV area = Lévy area between lead and lag
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                            '..', '..', 'examples', 'proof_of_concept'))
            from signature_features import RecurrentLeadLagLogSigMap

            # Input: log returns (scale-invariant)
            log_ret = dz / np.maximum(np.abs(z), 1e-10)

            sig_map = RecurrentLeadLagLogSigMap(state_dim=1, level=2,
                                                 forgetting_factor=sig_gamma)
            qv_area = np.zeros(n)
            for i in range(n):
                feats = sig_map.update(np.array([log_ret[i]]))
                # For d=1 lead-lag: features = [l1_lead, l1_lag, levy_area]
                # Lévy area = feats[2] = cumulative QV (with decay)
                qv_area[i] = feats[2]

            # QV area is cumulative with decay — take differences for
            # incremental vol estimate, or use level directly
            # Use the QV area as vol proxy (it's proportional to realized var)
            vol_proxy = np.abs(qv_area)
            vol_proxy = np.maximum(vol_proxy, 1e-12)

            # ScaleFunctionGP on log(vol_proxy)
            log_vol = np.log(vol_proxy)
            # Subsample to reduce autocorrelation from the decay
            subsample = max(1, int(1.0 / (1.0 - sig_gamma + 1e-8)) // 2)
            idx = np.arange(0, n - 1, subsample)
            if len(idx) < 30:
                idx = np.arange(0, n - 1)

            log_vol_sub = log_vol[idx]
            d_log_vol = np.diff(log_vol_sub)
            log_vol_t = log_vol_sub[:-1]
            sub_dt = subsample * self.dt

            if len(log_vol_t) < 20:
                return {'p_bubble': 0.5, 'method': 'hidden_vol_explosion',
                        'vol_method': vol_method,
                        'note': 'Too few points after subsampling'}

            gp = ScaleFunctionGP(n_landmarks=min(40, len(log_vol_t) // 5))
            gp.fit(log_vol_t, d_log_vol, sub_dt)

            return {
                'p_bubble': gp.p_bubble,
                'method': 'hidden_vol_explosion',
                'vol_method': 'signature',
                'vol_proxy': vol_proxy,
                'sig_gamma': sig_gamma,
                'n_points': len(log_vol_t),
                'gp': gp,
            }

        elif vol_method == 'block_rv':
            # Block realized vol: non-overlapping blocks
            log_ret = dz / np.maximum(np.abs(z), 1e-10)
            sq_log_ret = log_ret ** 2 / self.dt

            n_blocks = n // block_size
            if n_blocks < 20:
                return {'p_bubble': 0.5, 'method': 'hidden_vol_explosion',
                        'vol_method': vol_method,
                        'note': f'Too few blocks ({n_blocks}), need >= 20'}

            block_rv = np.array([
                np.mean(sq_log_ret[b * block_size:(b + 1) * block_size])
                for b in range(n_blocks)])

            valid = (block_rv > 1e-12) & np.isfinite(block_rv)
            if np.sum(valid) < 20:
                return {'p_bubble': 0.5, 'method': 'hidden_vol_explosion',
                        'vol_method': vol_method,
                        'note': 'Too few valid blocks'}
            block_rv = block_rv[valid]

            log_rv = np.log(block_rv)
            d_log_rv = np.diff(log_rv)
            log_rv_t = log_rv[:-1]
            block_dt = block_size * self.dt

            gp = ScaleFunctionGP(n_landmarks=min(40, len(log_rv_t) // 5))
            gp.fit(log_rv_t, d_log_rv, block_dt)

            return {
                'p_bubble': gp.p_bubble,
                'method': 'hidden_vol_explosion',
                'vol_method': 'block_rv',
                'block_rv': block_rv,
                'n_blocks': len(block_rv),
                'block_size': block_size,
                'gp': gp,
            }

        else:
            raise ValueError(f"Unknown vol_method: {vol_method}")

    def test_joint_feller(self, price_idx=0, vol_idx=None, n_landmarks=80,
                           regularization=1e-4):
        """L-JF: Joint diffusion test for non-separable SV bubbles.

        Two-pronged approach:

        1. **Direct σ²_X estimation**: NW regression of (ΔX)²/dt on the
           joint state (log|X|, Y). Tests α > 2 conditioned on vol.
           Handles separable SV and most practical cases.

        2. **Scale function on log(Y)**: For the JPS counter-example where
           α=2 at every vol level, the bubble comes from Y explosion.
           Run ScaleFunctionGP on Z = log(Y) where:
             μ_Z(z) = drift/Y - σ²_Y/(2Y²)  (Itô)
             σ²_Z(z) = σ²_Y/Y²
           The log transform compresses the explosive range, making NW
           estimation feasible. If the scale function converges → explosion
           → bubble.

        The two tests are combined via noisy-OR: either one detecting a
        bubble is sufficient.

        Returns:
            dict with p_bubble, sub-test details
        """
        if self.d < 2:
            return {'p_bubble': 0.0, 'method': 'joint_feller',
                    'note': 'Joint test requires d >= 2'}

        if vol_idx is None:
            vol_idx = self._get_vol_idx(price_idx)

        S = self.X_t[:, price_idx]
        V = self.X_t[:, vol_idx]
        dS = self.dX[:, price_idx]
        dV = self.dX[:, vol_idx]

        # ── Prong 1: Direct σ²_X conditioned on vol ──
        # FellerGP on price with vol_proxy already handles this via MLK.
        # Run a targeted version: bin by vol, test α at each level.
        n = len(S)
        n_vol_bins = min(5, max(2, n // 2000))
        vol_pcts = np.linspace(10, 90, n_vol_bins)
        vol_edges = np.percentile(V, vol_pcts)

        alpha_at_vol = []
        sd_at_vol = []
        for i in range(len(vol_edges)):
            if i == 0:
                mask = V < vol_edges[0]
            elif i == len(vol_edges) - 1:
                mask = V >= vol_edges[-1]
            else:
                mask = (V >= vol_edges[i - 1]) & (V < vol_edges[i])
            if np.sum(mask) < 200:
                continue
            gp = FellerGP(n_landmarks=min(60, np.sum(mask) // 5))
            gp.fit(S[mask], dS[mask], self.dt)
            if not np.isnan(gp.alpha_mean):
                alpha_at_vol.append(gp.alpha_mean)
                sd_at_vol.append(gp.alpha_sd)

        # Max α across vol bins
        if alpha_at_vol:
            alpha_at_vol = np.array(alpha_at_vol)
            sd_at_vol = np.array(sd_at_vol)
            z_scores = (alpha_at_vol - 2.0) / np.maximum(sd_at_vol, 0.01)
            z_max = np.max(z_scores)
            p_prong1 = float(stats.norm.cdf(z_max))
        else:
            p_prong1 = 0.0
            z_max = -np.inf

        # ── Prong 2: Scale function on log(Y) ──
        # Z = log(Y): compressed range, NW works on explosive processes.
        # ScaleFunctionGP estimates g(z) = 2μ_Z/σ²_Z and tests s(∞) < ∞.
        V_pos = np.maximum(V, 1e-10)
        logV = np.log(V_pos)
        # Increments of log(V): ΔZ ≈ dV/V - 0.5(dV/V)² (discrete Itô)
        dlogV = np.diff(logV)
        logV_t = logV[:-1]

        gp_scale = ScaleFunctionGP(n_landmarks=min(40, n // 20))
        gp_scale.fit(logV_t, dlogV, self.dt)
        p_prong2 = gp_scale.p_bubble

        # ── Combine via noisy-OR ──
        p_bubble = 1.0 - (1.0 - p_prong1) * (1.0 - p_prong2)

        return {
            'p_bubble': float(p_bubble),
            'p_conditional_alpha': float(p_prong1),
            'p_vol_explosion_logY': float(p_prong2),
            'alpha_at_vol_levels': alpha_at_vol.tolist() if isinstance(alpha_at_vol, np.ndarray) else alpha_at_vol,
            'z_max_conditional': float(z_max),
            'method': 'joint_feller',
            'gp_scale': gp_scale,
        }

    def test_jump_activity(self, price_idx=0, p_values=None, n_blocks=10):
        """L-JA: Jump activity test via power variation scaling.

        Estimates the Blumenthal-Getoor index β_BG from the scaling of
        p-th order power variation V^(p)_n = Σ|ΔX_i|^p as Δt → 0.

        For a process with continuous component σ² and jump activity β_BG:
          - V^(p) ~ n^{1-p/2}  when p > β_BG  (continuous dominates)
          - V^(p) ~ n^{1-p/β_BG}  when p < β_BG  (jumps dominate)

        The kink in log V^(p) vs p at p = β_BG reveals the activity index.

        Then runs the BV-robust Feller test (bipower NW) to separate
        continuous-component bubbles from jump contributions.

        Returns:
            dict with beta_bg, has_jumps, p_bubble_continuous,
            p_bubble_jump, method details
        """
        z = self.X_t[:, price_idx]
        dz = self.dX[:, price_idx]
        n = len(dz)

        if p_values is None:
            p_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])

        # Compute power variation at multiple time scales
        block_len = n // n_blocks
        if block_len < 20:
            return {'beta_bg': 0.0, 'has_jumps': False,
                    'p_bubble_continuous': 0.0, 'method': 'jump_activity',
                    'note': 'Insufficient data'}

        abs_dz = np.abs(dz)

        # V^(p) at full resolution
        V_p_full = np.array([np.mean(abs_dz ** p) for p in p_values])
        log_V = np.log(np.maximum(V_p_full, 1e-30))

        # V^(p) at coarsened resolution (sum adjacent increments)
        dz_coarse = dz[:(n // 2) * 2].reshape(-1, 2).sum(axis=1)
        abs_dz_c = np.abs(dz_coarse)
        V_p_coarse = np.array([np.mean(abs_dz_c ** p) for p in p_values])
        log_V_coarse = np.log(np.maximum(V_p_coarse, 1e-30))

        # Scaling exponent: log(V_full/V_coarse) / log(2) for each p
        # For pure diffusion: exponent = p/2
        # For jumps at activity β: exponent = p/β for p < β
        scaling = (log_V_coarse - log_V) / np.log(2.0)

        # Estimate β_BG from where scaling deviates from p/2
        # Diffusion scaling: ξ(p) = p/2. Jump scaling: ξ(p) = p/β for p < β
        # The ratio ξ(p) / (p/2) < 1 when jumps contribute
        diffusion_scaling = p_values / 2.0

        # BG index: fit piecewise. For p > β, scaling ≈ p/2.
        # For p ≤ β, scaling ≈ p/β_BG.
        # Use the deviation at p=0.5 and p=1.0 vs p=2.0
        ratio_low = scaling[0] / max(diffusion_scaling[0], 1e-8)  # p=0.5
        ratio_mid = scaling[1] / max(diffusion_scaling[1], 1e-8)  # p=1.0

        # If ratio < 0.8 at low p, jumps are present
        has_jumps = ratio_low < 0.8 or ratio_mid < 0.85

        # Estimate β_BG from slope change
        if has_jumps and scaling[0] > 0:
            # β_BG ≈ p / ξ(p) at low p where jumps dominate
            beta_bg_estimates = []
            for i, p in enumerate(p_values[:3]):  # p = 0.5, 1.0, 1.5
                if scaling[i] > 0.05:
                    beta_bg_estimates.append(p / scaling[i])
            beta_bg = float(np.median(beta_bg_estimates)) if beta_bg_estimates else 0.0
            beta_bg = np.clip(beta_bg, 0.0, 2.0)
        else:
            beta_bg = 0.0

        # Run BV-robust Feller test for continuous component
        if self.use_mlk:
            gp = MLKFellerGP(n_landmarks=self.n_landmarks, use_bipower=True)
            gp.fit(z, dz, self.dt)
        else:
            gp = FellerGP(n_landmarks=self.n_landmarks, use_bipower=True)
            gp.fit(z, dz, self.dt)

        p_bubble_continuous = gp.p_bubble

        # Jump bubble contribution: if β_BG > 1, jumps can create
        # strict local martingale behavior (Protter 2013)
        # Test: compare RV - BV (jump variation) scaling with price level
        if has_jumps:
            rv = dz ** 2
            bv = np.zeros(n)
            bv[1:] = (np.pi / 2.0) * abs_dz[1:] * abs_dz[:-1]
            jump_var = np.maximum(rv - bv, 0.0)

            # Regress log(jump_var + ε) ~ α_J·log|z| to see if jumps scale
            valid_jv = (jump_var > 1e-10) & (np.abs(z) > 1e-4)
            if np.sum(valid_jv) > 50:
                x_jv = np.log(np.abs(z[valid_jv]))
                y_jv = np.log(jump_var[valid_jv])
                H_jv = np.column_stack([x_jv, np.ones(len(x_jv))])
                try:
                    beta_jv = np.linalg.lstsq(H_jv, y_jv, rcond=None)[0]
                    alpha_jump = beta_jv[0]
                    # Jump bubble if jump intensity scales superlinearly
                    p_bubble_jump = float(stats.norm.cdf(
                        (alpha_jump - 2.0) / max(0.5, abs(alpha_jump) * 0.2)))
                except np.linalg.LinAlgError:
                    p_bubble_jump = 0.0
                    alpha_jump = np.nan
            else:
                p_bubble_jump = 0.0
                alpha_jump = np.nan
        else:
            p_bubble_jump = 0.0
            alpha_jump = np.nan

        return {
            'beta_bg': float(beta_bg),
            'has_jumps': bool(has_jumps),
            'p_bubble_continuous': float(p_bubble_continuous),
            'p_bubble_jump': float(p_bubble_jump),
            'alpha_continuous': gp.alpha_mean,
            'alpha_jump': float(alpha_jump) if not np.isnan(alpha_jump) else None,
            'scaling_exponents': scaling.tolist(),
            'p_values_used': p_values.tolist(),
            'method': 'jump_activity',
            'gp': gp,
        }

    def test_all(self, price_idx=0, vol_idx=None):
        """Run all applicable tiers and return unified result.

        Simplified: feller (with MLK) + directional + vol_explosion.
        Conditional Feller is subsumed by MLKFellerGP's ℓ_v.

        Returns:
            dict with per-tier P(bubble) and aggregate
        """
        results = {}

        # L2: per-asset Feller (MLK automatically handles time/vol/sig)
        results['L2_feller'] = self.test_feller(price_idx, vol_idx=vol_idx)

        # L2-dir: directional (if d >= 2)
        if self.d >= 2:
            results['L2_directional'] = self.test_directional(
                vol_idx=vol_idx)

        # L3: vol explosion (if d >= 2, observed vol)
        if self.d >= 2:
            results['L3_vol'] = self.test_vol_explosion(price_idx, vol_idx)

        # L-HV: hidden vol explosion (always available — uses price only)
        results['L_HV_sig'] = self.test_hidden_vol_explosion(
            price_idx, vol_method='signature')

        # L-JF: joint Feller (if d >= 2, observed vol)
        if self.d >= 2:
            results['L_JF_joint'] = self.test_joint_feller(price_idx, vol_idx)

        # Aggregate: maximum P(bubble) across non-degenerate tiers
        p_values = []
        for tier, res in results.items():
            p = res.get('p_bubble', 0.0)
            if not np.isnan(p):
                p_values.append(p)

        results['p_bubble'] = max(p_values) if p_values else 0.0

        return results
