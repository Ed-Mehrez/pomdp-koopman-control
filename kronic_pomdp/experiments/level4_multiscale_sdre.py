"""
Level 4 Multi-Scale Signature SDRE.

Key insight: Signatures at different forgetting factors ARE multi-scale
observations of the same latent state (volatility):
  - gamma=0.80: ~5 day window (high freq RV, noisy but responsive)
  - gamma=0.94: ~17 day window (monthly RV, moderate)
  - gamma=0.99: ~100 day window (quarterly RV, smooth but laggy)

Level-2 signature component encodes exponentially-weighted realized variance.
Different gamma = different observation noise levels.

Architecture:
  1. Multiple RecSig at different forgetting factors
  2. Each learns V_hat via RLS (target = r^2/dt)
  3. Kalman filter with CIR dynamics fuses the multi-scale observations
  4. SDRE policy from fused V_hat

The Kalman filter optimally weights fast (noisy) vs slow (smooth) scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from examples.proof_of_concept.signature_features import (
    RecurrentSignatureMap, RecurrentLogSignatureMap, RecurrentLeadLagLogSigMap
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

HestonParams = namedtuple('HestonParams',
    ['mu', 'r', 'kappa', 'theta', 'xi', 'rho', 'gamma', 'dt'])

P = HestonParams(mu=0.08, r=0.02, kappa=2.0, theta=0.04, xi=0.3,
                 rho=-0.5, gamma=2.0, dt=1/252)
T = 10000


def _make_crra(gamma):
    return (lambda W: W**(-gamma), lambda W: -gamma * W**(-gamma-1))


def _make_heston_sim(p):
    def sim(t, z1, z2, dt_val, sd):
        if 'V' not in sd:
            sd['V'] = p.theta
        V_prev = sd['V']
        sv = np.sqrt(max(V_prev, 1e-8))
        sdt = np.sqrt(dt_val)
        z2c = p.rho * z1 + np.sqrt(1 - p.rho**2) * z2
        ret = (p.mu - p.r) * dt_val + sv * sdt * z1
        V_new = max(V_prev + p.kappa * (p.theta - V_prev) * dt_val
                    + p.xi * sv * sdt * z2c, 1e-8)
        sd['V'] = V_new
        return V_new, ret
    return sim


class MultiScaleSigFilter:
    """Multi-scale signature filter for volatility estimation.

    Uses N signature maps at different forgetting factors (= timescales).
    Each produces a V_hat estimate via RLS. A Kalman filter with CIR
    dynamics fuses them, weighting by their effective observation noise.

    The slow signatures have noise ~ 2V^2 / (N_eff * dt) where
    N_eff = 1/(1-gamma), so they're N_eff times less noisy than r^2/dt.
    """
    def __init__(self, gammas=(0.80, 0.94, 0.99), input_dim=2,
                 rls_ff=0.999, kf_kappa=2.0, kf_theta=0.04, kf_xi=0.3,
                 sig_type='full'):
        self.gammas = gammas
        self.n_scales = len(gammas)
        self.sig_type = sig_type  # 'full', 'logsig', 'leadlag'

        # One signature map + RLS head per scale
        self.sig_maps = []
        self.rls_w = []
        self.rls_P = []
        for g in gammas:
            if sig_type == 'logsig':
                sm = RecurrentLogSignatureMap(state_dim=input_dim, level=2,
                                              forgetting_factor=g)
            elif sig_type == 'leadlag':
                sm = RecurrentLeadLagLogSigMap(state_dim=input_dim, level=2,
                                               forgetting_factor=g)
            else:
                sm = RecurrentSignatureMap(state_dim=input_dim, level=2,
                                           forgetting_factor=g)
            n_feat = sm.feature_dim + 1  # +bias
            self.sig_maps.append(sm)
            self.rls_w.append(np.zeros(n_feat))
            self.rls_P.append(np.eye(n_feat) * 100.0)

        self.rls_ff = rls_ff
        self.n_feat = self.sig_maps[0].feature_dim + 1

        # Kalman filter state (scalar V)
        self.kf_kappa = kf_kappa
        self.kf_theta = kf_theta
        self.kf_xi = kf_xi
        self.V_filt = kf_theta
        self.P_kf = kf_xi**2 * kf_theta / 252 * 10
        self._step_count = 0

    def reset(self):
        for sm in self.sig_maps:
            sm.reset()

    def update(self, dx, ret, dt):
        """Update all scales, fuse via Kalman filter.

        Returns: (V_hat_fused, list of per-scale V_hat)
        """
        self._step_count += 1
        scale_preds = []

        for i in range(self.n_scales):
            phi = self.sig_maps[i].update(dx)
            features = np.concatenate([phi, [1.0]])

            # RLS prediction
            pred = np.dot(self.rls_w[i], features)

            # RLS update (target = r^2/dt)
            target = min(ret**2 / dt, 2.0)
            z = features[:, np.newaxis]
            Pz = self.rls_P[i] @ z
            denom = self.rls_ff + (z.T @ Pz)[0, 0]
            k = Pz / denom
            self.rls_w[i] += k.flatten() * (target - pred)
            self.rls_P[i] = (self.rls_P[i] - k @ Pz.T) / self.rls_ff

            scale_preds.append(max(pred, 1e-8))

        # --- Kalman filter: fuse multi-scale observations ---
        # Predict: CIR dynamics
        V_pred = self.V_filt + self.kf_kappa * (self.kf_theta - self.V_filt) * dt
        V_pred = max(V_pred, 1e-6)

        # Process noise
        Q_kf = self.kf_xi**2 * max(self.V_filt, 1e-6) * dt
        P_pred = (1 - self.kf_kappa * dt)**2 * self.P_kf + Q_kf

        # Sequential Kalman update with each scale's observation
        V_upd = V_pred
        P_upd = P_pred

        for i, (gamma, v_obs) in enumerate(zip(self.gammas, scale_preds)):
            # Effective window: N_eff = 1/(1-gamma) for gamma<1, or step count for cumulative
            if gamma >= 1.0 - 1e-10:
                N_eff = max(self._step_count, 1.0)
            else:
                N_eff = 1.0 / (1.0 - gamma)
            # Observation noise: R ~ 2V^2 / (N_eff * dt)
            # The signature averages ~N_eff returns, reducing chi-sq noise by N_eff
            R_i = 2 * max(V_upd, 1e-6)**2 / (N_eff * dt)

            # Sequential scalar Kalman update
            K_i = P_upd / (P_upd + R_i)
            V_upd = V_upd + K_i * (v_obs - V_upd)
            V_upd = max(V_upd, 1e-6)
            P_upd = (1 - K_i) * P_upd

        self.V_filt = V_upd
        self.P_kf = P_upd

        return self.V_filt, scale_preds


def run_sdre_multiscale(p, T, seed, U_prime, U_double_prime,
                        known_mu=None, gammas=(0.80, 0.94, 0.99),
                        kf_xi=None, sig_type='full'):
    """SDRE controller with multi-scale signature Kalman filter."""
    rng = np.random.RandomState(seed)

    xi_kf = kf_xi if kf_xi is not None else p.xi
    ms_filter = MultiScaleSigFilter(
        gammas=gammas, kf_kappa=p.kappa, kf_theta=p.theta, kf_xi=xi_kf,
        sig_type=sig_type)

    ewma_mu, lam_mu = 0.0, 0.999
    V_hat = np.zeros(T)
    V_scales = np.zeros((T, len(gammas)))
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        dx = np.array([p.dt, ret])
        v_fused, v_per_scale = ms_filter.update(dx, ret, p.dt)
        V_hat[t] = v_fused
        V_scales[t] = v_per_scale

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)

        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi

        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {
        'V_hat': V_hat, 'V_scales': V_scales, 'pi_sdre': pi_sdre,
        'state_true': state_true, 'W_history': W_history,
    }


def run_sdre_kalman(p, T, seed, U_prime, U_double_prime,
                    known_mu=None, xi_kf=None):
    """Scalar Kalman baseline."""
    rng = np.random.RandomState(seed)
    kf_kappa, kf_theta = p.kappa, p.theta
    kf_xi = xi_kf if xi_kf is not None else p.xi
    V_filt = kf_theta
    P_kf = kf_xi**2 * kf_theta * p.dt * 10
    ewma_mu, lam_mu = 0.0, 0.999

    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        V_pred = max(V_filt + kf_kappa * (kf_theta - V_filt) * p.dt, 1e-6)
        Q_kf = kf_xi**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf
        y_obs = ret**2 / p.dt
        R_kf = 2 * max(V_pred, 1e-6)**2 / p.dt
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred
        V_hat[t] = V_filt

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)
        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi
        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
            'W_history': W_history}


class DirectMultiScaleFilter:
    """Use raw signature quadratic variation as observation — no RLS needed.

    Level-2 signature component s2[3] = Σ gamma^{t-i} r_i^2.
    Normalized by s1[0] = Σ gamma^{t-i} dt to get variance/dt.

    This IS the exponentially-weighted realized variance at timescale
    N_eff = 1/(1-gamma). No regression needed.
    """
    def __init__(self, gammas=(0.80, 0.94, 0.99), input_dim=2,
                 kf_kappa=2.0, kf_theta=0.04, kf_xi=0.3):
        self.gammas = gammas
        self.n_scales = len(gammas)

        # One signature map per scale
        self.sig_maps = [
            RecurrentSignatureMap(state_dim=input_dim, level=2, forgetting_factor=g)
            for g in gammas
        ]

        # Kalman filter
        self.kf_kappa = kf_kappa
        self.kf_theta = kf_theta
        self.kf_xi = kf_xi
        self.V_filt = kf_theta
        self.P_kf = kf_xi**2 * kf_theta / 252 * 10

    def reset(self):
        for sm in self.sig_maps:
            sm.reset()

    def update(self, dx, ret, dt):
        """Extract RV from signature level-2, fuse via Kalman."""
        scale_obs = []

        for i, (gamma, sm) in enumerate(zip(self.gammas, self.sig_maps)):
            phi = sm.update(dx)
            # phi layout for state_dim=2, level=2:
            # Level 1: [s1_time, s1_ret]  (indices 0, 1)
            # Level 2: [s2_tt, s2_tr, s2_rt, s2_rr]  (indices 2, 3, 4, 5)
            # s2_rr = Σ gamma^{t-i} r_i^2 (approx, from Chen's identity)
            # s1_time = Σ gamma^{t-i} dt

            s2_rr = phi[5]  # sum of gamma-weighted r^2
            s1_time = phi[0]  # sum of gamma-weighted dt

            if abs(s1_time) > 1e-12:
                # Realized variance per unit time
                rv = s2_rr / s1_time
            else:
                rv = self.kf_theta

            scale_obs.append(max(rv, 1e-8))

        # Kalman predict
        V_pred = self.V_filt + self.kf_kappa * (self.kf_theta - self.V_filt) * dt
        V_pred = max(V_pred, 1e-6)
        Q_kf = self.kf_xi**2 * max(self.V_filt, 1e-6) * dt
        P_pred = (1 - self.kf_kappa * dt)**2 * self.P_kf + Q_kf

        # Sequential Kalman update
        V_upd = V_pred
        P_upd = P_pred

        for i, (gamma, v_obs) in enumerate(zip(self.gammas, scale_obs)):
            N_eff = 1.0 / (1.0 - gamma)
            R_i = 2 * max(V_upd, 1e-6)**2 / (N_eff * dt)

            K_i = P_upd / (P_upd + R_i)
            V_upd = V_upd + K_i * (v_obs - V_upd)
            V_upd = max(V_upd, 1e-6)
            P_upd = (1 - K_i) * P_upd

        self.V_filt = V_upd
        self.P_kf = P_upd

        return self.V_filt, scale_obs


def run_sdre_direct_multiscale(p, T, seed, U_prime, U_double_prime,
                                known_mu=None, gammas=(0.80, 0.94, 0.99),
                                kf_xi=None):
    """SDRE with direct multi-scale signature RV + Kalman."""
    rng = np.random.RandomState(seed)

    xi_kf = kf_xi if kf_xi is not None else p.xi
    filt = DirectMultiScaleFilter(
        gammas=gammas, kf_kappa=p.kappa, kf_theta=p.theta, kf_xi=xi_kf)

    ewma_mu, lam_mu = 0.0, 0.999
    V_hat = np.zeros(T)
    V_scales = np.zeros((T, len(gammas)))
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        dx = np.array([p.dt, ret])
        v_fused, v_per_scale = filt.update(dx, ret, p.dt)
        V_hat[t] = v_fused
        V_scales[t] = v_per_scale

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)

        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi

        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {
        'V_hat': V_hat, 'V_scales': V_scales, 'pi_sdre': pi_sdre,
        'state_true': state_true, 'W_history': W_history,
    }


def run_sdre_levy_direct(p, T, seed, U_prime, U_double_prime,
                          known_mu=None, gamma_ll=0.99, kf_xi=None):
    """SDRE with direct Levy area as Kalman observation — no RLS.

    The Levy area of the lead-lag log-sig is analytically QV/2.
    We differentiate it: d(area)/dt gives instantaneous V estimate.
    This feeds directly into the CIR Kalman filter.

    Three variants tested by this function:
    - 'diff': Use d(2*area)/dt as observation (like r^2/dt but smoothed)
    - 'ewma_area': EWMA on the cumulative area, normalized by time weight
    """
    rng = np.random.RandomState(seed)
    kf_kappa, kf_theta = p.kappa, p.theta
    xi_kf = kf_xi if kf_xi is not None else p.xi
    V_filt = kf_theta
    P_kf = xi_kf**2 * kf_theta * p.dt * 10

    # Lead-lag log-sig (just for Levy area extraction)
    ll_sig = RecurrentLeadLagLogSigMap(
        state_dim=2, level=2, forgetting_factor=gamma_ll)
    # For d_input=2, lead-lag is 4D. Levy area indices for pairs (i<j):
    # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # ret_lead=1, ret_lag=3 -> index 4
    area_idx = 4
    prev_area = 0.0
    # Effective window for observation noise
    N_eff = 1.0 / (1.0 - gamma_ll) if gamma_ll < 1.0 - 1e-10 else 100.0

    ewma_mu, lam_mu = 0.0, 0.999
    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        # Update lead-lag log-sig
        dx = np.array([p.dt, ret])
        ll_sig.update(dx)
        area = ll_sig.l2[area_idx]

        # Differentiate: d(2*area) / dt ≈ instantaneous V
        # With decay gamma, area decays too, so diff captures local QV
        d_area = 2.0 * (area - gamma_ll**2 * prev_area) / p.dt
        prev_area = area
        y_obs = max(d_area, 1e-8)

        # Kalman predict (CIR dynamics)
        V_pred = max(V_filt + kf_kappa * (kf_theta - V_filt) * p.dt, 1e-6)
        Q_kf = xi_kf**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf

        # Observation noise: similar to r^2/dt but the lead-lag diff
        # has noise ~ 2V^2/dt (same as chi-sq observation)
        R_kf = 2 * max(V_pred, 1e-6)**2 / p.dt
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred
        V_hat[t] = V_filt

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)
        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi
        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
            'W_history': W_history}


def run_sdre_levy_ewma(p, T, seed, U_prime, U_double_prime,
                        known_mu=None, gamma_ll=0.99, kf_xi=None):
    """SDRE using EWMA-normalized Levy area as Kalman observation.

    Instead of differentiating the area, we normalize the cumulative area
    by the cumulative time weight: V_obs = 2*area / time_weight.
    This gives a smoother observation (EWMA of QV).
    """
    rng = np.random.RandomState(seed)
    kf_kappa, kf_theta = p.kappa, p.theta
    xi_kf = kf_xi if kf_xi is not None else p.xi
    V_filt = kf_theta
    P_kf = xi_kf**2 * kf_theta * p.dt * 10

    ll_sig = RecurrentLeadLagLogSigMap(
        state_dim=2, level=2, forgetting_factor=gamma_ll)
    area_idx = 4
    N_eff = 1.0 / (1.0 - gamma_ll) if gamma_ll < 1.0 - 1e-10 else 100.0

    # Track decayed time sum for normalization
    time_sum = 0.0

    ewma_mu, lam_mu = 0.0, 0.999
    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        dx = np.array([p.dt, ret])
        ll_sig.update(dx)
        area = ll_sig.l2[area_idx]

        # Decayed time sum (matches the decay in the area)
        time_sum = gamma_ll**2 * time_sum + p.dt

        # Normalized area = EWMA of r^2
        if time_sum > 1e-12:
            y_obs = max(2.0 * area / time_sum, 1e-8)
        else:
            y_obs = kf_theta

        # Kalman predict
        V_pred = max(V_filt + kf_kappa * (kf_theta - V_filt) * p.dt, 1e-6)
        Q_kf = xi_kf**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf

        # Observation noise: EWMA averages N_eff returns
        R_kf = 2 * max(V_pred, 1e-6)**2 / (N_eff * p.dt)
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred
        V_hat[t] = V_filt

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)
        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi
        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
            'W_history': W_history}


def run_sdre_blr_kalman(p, T, seed, U_prime, U_double_prime,
                         known_mu=None, gamma_ll=0.99, kf_xi=None,
                         n_features='qv+ret', obs_noise_init=0.01):
    """SDRE with Bayesian Linear Regression on log-sig features + Kalman on V.

    Two-level Bayesian hierarchy:
      Inner: BLR on log-sig features -> predictive mean + variance for V
      Outer: Scalar Kalman on V using BLR prediction as observation

    The BLR posterior variance gives a PRINCIPLED observation noise R
    for the Kalman filter — no ad-hoc formula needed.

    BLR is equivalent to online KRR with uncertainty tracking:
      Prior: w ~ N(0, sigma_w^2 I)
      Likelihood: y_t = w' phi_t + eps, eps ~ N(0, sigma_n^2)
      Posterior: updated via Kalman equations on w

    Args:
        n_features: 'qv+ret' (3 params), 'qv_only' (2), 'all' (11)
    """
    rng = np.random.RandomState(seed)
    kf_kappa, kf_theta = p.kappa, p.theta
    xi_kf = kf_xi if kf_xi is not None else p.xi
    V_filt = kf_theta
    P_kf = xi_kf**2 * kf_theta * p.dt * 10

    # Lead-lag log-sig
    ll_sig = RecurrentLeadLagLogSigMap(
        state_dim=2, level=2, forgetting_factor=gamma_ll)
    # Feature indices: area(ret_lead, ret_lag)=4, ret_lead=1
    area_idx = 4 + 4  # offset by level-1 dim (4)
    ret_idx = 1

    # BLR state (Kalman on regression weights)
    if n_features == 'all':
        nf = ll_sig.feature_dim + 1  # 10 + bias
    elif n_features == 'qv+ret':
        nf = 3  # QV area + ret_lead + bias
    else:  # qv_only
        nf = 2  # QV area + bias

    # BLR prior: w ~ N(0, sigma_w^2 I)
    sigma_w2 = 10.0  # broad prior on weights
    w_mean = np.zeros(nf)
    w_cov = np.eye(nf) * sigma_w2

    # Observation noise for BLR (variance of r^2/dt | V)
    # This is a hyperparameter; we use a running estimate
    sigma_n2 = obs_noise_init

    ewma_mu, lam_mu = 0.0, 0.999
    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    # For online noise estimation
    sq_err_ewma = obs_noise_init

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        # Extract features from lead-lag log-sig
        dx = np.array([p.dt, ret])
        feat_full = ll_sig.update(dx)

        if n_features == 'all':
            phi = np.concatenate([feat_full, [1.0]])
        elif n_features == 'qv+ret':
            phi = np.array([feat_full[area_idx], feat_full[ret_idx], 1.0])
        else:
            phi = np.array([feat_full[area_idx], 1.0])

        # --- BLR: predictive distribution ---
        # Predictive mean: y_hat = w_mean' phi
        y_pred = np.dot(w_mean, phi)

        # Predictive variance: sigma_pred^2 = phi' w_cov phi + sigma_n^2
        pred_var = phi @ w_cov @ phi + sigma_n2

        # --- BLR: posterior update (Kalman on weights) ---
        target = min(ret ** 2 / p.dt, 2.0)
        error = target - y_pred

        # Kalman gain for weights
        Cp = w_cov @ phi  # nf vector
        S = pred_var  # scalar innovation variance
        K_w = Cp / S  # nf vector

        # Update weight posterior
        w_mean = w_mean + K_w * error
        w_cov = w_cov - np.outer(K_w, Cp)
        # Symmetrize for numerical stability
        w_cov = 0.5 * (w_cov + w_cov.T)

        # Online noise estimation (EWMA of squared residuals)
        sq_err_ewma = 0.99 * sq_err_ewma + 0.01 * error ** 2
        sigma_n2 = max(sq_err_ewma, 1e-6)

        # --- Outer Kalman on V ---
        # Use BLR predictive mean as observation, predictive variance as R
        y_obs = max(y_pred, 1e-8)
        R_kf = max(pred_var, 1e-8)  # principled R from BLR

        # Kalman predict (CIR dynamics)
        V_pred = max(V_filt + kf_kappa * (kf_theta - V_filt) * p.dt, 1e-6)
        Q_kf = xi_kf**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf

        # Kalman update
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred
        V_hat[t] = V_filt

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)
        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi
        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
            'W_history': W_history}


def compare_all(p, n_seeds=5):
    """Compare multi-scale vs single-scale Kalman."""
    U_p, U_pp = _make_crra(p.gamma)
    test_s = 1000
    known_mu = p.mu - p.r

    methods = {
        # Baselines
        'Kalman(xi=0.3)': lambda s: run_sdre_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, xi_kf=0.3),
        'Kalman(xi=2.0)': lambda s: run_sdre_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, xi_kf=2.0),
        # LL + RLS + Kalman (current approach)
        'LL99+RLS+KF': lambda s: run_sdre_multiscale(
            p, T, s, U_p, U_pp, known_mu=known_mu,
            gammas=(0.99,), sig_type='leadlag'),
        'LL94+RLS+KF': lambda s: run_sdre_multiscale(
            p, T, s, U_p, U_pp, known_mu=known_mu,
            gammas=(0.94,), sig_type='leadlag'),
        # LL direct Levy area diff -> Kalman (no RLS)
        'LevyDiff99': lambda s: run_sdre_levy_direct(
            p, T, s, U_p, U_pp, known_mu=known_mu, gamma_ll=0.99),
        'LevyDiff94': lambda s: run_sdre_levy_direct(
            p, T, s, U_p, U_pp, known_mu=known_mu, gamma_ll=0.94),
        'LevyDiff99+xi1': lambda s: run_sdre_levy_direct(
            p, T, s, U_p, U_pp, known_mu=known_mu, gamma_ll=0.99, kf_xi=1.0),
        # LL EWMA-normalized area -> Kalman (no RLS)
        'LevyEWMA99': lambda s: run_sdre_levy_ewma(
            p, T, s, U_p, U_pp, known_mu=known_mu, gamma_ll=0.99),
        'LevyEWMA99+xi1': lambda s: run_sdre_levy_ewma(
            p, T, s, U_p, U_pp, known_mu=known_mu, gamma_ll=0.99, kf_xi=1.0),
        # BLR (Bayesian Linear Regression) on log-sig -> Kalman
        'BLR(qv+ret)': lambda s: run_sdre_blr_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_features='qv+ret'),
        'BLR(qv)': lambda s: run_sdre_blr_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_features='qv_only'),
        'BLR(all)': lambda s: run_sdre_blr_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_features='all'),
        'BLR(qv+ret)+xi1': lambda s: run_sdre_blr_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_features='qv+ret',
            kf_xi=1.0),
    }

    v_corrs = {m: [] for m in methods}
    pi_corrs = {m: [] for m in methods}
    pi_stds = {m: [] for m in methods}
    v_mses = {m: [] for m in methods}

    for seed in range(0, n_seeds * 1000, 1000):
        for name, fn in methods.items():
            res = fn(seed)
            V_true = res['state_true'][test_s:]
            V_h = res['V_hat'][test_s:]
            pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)

            v_corrs[name].append(np.corrcoef(V_h, V_true)[0, 1])
            v_mses[name].append(np.mean((V_h - V_true)**2))
            pi_corrs[name].append(
                np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1])
            pi_stds[name].append(np.std(res['pi_sdre'][test_s:]))

    print(f"\nMulti-seed comparison ({n_seeds} seeds, known mu):")
    print(f"{'Method':<22} {'V_corr':>14} {'V_MSE':>14} {'pi_corr':>14} {'pi_std':>14}")
    print("-" * 80)
    for m in methods:
        vc = f"{np.mean(v_corrs[m]):.3f}+/-{np.std(v_corrs[m]):.3f}"
        vm = f"{np.mean(v_mses[m]):.6f}"
        pc = f"{np.mean(pi_corrs[m]):.3f}+/-{np.std(pi_corrs[m]):.3f}"
        ps = f"{np.mean(pi_stds[m]):.3f}+/-{np.std(pi_stds[m]):.3f}"
        print(f"{m:<22} {vc:>14} {vm:>14} {pc:>14} {ps:>14}")

    # --- Visualization: single seed time series ---
    seed = 42
    t_days = np.arange(T) / 252

    # Run multi-scale for detailed output
    res_ms = run_sdre_multiscale(
        p, T, seed, U_p, U_pp, known_mu=known_mu,
        gammas=(0.80, 0.94, 0.99))
    res_k03 = run_sdre_kalman(p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=0.3)
    res_k20 = run_sdre_kalman(p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=2.0)

    V_true = res_ms['state_true']
    pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Panel 1: Per-scale V estimates + fused
    ax = axes[0]
    ax.plot(t_days[test_s:], np.sqrt(V_true[test_s:]),
            alpha=0.3, lw=0.5, color='gray', label='True $\\sqrt{V}$')
    scale_labels = ['$\\gamma$=0.80 (5d)', '$\\gamma$=0.94 (17d)', '$\\gamma$=0.99 (100d)']
    for i, lbl in enumerate(scale_labels):
        vs = res_ms['V_scales'][test_s:, i]
        corr = np.corrcoef(vs, V_true[test_s:])[0, 1]
        ax.plot(t_days[test_s:], np.sqrt(vs),
                alpha=0.5, lw=0.6, label=f'{lbl} ({corr:.3f})')
    corr_f = np.corrcoef(res_ms['V_hat'][test_s:], V_true[test_s:])[0, 1]
    ax.plot(t_days[test_s:], np.sqrt(res_ms['V_hat'][test_s:]),
            alpha=0.9, lw=1.5, color='black', label=f'Fused ({corr_f:.3f})')
    ax.set_ylabel('$\\sqrt{V}$')
    ax.set_title('Multi-Scale Signature Observations + Kalman Fusion')
    ax.legend(loc='upper right', fontsize=9)

    # Panel 2: Fused vs Kalman baselines
    ax = axes[1]
    ax.plot(t_days[test_s:], np.sqrt(V_true[test_s:]),
            alpha=0.3, lw=0.5, color='gray', label='True $\\sqrt{V}$')
    for lbl, res in [('Kalman($\\xi$=0.3)', res_k03), ('Kalman($\\xi$=2.0)', res_k20),
                     ('MultiScale(3)', res_ms)]:
        corr = np.corrcoef(res['V_hat'][test_s:], V_true[test_s:])[0, 1]
        ax.plot(t_days[test_s:], np.sqrt(res['V_hat'][test_s:]),
                alpha=0.8, lw=1.0, label=f'{lbl} ({corr:.3f})')
    ax.set_ylabel('$\\sqrt{V}$')
    ax.set_title('Fused Multi-Scale vs Single-Scale Kalman')
    ax.legend(loc='upper right', fontsize=9)

    # Panel 3: Policy comparison
    ax = axes[2]
    ax.plot(t_days[test_s:], pi_merton[test_s:],
            alpha=0.3, lw=0.5, color='gray', label='Merton $\\pi^*$')
    for lbl, res in [('Kalman($\\xi$=0.3)', res_k03), ('Kalman($\\xi$=2.0)', res_k20),
                     ('MultiScale(3)', res_ms)]:
        corr = np.corrcoef(res['pi_sdre'][test_s:], pi_merton[test_s:])[0, 1]
        ax.plot(t_days[test_s:], res['pi_sdre'][test_s:],
                alpha=0.8, lw=0.8, label=f'{lbl} ({corr:.3f})')
    ax.set_ylabel('$\\pi$')
    ax.set_xlabel('Years')
    ax.set_title('SDRE Policy')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 3.0)

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_multiscale.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_multiscale.png")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    method_names = list(methods.keys())
    x = np.arange(len(method_names))

    for ax_idx, (metric, label) in enumerate([
            (v_corrs, 'corr($\\hat{V}$, $V_{true}$)'),
            (pi_corrs, 'corr($\\pi_{SDRE}$, $\\pi_{Merton}$)'),
            (pi_stds, 'std($\\pi_{SDRE}$)')]):
        ax = axes[ax_idx]
        means = [np.mean(metric[m]) for m in method_names]
        stds = [np.std(metric[m]) for m in method_names]
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        ax.bar(x, means, 0.6, yerr=stds, capsize=3, color=colors[:len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=25, fontsize=8, ha='right')
        ax.set_ylabel(label)

    plt.suptitle('Multi-Scale Signature SDRE vs Baselines', fontsize=13)
    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_multiscale_bars.png',
                bbox_inches='tight')
    plt.close()
    print("Saved: level4_multiscale_bars.png")


if __name__ == '__main__':
    print("=== Multi-Scale Signature SDRE ===\n")
    compare_all(P, n_seeds=5)
    print("\n=== Done ===")
