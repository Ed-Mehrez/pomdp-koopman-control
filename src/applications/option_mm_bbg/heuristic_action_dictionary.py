"""Heuristic action directions for interpretability validation.

These are economically motivated directions in the 40D quote-distance space.
They are used ONLY for post-hoc alignment diagnostics, never during training.

Directions:
  1. Global width:  scale all bid+ask symmetrically
  2. Portfolio-vega skew:  widen bids / tighten asks (or vice versa),
     weighted by z_i V_i to maximally affect V^pi
  3. Maturity tilt:  widen long-dated, tighten short-dated
  4. Moneyness tilt:  widen OTM, tighten ATM (or vice versa)
"""

from __future__ import annotations

import numpy as np

from .pricing import bs_call_price, bs_call_vega_sqrt_nu
from .spec import BBGBenchmarkConfig


def build_heuristic_dictionary(config: BBGBenchmarkConfig) -> dict[str, np.ndarray]:
    """Return named unit-norm heuristic directions in R^{2N} action space.

    Action ordering: (bid_1, ..., bid_N, ask_1, ..., ask_N).
    """
    h = config.heston
    opts = config.book.options
    n = config.book.n_options

    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in opts
    ])
    prices = np.array([
        bs_call_price(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in opts
    ])
    trade_sizes = np.array([config.liquidity.trade_size(p) for p in prices])

    strikes = np.array([o.strike for o in opts])
    mats = np.array([o.maturity for o in opts])

    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-15 else v * 0.0

    # 1. Global width: increase all bid and ask distances equally
    d_width = np.ones(2 * n)
    d_width = _normalize(d_width)

    # 2. Portfolio-vega skew: widen bids, tighten asks, weighted by z_i V_i
    #    (+bid → fewer buys → V^pi decreases, -ask → more sells → V^pi decreases)
    w_vega = trade_sizes * vegas
    w_vega_norm = w_vega / (np.linalg.norm(w_vega) + 1e-15)
    d_skew = np.concatenate([w_vega_norm, -w_vega_norm])
    d_skew = _normalize(d_skew)

    # 3. Maturity tilt: widen long-dated, tighten short-dated
    mat_centered = mats - mats.mean()
    d_mat = np.concatenate([mat_centered, mat_centered])
    d_mat = _normalize(d_mat)

    # 4. Moneyness tilt: widen OTM (far from spot), tighten ATM
    money_dist = np.abs(strikes - h.spot0)
    money_centered = money_dist - money_dist.mean()
    d_money = np.concatenate([money_centered, money_centered])
    d_money = _normalize(d_money)

    return {
        "global_width": d_width,
        "vega_skew": d_skew,
        "maturity_tilt": d_mat,
        "moneyness_tilt": d_money,
    }


def principal_angles(
    U: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Principal angles (radians) between subspaces spanned by columns of U and V.

    Uses the SVD of U^T V to compute cos(theta_i).
    """
    # Orthonormalize
    Qu, _ = np.linalg.qr(U, mode="reduced")
    Qv, _ = np.linalg.qr(V, mode="reduced")
    M = Qu.T @ Qv
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)


def projection_fraction(
    U: np.ndarray,
    v: np.ndarray,
) -> float:
    """Fraction of v that lies in the column span of U: ||P_U v||^2 / ||v||^2."""
    v_norm_sq = np.dot(v, v)
    if v_norm_sq < 1e-30:
        return 0.0
    Q, _ = np.linalg.qr(U, mode="reduced")
    proj = Q @ (Q.T @ v)
    return float(np.dot(proj, proj) / v_norm_sq)
