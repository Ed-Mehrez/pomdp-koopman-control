"""Black-Scholes pricing helpers for the BBG benchmark.

Options are priced under Q using BS with variance = nu_t.
Vega is defined as V^i = d_{sqrt(nu)} O^i = 2 sqrt(nu) d_nu O^i.
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt

import numpy as np


SQRT_TWO = sqrt(2.0)
SQRT_TWO_PI = sqrt(2.0 * pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT_TWO))


def _normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / SQRT_TWO_PI


def bs_call_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    variance: float,
) -> float:
    """European call price under BS with instantaneous variance."""
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if tau <= 0.0:
        return max(spot - strike, 0.0)
    sigma = sqrt(max(variance, 1e-16))
    vol_sqrt_tau = sigma * sqrt(tau)
    if vol_sqrt_tau <= 1e-12:
        return max(spot - strike * exp(-rate * tau), 0.0)
    d1 = (log(spot / strike) + (rate + 0.5 * variance) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return spot * _normal_cdf(d1) - strike * exp(-rate * tau) * _normal_cdf(d2)


def bs_call_delta(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    variance: float,
) -> float:
    """European call delta."""
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if tau <= 0.0:
        return 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
    sigma = sqrt(max(variance, 1e-16))
    vol_sqrt_tau = sigma * sqrt(tau)
    if vol_sqrt_tau <= 1e-12:
        return 1.0 if spot > strike * exp(-rate * tau) else 0.0
    d1 = (log(spot / strike) + (rate + 0.5 * variance) * tau) / vol_sqrt_tau
    return _normal_cdf(d1)


def bs_call_vega_sqrt_nu(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    variance: float,
) -> float:
    """BBG vega: V^i = d_{sqrt(nu)} O = 2 sqrt(nu) * d_nu O.

    This is spot * sqrt(tau) * phi(d1), the standard BS vega per share.
    BBG defines V^i = d_{sqrt(nu)} O^i(t, S, nu), which for BS call equals
    S * sqrt(tau) * phi(d1).
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if tau <= 0.0:
        return 0.0
    sigma = sqrt(max(variance, 1e-16))
    vol_sqrt_tau = sigma * sqrt(tau)
    if vol_sqrt_tau <= 1e-12:
        return 0.0
    d1 = (log(spot / strike) + (rate + 0.5 * variance) * tau) / vol_sqrt_tau
    return spot * sqrt(tau) * _normal_pdf(d1)


def price_option_book(
    spot: float,
    variance: float,
    rate: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Price all options in the book.

    Returns (prices, deltas, vegas) each of shape (n_options,).
    Options are ordered as strikes x maturities (outer x inner).
    """
    n = strikes.shape[0]
    prices = np.empty(n)
    deltas = np.empty(n)
    vegas = np.empty(n)

    for i in range(n):
        prices[i] = bs_call_price(spot, strikes[i], maturities[i], rate, variance)
        deltas[i] = bs_call_delta(spot, strikes[i], maturities[i], rate, variance)
        vegas[i] = bs_call_vega_sqrt_nu(spot, strikes[i], maturities[i], rate, variance)

    return prices, deltas, vegas
