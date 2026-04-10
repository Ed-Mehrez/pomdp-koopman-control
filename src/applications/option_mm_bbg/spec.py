"""BBG benchmark parameter containers.

All parameters match Baldacci, Bergault & Gueant (2020), Section 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class BBGHestonSpec:
    """Heston dynamics under both P and Q measures."""

    spot0: float = 10.0
    nu0: float = 0.0225
    kappa_p: float = 2.0
    theta_p: float = 0.04
    kappa_q: float = 3.0
    theta_q: float = 0.0225
    xi: float = 0.2
    rho: float = -0.5
    rate: float = 0.0  # assumed zero in the paper

    def a_p(self, nu: float) -> float:
        """P-measure drift of nu."""
        return self.kappa_p * (self.theta_p - nu)

    def a_q(self, nu: float) -> float:
        """Q-measure drift of nu."""
        return self.kappa_q * (self.theta_q - nu)

    def validate(self) -> None:
        if self.spot0 <= 0.0:
            raise ValueError("spot0 must be positive")
        if self.nu0 < 0.0:
            raise ValueError("nu0 must be nonneg")
        if self.xi < 0.0:
            raise ValueError("xi must be nonneg")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1,1]")


@dataclass(frozen=True)
class BBGOptionSpec:
    """One option in the book."""

    strike: float
    maturity: float  # years to expiry

    def validate(self) -> None:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")


@dataclass(frozen=True)
class BBGOptionBookSpec:
    """Full option book: strikes x maturities."""

    strikes: tuple[float, ...] = (8.0, 9.0, 10.0, 11.0, 12.0)
    maturities: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

    @property
    def options(self) -> list[BBGOptionSpec]:
        return [
            BBGOptionSpec(strike=k, maturity=t)
            for k in self.strikes
            for t in self.maturities
        ]

    @property
    def n_options(self) -> int:
        return len(self.strikes) * len(self.maturities)


@dataclass(frozen=True)
class BBGLiquiditySpec:
    """Logistic intensity and Dirac trade sizes from BBG Section 4.1.

    Intensity: Lambda^{i,j}(delta) = lambda_i / (1 + exp(alpha + beta * delta / V_i))
    Trade size: z_i = notional / O_i_0  (Dirac mass)
    """

    alpha: float = 0.7
    beta: float = 150.0  # year^{1/2}
    lambda_base_rate: float = 30.0  # trades per day for ATM
    lambda_moneyness_decay: float = 0.7
    notional_per_trade: float = 5.0e5  # EUR per transaction (paper: ~500,000€)
    trading_days_per_year: float = 252.0

    def lambda_i(self, spot0: float, strike: float) -> float:
        """Annualized base intensity for option with given strike."""
        return (
            self.trading_days_per_year
            * self.lambda_base_rate
            / (1.0 + self.lambda_moneyness_decay * abs(spot0 - strike))
        )

    def trade_size(self, option_price: float) -> float:
        """Number of contracts per transaction (Dirac mass)."""
        if option_price <= 0.0:
            return 0.0
        return self.notional_per_trade / option_price

    def intensity(
        self,
        delta: float,
        lambda_i: float,
        vega_i: float,
    ) -> float:
        """Logistic intensity at quote distance delta for option i."""
        if lambda_i <= 0.0 or vega_i <= 0.0:
            return 0.0
        arg = self.alpha + self.beta * delta / vega_i
        arg = float(np.clip(arg, -700.0, 700.0))
        return lambda_i / (1.0 + np.exp(arg))


@dataclass(frozen=True)
class BBGControlSpec:
    """Control horizon and risk parameters."""

    horizon: float = 0.0012  # years (~0.3 trading day)
    gamma: float = 1.0e-3    # EUR^{-1}
    vega_limit: float = 1.0e7  # EUR * year^{1/2}


@dataclass(frozen=True)
class BBGBenchmarkConfig:
    """Complete benchmark configuration."""

    heston: BBGHestonSpec = field(default_factory=BBGHestonSpec)
    book: BBGOptionBookSpec = field(default_factory=BBGOptionBookSpec)
    liquidity: BBGLiquiditySpec = field(default_factory=BBGLiquiditySpec)
    control: BBGControlSpec = field(default_factory=BBGControlSpec)

    @classmethod
    def paper_default(cls) -> "BBGBenchmarkConfig":
        """Return the exact configuration from BBG Section 4.1."""
        return cls()

    def validate(self) -> None:
        self.heston.validate()
        for opt in self.book.options:
            opt.validate()
        if self.control.horizon <= 0.0:
            raise ValueError("horizon must be positive")
        if self.control.gamma < 0.0:
            raise ValueError("gamma must be nonneg")
        if self.control.vega_limit <= 0.0:
            raise ValueError("vega_limit must be positive")
