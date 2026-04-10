"""BBG benchmark scale audit.

For the paper-default config, print per-option: price, vega, lambda,
trade size z_i, z_i*V_i, and whether a single fill from zero inventory
is admissible under the vega risk limit V_bar.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.pricing import (
    bs_call_price,
    bs_call_vega_sqrt_nu,
)


def main() -> int:
    config = BBGBenchmarkConfig.paper_default()
    h = config.heston
    liq = config.liquidity
    ctrl = config.control

    options = config.book.options
    N = config.book.n_options

    out: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        out.append(msg)

    log("=" * 90)
    log("  BBG Scale Audit — paper_default()")
    log("=" * 90)
    log(f"  notional_per_trade = {liq.notional_per_trade:.0f} EUR")
    log(f"  V_bar = {ctrl.vega_limit:.0e}")
    log()

    header = (
        f"{'K':>5s} {'T':>5s} {'O0':>8s} {'V_i':>8s} {'lam_i':>8s} "
        f"{'z_i':>12s} {'z_i*V_i':>14s} {'admiss':>8s}"
    )
    log(header)
    log("-" * len(header))

    prices = []
    vegas = []
    lambdas = []
    trade_sizes = []
    vega_jumps = []
    admissible = []

    for opt in options:
        p = bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        v = bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        lam = liq.lambda_i(h.spot0, opt.strike)
        z = liq.trade_size(p)
        zv = z * v
        adm = abs(zv) <= ctrl.vega_limit

        prices.append(p)
        vegas.append(v)
        lambdas.append(lam)
        trade_sizes.append(z)
        vega_jumps.append(zv)
        admissible.append(adm)

        log(
            f"{opt.strike:>5.0f} {opt.maturity:>5.1f} {p:>8.3f} {v:>8.3f} {lam:>8.0f} "
            f"{z:>12.0f} {zv:>14.0f} {'YES' if adm else '**NO**':>8s}"
        )

    log()
    log("  Summary:")
    log(f"    N options:         {N}")
    log(f"    price range:       [{min(prices):.3f}, {max(prices):.3f}]")
    log(f"    vega range:        [{min(vegas):.3f}, {max(vegas):.3f}]")
    log(f"    lambda range:      [{min(lambdas):.0f}, {max(lambdas):.0f}]")
    log(f"    z_i range:         [{min(trade_sizes):.0f}, {max(trade_sizes):.0f}]")
    log(f"    z_i*V_i range:     [{min(vega_jumps):.0f}, {max(vega_jumps):.0f}]")
    log(f"    V_bar:             {ctrl.vega_limit:.0f}")
    log(f"    count z_i*V_i > V_bar: {sum(not a for a in admissible)} of {N}")
    log(f"    max z_i*V_i / V_bar:   {max(vega_jumps) / ctrl.vega_limit:.3f}")

    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_scale_audit_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\n  Saved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
