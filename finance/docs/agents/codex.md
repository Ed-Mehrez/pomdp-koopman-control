We are switching context from the sibling repo `koopman-pricing` to this repo, `pomdp-koopman-control`.

Background:
In `koopman-pricing`, a scalar Heston-style Q-transfer model worked well in homogeneous Heston simulations and produced strong simulated option-pricing errors, but it did not transfer well to Alpaca real options data. Real-data experiments showed:

- premium persistence is a very strong baseline
- OHLC features add only modest incremental value
- signatures did not help as direct premium regressors
- the strict cross-asset scalar transfer model underperformed the RV null

So I want to explore a different framing:
not pure option-price prediction, but **adaptive control / market making under partial observation**, where the pricing model is only a belief input and the actual objective is controlled wealth / inventory-adjusted P&L.

Important:
Please audit this repo skeptically. Some previous results may be WIP or have baseline / leakage / discretization issues. Do not assume reported wins are valid without checking.

I do NOT want generic deep RL first.

My preferred order is:

1. inventory-aware analytic controller
2. belief-state + SDRE / KRONIC controller
3. only later, domain-randomized RL / meta-RL if the structured control baselines leave clear room

The main question:

Can we use the POMDP / signature-belief / Koopman-SDRE tooling here to build a small option market-making or hedging-control benchmark where the method is evaluated by a wealth process, not IV RMSE?

Please first produce a candid design proposal before implementing.

Please inspect at least:

- `src/sskf/streaming_sig_kkf.py`
- `src/finance/inventory_averse_mm.py`
- `src/finance/adaptive_kyle.py`
- `kronic_pomdp/`
- any SDRE / KRONIC controller files in this repo

I want you to answer:

1. What code is actually reusable?
   Separate:
   - production-ish / reusable
   - prototype-only
   - likely misleading or too WIP

2. What is the smallest credible benchmark?
   My current guess:
   - option market-making or option hedging simulator
   - state/belief includes fair IV estimate, uncertainty, inventory, hedge error, realized vol regime, market VRP factor
   - action includes quote offset, spread multiplier, hedge ratio, or inventory target
   - reward is spread capture + mark-to-market P&L + hedge P&L − inventory penalty − transaction costs

3. How should SDRE/KRONIC enter?
   I want a concrete low-dimensional state equation if possible.
   For example:
   - state = [pricing_error_belief, inventory, hedge_error, vol_regime, uncertainty]
   - action = [quote_offset, hedge_ratio]
   - running cost = inventory² + hedge_error² + uncertainty-scaled position risk − spread capture
     Then use SDRE / local Koopman control to compute the action.

4. What are the baselines?
   At minimum:
   - no-trade / no-inventory
   - fixed quote width + fixed hedge rule
   - premium_AR fair value + inventory rule
   - Factor-Q-head fair value + inventory rule, if we can import from `koopman-pricing`
   - inventory-averse market maker already in this repo
   - SDRE/KRONIC controller

5. What data environment should we use first?
   Please recommend one:
   - pure simulator first
   - historical Alpaca replay from the sibling `koopman-pricing` cache
   - or hybrid: train/tune in simulator, evaluate on Alpaca replay

My instinct:
start with pure simulator or hybrid, not raw Alpaca only.

6. What is the role of signatures?
   I do not want another direct `signature -> premium` regression.
   Use signatures only if they are serving as:
   - belief-state summaries
   - uncertainty/state filters
   - or online path features for the controller

7. What metrics should determine success?
   Please propose metrics like:
   - certainty-equivalent P&L
   - mean P&L
   - P&L volatility
   - max drawdown
   - inventory distribution
   - hedge error
   - fill-adjusted spread capture
   - not just pricing RMSE

8. What should wait?
   Please explicitly say what not to do in v1.
   My expectation:
   - no SAC / deep RL yet
   - no high-dimensional action space
   - no live trading
   - no full options surface market making
   - no untested import of all old POMDP results

Please return:

- a short repo audit
- the proposed MVP benchmark
- the SDRE/KRONIC formulation if feasible
- exact files you would modify or add
- and a recommendation on whether to implement v1 immediately.
