We are moving the control/RL exploration to `pomdp-koopman-control`.

Please act as the skeptical research-design reviewer, not the implementer.

Context:
The sibling `koopman-pricing` repo found that scalar Heston-style Q-transfer worked in homogeneous simulations but failed on Alpaca real options data. Real-data experiments showed:

- premium persistence is a very strong baseline
- OHLC features add only modest incremental value
- signatures did not help as direct premium regressors
- strict cross-asset scalar transfer underperformed the RV null

We are considering a pivot from pure option-price prediction to adaptive control / market making under partial observation.

I do NOT want generic deep RL first.

The proposed ordering is:

1. inventory-aware analytic controller
2. belief-state + SDRE / KRONIC controller
3. only later, domain-randomized RL / meta-RL if structured control baselines leave clear room

Please review this direction skeptically.

Focus on:

- whether SDRE/KRONIC is a coherent fit for option market making / hedging
- what the low-dimensional state and action should be
- what baselines would make the benchmark fair
- what leakage or hindsight traps to avoid in historical Alpaca replay
- whether a simulator-first or hybrid simulator-to-replay protocol is better
- how to phrase this so it does not overclaim pricing accuracy when the actual objective is wealth/inventory control

Important:
Some old results in `pomdp-koopman-control` may be WIP, have baseline issues, or contain leakage/discretization mistakes. Please do not assume previous reported wins are valid.

I want your concise design critique and recommended MVP before any implementation.

Please include:

- what should be built first
- what should definitely wait
- what success metrics matter
- and what result would convince you the control framing is genuinely adding value beyond a pricing model plus simple inventory rule.
