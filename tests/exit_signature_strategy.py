"""
Exit-side log-signature filter for ride-bubble strategy.

Instead of a fixed vol_P exit threshold, use the cumulative (γ=1.0)
lead-lag log-sig of (P, vol_P) to detect when a smooth bubble turns rough.

Within-trade signature dynamics (all causal, no look-ahead):
  - QV_trade = A(P_lead, P_lag)(t) - A(P_lead, P_lag)(entry)
             = Σ (ΔP)² during the trade — cumulative roughness
  - disp_trade = P(t) - P(entry) — net P change during trade
  - roughness = QV_trade / max(disp_trade², eps) — how noisy vs trending

Exit signals tested:
  1. roughness > threshold (P got choppy relative to its trend)
  2. QV rate = QV_trade / duration > threshold (absolute choppiness)
  3. disp reversal: P_lead starts decreasing (bubble deflating)
  4. Combined: original vol_P exit OR signature exit (first to fire)

Uses cumulative signatures (γ=1.0) throughout — no forgetting, no decay.
"""

import numpy as np
from scipy import stats as sp_stats
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'examples', 'proof_of_concept'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kronic_pomdp', 'experiments'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'sskf'))

from signature_features import RecurrentLeadLagLogSigMap
from bubble_trading_strategies import (
    TICKERS, DT_5MIN, STEP_DAYS, load_5min, load_market_iv,
    run_streaming, add_empirical_vol_p, trade_stats,
)


# ═══════════════════════════════════════════════════════════════════════
# Cumulative signature tracker for (P, vol_P)
# ═══════════════════════════════════════════════════════════════════════

class CumulativePSignature:
    """Maintain cumulative lead-lag log-sig of (P, vol_P).

    gamma=1.0 (no forgetting). Each update adds to the signature.

    For d=2 (P, vol_P), lead-lag gives 4D path.
    Level-2 log-sig: 4 level-1 + 6 level-2 = 10 features.

    Lead-lag channels: (P_lead, vP_lead, P_lag, vP_lag)
    Indices:            0       1        2      3

    Level-2 areas (i<j pairs):
      (0,1)→4  (0,2)→5  (0,3)→6  (1,2)→7  (1,3)→8  (2,3)→9

    Key features:
      feat[0], feat[2]: P displacement (lead, lag)
      feat[5]: A(P_lead, P_lag) = QV of P = Σ(dP)²
      feat[9]: A(vP_lead, vP_lag) = QV of vol(P)
    """

    IDX_P_LEAD = 0
    IDX_P_LAG = 2
    IDX_QV_P = 5        # A(P_lead, P_lag)
    IDX_QV_VP = 9       # A(vP_lead, vP_lag)
    IDX_CROSS_P_VP = 6  # A(P_lead, vP_lag) — how P and vol(P) co-rotate

    def __init__(self):
        self.sig = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=1.0)  # cumulative
        self.prev_p = None
        self.prev_vp = None
        # Running normalization: use expanding std of increments
        self.dp_history = []
        self.dvp_history = []
        self.dp_std = 0.05  # initial guess
        self.dvp_std = 0.05

    def reset(self):
        self.sig.reset()
        self.prev_p = None
        self.prev_vp = None
        self.dp_history = []
        self.dvp_history = []
        self.dp_std = 0.05
        self.dvp_std = 0.05

    def update(self, p_bubble, vol_p):
        """Update with new observation. Returns current feature vector."""
        if self.prev_p is None:
            self.prev_p = p_bubble
            self.prev_vp = vol_p
            return self.sig.get_features()

        dp = p_bubble - self.prev_p
        dvp = vol_p - self.prev_vp

        # Track raw increments for normalization
        self.dp_history.append(dp)
        self.dvp_history.append(dvp)

        # Update running std (expanding window)
        if len(self.dp_history) >= 5:
            self.dp_std = max(np.std(self.dp_history), 1e-6)
            self.dvp_std = max(np.std(self.dvp_history), 1e-6)

        # Normalize increments
        dp_norm = dp / self.dp_std
        dvp_norm = dvp / self.dvp_std

        feat = self.sig.update(np.array([dp_norm, dvp_norm]))
        self.prev_p = p_bubble
        self.prev_vp = vol_p
        return feat

    def get_features(self):
        return self.sig.get_features()

    @staticmethod
    def p_displacement(feat):
        return 0.5 * (feat[CumulativePSignature.IDX_P_LEAD] +
                       feat[CumulativePSignature.IDX_P_LAG])

    @staticmethod
    def p_qv(feat):
        return feat[CumulativePSignature.IDX_QV_P]

    @staticmethod
    def vp_qv(feat):
        return feat[CumulativePSignature.IDX_QV_VP]


# ═══════════════════════════════════════════════════════════════════════
# Strategy: ride bubble with signature-based exit
# ═══════════════════════════════════════════════════════════════════════

def ride_bubble_sigexit(ticker, prices, results,
                         p_entry=0.6, vol_p_entry_max=0.05,
                         p_exit=0.4, max_hold=60,
                         # Standard exit (can disable by setting high)
                         vol_p_exit=999.0,
                         # Signature exit params
                         roughness_exit=None,
                         qv_rate_exit=None,
                         disp_reversal_exit=False,
                         # Normalization
                         sig_gamma=1.0):
    """Ride bubble with signature-based exit signals.

    The cumulative signature runs over the ENTIRE results sequence.
    At trade entry, we snapshot the signature state.
    During the trade, we compute within-trade deltas (causal).

    Exit triggers (any one fires → exit):
      - vol_p_exit: standard vol(P) threshold (set high to disable)
      - roughness_exit: QV_trade / disp_trade² > threshold
      - qv_rate_exit: QV_trade / n_steps > threshold
      - disp_reversal_exit: P displacement went negative (bubble deflating)
      - p_exit: P dropped below threshold
      - max_hold: time limit
    """
    # Pre-compute cumulative signature at each result index
    sig_tracker = CumulativePSignature()
    sig_at = {}  # index → feature vector
    for idx, r in enumerate(results):
        feat = sig_tracker.update(r['p_bubble'], r['vol_p'])
        sig_at[idx] = feat.copy()

    trades = []
    n = len(results)
    i = 0

    while i < n:
        r = results[i]
        vol_p = r['vol_p']

        # Entry conditions (same as standard)
        if r['p_bubble'] <= p_entry or vol_p > vol_p_entry_max:
            i += 1
            continue

        S_entry = r['price']
        entry_feat = sig_at[i]
        entry_qv = CumulativePSignature.p_qv(entry_feat)
        entry_disp = CumulativePSignature.p_displacement(entry_feat)

        j = i + 1
        exit_reason = None

        while j < n:
            elapsed = (j - i) * STEP_DAYS
            rj = results[j]
            vp_j = rj['vol_p']

            # Compute within-trade signature deltas
            current_feat = sig_at[j]
            qv_trade = CumulativePSignature.p_qv(current_feat) - entry_qv
            disp_trade = CumulativePSignature.p_displacement(current_feat) - entry_disp
            n_steps = j - i

            # ── Exit checks (order matters: first to fire wins) ──

            # 1. Standard vol(P) exit
            if vp_j > vol_p_exit:
                exit_reason = 'vol_spike'

            # 2. Roughness exit: QV grew much faster than displacement²
            elif roughness_exit is not None and n_steps >= 2:
                roughness = qv_trade / max(disp_trade ** 2, 0.01)
                if roughness > roughness_exit:
                    exit_reason = 'sig_rough'

            # 3. QV rate exit: absolute choppiness per step
            elif qv_rate_exit is not None and n_steps >= 2:
                qv_rate = qv_trade / n_steps
                if qv_rate > qv_rate_exit:
                    exit_reason = 'sig_qv_rate'

            # 4. Displacement reversal: P trending down during trade
            elif disp_reversal_exit and n_steps >= 3:
                if disp_trade < -0.5:  # meaningful negative displacement
                    exit_reason = 'sig_disp_rev'

            # 5. P dropped
            elif rj['p_bubble'] < p_exit:
                exit_reason = 'p_dropped'

            # 6. Max hold
            elif elapsed >= max_hold:
                exit_reason = 'max_hold'

            if exit_reason:
                S_exit = rj['price']
                trades.append({
                    'ticker': ticker, 'strategy': 'ride_sigexit',
                    'entry_bar': r['bar'], 'exit_bar': rj['bar'],
                    'entry_price': S_entry, 'exit_price': S_exit,
                    'ret': (S_exit - S_entry) / S_entry,
                    'hold_days': elapsed, 'exit_reason': exit_reason,
                    'p_entry': r['p_bubble'], 'vol_p_entry': vol_p,
                    'qv_trade': qv_trade,
                    'disp_trade': disp_trade,
                    'roughness': qv_trade / max(disp_trade ** 2, 0.01),
                })
                i = j + 1
                break
            j += 1
        else:
            break

    return trades


def main():
    print("=" * 90)
    print("  EXIT-SIDE SIGNATURE STRATEGY")
    print("  Cumulative log-sig (γ=1.0), no look-ahead")
    print("=" * 90)

    # ── Load data ──
    print("\n  Loading data (9 years, 20 tickers)...")
    t0 = time.time()
    all_data = {}
    for ticker in TICKERS:
        prices, timestamps = load_5min(ticker)
        if prices is None:
            continue
        mkt_iv = load_market_iv(ticker)
        results = run_streaming(prices, timestamps, mkt_iv)
        if len(results) < 10:
            continue
        add_empirical_vol_p(results)
        all_data[ticker] = (prices, results)
    print(f"  Loaded {len(all_data)} tickers ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # 1. Characterize within-trade signature dynamics
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  WITHIN-TRADE SIGNATURE DYNAMICS")
    print(f"  How do QV, roughness, displacement evolve during trades?")
    print(f"{'─' * 90}")

    # Run baseline (standard exit) to get trade-level sig features
    base_trades = []
    for ticker, (prices, results) in all_data.items():
        trades = ride_bubble_sigexit(
            ticker, prices, results,
            vol_p_exit=0.30,  # standard exit
            roughness_exit=None, qv_rate_exit=None)
        base_trades.extend(trades)

    print(f"\n  Baseline (vol_P exit=0.30): {len(base_trades)} trades")

    # Split by profitability and compare sig features at exit
    profitable = [t for t in base_trades if t['ret'] > 0.02]
    unprofitable = [t for t in base_trades if t['ret'] < -0.02]

    print(f"  Profitable (>2%): {len(profitable)}")
    print(f"  Unprofitable (<-2%): {len(unprofitable)}")

    if profitable and unprofitable:
        for feat_name in ['qv_trade', 'disp_trade', 'roughness']:
            prof_vals = [t[feat_name] for t in profitable
                         if np.isfinite(t[feat_name])]
            unprof_vals = [t[feat_name] for t in unprofitable
                           if np.isfinite(t[feat_name])]
            if prof_vals and unprof_vals:
                stat, pv = sp_stats.mannwhitneyu(
                    prof_vals, unprof_vals, alternative='two-sided')
                print(f"\n  {feat_name}:")
                print(f"    Profitable:   mean={np.mean(prof_vals):+.3f}, "
                      f"median={np.median(prof_vals):+.3f}")
                print(f"    Unprofitable: mean={np.mean(unprof_vals):+.3f}, "
                      f"median={np.median(unprof_vals):+.3f}")
                print(f"    Mann-Whitney p={pv:.4f} "
                      f"{'***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.1 else ''}")

    # ── Correlation: early exit sig features vs return ──
    print(f"\n  Correlation of within-trade sig features with trade return:")
    for feat_name in ['qv_trade', 'disp_trade', 'roughness']:
        vals = np.array([t[feat_name] for t in base_trades
                         if np.isfinite(t[feat_name])])
        rets = np.array([t['ret'] for t in base_trades
                         if np.isfinite(t[feat_name])])
        if len(vals) >= 10:
            r, pv = sp_stats.spearmanr(vals, rets)
            print(f"    {feat_name:<15}: r={r:+.3f}, p={pv:.4f}")

    # ══════════════════════════════════════════════════════════════
    # 2. Walk-forward CV: test exit strategies
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  WALK-FORWARD CV: Exit Strategy Comparison")
    print(f"  5-fold expanding window, p_entry=0.6, vol_p_entry_max=0.05")
    print(f"{'─' * 90}")

    n_folds = 5
    base_params = {'p_entry': 0.6, 'vol_p_entry_max': 0.05}

    ticker_folds = {}
    for ticker, (prices, results) in all_data.items():
        n = len(results)
        fold_size = n // n_folds
        if fold_size < 5:
            continue
        folds = []
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else n
            folds.append(results[start:end])
        ticker_folds[ticker] = (prices, folds)

    # Define exit strategies to test
    exit_strategies = [
        # (label, vol_p_exit, roughness_exit, qv_rate_exit, disp_rev)
        ("vol_P=0.30 (baseline)",    0.30, None, None, False),
        ("vol_P=0.20",               0.20, None, None, False),
        ("vol_P=0.15",               0.15, None, None, False),
        ("rough≤5",                  999., 5.0,  None, False),
        ("rough≤10",                 999., 10.0, None, False),
        ("rough≤20",                 999., 20.0, None, False),
        ("rough≤50",                 999., 50.0, None, False),
        ("qv_rate≤0.5",             999., None, 0.5,  False),
        ("qv_rate≤1.0",             999., None, 1.0,  False),
        ("qv_rate≤2.0",             999., None, 2.0,  False),
        ("disp_reversal",           999., None, None, True),
        ("vol0.30 + rough≤10",      0.30, 10.0, None, False),
        ("vol0.30 + rough≤20",      0.30, 20.0, None, False),
        ("vol0.30 + rough≤50",      0.30, 50.0, None, False),
        ("vol0.30 + qv≤1.0",        0.30, None, 1.0,  False),
        ("vol0.30 + qv≤2.0",        0.30, None, 2.0,  False),
        ("vol0.30 + disp_rev",      0.30, None, None, True),
        ("vol0.20 + rough≤20",      0.20, 20.0, None, False),
        ("vol0.15 + rough≤10",      0.15, 10.0, None, False),
    ]

    print(f"\n  {'Strategy':<28} {'N':>4} {'SR_ann':>7} {'Win':>5} "
          f"{'Mean':>6} {'AvgHold':>7} {'Worst':>7} {'ExitSig%':>8}")
    print(f"  {'-'*80}")

    results_table = []
    for label, vp_exit, rough, qv_rate, disp_rev in exit_strategies:
        oos_trades = []
        for fold_k in range(1, n_folds):
            for ticker, (prices, folds) in ticker_folds.items():
                test_res = folds[fold_k]
                trades = ride_bubble_sigexit(
                    ticker, prices, test_res,
                    vol_p_exit=vp_exit,
                    roughness_exit=rough,
                    qv_rate_exit=qv_rate,
                    disp_reversal_exit=disp_rev,
                    **base_params)
                oos_trades.extend(trades)

        s = trade_stats(oos_trades)
        if s and s['n'] >= 2:
            sr = s.get('sharpe_ann', float('nan'))
            sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
            # What fraction exited via signature signal?
            sig_exits = sum(1 for t in oos_trades
                            if t['exit_reason'].startswith('sig_'))
            sig_pct = sig_exits / max(len(oos_trades), 1)
            print(f"  {label:<28} {s['n']:>4} {sr_str:>7} "
                  f"{s['win_rate']:>5.0%} {s['mean_ret']:>+5.1%} "
                  f"{s['avg_hold']:>6.0f}d {s['worst']:>+6.1%} "
                  f"{sig_pct:>7.0%}")
            results_table.append({
                'label': label, 'n': s['n'], 'sr': sr,
                'win': s['win_rate'], 'mean': s['mean_ret'],
                'avg_hold': s['avg_hold'], 'worst': s['worst'],
                'sig_pct': sig_pct,
            })
        else:
            print(f"  {label:<28} {'<2 trades':>4}")

    # ══════════════════════════════════════════════════════════════
    # 3. Adaptive: train exit params per fold
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  ADAPTIVE WALK-FORWARD: Train exit params per fold")
    print(f"{'─' * 90}")

    candidate_exits = [
        (0.30, None, None, False),   # baseline
        (0.20, None, None, False),
        (0.15, None, None, False),
        (0.30, 10.0, None, False),
        (0.30, 20.0, None, False),
        (0.30, 50.0, None, False),
        (0.30, None, 1.0, False),
        (0.30, None, 2.0, False),
        (0.30, None, None, True),
        (999., 10.0, None, False),
        (999., 20.0, None, False),
        (999., None, 1.0, False),
        (999., None, 2.0, False),
    ]

    oos_adaptive = []
    for fold_k in range(1, n_folds):
        best_exit = (0.30, None, None, False)
        best_sr = -999

        for vp_exit, rough, qv_rate, disp_rev in candidate_exits:
            train_trades = []
            for train_fold in range(fold_k):
                for ticker, (prices, folds) in ticker_folds.items():
                    train_res = folds[train_fold]
                    trades = ride_bubble_sigexit(
                        ticker, prices, train_res,
                        vol_p_exit=vp_exit, roughness_exit=rough,
                        qv_rate_exit=qv_rate,
                        disp_reversal_exit=disp_rev,
                        **base_params)
                    train_trades.extend(trades)

            s = trade_stats(train_trades)
            if s and s['n'] >= 5:
                sr = s.get('sharpe_ann', float('nan'))
                if np.isfinite(sr) and sr > best_sr:
                    best_sr = sr
                    best_exit = (vp_exit, rough, qv_rate, disp_rev)

        # Test on fold k
        test_trades = []
        for ticker, (prices, folds) in ticker_folds.items():
            test_res = folds[fold_k]
            trades = ride_bubble_sigexit(
                ticker, prices, test_res,
                vol_p_exit=best_exit[0], roughness_exit=best_exit[1],
                qv_rate_exit=best_exit[2],
                disp_reversal_exit=best_exit[3],
                **base_params)
            test_trades.extend(trades)

        oos_adaptive.extend(test_trades)
        s = trade_stats(test_trades)
        sr = s.get('sharpe_ann', float('nan')) if s else float('nan')
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        n_test = s.get('n', 0) if s else 0

        sig_exits = sum(1 for t in test_trades
                        if t['exit_reason'].startswith('sig_'))
        sig_pct = sig_exits / max(len(test_trades), 1)

        exit_label = f"vp={best_exit[0]}"
        if best_exit[1]: exit_label += f",rough={best_exit[1]}"
        if best_exit[2]: exit_label += f",qv={best_exit[2]}"
        if best_exit[3]: exit_label += ",disp_rev"
        print(f"  Fold {fold_k}: {exit_label:<35} "
              f"N={n_test}, SR_ann={sr_str}, sig_exits={sig_pct:.0%}")

    # ══════════════════════════════════════════════════════════════
    # 4. Final comparison
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  FINAL COMPARISON")
    print(f"{'═' * 90}")

    # Baseline
    oos_base = []
    for fold_k in range(1, n_folds):
        for ticker, (prices, folds) in ticker_folds.items():
            test_res = folds[fold_k]
            trades = ride_bubble_sigexit(
                ticker, prices, test_res,
                vol_p_exit=0.30, **base_params)
            oos_base.extend(trades)

    strategies = [
        ("vol_P=0.30 baseline", oos_base),
        ("Adaptive sig exit", oos_adaptive),
    ]

    print(f"\n  {'Strategy':<28} {'N':>4} {'SR_ann':>7} {'Win':>5} "
          f"{'Mean':>6} {'AvgHold':>7} {'Worst':>7}")
    print(f"  {'-'*70}")
    for name, trades in strategies:
        s = trade_stats(trades)
        if not s:
            continue
        sr = s.get('sharpe_ann', float('nan'))
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        print(f"  {name:<28} {s['n']:>4} {sr_str:>7} {s['win_rate']:>5.0%} "
              f"{s['mean_ret']:>+5.1%} {s['avg_hold']:>6.0f}d "
              f"{s['worst']:>+6.1%}")

    # Bootstrap comparison
    if len(oos_base) >= 5 and len(oos_adaptive) >= 5:
        rng = np.random.default_rng(42)
        n_boot = 5000
        base_rets = np.array([t['ret'] for t in oos_base])
        filt_rets = np.array([t['ret'] for t in oos_adaptive])

        base_srs, filt_srs = [], []
        for _ in range(n_boot):
            b_idx = rng.integers(0, len(base_rets), len(base_rets))
            f_idx = rng.integers(0, len(filt_rets), len(filt_rets))
            b_std = np.std(base_rets[b_idx])
            f_std = np.std(filt_rets[f_idx])
            base_srs.append(np.mean(base_rets[b_idx]) / max(b_std, 1e-8))
            filt_srs.append(np.mean(filt_rets[f_idx]) / max(f_std, 1e-8))

        base_srs = np.array(base_srs)
        filt_srs = np.array(filt_srs)
        p_better = np.mean(filt_srs > base_srs)
        print(f"\n  Bootstrap P(sig_exit > baseline): {p_better:.1%}")

    # ── Exit reason breakdown ──
    print(f"\n  Exit reason breakdown (adaptive):")
    reasons = {}
    for t in oos_adaptive:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, [])
        reasons[r].append(t['ret'])

    for reason in sorted(reasons.keys()):
        rets = reasons[reason]
        print(f"    {reason:<15}: N={len(rets):>3}, "
              f"mean={np.mean(rets):+.1%}, win={np.mean(np.array(rets)>0):.0%}")

    print(f"\n{'=' * 90}")


if __name__ == '__main__':
    main()
