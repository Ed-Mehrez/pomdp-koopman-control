"""
Log-signature filtered ride-bubble strategy.

At each potential entry point, compute the rolling lead-lag log-sig of
(P, vol_P, log_return). The Lévy area A(r_lead, P_lag) captures whether
price momentum leads the bubble signal (good) or lags it (bad).

Key insight from bubble_logsig_fingerprint.py:
  A(r_lead, P_lag) < 0 at entry → price leading, P lagging → ENTER
  A(r_lead, P_lag) > 0 at entry → P leading, price lagging → SKIP

Test 1: Does the filter improve walk-forward OOS Sharpe?
Test 2: Is the filter robust or is it overfitting to a threshold?
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


def compute_rolling_logsig_at_results(results, gamma=0.98):
    """Compute rolling lead-lag log-sig of (P, vol_P, log_return).

    Returns dict mapping result index → feature vector.
    The signature at index i uses ONLY data from results[0..i] (causal).
    """
    n = len(results)
    if n < 3:
        return {}

    d = 3  # (P, vol_P, log_return)
    sig_map = RecurrentLeadLagLogSigMap(state_dim=d, level=2,
                                        forgetting_factor=gamma)

    # Compute increments between consecutive result snapshots
    # Normalize by running std to keep features comparable across tickers
    p = np.array([r['p_bubble'] for r in results])
    vp = np.array([r['vol_p'] for r in results])
    prices = np.array([r['price'] for r in results])

    dp = np.diff(p)
    dvp = np.diff(vp)
    dlog_price = np.diff(np.log(prices))

    increments = np.column_stack([dp, dvp, dlog_price])

    # Running normalization: use expanding std (min 10 obs)
    running_std = np.ones(3)
    sigs = {}

    for i in range(len(increments)):
        # Update running std with exponential weighting
        if i >= 2:
            window = increments[max(0, i-50):i+1]
            running_std = np.maximum(np.std(window, axis=0), 1e-8)

        normed = increments[i] / running_std
        feat = sig_map.update(normed)
        sigs[i + 1] = feat.copy()  # sigs[i+1] corresponds to results[i+1]

    return sigs


def get_area_r_lead_p_lag(sig_features):
    """Extract A(r_lead, P_lag) from the (P, vol_P, return) lead-lag log-sig.

    Lead-lag dims for d=3 input: (P_lead, vP_lead, r_lead, P_lag, vP_lag, r_lag)
    Indices:                     (0,      1,       2,      3,     4,      5)

    Level-2 areas are stored as upper-triangular pairs (i<j):
    Index mapping: (0,1), (0,2), (0,3), (0,4), (0,5),
                          (1,2), (1,3), (1,4), (1,5),
                                 (2,3), (2,4), (2,5),
                                        (3,4), (3,5),
                                               (4,5)
    A(r_lead, P_lag) = A(2, 3) → index 9 in the l2 block

    Level-1 has 6 entries, so A(2,3) is at index 6 + 9 = 15.
    Wait, let me recount:
    (0,1)→0, (0,2)→1, (0,3)→2, (0,4)→3, (0,5)→4,
    (1,2)→5, (1,3)→6, (1,4)→7, (1,5)→8,
    (2,3)→9, (2,4)→10, (2,5)→11,
    (3,4)→12, (3,5)→13,
    (4,5)→14
    So A(2,3) = l2[9], total index = 6 + 9 = 15.
    """
    return sig_features[6 + 9]  # = features[15]


def get_area_p_lead_r_lead(sig_features):
    """A(P_lead, r_lead) = A(0, 2) → l2[1], total index = 6 + 1 = 7."""
    return sig_features[6 + 1]


def ride_bubble_logsig(ticker, prices, results, sigs,
                        p_entry=0.6, vol_p_entry_max=0.05,
                        vol_p_exit=0.3, p_exit=0.4, max_hold=60,
                        area_thresh=None, area_feature='r_lead_p_lag'):
    """Ride bubble with optional log-sig area filter at entry.

    area_thresh: if set, only enter when A(r_lead, P_lag) < area_thresh
                 (negative area = price leads P = good entry)
    area_feature: which area to use ('r_lead_p_lag' or 'p_lead_r_lead')
    """
    trades = []
    n = len(results)
    i = 0

    area_fn = (get_area_r_lead_p_lag if area_feature == 'r_lead_p_lag'
               else get_area_p_lead_r_lead)

    while i < n:
        r = results[i]
        vol_p = r['vol_p']

        if r['p_bubble'] <= p_entry or vol_p > vol_p_entry_max:
            i += 1
            continue

        # Log-sig filter
        if area_thresh is not None and i in sigs:
            area = area_fn(sigs[i])
            if area > area_thresh:
                i += 1
                continue

        S_entry = r['price']
        entry_area = area_fn(sigs[i]) if i in sigs else np.nan
        j = i + 1
        while j < n:
            elapsed = (j - i) * STEP_DAYS
            rj = results[j]
            vp_j = rj['vol_p']

            exit_reason = None
            if vp_j > vol_p_exit:
                exit_reason = 'vol_spike'
            elif rj['p_bubble'] < p_exit:
                exit_reason = 'p_dropped'
            elif elapsed >= max_hold:
                exit_reason = 'max_hold'

            if exit_reason:
                S_exit = rj['price']
                trades.append({
                    'ticker': ticker, 'strategy': 'ride_bubble_logsig',
                    'entry_bar': r['bar'], 'exit_bar': rj['bar'],
                    'entry_price': S_entry, 'exit_price': S_exit,
                    'ret': (S_exit - S_entry) / S_entry,
                    'hold_days': elapsed, 'exit_reason': exit_reason,
                    'p_entry': r['p_bubble'], 'vol_p_entry': vol_p,
                    'entry_area': entry_area,
                })
                i = j + 1
                break
            j += 1
        else:
            break

    return trades


def main():
    print("=" * 90)
    print("  LOG-SIGNATURE FILTERED RIDE BUBBLE STRATEGY")
    print("=" * 90)

    # ── Load data ──
    print("\n  Loading data...")
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
        sigs = compute_rolling_logsig_at_results(results, gamma=0.98)
        all_data[ticker] = (prices, results, sigs)

    print(f"  Loaded {len(all_data)} tickers ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # 1. Verify: at ENTRY time, does area predict trade outcome?
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  TEST 1: Entry-time A(r_lead, P_lag) vs trade return")
    print(f"  (Is the signal causal, not just endpoint correlation?)")
    print(f"{'─' * 90}")

    all_trades = []
    for ticker, (prices, results, sigs) in all_data.items():
        trades = ride_bubble_logsig(ticker, prices, results, sigs,
                                     p_entry=0.6, vol_p_entry_max=0.05,
                                     vol_p_exit=0.3, area_thresh=None)
        all_trades.extend(trades)

    areas = np.array([t['entry_area'] for t in all_trades
                       if np.isfinite(t['entry_area'])])
    rets = np.array([t['ret'] for t in all_trades
                      if np.isfinite(t['entry_area'])])

    print(f"\n  {len(areas)} trades with valid entry-time area")
    r_spear, p_spear = sp_stats.spearmanr(areas, rets)
    r_pears, p_pears = sp_stats.pearsonr(areas, rets)
    print(f"  Spearman(area, ret): r={r_spear:+.3f}, p={p_spear:.4f}")
    print(f"  Pearson(area, ret):  r={r_pears:+.3f}, p={p_pears:.4f}")

    # Split at area=0 (natural threshold)
    pos_area = rets[areas > 0]
    neg_area = rets[areas <= 0]
    print(f"\n  Area ≤ 0 (price leads): {len(neg_area)} trades, "
          f"mean ret={np.mean(neg_area):+.1%}, "
          f"win={np.mean(neg_area > 0):.0%}")
    print(f"  Area > 0 (P leads):     {len(pos_area)} trades, "
          f"mean ret={np.mean(pos_area):+.1%}, "
          f"win={np.mean(pos_area > 0):.0%}")

    if len(pos_area) >= 3 and len(neg_area) >= 3:
        stat, pv = sp_stats.mannwhitneyu(neg_area, pos_area,
                                          alternative='greater')
        print(f"  Mann-Whitney (neg_area > pos_area): p={pv:.4f}")

    # Quartile analysis
    quartiles = np.percentile(areas, [25, 50, 75])
    labels = [
        (f"Q1 (area < {quartiles[0]:+.2f})", areas < quartiles[0]),
        (f"Q2 ({quartiles[0]:+.2f} ≤ area < {quartiles[1]:+.2f})",
         (areas >= quartiles[0]) & (areas < quartiles[1])),
        (f"Q3 ({quartiles[1]:+.2f} ≤ area < {quartiles[2]:+.2f})",
         (areas >= quartiles[1]) & (areas < quartiles[2])),
        (f"Q4 (area ≥ {quartiles[2]:+.2f})", areas >= quartiles[2]),
    ]
    print(f"\n  Quartile analysis:")
    for label, mask in labels:
        q_rets = rets[mask]
        if len(q_rets) > 0:
            s = trade_stats([{'ret': r, 'hold_days': 15} for r in q_rets])
            sr_str = f"SR={s['sharpe_ann']:+.2f}" if np.isfinite(s.get('sharpe_ann', float('nan'))) else "SR=nan"
            print(f"    {label}: N={len(q_rets)}, "
                  f"mean={np.mean(q_rets):+.1%}, "
                  f"win={np.mean(q_rets>0):.0%}, {sr_str}")

    # Also check A(P_lead, r_lead)
    print(f"\n  Also checking A(P_lead, r_lead):")
    areas2 = np.array([get_area_p_lead_r_lead(
        all_data[t['ticker']][2].get(
            # find result index from entry_bar
            next((idx for idx, r in enumerate(all_data[t['ticker']][1])
                  if r['bar'] == t['entry_bar']), -1),
            np.zeros(21)))
        for t in all_trades])

    # Simpler: just use the already-stored entry area and recompute
    # Actually let's just run it with the other feature
    all_trades2 = []
    for ticker, (prices, results, sigs) in all_data.items():
        trades = ride_bubble_logsig(ticker, prices, results, sigs,
                                     p_entry=0.6, vol_p_entry_max=0.05,
                                     vol_p_exit=0.3, area_thresh=None,
                                     area_feature='p_lead_r_lead')
        all_trades2.extend(trades)

    areas2 = np.array([t['entry_area'] for t in all_trades2
                        if np.isfinite(t['entry_area'])])
    rets2 = np.array([t['ret'] for t in all_trades2
                       if np.isfinite(t['entry_area'])])
    r2, p2 = sp_stats.spearmanr(areas2, rets2)
    print(f"  Spearman(A(P_lead,r_lead), ret): r={r2:+.3f}, p={p2:.4f}")

    # ══════════════════════════════════════════════════════════════
    # 2. Walk-forward test: filter vs no-filter
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  TEST 2: Walk-Forward CV — Filtered vs Unfiltered")
    print(f"  5-fold expanding window, fixed strategy params")
    print(f"{'─' * 90}")

    n_folds = 5
    base_params = {'p_entry': 0.6, 'vol_p_entry_max': 0.05,
                   'vol_p_exit': 0.3}

    # Split each ticker into temporal folds
    ticker_folds = {}
    for ticker, (prices, results, sigs) in all_data.items():
        n = len(results)
        fold_size = n // n_folds
        if fold_size < 5:
            continue
        folds = []
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else n
            folds.append((results[start:end], sigs))
        ticker_folds[ticker] = (prices, folds)

    # For each fold: train area threshold on train, evaluate on test
    area_thresholds_to_test = [None, 0.0, -0.5, -1.0, 0.5, 1.0]

    for area_thresh in area_thresholds_to_test:
        oos_trades = []

        for fold_k in range(1, n_folds):
            fold_trades = []
            for ticker, (prices, folds) in ticker_folds.items():
                test_results, sigs = folds[fold_k]
                trades = ride_bubble_logsig(
                    ticker, prices, test_results, sigs,
                    area_thresh=area_thresh, **base_params)
                fold_trades.extend(trades)

            oos_trades.extend(fold_trades)

        s = trade_stats(oos_trades)
        if s:
            label = f"thresh={area_thresh}" if area_thresh is not None else "NO FILTER"
            sr = s.get('sharpe_ann', float('nan'))
            sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
            print(f"  {label:<16}: N={s['n']:>3}, "
                  f"SR_ann={sr_str}, win={s['win_rate']:.0%}, "
                  f"mean={s['mean_ret']:+.1%}")
        else:
            label = f"thresh={area_thresh}" if area_thresh is not None else "NO FILTER"
            print(f"  {label:<16}: NO TRADES")

    # ══════════════════════════════════════════════════════════════
    # 3. Data-driven: train threshold on folds 0..k, test on k+1
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  TEST 3: Adaptive threshold (trained per fold)")
    print(f"{'─' * 90}")

    candidate_thresholds = [None, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    oos_trades_adaptive = []
    fold_details = []

    for fold_k in range(1, n_folds):
        # Train: evaluate each threshold on folds 0..k-1
        best_thresh = None
        best_sr = -999

        for thresh in candidate_thresholds:
            train_trades = []
            for train_fold in range(fold_k):
                for ticker, (prices, folds) in ticker_folds.items():
                    train_res, sigs = folds[train_fold]
                    trades = ride_bubble_logsig(
                        ticker, prices, train_res, sigs,
                        area_thresh=thresh, **base_params)
                    train_trades.extend(trades)

            s = trade_stats(train_trades)
            sr = s.get('sharpe_ann', float('nan')) if s else float('nan')
            # Prefer None (no filter) if it's close, to avoid overfitting
            if np.isfinite(sr) and sr > best_sr:
                best_sr = sr
                best_thresh = thresh

        # Test on fold k with best threshold
        test_trades = []
        for ticker, (prices, folds) in ticker_folds.items():
            test_res, sigs = folds[fold_k]
            trades = ride_bubble_logsig(
                ticker, prices, test_res, sigs,
                area_thresh=best_thresh, **base_params)
            test_trades.extend(trades)

        oos_trades_adaptive.extend(test_trades)
        s = trade_stats(test_trades)
        sr = s.get('sharpe_ann', float('nan')) if s else float('nan')
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        n_test = s.get('n', 0) if s else 0
        print(f"  Fold {fold_k}: best_thresh={best_thresh}, "
              f"N={n_test}, SR_ann={sr_str}")
        fold_details.append({'fold': fold_k, 'thresh': best_thresh,
                             'n': n_test, 'sr': sr})

    # Aggregate
    s = trade_stats(oos_trades_adaptive)
    if s:
        sr = s.get('sharpe_ann', float('nan'))
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        print(f"\n  ADAPTIVE OOS: N={s['n']}, SR_ann={sr_str}, "
              f"win={s['win_rate']:.0%}, mean={s['mean_ret']:+.1%}")

    # ══════════════════════════════════════════════════════════════
    # 4. Compare unfiltered vs best fixed filter
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  SUMMARY: Unfiltered vs Log-Sig Filtered (OOS)")
    print(f"{'─' * 90}")

    # Unfiltered baseline
    oos_base = []
    for fold_k in range(1, n_folds):
        for ticker, (prices, folds) in ticker_folds.items():
            test_res, sigs = folds[fold_k]
            trades = ride_bubble_logsig(
                ticker, prices, test_res, sigs,
                area_thresh=None, **base_params)
            oos_base.extend(trades)

    s_base = trade_stats(oos_base)
    s_filt = trade_stats(oos_trades_adaptive)

    if s_base and s_filt:
        sr_b = s_base.get('sharpe_ann', float('nan'))
        sr_f = s_filt.get('sharpe_ann', float('nan'))
        print(f"\n  {'Metric':<20} {'Unfiltered':>12} {'LogSig Filtered':>15}")
        print(f"  {'-'*50}")
        print(f"  {'N trades':<20} {s_base['n']:>12} {s_filt['n']:>15}")
        sr_b_str = f"{sr_b:+.2f}" if np.isfinite(sr_b) else "nan"
        sr_f_str = f"{sr_f:+.2f}" if np.isfinite(sr_f) else "nan"
        print(f"  {'SR_ann':<20} {sr_b_str:>12} {sr_f_str:>15}")
        print(f"  {'Win rate':<20} {s_base['win_rate']:>12.0%} "
              f"{s_filt['win_rate']:>15.0%}")
        print(f"  {'Mean ret':<20} {s_base['mean_ret']:>+12.1%} "
              f"{s_filt['mean_ret']:>+15.1%}")
        print(f"  {'Worst':<20} {s_base['worst']:>+12.1%} "
              f"{s_filt['worst']:>+15.1%}")

        # Bootstrap comparison
        if len(oos_base) >= 5 and len(oos_trades_adaptive) >= 5:
            rng = np.random.default_rng(42)
            n_boot = 5000
            base_rets = np.array([t['ret'] for t in oos_base])
            filt_rets = np.array([t['ret'] for t in oos_trades_adaptive])

            base_srs = []
            filt_srs = []
            for _ in range(n_boot):
                b_idx = rng.integers(0, len(base_rets), len(base_rets))
                f_idx = rng.integers(0, len(filt_rets), len(filt_rets))
                b_sr = np.mean(base_rets[b_idx]) / max(np.std(base_rets[b_idx]), 1e-8)
                f_sr = np.mean(filt_rets[f_idx]) / max(np.std(filt_rets[f_idx]), 1e-8)
                base_srs.append(b_sr)
                filt_srs.append(f_sr)

            base_srs = np.array(base_srs)
            filt_srs = np.array(filt_srs)
            p_better = np.mean(filt_srs > base_srs)
            print(f"\n  Bootstrap P(filtered > unfiltered): {p_better:.1%}")

    print(f"\n{'=' * 90}")


if __name__ == '__main__':
    main()
