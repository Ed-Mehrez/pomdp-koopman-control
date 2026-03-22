"""
Momentum stock selection via rolling log-signatures.

Key insight from bubble_signature_analysis.py:
  Profitable tickers share: AC(P) > 0.7, moderate σ, smooth P ramp.
  Meme stocks: spiky P (low AC), extreme vol, high QV of P.
  Boring stocks: P never triggers.

Signature view:
  Lead-lag log-sig of P path captures BOTH displacement and roughness.
  - Level-1 displacement of P_lead/P_lag: net P trend
  - Lévy area A(P_lead, P_lag) = QV of P: signal roughness
  - Ratio = displacement / sqrt(QV) = "signature Sharpe of P"

A stock with high P displacement but low QV has SMOOTH momentum.
A stock with high P displacement AND high QV is noisy meme.

Cross-asset protocol:
  1. At each time step, compute rolling log-sig per ticker
  2. Score = P_displacement / sqrt(QV_of_P + eps) — "signal quality"
  3. Only enter ride-bubble on tickers with score > threshold
  4. Walk-forward: train threshold on past folds, test on next
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
# Signature-based momentum scoring
# ═══════════════════════════════════════════════════════════════════════

class MomentumSignatureScorer:
    """Rolling log-sig of (P, vol_P) per ticker → momentum score.

    For d=2 (P, vol_P), lead-lag gives 4D path.
    Level-2 log-sig: 4 level-1 + 6 level-2 = 10 features.

    Lead-lag channels: (P_lead, vP_lead, P_lag, vP_lag)
    Indices:           (0,      1,       2,     3)

    Key features:
      P_lead (idx 0): cumulative P displacement (lead)
      P_lag  (idx 2): cumulative P displacement (lag)
      A(P_lead, P_lag) (idx 4+0=4): QV of P signal = sum of (dP)^2

    Level-2 area indices (i<j):
      (0,1)→0, (0,2)→1, (0,3)→2, (1,2)→3, (1,3)→4, (2,3)→5
      So A(P_lead, P_lag) = l2[1] → total index = 4 + 1 = 5

      A(vP_lead, vP_lag) = A(1, 3) → l2[4] → total index = 4 + 4 = 8
    """

    def __init__(self, gamma=0.97):
        self.gamma = gamma
        self.sig_map = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=gamma)
        self.prev_p = None
        self.prev_vp = None
        # Running normalization
        self.dp_var = 0.01
        self.dvp_var = 0.01
        self.n_obs = 0
        self.ema_alpha = 0.02  # for running variance

    def reset(self):
        self.sig_map.reset()
        self.prev_p = None
        self.prev_vp = None
        self.dp_var = 0.01
        self.dvp_var = 0.01
        self.n_obs = 0

    def update(self, p_bubble, vol_p):
        """Feed one observation. Returns current feature vector."""
        if self.prev_p is None:
            self.prev_p = p_bubble
            self.prev_vp = vol_p
            return self.sig_map.get_features()

        dp = p_bubble - self.prev_p
        dvp = vol_p - self.prev_vp

        # Update running variance for normalization
        self.n_obs += 1
        self.dp_var = (1 - self.ema_alpha) * self.dp_var + self.ema_alpha * dp**2
        self.dvp_var = (1 - self.ema_alpha) * self.dvp_var + self.ema_alpha * dvp**2

        # Normalize
        dp_norm = dp / max(np.sqrt(self.dp_var), 1e-6)
        dvp_norm = dvp / max(np.sqrt(self.dvp_var), 1e-6)

        feat = self.sig_map.update(np.array([dp_norm, dvp_norm]))

        self.prev_p = p_bubble
        self.prev_vp = vol_p
        return feat

    @staticmethod
    def p_displacement(feat):
        """Net P displacement (average of lead and lag)."""
        return 0.5 * (feat[0] + feat[2])  # (P_lead + P_lag) / 2

    @staticmethod
    def p_qv(feat):
        """QV of P = Lévy area A(P_lead, P_lag)."""
        return feat[5]  # index 4 + 1 = 5

    @staticmethod
    def vp_qv(feat):
        """QV of vol(P) = A(vP_lead, vP_lag)."""
        return feat[8]  # index 4 + 4 = 8

    @staticmethod
    def momentum_score(feat):
        """Signature 'Sharpe' of P: displacement / sqrt(roughness).

        High score = smooth, persistent P ramp (momentum).
        Low score = choppy, mean-reverting P (meme or boring).
        """
        disp = 0.5 * (feat[0] + feat[2])
        qv = feat[5]
        # Only consider positive displacement (P trending up = bubble forming)
        if disp <= 0:
            return -1.0  # not in bubble trend
        return disp / max(np.sqrt(abs(qv)), 0.1)

    @staticmethod
    def roughness_score(feat):
        """QV(P) / displacement^2 — high = rough/noisy, low = smooth."""
        disp = 0.5 * (feat[0] + feat[2])
        qv = feat[5]
        if abs(disp) < 0.01:
            return float('inf')
        return abs(qv) / (disp ** 2)


def compute_ticker_scores(results, gamma=0.97):
    """Compute momentum scores at each result snapshot."""
    scorer = MomentumSignatureScorer(gamma=gamma)
    scores = []
    for r in results:
        feat = scorer.update(r['p_bubble'], r['vol_p'])
        scores.append({
            'bar': r['bar'],
            'momentum_score': scorer.momentum_score(feat),
            'p_displacement': scorer.p_displacement(feat),
            'p_qv': scorer.p_qv(feat),
            'roughness': scorer.roughness_score(feat),
            'feat': feat.copy(),
        })
    return scores


def ride_bubble_momentum(ticker, prices, results, scores,
                          p_entry=0.6, vol_p_entry_max=0.05,
                          vol_p_exit=0.3, p_exit=0.4, max_hold=60,
                          min_momentum_score=None,
                          max_roughness=None):
    """Ride bubble with momentum signature filter."""
    trades = []
    n = len(results)
    i = 0

    while i < n:
        r = results[i]
        vol_p = r['vol_p']

        if r['p_bubble'] <= p_entry or vol_p > vol_p_entry_max:
            i += 1
            continue

        # Momentum filter
        if min_momentum_score is not None and i < len(scores):
            if scores[i]['momentum_score'] < min_momentum_score:
                i += 1
                continue

        if max_roughness is not None and i < len(scores):
            if scores[i]['roughness'] > max_roughness:
                i += 1
                continue

        S_entry = r['price']
        entry_score = scores[i] if i < len(scores) else {}
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
                    'ticker': ticker, 'strategy': 'ride_momentum',
                    'entry_bar': r['bar'], 'exit_bar': rj['bar'],
                    'entry_price': S_entry, 'exit_price': S_exit,
                    'ret': (S_exit - S_entry) / S_entry,
                    'hold_days': elapsed, 'exit_reason': exit_reason,
                    'p_entry': r['p_bubble'], 'vol_p_entry': vol_p,
                    'momentum_score': entry_score.get('momentum_score', np.nan),
                    'roughness': entry_score.get('roughness', np.nan),
                    'p_displacement': entry_score.get('p_displacement', np.nan),
                })
                i = j + 1
                break
            j += 1
        else:
            break

    return trades


def main():
    print("=" * 90)
    print("  MOMENTUM SIGNATURE SELECTION")
    print("  Isolating momentum from meme via log-sig of P dynamics")
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
        scores = compute_ticker_scores(results, gamma=0.97)
        all_data[ticker] = (prices, results, scores)

    print(f"  Loaded {len(all_data)} tickers ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # 1. Cross-asset signature characterization
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  CROSS-ASSET SIGNATURE SCORES (time-averaged)")
    print(f"{'─' * 90}")

    # Compute average momentum score during bubble episodes (P > 0.5)
    print(f"\n  {'Ticker':<8} {'Mom Score':>9} {'P disp':>7} {'P QV':>7} "
          f"{'Rough':>7} {'Frac P>.5':>9} {'N trades':>8} {'SR':>7}")
    print(f"  {'-'*75}")

    ticker_profiles = []
    for ticker in sorted(all_data.keys()):
        prices, results, scores = all_data[ticker]

        # Average scores during bubble episodes
        bubble_scores = [s for s, r in zip(scores, results)
                         if r['p_bubble'] > 0.5]

        if bubble_scores:
            avg_mom = np.mean([s['momentum_score'] for s in bubble_scores])
            avg_disp = np.mean([s['p_displacement'] for s in bubble_scores])
            avg_qv = np.mean([s['p_qv'] for s in bubble_scores])
            avg_rough = np.median([s['roughness'] for s in bubble_scores
                                   if np.isfinite(s['roughness'])])
        else:
            avg_mom = avg_disp = avg_qv = 0.0
            avg_rough = float('inf')

        frac_bubble = np.mean([r['p_bubble'] > 0.5 for r in results])

        # Get trade stats
        trades = ride_bubble_momentum(ticker, prices, results, scores,
                                       p_entry=0.6, vol_p_entry_max=0.05,
                                       vol_p_exit=0.3)
        s = trade_stats(trades)
        n_trades = s.get('n', 0) if s else 0
        sr = s.get('sharpe_ann', float('nan')) if s else float('nan')

        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "   —"
        rough_str = f"{avg_rough:.2f}" if np.isfinite(avg_rough) else "  inf"

        print(f"  {ticker:<8} {avg_mom:>+9.2f} {avg_disp:>+7.2f} "
              f"{avg_qv:>7.2f} {rough_str:>7} {frac_bubble:>9.0%} "
              f"{n_trades:>8} {sr_str:>7}")

        ticker_profiles.append({
            'ticker': ticker, 'avg_momentum': avg_mom,
            'avg_disp': avg_disp, 'avg_qv': avg_qv,
            'avg_roughness': avg_rough, 'frac_bubble': frac_bubble,
            'n_trades': n_trades, 'sr': sr,
        })

    # ── Correlation: momentum score vs profitability ──
    valid = [(p['avg_momentum'], p['sr']) for p in ticker_profiles
             if p['n_trades'] >= 2 and np.isfinite(p['sr'])]
    if len(valid) >= 5:
        moms, srs = zip(*valid)
        r, pv = sp_stats.spearmanr(moms, srs)
        print(f"\n  Spearman(avg_momentum_score, SR): r={r:+.3f}, p={pv:.3f}")

        valid_r = [(p['avg_roughness'], p['sr']) for p in ticker_profiles
                   if p['n_trades'] >= 2 and np.isfinite(p['sr'])
                   and np.isfinite(p['avg_roughness'])]
        if len(valid_r) >= 5:
            roughs, srs_r = zip(*valid_r)
            r2, pv2 = sp_stats.spearmanr(roughs, srs_r)
            print(f"  Spearman(avg_roughness, SR):       r={r2:+.3f}, p={pv2:.3f}")

    # ══════════════════════════════════════════════════════════════
    # 2. Natural grouping by momentum score
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  NATURAL GROUPS BY MOMENTUM SCORE")
    print(f"{'─' * 90}")

    # Sort by momentum score
    sorted_profiles = sorted(ticker_profiles, key=lambda p: p['avg_momentum'],
                              reverse=True)
    n_tickers = len(sorted_profiles)
    top_half = sorted_profiles[:n_tickers//2]
    bot_half = sorted_profiles[n_tickers//2:]

    for label, group in [("TOP half (high momentum score)", top_half),
                         ("BOTTOM half (low momentum score)", bot_half)]:
        tickers = [p['ticker'] for p in group]
        valid_srs = [p['sr'] for p in group
                     if p['n_trades'] >= 2 and np.isfinite(p['sr'])]
        n_trades_total = sum(p['n_trades'] for p in group)

        print(f"\n  {label}:")
        print(f"    Tickers: {', '.join(tickers)}")
        if valid_srs:
            print(f"    Mean SR: {np.mean(valid_srs):+.2f}, "
                  f"Median SR: {np.median(valid_srs):+.2f}, "
                  f"P(SR>0): {np.mean(np.array(valid_srs)>0):.0%}")
        print(f"    Total trades: {n_trades_total}")

    # ══════════════════════════════════════════════════════════════
    # 3. Walk-forward: momentum filter on ride bubble
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  WALK-FORWARD CV: Momentum-Filtered Ride Bubble")
    print(f"  5 folds, test momentum_score and roughness thresholds")
    print(f"{'─' * 90}")

    n_folds = 5
    base_params = {'p_entry': 0.6, 'vol_p_entry_max': 0.05,
                   'vol_p_exit': 0.3}

    # Split each ticker into temporal folds
    ticker_folds = {}
    for ticker, (prices, results, scores) in all_data.items():
        n = len(results)
        fold_size = n // n_folds
        if fold_size < 5:
            continue
        folds = []
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else n
            folds.append((results[start:end], scores[start:end]))
        ticker_folds[ticker] = (prices, folds)

    # Test: fixed momentum thresholds
    print(f"\n  Fixed momentum_score thresholds:")
    for mom_thresh in [None, 0.0, 0.5, 1.0, 2.0, 3.0]:
        oos_trades = []
        for fold_k in range(1, n_folds):
            for ticker, (prices, folds) in ticker_folds.items():
                test_res, test_scores = folds[fold_k]
                trades = ride_bubble_momentum(
                    ticker, prices, test_res, test_scores,
                    min_momentum_score=mom_thresh, **base_params)
                oos_trades.extend(trades)

        s = trade_stats(oos_trades)
        if s and s['n'] > 0:
            label = f"mom≥{mom_thresh}" if mom_thresh is not None else "NO FILTER"
            sr = s.get('sharpe_ann', float('nan'))
            sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
            print(f"    {label:<14}: N={s['n']:>3}, SR_ann={sr_str}, "
                  f"win={s['win_rate']:.0%}, mean={s['mean_ret']:+.1%}")
        else:
            label = f"mom≥{mom_thresh}" if mom_thresh is not None else "NO FILTER"
            print(f"    {label:<14}: NO TRADES")

    # Test: fixed roughness thresholds
    print(f"\n  Fixed roughness thresholds (lower = smoother):")
    for rough_max in [None, 50.0, 20.0, 10.0, 5.0, 2.0]:
        oos_trades = []
        for fold_k in range(1, n_folds):
            for ticker, (prices, folds) in ticker_folds.items():
                test_res, test_scores = folds[fold_k]
                trades = ride_bubble_momentum(
                    ticker, prices, test_res, test_scores,
                    max_roughness=rough_max, **base_params)
                oos_trades.extend(trades)

        s = trade_stats(oos_trades)
        if s and s['n'] > 0:
            label = f"rough≤{rough_max}" if rough_max is not None else "NO FILTER"
            sr = s.get('sharpe_ann', float('nan'))
            sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
            print(f"    {label:<14}: N={s['n']:>3}, SR_ann={sr_str}, "
                  f"win={s['win_rate']:.0%}, mean={s['mean_ret']:+.1%}")

    # Test: combined momentum + roughness
    print(f"\n  Combined filters:")
    combos = [
        (None, None, "No filter"),
        (0.0, None, "mom≥0"),
        (0.0, 20.0, "mom≥0 + rough≤20"),
        (0.5, 10.0, "mom≥0.5 + rough≤10"),
        (1.0, 10.0, "mom≥1 + rough≤10"),
        (1.0, 5.0, "mom≥1 + rough≤5"),
        (2.0, 10.0, "mom≥2 + rough≤10"),
    ]
    for mom_thresh, rough_max, label in combos:
        oos_trades = []
        for fold_k in range(1, n_folds):
            for ticker, (prices, folds) in ticker_folds.items():
                test_res, test_scores = folds[fold_k]
                trades = ride_bubble_momentum(
                    ticker, prices, test_res, test_scores,
                    min_momentum_score=mom_thresh,
                    max_roughness=rough_max, **base_params)
                oos_trades.extend(trades)

        s = trade_stats(oos_trades)
        if s and s['n'] > 0:
            sr = s.get('sharpe_ann', float('nan'))
            sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
            # Per-ticker breakdown
            by_ticker = {}
            for t in oos_trades:
                by_ticker.setdefault(t['ticker'], []).append(t)
            n_tickers_trading = len(by_ticker)
            print(f"    {label:<22}: N={s['n']:>3}, SR_ann={sr_str}, "
                  f"win={s['win_rate']:.0%}, mean={s['mean_ret']:+.1%}, "
                  f"tickers={n_tickers_trading}")
        else:
            print(f"    {label:<22}: NO TRADES")

    # ══════════════════════════════════════════════════════════════
    # 4. Adaptive walk-forward: train filter per fold
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print(f"  ADAPTIVE WALK-FORWARD: Train momentum filter per fold")
    print(f"{'─' * 90}")

    candidate_combos = [
        (None, None),
        (0.0, None), (0.5, None), (1.0, None), (2.0, None),
        (None, 20.0), (None, 10.0), (None, 5.0),
        (0.0, 20.0), (0.5, 10.0), (1.0, 10.0), (1.0, 5.0),
    ]

    oos_adaptive = []
    for fold_k in range(1, n_folds):
        best_combo = (None, None)
        best_sr = -999

        for mom_thresh, rough_max in candidate_combos:
            train_trades = []
            for train_fold in range(fold_k):
                for ticker, (prices, folds) in ticker_folds.items():
                    train_res, train_scores = folds[train_fold]
                    trades = ride_bubble_momentum(
                        ticker, prices, train_res, train_scores,
                        min_momentum_score=mom_thresh,
                        max_roughness=rough_max, **base_params)
                    train_trades.extend(trades)

            s = trade_stats(train_trades)
            # Require minimum trades to avoid overfitting
            if s and s['n'] >= 5:
                sr = s.get('sharpe_ann', float('nan'))
                if np.isfinite(sr) and sr > best_sr:
                    best_sr = sr
                    best_combo = (mom_thresh, rough_max)

        # Test on fold k
        test_trades = []
        for ticker, (prices, folds) in ticker_folds.items():
            test_res, test_scores = folds[fold_k]
            trades = ride_bubble_momentum(
                ticker, prices, test_res, test_scores,
                min_momentum_score=best_combo[0],
                max_roughness=best_combo[1], **base_params)
            test_trades.extend(trades)

        oos_adaptive.extend(test_trades)
        s = trade_stats(test_trades)
        sr = s.get('sharpe_ann', float('nan')) if s else float('nan')
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        n_test = s.get('n', 0) if s else 0
        print(f"  Fold {fold_k}: best={best_combo}, "
              f"N={n_test}, SR_ann={sr_str}")

    s_adapt = trade_stats(oos_adaptive)
    if s_adapt:
        sr = s_adapt.get('sharpe_ann', float('nan'))
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        print(f"\n  ADAPTIVE OOS: N={s_adapt['n']}, SR_ann={sr_str}, "
              f"win={s_adapt['win_rate']:.0%}, mean={s_adapt['mean_ret']:+.1%}")

    # ══════════════════════════════════════════════════════════════
    # 5. Final comparison
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  FINAL COMPARISON")
    print(f"{'═' * 90}")

    # Unfiltered baseline
    oos_base = []
    for fold_k in range(1, n_folds):
        for ticker, (prices, folds) in ticker_folds.items():
            test_res, test_scores = folds[fold_k]
            trades = ride_bubble_momentum(
                ticker, prices, test_res, test_scores,
                min_momentum_score=None, max_roughness=None,
                **base_params)
            oos_base.extend(trades)

    s_base = trade_stats(oos_base)

    strategies = [
        ("Unfiltered", oos_base),
        ("Adaptive momentum", oos_adaptive),
    ]

    print(f"\n  {'Strategy':<25} {'N':>4} {'SR_ann':>7} {'Win':>5} "
          f"{'Mean':>6} {'Worst':>7} {'Tickers':>8}")
    print(f"  {'-'*70}")
    for name, trades in strategies:
        s = trade_stats(trades)
        if not s:
            continue
        sr = s.get('sharpe_ann', float('nan'))
        sr_str = f"{sr:+.2f}" if np.isfinite(sr) else "  nan"
        n_tickers = len(set(t['ticker'] for t in trades))
        print(f"  {name:<25} {s['n']:>4} {sr_str:>7} {s['win_rate']:>5.0%} "
              f"{s['mean_ret']:>+5.1%} {s['worst']:>+6.1%} {n_tickers:>8}")

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
        print(f"\n  Bootstrap P(momentum > unfiltered): {p_better:.1%}")

    # Which tickers survived the momentum filter?
    if oos_adaptive:
        surviving = set(t['ticker'] for t in oos_adaptive)
        excluded = set(t['ticker'] for t in oos_base) - surviving
        print(f"\n  Tickers in momentum-filtered strategy: "
              f"{', '.join(sorted(surviving))}")
        if excluded:
            print(f"  Excluded by filter: {', '.join(sorted(excluded))}")

    print(f"\n{'=' * 90}")


if __name__ == '__main__':
    main()
