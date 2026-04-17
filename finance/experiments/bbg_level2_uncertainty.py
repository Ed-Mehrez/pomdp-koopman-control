"""Level-2 uncertainty: deterministic encoder + Bayesian KRR head.

Compares three encoder families on the fixed ActionPCA r3 basis:
  1. compact (3D) — current winner, now with posterior variance
  2. rich (7D) — hand-built grouped exposure
  3. deepsets (learned) — permutation-invariant option-book encoder

Two-slice evaluation:
  - Slice A (in-split): seeds 2000-2399
  - Slice B (hard):     seeds 4000-4399

Diagnostics:
  - CE gap vs BBG on each slice
  - Reduced-coordinate prediction error (true BBG projected vs predicted)
  - Head posterior variance (mean, by slice)
  - Error-uncertainty correlation
  - Latent coverage (train vs in-split vs hard)

Usage:
    python finance/experiments/bbg_level2_uncertainty.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)
from applications.option_mm.metrics import cara_utility, paired_ce_posterior
from applications.option_mm_bbg.sdre_recovery import (
    _compute_rn_distances,
    extract_state_compact,
    extract_state_rich,
    extract_state_features,
    BilinearControlModel,
    ActionPCAModel,
    make_level2_controller,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cara_ce(wealths: np.ndarray, gamma: float) -> float:
    if gamma == 0.0:
        return float(np.mean(wealths))
    neg_gw = -gamma * wealths
    mx = float(np.max(neg_gw))
    return -(mx + np.log(np.mean(np.exp(neg_gw - mx)))) / gamma


def eval_seeds(config, ctrl, seeds):
    wealths = []
    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            state, _, _, _ = env.step(ctrl(state))
        wealths.append(state.wealth)
    return np.array(wealths)


def mode_seed_slices(mode: str) -> tuple[list[int], list[int]]:
    sizes = {
        "smoke": 50,
        "dev": 100,
        "formal": 400,
    }
    n = sizes[mode]
    return list(range(2000, 2000 + n)), list(range(4000, 4000 + n))


def paired_recovery_gate(w_cand, w_bbg, gamma, h, s_max, method="delta"):
    from scipy.stats import norm
    utility = cara_utility(gamma)
    post = paired_ce_posterior(w_cand, w_bbg, utility=utility, method=method)
    p_rope = float(
        norm.cdf(h, loc=post.mean, scale=post.sd_post)
        - norm.cdf(-h, loc=post.mean, scale=post.sd_post)
    )
    return {
        "mean": post.mean,
        "sd_post": post.sd_post,
        "p_rope": p_rope,
        "passes_a": p_rope >= 0.95,
        "passes_b": post.sd_post <= s_max,
    }


def load_or_build_hjb(config, cache_dir: Path, log, force: bool = False):
    cache_file = cache_dir / "hjb_paper_default_nnu20_nvpi40_ntime120.npz"
    if cache_file.exists() and not force:
        data = np.load(cache_file)
        log(f"\nLoaded cached HJB from {cache_file.name}")
        return data["t_grid"], data["nu_grid"], data["vpi_grid"], data["values"]

    log("\nSolving HJB...")
    t0 = time.time()
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    np.savez_compressed(
        cache_file,
        t_grid=t_grid,
        nu_grid=nu_grid,
        vpi_grid=vpi_grid,
        values=values,
    )
    log(f"  Solved in {time.time() - t0:.1f}s and cached to {cache_file.name}")
    return t_grid, nu_grid, vpi_grid, values


def get_action_pca_r3_basis(config, bbg_ctrl, rn_distances):
    """Build the ActionPCA r3 basis from BBG-regime demonstrations.

    This must match the basis construction used by the successful
    kernelized recovery experiment. Using random exploratory actions here
    produces a different basis that is not aligned with the BBG operating
    regime and contaminates the Level-2 comparison.
    """
    from applications.option_mm_bbg.sdre_recovery import ExplorationData

    ridge_alpha = 1e-3
    n_explore = 500
    max_dist = 10.0 * np.max(rn_distances)

    all_sf, all_u, all_vpi_pre, all_vpi_post = [], [], [], []
    all_dinv, all_spread = [], []

    for ep in range(n_explore):
        env = OptionBookMarketMakingEnv(config, seed=ep)
        state = env.reset()
        while not state.done:
            sf = extract_state_features(state, config)
            vpi_pre = state.portfolio_vega
            inv_pre = state.option_inventories.copy()
            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])
            u = np.minimum(u, max_dist)
            next_state, _, _, info = env.step(action)

            all_sf.append(sf)
            all_u.append(u)
            all_vpi_pre.append(vpi_pre)
            all_vpi_post.append(next_state.portfolio_vega)
            all_dinv.append(next_state.option_inventories - inv_pre)
            all_spread.append(info["spread_capture"])
            state = next_state

    data = ExplorationData(
        state_features=np.array(all_sf),
        actions=np.array(all_u),
        vpi_pre=np.array(all_vpi_pre),
        vpi_post=np.array(all_vpi_post),
        inventory_changes=np.array(all_dinv),
        spread_captures=np.array(all_spread),
    )

    bilinear = BilinearControlModel(config, ridge_alpha)
    bilinear.fit(data)
    env_dt = config.control.horizon / 30
    apca = ActionPCAModel(config, ridge_alpha)
    apca.fit(bilinear, config.control.gamma, config.heston.xi, env_dt)
    apca.reduce(3)
    return apca.U_r


def load_or_build_basis(config, bbg_ctrl, rn_distances, cache_dir: Path, log,
                        force: bool = False) -> np.ndarray:
    cache_file = cache_dir / "actionpca_r3_bbg_basis_500eps.npz"
    if cache_file.exists() and not force:
        data = np.load(cache_file)
        log(f"\nLoaded cached BBG-regime basis from {cache_file.name}")
        return data["U_r"]

    log("\nBuilding ActionPCA r3 basis...")
    t0 = time.time()
    U_r = get_action_pca_r3_basis(config, bbg_ctrl, rn_distances)
    np.savez_compressed(cache_file, U_r=U_r)
    log(f"  Done in {time.time() - t0:.1f}s and cached to {cache_file.name}")
    return U_r


def _basis_hash(U_r: np.ndarray) -> str:
    return hashlib.sha1(U_r.tobytes()).hexdigest()[:10]


def load_or_build_train_demo_cache(
    config,
    bbg_ctrl,
    rn_distances,
    U_r,
    train_seeds,
    cache_dir: Path,
    log,
    force: bool = False,
):
    from applications.option_mm_bbg.state_encoder import (
        extract_per_option_features,
        extract_global_features,
    )

    bkey = _basis_hash(U_r)
    cache_file = cache_dir / f"train_demos_bbg_basis_{bkey}_n{len(train_seeds)}.npz"
    if cache_file.exists() and not force:
        data = np.load(cache_file)
        log(f"\nLoaded cached training demos from {cache_file.name}")
        return {k: data[k] for k in data.files}

    log("\nCollecting and caching BBG training demonstrations...")
    t0 = time.time()
    u_baseline = np.concatenate([rn_distances, rn_distances])
    max_dist = 10.0 * np.max(rn_distances)

    # `global_all` here is the DeepSets global feature block from
    # `extract_global_features`, not the compact/rich summary state. The
    # compact/rich encoder paths use `compact_all` / `rich_all` directly.
    compact_all, rich_all, per_opt_all, global_all, y_all = [], [], [], [], []
    for seed in train_seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])
            u_clipped = np.minimum(u, max_dist)
            delta_u = u_clipped - u_baseline
            a_target = U_r.T @ delta_u

            compact_all.append(extract_state_compact(state, config))
            rich_all.append(extract_state_rich(state, config))
            per_opt_all.append(extract_per_option_features(state, config))
            global_all.append(extract_global_features(state, config))
            y_all.append(a_target)

            state, _, _, _ = env.step(action)

    bundle = {
        "compact_all": np.array(compact_all),
        "rich_all": np.array(rich_all),
        "per_opt_all": np.array(per_opt_all),
        "global_all": np.array(global_all),
        "Y_all": np.array(y_all),
    }
    np.savez_compressed(cache_file, **bundle)
    log(f"  Cached in {time.time() - t0:.1f}s to {cache_file.name}")
    return bundle


def load_or_build_slice_cache(
    config,
    bbg_ctrl,
    rn_distances,
    U_r,
    seeds,
    label: str,
    cache_dir: Path,
    log,
    force: bool = False,
):
    from applications.option_mm_bbg.state_encoder import (
        extract_per_option_features,
        extract_global_features,
    )

    bkey = _basis_hash(U_r)
    cache_file = cache_dir / f"{label}_slice_bbg_basis_{bkey}_n{len(seeds)}.npz"
    if cache_file.exists() and not force:
        data = np.load(cache_file)
        log(f"\nLoaded cached {label} slice diagnostics from {cache_file.name}")
        return {k: data[k] for k in data.files}

    log(f"\nCollecting and caching {label} slice diagnostics...")
    t0 = time.time()
    u_baseline = np.concatenate([rn_distances, rn_distances])
    max_dist = 10.0 * np.max(rn_distances)

    compact_all, rich_all, per_opt_all, global_all, a_true_all = [], [], [], [], []
    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            bbg_action = bbg_ctrl(state)
            u_bbg = np.concatenate([bbg_action.bid_distances, bbg_action.ask_distances])
            u_bbg = np.minimum(u_bbg, max_dist)
            a_true = U_r.T @ (u_bbg - u_baseline)

            compact_all.append(extract_state_compact(state, config))
            rich_all.append(extract_state_rich(state, config))
            per_opt_all.append(extract_per_option_features(state, config))
            global_all.append(extract_global_features(state, config))
            a_true_all.append(a_true)

            state, _, _, _ = env.step(bbg_action)

    bundle = {
        "compact_all": np.array(compact_all),
        "rich_all": np.array(rich_all),
        "per_opt_all": np.array(per_opt_all),
        "global_all": np.array(global_all),
        "a_true_all": np.array(a_true_all),
    }
    np.savez_compressed(cache_file, **bundle)
    log(f"  Cached in {time.time() - t0:.1f}s to {cache_file.name}")
    return bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "dev", "formal"], default="dev")
    parser.add_argument(
        "--encoders",
        default="compact,deepsets",
        help="Comma-separated encoder list from {compact,rich,deepsets}",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run heavy per-step diagnostics and latent coverage",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore caches and rebuild deterministic artifacts",
    )
    args = parser.parse_args()

    out: list[str] = []
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    cache_dir = results_dir / "cache" / "bbg_level2_uncertainty"
    results_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / f"bbg_level2_uncertainty_{date.today()}_{args.mode}.txt"

    def flush_report():
        result_file.write_text("\n".join(out))

    def log(msg=""):
        print(msg, flush=True)
        out.append(msg)
        flush_report()

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma
    rn_dists = _compute_rn_distances(config)
    train_seeds = list(range(500))
    in_split_seeds, hard_seeds = mode_seed_slices(args.mode)
    encoder_types = [e.strip() for e in args.encoders.split(",") if e.strip()]
    run_diagnostics = args.diagnostics or args.mode == "formal"

    log("=" * 70)
    log("  Level-2 Uncertainty: Encoder + Bayesian KRR Head")
    log("=" * 70)
    log(f"  Mode: {args.mode}")
    log(f"  Train: {len(train_seeds)} eps")
    log(f"  In-split eval: seeds {in_split_seeds[0]}-{in_split_seeds[-1]} "
        f"({len(in_split_seeds)} eps)")
    log(f"  Hard eval:     seeds {hard_seeds[0]}-{hard_seeds[-1]} "
        f"({len(hard_seeds)} eps)")
    log(f"  Encoders: {', '.join(encoder_types)}")
    log(f"  Diagnostics: {'on' if run_diagnostics else 'off'}")

    # === Solve BBG HJB ===
    t_grid, nu_grid, vpi_grid, values = load_or_build_hjb(
        config, cache_dir, log, force=args.force_recompute,
    )
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # === ROPE calibration ===
    pilot_seeds = list(range(5000, 5500))
    w_rn_p = eval_seeds(config, rn_ctrl, pilot_seeds)
    w_bbg_p = eval_seeds(config, bbg_ctrl, pilot_seeds)
    gap_pilot = cara_ce(w_bbg_p, gamma) - cara_ce(w_rn_p, gamma)
    h = max(abs(gap_pilot) * 0.40, 1000.0)
    s_max = h
    log(f"\n  ROPE: h={h:.0f}, s_max={s_max:.0f} (pilot gap={gap_pilot:.0f})")

    # === Build/load ActionPCA r3 basis ===
    U_r = load_or_build_basis(
        config, bbg_ctrl, rn_dists, cache_dir, log, force=args.force_recompute,
    )
    log(f"  Basis shape={U_r.shape}")

    # === Cached demonstrations ===
    train_demo_cache = load_or_build_train_demo_cache(
        config, bbg_ctrl, rn_dists, U_r, train_seeds, cache_dir, log,
        force=args.force_recompute,
    )

    # === Evaluate BBG and RN on both slices ===
    log("\nEvaluating BBG and RN references...")
    w_bbg_in = eval_seeds(config, bbg_ctrl, in_split_seeds)
    w_bbg_hard = eval_seeds(config, bbg_ctrl, hard_seeds)
    w_rn_in = eval_seeds(config, rn_ctrl, in_split_seeds)
    w_rn_hard = eval_seeds(config, rn_ctrl, hard_seeds)
    ce_bbg_in = cara_ce(w_bbg_in, gamma)
    ce_bbg_hard = cara_ce(w_bbg_hard, gamma)
    log(f"  BBG CE  in-split={ce_bbg_in:.0f}  hard={ce_bbg_hard:.0f}")
    log(f"  RN CE   in-split={cara_ce(w_rn_in, gamma):.0f}  "
        f"hard={cara_ce(w_rn_hard, gamma):.0f}")

    # === Build Level-2 controllers ===
    controllers = {}
    diagnostics_all = {}

    for enc_type in encoder_types:
        log(f"\nBuilding Level-2 controller: {enc_type}...")
        t0 = time.time()
        ctrl, diag = make_level2_controller(
            config,
            U_r=U_r,
            bbg_ctrl=bbg_ctrl,
            rn_distances=rn_dists,
            train_seeds=train_seeds,
            demo_cache=train_demo_cache,
            encoder_type=enc_type,
            n_subsample=3000,
            krr_alpha=1e-2,
            ls_multiplier=1.0,
            deepsets_latent_dim=8,
            deepsets_hidden_dim=32,
            deepsets_element_dim=16,
            deepsets_epochs=200,
            deepsets_lr=1e-3,
            deepsets_seed=0,
            return_diagnostics=True,
        )
        elapsed = time.time() - t0
        controllers[enc_type] = ctrl
        diagnostics_all[enc_type] = diag
        log(f"  {enc_type}: {elapsed:.1f}s, n_train={diag['n_train']}")
        if "encoder_train_info" in diag:
            info = diag["encoder_train_info"]
            log(f"    DeepSets train R²={info['train_r2']:.4f}, "
                f"final_loss={info['final_loss']:.6f}")

    # =====================================================================
    # Part 1: CE evaluation on both slices
    # =====================================================================
    log(f"\n{'='*70}")
    log("  Part 1: CE Evaluation")
    log(f"{'='*70}")

    log(f"\n  {'Controller':<25s} {'in-split CE':>12s} {'gap':>8s} "
        f"{'hard CE':>12s} {'gap':>8s}")
    log(f"  {'-'*70}")

    log(f"  {'BBG':<25s} {ce_bbg_in:>12.0f} {'---':>8s} "
        f"{ce_bbg_hard:>12.0f} {'---':>8s}")

    for enc_type in encoder_types:
        ctrl = controllers[enc_type]
        w_in = eval_seeds(config, ctrl, in_split_seeds)
        w_hard = eval_seeds(config, ctrl, hard_seeds)
        ce_in = cara_ce(w_in, gamma)
        ce_hard = cara_ce(w_hard, gamma)
        gap_in = ce_in - ce_bbg_in
        gap_hard = ce_hard - ce_bbg_hard

        log(f"  {enc_type:<25s} {ce_in:>12.0f} {gap_in:>+8.0f} "
            f"{ce_hard:>12.0f} {gap_hard:>+8.0f}")

        # Recovery gate on in-split
        g_in = paired_recovery_gate(w_in, w_bbg_in, gamma, h, s_max)
        g_hard = paired_recovery_gate(w_hard, w_bbg_hard, gamma, h, s_max)
        log(f"    in-split gate: mean={g_in['mean']:+.0f} sd={g_in['sd_post']:.0f} "
            f"P(ROPE)={g_in['p_rope']:.3f} "
            f"GA={'P' if g_in['passes_a'] else 'F'} "
            f"GB={'P' if g_in['passes_b'] else 'F'}")
        log(f"    hard gate:     mean={g_hard['mean']:+.0f} sd={g_hard['sd_post']:.0f} "
            f"P(ROPE)={g_hard['p_rope']:.3f} "
            f"GA={'P' if g_hard['passes_a'] else 'F'} "
            f"GB={'P' if g_hard['passes_b'] else 'F'}")

    if not run_diagnostics:
        log(f"\nSaved to {result_file}")
        return 0

    # =====================================================================
    # Part 2: Prediction error + posterior variance diagnostics
    # =====================================================================
    log(f"\n{'='*70}")
    log("  Part 2: Prediction Error & Posterior Variance")
    log(f"{'='*70}")

    import torch

    slice_caches = {
        "in-split": load_or_build_slice_cache(
            config, bbg_ctrl, rn_dists, U_r, in_split_seeds, "in_split",
            cache_dir, log, force=args.force_recompute,
        ),
        "hard": load_or_build_slice_cache(
            config, bbg_ctrl, rn_dists, U_r, hard_seeds, "hard",
            cache_dir, log, force=args.force_recompute,
        ),
    }

    for enc_type in encoder_types:
        diag = diagnostics_all[enc_type]
        krr = diag["krr"]

        ds_encoder = diag.get("encoder")
        ds_normalizer = diag.get("normalizer")

        log(f"\n  --- {enc_type} ---")

        for slice_label, bundle in slice_caches.items():
            a_true = bundle["a_true_all"]
            if enc_type == "compact":
                x_eval = bundle["compact_all"]
            elif enc_type == "rich":
                x_eval = bundle["rich_all"]
            else:
                po = bundle["per_opt_all"]
                gl = bundle["global_all"]
                po_n = ds_normalizer.normalize_per_opt(
                    po.reshape(-1, po.shape[-1]),
                ).reshape(po.shape)
                gl_n = ds_normalizer.normalize_global(gl)
                with torch.no_grad():
                    x_eval = ds_encoder(
                        torch.as_tensor(po_n, dtype=torch.float32),
                        torch.as_tensor(gl_n, dtype=torch.float32),
                    ).numpy().astype(np.float64)

            a_pred, variances = krr.predict_with_variance(x_eval)
            errors = a_true - a_pred

            # Per-coordinate RMSE
            rmse = np.sqrt(np.mean(errors ** 2, axis=0))
            mean_var = float(np.mean(variances))
            median_var = float(np.median(variances))

            log(f"  {slice_label}:")
            log(f"    n_steps = {len(errors)}")
            log(f"    per-coord RMSE: {rmse}")
            log(f"    mean GP var = {mean_var:.6f}, median = {median_var:.6f}")

            # Error-uncertainty correlation
            error_norm = np.sqrt(np.sum(errors ** 2, axis=1))
            if np.std(variances) > 1e-12 and np.std(error_norm) > 1e-12:
                corr = float(np.corrcoef(error_norm, variances)[0, 1])
            else:
                corr = 0.0
            log(f"    error-variance correlation = {corr:.4f}")

            # Variance quantiles
            q25, q50, q75, q95 = np.percentile(variances, [25, 50, 75, 95])
            log(f"    variance quantiles: "
                f"25%={q25:.6f} 50%={q50:.6f} 75%={q75:.6f} 95%={q95:.6f}")

    # =====================================================================
    # Part 3: Latent coverage diagnostics
    # =====================================================================
    log(f"\n{'='*70}")
    log("  Part 3: Latent Coverage (train vs in-split vs hard)")
    log(f"{'='*70}")

    for enc_type in ["compact", "rich"]:
        if enc_type not in diagnostics_all:
            continue
        diag = diagnostics_all[enc_type]
        X_train = diag["X_train"]

        log(f"\n  --- {enc_type} (d={X_train.shape[1]}) ---")

        for slice_label, bundle in slice_caches.items():
            X_eval = bundle[f"{enc_type}_all"]
            # Per-dimension comparison
            log(f"  {slice_label} (n={len(X_eval)}):")
            dim_names = {
                "compact": ["tau_frac", "nu_norm", "vpi_norm"],
                "rich": ["tau_frac", "nu_norm", "vpi_norm", "vpi_short",
                          "vpi_long", "vega_conc", "dist_to_limit"],
            }[enc_type]

            for d, name in enumerate(dim_names):
                train_mean = X_train[:, d].mean()
                train_std = X_train[:, d].std()
                eval_mean = X_eval[:, d].mean()
                eval_std = X_eval[:, d].std()
                # Shift in units of training std
                shift = abs(eval_mean - train_mean) / max(train_std, 1e-8)
                log(f"    {name:>15s}: train={train_mean:+.4f}±{train_std:.4f}  "
                    f"eval={eval_mean:+.4f}±{eval_std:.4f}  "
                    f"shift={shift:.2f}σ")

    # DeepSets latent coverage
    if "deepsets" in diagnostics_all:
        ds_diag = diagnostics_all["deepsets"]
        if "Z_all" in ds_diag and "encoder" in ds_diag and "normalizer" in ds_diag:
            Z_train = ds_diag["X_train"]  # subsampled latent from KRR training
            ds_enc = ds_diag["encoder"]
            ds_norm = ds_diag["normalizer"]
            latent_dim = Z_train.shape[1]
            log(f"\n  --- deepsets (d={latent_dim}) ---")
            log(f"  train latent: mean_norm={np.linalg.norm(Z_train.mean(axis=0)):.4f}, "
                f"per-dim std range=[{Z_train.std(axis=0).min():.4f}, "
                f"{Z_train.std(axis=0).max():.4f}]")

            for slice_label, bundle in slice_caches.items():
                po = bundle["per_opt_all"]
                gl = bundle["global_all"]
                po_n = ds_norm.normalize_per_opt(
                    po.reshape(-1, po.shape[-1]),
                ).reshape(po.shape)
                gl_n = ds_norm.normalize_global(gl)
                with torch.no_grad():
                    Z_eval = ds_enc(
                        torch.as_tensor(po_n, dtype=torch.float32),
                        torch.as_tensor(gl_n, dtype=torch.float32),
                    ).numpy()
                # Per-dim shift
                for d in range(min(latent_dim, 4)):  # show first 4 dims
                    t_m, t_s = Z_train[:, d].mean(), Z_train[:, d].std()
                    e_m, e_s = Z_eval[:, d].mean(), Z_eval[:, d].std()
                    shift = abs(e_m - t_m) / max(t_s, 1e-8)
                    log(f"    z[{d}] {slice_label}: train={t_m:+.4f}±{t_s:.4f}  "
                        f"eval={e_m:+.4f}±{e_s:.4f}  shift={shift:.2f}σ")

    # =====================================================================
    # Part 4: Summary and Level 3 escalation indicators
    # =====================================================================
    log(f"\n{'='*70}")
    log("  Part 4: Summary & Level 3 Escalation Check")
    log(f"{'='*70}")

    log("\n  Level 3 escalation indicators:")
    log("  1. Head variance modest on hard seeds that still fail badly?")
    log("  2. Different encoders give very different hard-slice behavior?")
    log("  3. Train vs hard latent distributions badly shifted?")
    log("  4. Richer encoder + Bayesian head still fails same pattern?")
    log("  5. Encoder does not separate easy vs hard in latent space?")
    log("\n  (Manual interpretation required — see diagnostics above)")

    # === Save ===
    flush_report()
    log(f"\nSaved to {result_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
