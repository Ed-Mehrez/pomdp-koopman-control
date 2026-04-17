"""DeepSets state encoder for the option book.

Provides a permutation-invariant encoder that maps the full option-book
state to a compact latent vector z.  Three feature extraction levels:

  - per-option features: (n_options, per_option_dim) — moneyness, maturity,
    inventory, vega, delta per option
  - global features: (global_dim,) — time-to-go, variance regime, aggregate
    exposure, constraint slack
  - DeepSets encoder: per-option MLP → mean pool → concat global → MLP → z

Training is end-to-end supervised: the encoder + linear head predict reduced
action coordinates from BBG demonstrations.  After training, the linear head
is replaced by a Bayesian KRR for Level-2 uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .env import OptionBookMMState
from .spec import BBGBenchmarkConfig


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

PER_OPTION_DIM = 5   # moneyness, maturity, inventory, vega, delta
GLOBAL_DIM = 4        # tau_frac, nu_norm, vpi_norm, dist_to_limit


def extract_per_option_features(
    state: OptionBookMMState,
    config: BBGBenchmarkConfig,
) -> np.ndarray:
    """Per-option feature matrix (n_options, PER_OPTION_DIM).

    Features per option:
      0: log-moneyness  log(K/S)
      1: maturity (years)
      2: signed inventory (contracts)
      3: vega (BBG sqrt-nu vega)
      4: delta
    """
    spot = state.spot
    n_opt = config.book.n_options
    feats = np.empty((n_opt, PER_OPTION_DIM))
    for i, opt in enumerate(config.book.options):
        feats[i, 0] = np.log(opt.strike / spot)
        feats[i, 1] = opt.maturity
        feats[i, 2] = state.option_inventories[i]
        feats[i, 3] = state.option_vegas[i]
        feats[i, 4] = state.option_deltas[i]
    return feats


def extract_global_features(
    state: OptionBookMMState,
    config: BBGBenchmarkConfig,
) -> np.ndarray:
    """Global state features (GLOBAL_DIM,).

    Features:
      0: tau_frac — horizon progress
      1: nu_norm — variance / nu0
      2: vpi_norm — portfolio vega / vega_limit
      3: dist_to_limit — 1 - |vega| / vega_limit
    """
    horizon = config.control.horizon
    tau_frac = state.time / horizon if horizon > 0 else 0.0
    nu_norm = state.variance / config.heston.nu0
    vl = config.control.vega_limit
    vpi_norm = state.portfolio_vega / vl
    dist_to_limit = 1.0 - abs(state.portfolio_vega) / vl
    return np.array([tau_frac, nu_norm, vpi_norm, dist_to_limit])


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------


class FeatureNormalizer:
    """Online mean/std normalizer for per-option and global features."""

    def __init__(self) -> None:
        self.per_opt_mean: np.ndarray | None = None
        self.per_opt_std: np.ndarray | None = None
        self.global_mean: np.ndarray | None = None
        self.global_std: np.ndarray | None = None

    def fit(
        self,
        per_opt_all: np.ndarray,
        global_all: np.ndarray,
    ) -> None:
        """Compute mean/std from training data.

        per_opt_all: (N * n_options, PER_OPTION_DIM)  — flattened across steps
        global_all:  (N, GLOBAL_DIM)
        """
        self.per_opt_mean = per_opt_all.mean(axis=0)
        self.per_opt_std = np.maximum(per_opt_all.std(axis=0), 1e-8)
        self.global_mean = global_all.mean(axis=0)
        self.global_std = np.maximum(global_all.std(axis=0), 1e-8)

    def normalize_per_opt(self, x: np.ndarray) -> np.ndarray:
        return (x - self.per_opt_mean) / self.per_opt_std

    def normalize_global(self, x: np.ndarray) -> np.ndarray:
        return (x - self.global_mean) / self.global_std


# ---------------------------------------------------------------------------
# DeepSets encoder
# ---------------------------------------------------------------------------


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant encoder for the option book.

    Architecture:
      per-option MLP → mean pool → concat global → global MLP → z

    The latent z is used as input to a downstream Bayesian KRR head.
    """

    def __init__(
        self,
        per_option_dim: int = PER_OPTION_DIM,
        global_dim: int = GLOBAL_DIM,
        hidden_dim: int = 32,
        element_dim: int = 16,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.element_mlp = nn.Sequential(
            nn.Linear(per_option_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, element_dim),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(element_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        per_option_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the option book state to a latent vector.

        Parameters
        ----------
        per_option_features : (batch, n_options, per_option_dim)
        global_features : (batch, global_dim)

        Returns
        -------
        z : (batch, latent_dim)
        """
        elements = self.element_mlp(per_option_features)   # (B, N, element_dim)
        pooled = elements.mean(dim=1)                       # (B, element_dim)
        combined = torch.cat([pooled, global_features], dim=1)
        return self.global_mlp(combined)                    # (B, latent_dim)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class DeepSetsWithHead(nn.Module):
    """DeepSets encoder + linear head for end-to-end supervised training.

    The linear head maps z → reduced action coordinates.  After training,
    the encoder is frozen and the head is replaced by a Bayesian KRR.
    """

    def __init__(self, encoder: DeepSetsEncoder, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, output_dim)

    def forward(
        self,
        per_option_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(per_option_features, global_features)
        return self.head(z)

    def encode(
        self,
        per_option_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(per_option_features, global_features)


def train_deepsets_encoder(
    per_opt_train: np.ndarray,
    global_train: np.ndarray,
    targets: np.ndarray,
    latent_dim: int = 8,
    hidden_dim: int = 32,
    element_dim: int = 16,
    n_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[DeepSetsEncoder, FeatureNormalizer, dict]:
    """Train a DeepSets encoder end-to-end on reduced-coordinate targets.

    Parameters
    ----------
    per_opt_train : (N, n_options, PER_OPTION_DIM)
    global_train : (N, GLOBAL_DIM)
    targets : (N, rank) — reduced action coordinates
    latent_dim : dimension of the latent z
    hidden_dim : MLP hidden width
    element_dim : per-element output dimension
    n_epochs : training epochs
    batch_size : mini-batch size
    lr : learning rate
    weight_decay : L2 regularization
    seed : RNG seed
    verbose : print loss each epoch

    Returns
    -------
    encoder : trained DeepSetsEncoder (frozen)
    normalizer : fitted FeatureNormalizer
    info : dict with training losses and final metrics
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    N, n_options, per_opt_dim = per_opt_train.shape
    global_dim = global_train.shape[1]
    output_dim = targets.shape[1]

    # Normalize features
    normalizer = FeatureNormalizer()
    normalizer.fit(
        per_opt_train.reshape(-1, per_opt_dim),
        global_train,
    )
    po_norm = normalizer.normalize_per_opt(per_opt_train.reshape(-1, per_opt_dim))
    po_norm = po_norm.reshape(N, n_options, per_opt_dim)
    gl_norm = normalizer.normalize_global(global_train)

    # Tensors
    po_t = torch.as_tensor(po_norm, dtype=torch.float32)
    gl_t = torch.as_tensor(gl_norm, dtype=torch.float32)
    y_t = torch.as_tensor(targets, dtype=torch.float32)

    # Model
    encoder = DeepSetsEncoder(
        per_option_dim=per_opt_dim,
        global_dim=global_dim,
        hidden_dim=hidden_dim,
        element_dim=element_dim,
        latent_dim=latent_dim,
    )
    model = DeepSetsWithHead(encoder, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(po_t[idx], gl_t[idx])
            loss = loss_fn(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:4d}  loss={avg_loss:.6f}")

    # Final train R²
    model.eval()
    with torch.no_grad():
        pred_all = model(po_t, gl_t).numpy()
    ss_res = np.sum((targets - pred_all) ** 2)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
    train_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    encoder.eval()
    info = {"losses": losses, "train_r2": float(train_r2), "final_loss": losses[-1]}

    return encoder, normalizer, info
