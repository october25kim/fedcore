"""FOOGD SM3D semantic-shift score head (faithful, feature-space) for Fed-CORE.

This is the NATIVE open-set score of FOOGD (NeurIPS 2024,
https://github.com/XeniaLLL/FOOGD-main), lifted faithfully and run on the SHARED
federated backbone's penultimate features:

  - score model: ``Energy(MLPScore)`` -- an MLP energy net (verbatim structure from
    FOOGD ``src/models/score.py``, with the input width set to our feature dim);
  - training objective: denoising score matching (DSM), verbatim from FOOGD
    ``ODGClient.dsm_loss2`` (``src/algorithms/FOOGD/client_foogd.py``);
  - NATIVE detection score (FOOGD ``score_method="sm"``):
        ``sm = score_model(latents).norm(dim=-1)``  (higher => more OOD / semantic-shift).
    For Fed-CORE (higher score => more ID => accept) we feed ``accept = -sm``.

The score model is trained FEDERATED (per-client DSM local steps + FedAvg of the
score-model parameters), mirroring FOOGD's federated score estimation. The backbone
here is the shared FedAvg backbone (NOT FOOGD's full SAG generalization training), so
runs using this head are labeled ``base_model_kind=representative`` ("FOOGD-style
SM3D score on shared features"), never passed off as the full FOOGD method.

Source of truth: FOOGD-main commit cloned to third_party/FOOGD-main.
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn


class MLPScore(nn.Module):
    """Energy MLP over features (FOOGD src/models/score.py, width set to feat_dim)."""

    def __init__(self, feat_dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x):
        return self.main(x)


class Energy(nn.Module):
    """Energy wrapper (FOOGD): forward returns the per-dim score field s(x)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def dsm_loss(score_model: nn.Module, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Denoising score matching, verbatim from FOOGD ODGClient.dsm_loss2.

    x_ = x + v*sigma ; s = score_model(x_) ; loss = ||s*sigma^2 + v||^2 / 2 .
    """
    v = torch.randn_like(x)
    x_ = x + v * sigma
    s = score_model(x_)
    loss = torch.norm(s * (sigma ** 2) + v, dim=-1) ** 2
    return loss.mean() / 2.0


def _avg_state(states: List[dict], weights) -> dict:
    w = np.asarray(weights, float)
    w = w / w.sum()
    avg = copy.deepcopy(states[0])
    for k, ref in avg.items():
        if torch.is_floating_point(ref):
            avg[k] = torch.stack([s[k].float() * wi for s, wi in zip(states, w)], 0).sum(0).to(ref.dtype)
        else:
            avg[k] = states[0][k]
    return avg


def feature_stats(client_feats: List[np.ndarray]):
    """Per-dim mean/std over POOLED TRAIN ID features (secure-aggregatable: sum, sumsq).

    Standardization is fit on TRAIN features only -- no audit-fold leakage. The same
    (mu, sd) standardize the audit features at inference (see :func:`sm_score`).
    """
    allf = np.concatenate([np.asarray(f, dtype=np.float64) for f in client_feats if len(f)], 0)
    mu = allf.mean(0)
    sd = allf.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def train_federated_score_model(
    client_feats: List[np.ndarray],
    feat_dim: int,
    *,
    rounds: int = 40,
    local_steps: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    sigma: float = 0.5,
    device: str = "cpu",
    seed: int = 0,
):
    """Federated DSM training of the FOOGD score model on per-client ID features.

    Features are standardized by the shared TRAIN-pooled (mu, sd); each round every
    client runs ``local_steps`` Adam steps of denoising score matching on its own
    standardized penultimate features, and the score-model parameters are
    FedAvg-averaged (size weighted). Standardization + Adam + sigma=0.5 are required
    for the 512-dim ResNet feature space (raw-feature SGD does not fit). Returns
    ``(global_score_model, mu, sd)``.
    """
    torch.manual_seed(seed)
    mu, sd = feature_stats(client_feats)
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)
    g = Energy(MLPScore(feat_dim=feat_dim)).to(device)
    sizes = [len(f) for f in client_feats]
    tensors = [((torch.from_numpy(np.asarray(f, dtype=np.float32)).to(device) - mu_t) / sd_t)
               for f in client_feats]
    rng = np.random.default_rng(seed)

    for r in range(rounds):
        states, used = [], []
        for feat, size in zip(tensors, sizes):
            if size < 2:
                continue
            local = Energy(MLPScore(feat_dim=feat_dim)).to(device)
            local.load_state_dict(copy.deepcopy(g.state_dict()))
            opt = torch.optim.Adam(local.parameters(), lr=lr)
            local.train()
            for _ in range(local_steps):
                idx = rng.integers(0, size, size=min(batch_size, size))
                xb = feat[idx]
                opt.zero_grad()
                loss = dsm_loss(local, xb, sigma=sigma)
                loss.backward()
                opt.step()
            states.append({k: v.cpu() for k, v in local.state_dict().items()})
            used.append(size)
        if states:
            g.load_state_dict(_avg_state(states, used))
    g.eval()
    return g, mu, sd


@torch.no_grad()
def sm_score(score_model: nn.Module, feats: np.ndarray, mu, sd, device: str = "cpu") -> np.ndarray:
    """FOOGD native sm score = ||score_model(standardize(latents))||_2 (higher => more OOD)."""
    score_model.to(device).eval()
    mu_t = torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(device)
    sd_t = torch.from_numpy(np.asarray(sd, dtype=np.float32)).to(device)
    f = (torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(device) - mu_t) / sd_t
    out = []
    for i in range(0, len(f), 1024):
        out.append(score_model(f[i:i + 1024]).norm(dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)
