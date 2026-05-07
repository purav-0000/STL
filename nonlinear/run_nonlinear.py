# -*- coding: utf-8 -*-
"""
run_nonlinear.py
================

Autonomous, single-file driver for the STL nonlinear experiments
(Selective Tie Learning; UTL = Uniform Tie Learning).
Drop in Colab or run locally; no other files required.

USAGE:
    python run_nonlinear.py --experiment smoke
    python run_nonlinear.py --experiment e1
    python run_nonlinear.py --experiment e2
    python run_nonlinear.py --experiment e3
    python run_nonlinear.py --experiment e4                # score-level only
    python run_nonlinear.py --experiment e4 --include-ood  # plus OOD transfer

    # Tiny sanity-check config for any experiment:
    python run_atl.py --experiment e2 --quick

EXPERIMENTS:
    smoke : end-to-end pipeline check in both regimes (~2 min).
    e1    : per-anchor monotone suppression (claim 1: mechanism).
            Two metrics: cf_margin (Monte Carlo E_cf) and adversarial accuracy.
    e2    : budget-optimal selection (claim 2: comparison).
            STL-{PM, GA, HA} vs UTL (uniform random ties), sweep over budget k.
            Cells: regime in {decoupled, coupled} (feature-known CF only).
    e3    : score-variant agreement under coupling (necessity-of-HA claim).
            Sweep rho, report Spearman PM-GA, PM-HA, GA-HA.
    e4    : HA self-consistency at deployment budget (claim 3: deployability).
            Score-level: 4-cell (N_v, N_sub) ablation vs large-budget reference.
            OOD (with --include-ood): retrain under HA-reference and HA-deployment,
            evaluate cf_margin and adversarial accuracy, overlay strict-only and
            UTL reference lines.

OUTPUT:
    Each experiment writes to results/<exp>/ : JSON metrics + per-panel PNG figures.
    JSON storage uses internal labels ("random", "PM", "GA", "HA"); plot labels
    use the paper's display names (UTL, STL-PM, STL-GA, STL-HA).

NOTE on regime control:
    The "coupling" knob is rho in GenConfig. We set gamma=0, alpha_spur=0,
    noise_t=0 in the regime configs so rho is the SOLE driver of latent
    causal-spurious coupling (the scalar shortcut t is disabled). This makes
    the decoupled-vs-coupled ablation interpretable as a data-generating
    quantity, not a model artifact.
"""

# =======
# imports
# =======
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================================================================
# === LIBRARY: utilities ==================================================
# =========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================
# === LIBRARY: data generation ============================================
# =========================================================================
@dataclass
class GenConfig:
    dim_c: int = 10
    dim_s: int = 10
    mix_hidden: int = 64
    noise_c: float = 1.0
    noise_s: float = 1.0
    noise_t: float = 0.15
    rho: float = 0.9
    gamma: float = 10.0
    alpha_spur: float = 6.0


class FixedMixer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim), nn.Tanh(),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.net(x)


class ShortcutGenerator:
    """
    Latents: q ~ N(0,1), c = q + eps_c, s = sign*rho*q + eps_s,
             t = sign*alpha_spur*q + eps_t
    Features: phi = [ mixer([c;s]) ; gamma*t ]
    Counterfactual T flips (s, t); keeps c.

    To make rho the sole driver of latent causal-spurious coupling, set
    gamma=0, alpha_spur=0, noise_t=0 (the scalar-shortcut t is disabled).
    """
    def __init__(self, cfg: GenConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.mixer = FixedMixer(cfg.dim_c + cfg.dim_s, cfg.mix_hidden,
                                cfg.dim_c + cfg.dim_s).to(device)

    def sample_latents(self, n: int, rho_sign: float) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        q = torch.randn(n, 1, device=self.device)
        eps_c = cfg.noise_c * torch.randn(n, cfg.dim_c, device=self.device)
        eps_s = cfg.noise_s * torch.randn(n, cfg.dim_s, device=self.device)
        eps_t = cfg.noise_t * torch.randn(n, 1, device=self.device)
        c = q.repeat(1, cfg.dim_c) + eps_c
        s = (rho_sign * cfg.rho) * q.repeat(1, cfg.dim_s) + eps_s
        t = (rho_sign * cfg.alpha_spur) * q + eps_t
        return {"q": q, "c": c, "s": s, "t": t}

    def build_phi(self, c, s, t):
        base = self.mixer(torch.cat([c, s], dim=1))
        return torch.cat([base, self.cfg.gamma * t], dim=1)

    def sample_phi(self, n, rho_sign):
        lat = self.sample_latents(n, rho_sign)
        lat["phi"] = self.build_phi(lat["c"], lat["s"], lat["t"])
        return lat

    def counterfactual_phi(self, lat):
        return self.build_phi(lat["c"], -lat["s"], -lat["t"])

    @property
    def feature_dim(self) -> int:
        return (self.cfg.dim_c + self.cfg.dim_s) + 1


def regime_config(rho: float) -> GenConfig:
    """Vector-only spurious: rho is the sole knob."""
    return GenConfig(rho=rho, gamma=0.0, alpha_spur=0.0, noise_t=0.0)


def sample_strict_pairs(gen, n_pairs, beta_teacher, rho_sign):
    a = gen.sample_phi(n_pairs, rho_sign=rho_sign)
    b = gen.sample_phi(n_pairs, rho_sign=rho_sign)
    q1 = a["q"].squeeze(1); q2 = b["q"].squeeze(1)
    logits = beta_teacher * (q1 - q2)
    y = torch.bernoulli(torch.sigmoid(logits)).float().unsqueeze(1)
    return a["phi"], b["phi"], y


# =========================================================================
# === LIBRARY: scorer + training ==========================================
# =========================================================================
class ScorerMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, phi):
        return self.net(phi)


def train_pairwise(model, phi1, phi2, y,
                   beta_model=1.0, lr=2e-3, weight_decay=1e-4,
                   epochs=10, batch_size=1024, verbose=False):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n = phi1.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=phi1.device)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            r1 = model(phi1[idx]).squeeze(1)
            r2 = model(phi2[idx]).squeeze(1)
            logits = beta_model * (r1 - r2)
            loss = loss_fn(logits, y[idx].squeeze(1))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        if verbose:
            print(f"  epoch {ep}: loss={ep_loss / max(nb, 1):.4f}")


# =========================================================================
# === LIBRARY: metrics ====================================================
# =========================================================================
@torch.no_grad()
def counterfactual_margin(model, gen, n, rho_sign=1.0):
    """Monte Carlo estimate of E_cf in L1 form: mean |r(phi) - r(T phi)|."""
    model.eval()
    lat = gen.sample_phi(n, rho_sign=rho_sign)
    phi_cf = gen.counterfactual_phi(lat)
    r = model(lat["phi"]).squeeze(1); r_cf = model(phi_cf).squeeze(1)
    return (r - r_cf).abs().mean().item()


@torch.no_grad()
def counterfactual_energy_sq(model, gen, n, rho_sign=1.0):
    """
    Monte Carlo estimate of squared counterfactual energy:
        E_cf^(2) = 0.5 * E[(r(phi) - r(T phi))^2].
    This is the metric whose gradient defines v_s, so it is the theory-aligned
    energy quantity. Reported alongside cf_margin (L1 form) so E1's signed
    HA Pearson can be compared against the metric the influence formula
    actually predicts.
    """
    model.eval()
    lat = gen.sample_phi(n, rho_sign=rho_sign)
    phi_cf = gen.counterfactual_phi(lat)
    r = model(lat["phi"]).squeeze(1); r_cf = model(phi_cf).squeeze(1)
    gap = r - r_cf
    return 0.5 * (gap ** 2).mean().item()


@torch.no_grad()
def adversarial_accuracy(model, gen, n_pairs, beta_teacher):
    """OOD accuracy: held-out preference accuracy under sign-flipped spurious."""
    model.eval()
    phi1, phi2, y = sample_strict_pairs(gen, n_pairs,
                                        beta_teacher=beta_teacher, rho_sign=-1.0)
    r1 = model(phi1).squeeze(1); r2 = model(phi2).squeeze(1)
    pred = (r1 > r2).float()
    return (pred == y.squeeze(1)).float().mean().item()


@torch.no_grad()
def in_distribution_accuracy(model, gen, n_pairs, beta_teacher):
    """In-distribution accuracy: held-out preference accuracy under P (rho_sign=+1).

    Diagnostic: separates "model fits training distribution but uses spurious shortcuts"
    (id_acc high, adv_acc low) from "model is underfit" (id_acc low, adv_acc low).
    """
    model.eval()
    phi1, phi2, y = sample_strict_pairs(gen, n_pairs,
                                        beta_teacher=beta_teacher, rho_sign=+1.0)
    r1 = model(phi1).squeeze(1); r2 = model(phi2).squeeze(1)
    pred = (r1 > r2).float()
    return (pred == y.squeeze(1)).float().mean().item()


def evaluate_all(model, gen, n=10000, beta_teacher=1.0):
    """Compute cf_margin (L1), cf_energy_sq (theory-aligned), adv_acc, id_acc."""
    return {
        "cf_margin":    counterfactual_margin(model, gen, n=n),
        "cf_energy_sq": counterfactual_energy_sq(model, gen, n=n),
        "adv_acc":      adversarial_accuracy(model, gen, n_pairs=n, beta_teacher=beta_teacher),
        "id_acc":       in_distribution_accuracy(model, gen, n_pairs=n, beta_teacher=beta_teacher),
    }


@torch.no_grad()
def latent_coupling_ratio(gen, n=50000):
    """||Cov(c,s)||_F / ||Cov(c)||_F in latent space (data-generating diagnostic)."""
    lat = gen.sample_latents(n, rho_sign=+1.0)
    c = lat["c"]; s = lat["s"]
    cc = c - c.mean(0, keepdim=True)
    ss = s - s.mean(0, keepdim=True)
    Cov_cc = (cc.T @ cc) / n
    Cov_cs = (cc.T @ ss) / n
    return (torch.linalg.norm(Cov_cs).item() /
            (torch.linalg.norm(Cov_cc).item() + 1e-12))


# =========================================================================
# === LIBRARY: STL machinery ==============================================
# =========================================================================
def _flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _params_list(model):
    return [p for p in model.parameters() if p.requires_grad]


def _flat_grad(loss, params, create_graph=False):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return _flatten(grads)


class CounterfactualOperator:
    def __call__(self, anchor):
        raise NotImplementedError


class FeatureKnownCF(CounterfactualOperator):
    """Flips (s, t) latents through the fixed mixer; keeps c. Requires latents."""
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, anchor):
        return self.gen.counterfactual_phi(anchor)


def compute_v_s(model, cf_op, val_anchors, normalize=True):
    """v_s = grad_theta E_cf, optionally unit-normalized."""
    model.eval()
    params = _params_list(model)
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    phi = val_anchors["phi"]
    phi_cf = cf_op(val_anchors)
    gap = model(phi).squeeze(1) - model(phi_cf).squeeze(1)
    E_cf = 0.5 * (gap ** 2).mean()
    v_s = _flat_grad(E_cf, params, create_graph=False)
    if normalize:
        v_s = v_s / (v_s.norm() + 1e-12)
    return v_s.detach()


def _packet_loss(model, phi_a, phi_b, beta):
    """Antisymmetric tie packet: 0.5 * [softplus(-m) + softplus(m)]."""
    r_a = model(phi_a).squeeze(-1)
    r_b = model(phi_b).squeeze(-1)
    margin = beta * (r_a - r_b)
    return 0.5 * (F.softplus(-margin) + F.softplus(margin)).mean()


def packet_gradient(model, phi_a, phi_b, beta):
    params = _params_list(model)
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    L = _packet_loss(model, phi_a, phi_b, beta)
    return _flat_grad(L, params, create_graph=False).detach()


class FisherOracle:
    """
    Empirical Fisher F = (1/N_sub) G^T G + lambda * I.
    PSD by construction. Per-sample gradients via torch.func.vmap.
    """
    def __init__(self, model, phi1, phi2, y, beta=1.0,
                 damping=1e-3, hvp_subsample=None):
        self.model = model
        self.params = _params_list(model)
        self.beta = beta
        self.damping = damping
        if hvp_subsample is not None and hvp_subsample < phi1.shape[0]:
            idx = torch.randperm(phi1.shape[0], device=phi1.device)[:hvp_subsample]
            phi1 = phi1[idx]; phi2 = phi2[idx]; y = y[idx]
        self.G = self._per_sample_grads(phi1, phi2, y)
        self.N = self.G.shape[0]

    def _per_sample_grads(self, phi1, phi2, y):
        from torch.func import functional_call, grad, vmap
        params_dict = {n: p.detach() for n, p in self.model.named_parameters()
                       if p.requires_grad}
        param_names = list(params_dict.keys())
        beta = self.beta

        def single_loss(params, p1, p2, y_i):
            r1 = functional_call(self.model, params, (p1.unsqueeze(0),)).squeeze()
            r2 = functional_call(self.model, params, (p2.unsqueeze(0),)).squeeze()
            margin = beta * (r1 - r2)
            return F.binary_cross_entropy_with_logits(margin, y_i.squeeze())

        grad_fn = grad(single_loss)
        per_sample = vmap(grad_fn, in_dims=(None, 0, 0, 0))(params_dict, phi1, phi2, y)
        flat_rows = []
        for name in param_names:
            t = per_sample[name]
            flat_rows.append(t.reshape(t.shape[0], -1))
        return torch.cat(flat_rows, dim=1).detach()

    def __call__(self, v):
        Gv = self.G @ v
        out = (self.G.T @ Gv) / self.N
        if self.damping > 0.0:
            out = out + self.damping * v
        return out


def conjugate_gradients(hvp, b, max_iter=50, tol=1e-4):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs_old = (r * r).sum()
    b_norm = b.norm() + 1e-12
    for _ in range(max_iter):
        Ap = hvp(p)
        pAp = (p * Ap).sum() + 1e-20
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if (rs_new.sqrt() / b_norm).item() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x.detach()


def score_pure_margin(model, anchors, cf_op, beta=1.0, signed=False):
    """PM_j = beta * |r(phi_j) - r(T phi_j)|, or signed if signed=True."""
    model.eval()
    with torch.no_grad():
        phi = anchors["phi"]
        phi_cf = cf_op(anchors)
        u = model(phi).squeeze(-1) - model(phi_cf).squeeze(-1)
        return (u if signed else u.abs()) * beta


def score_gradient_alignment(model, anchors, cf_op, v_s, beta=1.0, signed=False):
    """GA_j = |v_s^T g_j|, or v_s^T g_j if signed=True."""
    n = anchors["phi"].shape[0]
    scores = torch.zeros(n, device=anchors["phi"].device)
    for j in range(n):
        anchor_j = {k: v[j:j+1] for k, v in anchors.items()}
        phi_a = anchor_j["phi"]
        phi_b = cf_op(anchor_j)
        g_j = packet_gradient(model, phi_a, phi_b, beta)
        s = (v_s * g_j).sum().item()
        scores[j] = s if signed else abs(s)
    return scores


def score_hessian_aware(model, anchors, cf_op, v_s, hvp_oracle,
                        beta=1.0, cg_max_iter=50, cg_tol=1e-4, signed=False):
    """HA_j = |v_s^T F^{-1} g_j| via CG, or v_s^T F^{-1} g_j if signed=True."""
    n = anchors["phi"].shape[0]
    scores = torch.zeros(n, device=anchors["phi"].device)
    for j in range(n):
        anchor_j = {k: v[j:j+1] for k, v in anchors.items()}
        phi_a = anchor_j["phi"]
        phi_b = cf_op(anchor_j)
        g_j = packet_gradient(model, phi_a, phi_b, beta)
        u_j = conjugate_gradients(hvp_oracle, g_j, max_iter=cg_max_iter, tol=cg_tol)
        s = (v_s * u_j).sum().item()
        scores[j] = s if signed else abs(s)
    return scores


def score_all_three(model, phi1, phi2, y, anchors, cf_op,
                    n_val=2000, n_fisher=2000, beta=1.0,
                    fisher_damping=1e-3, cg_max_iter=50, cg_tol=1e-4,
                    gen_for_val=None, rho_sign_val=+1.0, signed=False):
    """
    Convenience wrapper: compute v_s and Fisher, return (PM, GA, HA) for anchors.
    `gen_for_val` is the generator used to sample fresh validation anchors.
    If signed=True, returns signed (not absolute-value) scores; useful for
    testing the influence-function directional prediction.
    """
    pm = score_pure_margin(model, anchors, cf_op, beta=beta, signed=signed)
    val_anchors = gen_for_val.sample_phi(n_val, rho_sign=rho_sign_val)
    v_s = compute_v_s(model, cf_op, val_anchors)
    ga = score_gradient_alignment(model, anchors, cf_op, v_s, beta=beta, signed=signed)
    fisher = FisherOracle(model, phi1, phi2, y, beta=beta,
                          damping=fisher_damping, hvp_subsample=n_fisher)
    ha = score_hessian_aware(model, anchors, cf_op, v_s, fisher,
                             beta=beta, cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                             signed=signed)
    return pm, ga, ha


# =========================================================================
# === EXPERIMENT: smoke ===================================================
# =========================================================================
def run_smoke(args):
    """End-to-end pipeline check in both regimes."""
    print(f"[smoke] device={args.device}")
    out_dir = os.path.join(args.out_root, "smoke")
    ensure_dir(out_dir)

    from scipy.stats import spearmanr
    rows = []
    for rho, label in [(0.0, "decoupled"), (0.9, "coupled")]:
        print(f"\n--- regime: {label} (rho={rho}) ---")
        set_seed(0)
        gen = ShortcutGenerator(regime_config(rho), device=args.device)
        coupling = latent_coupling_ratio(gen, n=20000)
        print(f"  latent coupling: {coupling:.3f}")

        n_train = 5000
        phi1, phi2, y = sample_strict_pairs(gen, n_train, beta_teacher=args.beta_teacher, rho_sign=+1.0)
        model = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
        train_pairwise(model, phi1, phi2, y, epochs=args.epochs, batch_size=512)
        ev = evaluate_all(model, gen, n=5000, beta_teacher=args.beta_teacher)
        print(f"  strict-only: cf_margin={ev['cf_margin']:.3f}, "
              f"id_acc={ev['id_acc']:.3f}, adv_acc={ev['adv_acc']:.3f}")

        cf = FeatureKnownCF(gen)
        anchors = gen.sample_phi(100, rho_sign=+1.0)
        pm, ga, ha = score_all_three(model, phi1, phi2, y, anchors, cf,
                                     n_val=2000, n_fisher=2000,
                                     gen_for_val=gen)
        sp_pmga = spearmanr(pm.cpu().numpy(), ga.cpu().numpy()).correlation
        sp_pmha = spearmanr(pm.cpu().numpy(), ha.cpu().numpy()).correlation
        sp_gaha = spearmanr(ga.cpu().numpy(), ha.cpu().numpy()).correlation
        print(f"  Spearman: PM-GA={sp_pmga:.3f}, PM-HA={sp_pmha:.3f}, GA-HA={sp_gaha:.3f}")

        rows.append({
            "regime": label, "rho": rho, "coupling": coupling,
            "cf_margin": ev["cf_margin"], "adv_acc": ev["adv_acc"],
            "id_acc": ev["id_acc"],
            "spearman_pm_ga": sp_pmga,
            "spearman_pm_ha": sp_pmha,
            "spearman_ga_ha": sp_gaha,
        })

    with open(os.path.join(out_dir, "smoke_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[smoke] wrote {os.path.join(out_dir, 'smoke_results.json')}")
    return rows


# =========================================================================
# === EXPERIMENT: E1 ======================================================
# Per-anchor monotone suppression test. Tests claim 1 (mechanism).
# Two metric panels: cf_margin (Monte Carlo E_cf) and adversarial accuracy.
# =========================================================================
def stratified_subsample_idx(scores, n_per_bin, n_bins=5):
    sorted_idx = torch.argsort(scores)
    n = len(sorted_idx); bin_size = n // n_bins
    selected = []
    for b in range(n_bins):
        start = b * bin_size
        stop = start + bin_size if b < n_bins - 1 else n
        bin_idx = sorted_idx[start:stop]
        if len(bin_idx) <= n_per_bin:
            selected.extend(bin_idx.tolist())
        else:
            choice = torch.randperm(len(bin_idx))[:n_per_bin]
            selected.extend(bin_idx[choice].tolist())
    return torch.tensor(selected)


def measure_delta_metrics(strict_data, anchor, cf_op, gen, in_dim,
                          baseline_cm, baseline_adv, baseline_cf_sq,
                          epsilon_packets=200, n_eval=10000,
                          n_seeds=1, device="cpu", beta_teacher=1.0,
                          epochs=10):
    """
    Augment strict data with epsilon_packets copies of this anchor's tie,
    retrain, return (Delta cf_margin, Delta adv_acc, Delta cf_energy_sq)
    averaged over seeds.

    cf_energy_sq is the metric whose gradient defines v_s, so it is the
    quantity the signed influence formula predicts. cf_margin (L1) is the
    operational proxy. Reporting both lets the directional-influence test
    distinguish theory-predicted from operational behavior.
    """
    phi1_s, phi2_s, y_s = strict_data
    phi_a = anchor["phi"]
    phi_b = cf_op(anchor)
    n = epsilon_packets
    # Deterministic symmetric packet: split n total rows into n//2 with y=1
    # and (n - n//2) with y=0. Realizes the symmetric tie measure exactly
    # per anchor, with no Monte Carlo label noise. Matches the symmetric
    # tie loss used at scoring time.
    n_pos = n // 2
    n_neg = n - n_pos
    phi1_t = phi_a.repeat(n, 1)
    phi2_t = phi_b.repeat(n, 1)
    y_t = torch.cat([
        torch.ones(n_pos, 1, device=device),
        torch.zeros(n_neg, 1, device=device),
    ], dim=0)

    phi1 = torch.cat([phi1_s, phi1_t], dim=0)
    phi2 = torch.cat([phi2_s, phi2_t], dim=0)
    y = torch.cat([y_s, y_t], dim=0)

    cms, advs, cfsqs = [], [], []
    for sd in range(n_seeds):
        set_seed(7000 + sd)
        model = ScorerMLP(in_dim=in_dim, hidden=64).to(device)
        train_pairwise(model, phi1, phi2, y, epochs=epochs, batch_size=512)
        cms.append(counterfactual_margin(model, gen, n=n_eval))
        advs.append(adversarial_accuracy(model, gen, n_pairs=n_eval,
                                         beta_teacher=beta_teacher))
        cfsqs.append(counterfactual_energy_sq(model, gen, n=n_eval))
    return (float(np.mean(cms)) - baseline_cm,
            float(np.mean(advs)) - baseline_adv,
            float(np.mean(cfsqs)) - baseline_cf_sq)


# =========================================================================
# === Plotting display utilities ==========================================
# Centralized mapping from internal storage labels (used in JSON/code) to
# paper display names. Keep JSON storage stable; vary only display.
# =========================================================================
DISPLAY_LABEL = {
    "random":     "UTL",
    "PM":         "STL-PM",
    "GA":         "STL-GA",
    "HA":         "STL-HA",
    "HA_ref":     "STL-HA (ref)",
    "HA_deploy":  "STL-HA (deploy)",
    "strict_only":"strict-only",
}

DISPLAY_COLOR = {
    "random":     "#888",
    "PM":         "#1f77b4",
    "GA":         "#2ca02c",
    "HA":         "#d62728",
    "HA_ref":     "#1f77b4",
    "HA_deploy":  "#d62728",
    "strict_only":"black",
}

DISPLAY_MARKER = {
    "random":     "s",
    "PM":         "o",
    "GA":         "^",
    "HA":         "D",
    "HA_ref":     "o",
    "HA_deploy":  "D",
}


def run_e1_regime(rho, regime_label, args, out_dir):
    from scipy.stats import spearmanr
    print(f"\n{'='*70}\nE1 regime: {regime_label} (rho={rho})\n{'='*70}")
    set_seed(0)
    gen = ShortcutGenerator(regime_config(rho), device=args.device)

    print(f"training strict-only on {args.n_train} pairs...")
    phi1, phi2, y = sample_strict_pairs(gen, args.n_train,
                                        beta_teacher=args.beta_teacher, rho_sign=+1.0)
    model = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
    train_pairwise(model, phi1, phi2, y, epochs=args.epochs, batch_size=512)
    cm_strict = counterfactual_margin(model, gen, n=10000)
    cf_sq_strict = counterfactual_energy_sq(model, gen, n=10000)
    adv_strict = adversarial_accuracy(model, gen, n_pairs=10000, beta_teacher=args.beta_teacher)
    id_strict = in_distribution_accuracy(model, gen, n_pairs=10000, beta_teacher=args.beta_teacher)
    print(f"  strict-only: cf_margin={cm_strict:.4f}, "
          f"cf_energy_sq={cf_sq_strict:.4f}, "
          f"id_acc={id_strict:.4f}, adv_acc={adv_strict:.4f}")

    print(f"scoring {args.n_anchors} anchors...")
    anchors = gen.sample_phi(args.n_anchors, rho_sign=+1.0)
    cf = FeatureKnownCF(gen)
    # Unsigned scores (for ranking analysis).
    pm, ga, ha = score_all_three(model, phi1, phi2, y, anchors, cf,
                                 n_val=2000, n_fisher=2000,
                                 gen_for_val=gen,
                                 fisher_damping=1e-3,
                                 cg_max_iter=50, cg_tol=1e-4)
    # Signed scores (for influence-direction test).
    pm_s, ga_s, ha_s = score_all_three(model, phi1, phi2, y, anchors, cf,
                                       n_val=2000, n_fisher=2000,
                                       gen_for_val=gen,
                                       fisher_damping=1e-3,
                                       cg_max_iter=50, cg_tol=1e-4,
                                       signed=True)
    print(f"  PM: [{pm.min():.3f}, {pm.max():.3f}]")

    n_per_bin = max(1, args.n_subsample // 5)
    sub_idx = stratified_subsample_idx(pm, n_per_bin=n_per_bin, n_bins=5)
    print(f"subsampled {len(sub_idx)} anchors for retraining")

    pm_sub = pm[sub_idx].cpu().numpy()
    ga_sub = ga[sub_idx].cpu().numpy()
    ha_sub = ha[sub_idx].cpu().numpy()
    pm_s_sub = pm_s[sub_idx].cpu().numpy()
    ga_s_sub = ga_s[sub_idx].cpu().numpy()
    ha_s_sub = ha_s[sub_idx].cpu().numpy()
    dE_cm, dE_adv, dE_cf_sq = [], [], []

    for k, j in enumerate(sub_idx):
        anchor_j = {key: val[j:j+1] for key, val in anchors.items()}
        d_cm, d_adv, d_cf_sq = measure_delta_metrics(
            (phi1, phi2, y), anchor_j, cf, gen,
            in_dim=gen.feature_dim,
            baseline_cm=cm_strict, baseline_adv=adv_strict,
            baseline_cf_sq=cf_sq_strict,
            epsilon_packets=args.epsilon_packets,
            n_eval=10000, n_seeds=args.n_retrain_seeds,
            device=args.device, beta_teacher=args.beta_teacher,
            epochs=args.epochs,
        )
        dE_cm.append(d_cm); dE_adv.append(d_adv); dE_cf_sq.append(d_cf_sq)
        if (k + 1) % 5 == 0 or k == len(sub_idx) - 1:
            print(f"  retrained {k+1}/{len(sub_idx)}: PM={pm[j]:.3f}, "
                  f"d_cm={d_cm:+.4f}, d_cf_sq={d_cf_sq:+.4f}, d_adv={d_adv:+.4f}")

    dE_cm = np.array(dE_cm); dE_adv = np.array(dE_adv); dE_cf_sq = np.array(dE_cf_sq)

    # Spearman: cf_margin and cf_energy_sq should DECREASE with score (negative);
    # adv_acc should INCREASE (positive).
    sp = {}
    for sname, scores in [("PM", pm_sub), ("GA", ga_sub), ("HA", ha_sub)]:
        sp[f"{sname}_dCM"] = spearmanr(scores, dE_cm).correlation
        sp[f"{sname}_dCFsq"] = spearmanr(scores, dE_cf_sq).correlation
        sp[f"{sname}_dAdv"] = spearmanr(scores, dE_adv).correlation

    # Directional influence test: Pearson(signed score, Delta).
    # Theory predicts Delta E_cf^(2) ~ -epsilon * v_s^T H^-1 g_j, so signed HA
    # should NEGATIVELY correlate with empirical Delta cf_energy_sq.
    # cf_margin (L1) is an operational proxy; correlation with signed HA may be
    # weaker, especially at low coupling.
    pe = {}
    for sname, scores_signed in [("PM", pm_s_sub), ("GA", ga_s_sub), ("HA", ha_s_sub)]:
        pe[f"{sname}_signed_dCM"] = float(np.corrcoef(scores_signed, dE_cm)[0, 1])
        pe[f"{sname}_signed_dCFsq"] = float(np.corrcoef(scores_signed, dE_cf_sq)[0, 1])
        pe[f"{sname}_signed_dAdv"] = float(np.corrcoef(scores_signed, dE_adv)[0, 1])
    print(f"\nSpearman(score, delta cf_margin) [theory: NEGATIVE]:")
    for k in ("PM_dCM", "GA_dCM", "HA_dCM"):
        print(f"  {k}: {sp[k]:+.3f}")
    print(f"Spearman(score, delta cf_energy_sq) [theory: NEGATIVE; theory-aligned metric]:")
    for k in ("PM_dCFsq", "GA_dCFsq", "HA_dCFsq"):
        print(f"  {k}: {sp[k]:+.3f}")
    print(f"Spearman(score, delta adv_acc)   [theory: POSITIVE]:")
    for k in ("PM_dAdv", "GA_dAdv", "HA_dAdv"):
        print(f"  {k}: {sp[k]:+.3f}")
    print(f"Pearson(SIGNED score, delta cf_margin) [first-order influence; L1 proxy]:")
    for k in ("PM_signed_dCM", "GA_signed_dCM", "HA_signed_dCM"):
        print(f"  {k}: {pe[k]:+.3f}")
    print(f"Pearson(SIGNED score, delta cf_energy_sq) [first-order influence; theory-aligned]:")
    for k in ("PM_signed_dCFsq", "GA_signed_dCFsq", "HA_signed_dCFsq"):
        print(f"  {k}: {pe[k]:+.3f}")
    print(f"Pearson(SIGNED score, delta adv_acc):")
    for k in ("PM_signed_dAdv", "GA_signed_dAdv", "HA_signed_dAdv"):
        print(f"  {k}: {pe[k]:+.3f}")

    payload = {
        "regime": regime_label, "rho": rho,
        "cm_strict": float(cm_strict),
        "cf_sq_strict": float(cf_sq_strict),
        "adv_strict": float(adv_strict),
        "pm": pm_sub.tolist(), "ga": ga_sub.tolist(), "ha": ha_sub.tolist(),
        "pm_signed": pm_s_sub.tolist(),
        "ga_signed": ga_s_sub.tolist(),
        "ha_signed": ha_s_sub.tolist(),
        "delta_cm": dE_cm.tolist(),
        "delta_cf_sq": dE_cf_sq.tolist(),
        "delta_adv": dE_adv.tolist(),
        "spearman": {k: float(v) for k, v in sp.items()},
        "pearson_signed": pe,
    }
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, f"e1_{regime_label}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # Per-panel plots: 6 separate PNGs per regime (3 score variants x 2 metrics).
    # Filenames: e1_{regime}_{metric}_{score}.png with metric in
    # {dCM, dCFsq, dAdv}. Storage label "PM"/"GA"/"HA" maps to display
    # "STL-PM"/"STL-GA"/"STL-HA".
    import matplotlib.pyplot as plt
    score_arr = [(pm_sub, "PM"), (ga_sub, "GA"), (ha_sub, "HA")]
    panel_specs = [
        ("dCM",   dE_cm,    r"$\Delta$ cf_margin (L1)",            "tab:blue"),
        ("dCFsq", dE_cf_sq, r"$\Delta E_{\rm cf}^{(2)}$ (theory)", "tab:green"),
        ("dAdv",  dE_adv,   r"$\Delta$ adv_acc (OOD)",             "tab:orange"),
    ]
    for sc, name in score_arr:
        for metric_tag, dE_arr, ylabel, color in panel_specs:
            fig, ax = plt.subplots(figsize=(4.5, 3.7))
            ax.scatter(sc, dE_arr, alpha=0.7, s=28, color=color)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"{DISPLAY_LABEL[name]} score")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{DISPLAY_LABEL[name]} ({regime_label}, "
                         rf"$\rho$={rho}), Spearman={sp[name+'_d'+metric_tag[1:]]:+.2f}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"e1_{regime_label}_{metric_tag}_{name}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=120)
            plt.close(fig)
    print(f"[e1] wrote 9 per-panel plots to {out_dir}/e1_{regime_label}_*.png")
    return payload


def run_e1(args):
    out_dir = os.path.join(args.out_root, "e1")
    ensure_dir(out_dir)
    if args.quick:
        args.n_train = 1500; args.n_anchors = 100
        args.n_subsample = 15; args.epsilon_packets = 200
        args.n_retrain_seeds = 1
    # Mid-coupled (rho=0.5) and strong-coupled (rho=0.9): both deployment-relevant.
    # Decoupled (rho=0.0) is a control regime where no shortcut exists; per-anchor
    # signal is noise-dominated and not informative about the mechanism.
    res = {}
    for rho, label in [(0.5, "mid_coupled"), (0.9, "coupled")]:
        res[label] = run_e1_regime(rho, label, args, out_dir)
    return res


# =========================================================================
# === EXPERIMENT: E2 ======================================================
# Budget-optimal selection: STL-{PM,GA,HA} vs UTL across budgets.
# Two regimes (decoupled, coupled), feature-known counterfactual only.
# =========================================================================
def build_tie_packet(anchors, cf_op, device):
    """
    Build a deterministic symmetric tie packet for the given anchors.

    For each of the n input anchors, this emits TWO training rows: one with
    label y=1 (phi_a preferred over phi_b) and one with label y=0 (phi_b
    preferred over phi_a). This realizes the symmetric tie measure
        Q_j = 0.5 * delta_{a > b} + 0.5 * delta_{b > a}
    exactly per anchor, with the BCE losses canceling antisymmetrically to
    produce the tanh(beta * m / 2) gradient that matches the scoring-time
    tie packet loss.

    Returns three tensors of length 2n.

    Note: prior versions sampled a single random label per anchor (Monte
    Carlo realization of Q_j), which is unbiased in expectation but adds
    finite-sample label variance. The deterministic version is variance-free
    and exactly matches the theory's antisymmetric packet construction.
    """
    phi_a = anchors["phi"]
    phi_b = cf_op(anchors)
    n = phi_a.shape[0]
    # Stack both orientations: first n rows have y=1, next n have y=0.
    phi1 = torch.cat([phi_a, phi_a], dim=0)
    phi2 = torch.cat([phi_b, phi_b], dim=0)
    y = torch.cat([
        torch.ones(n, 1, device=device),
        torch.zeros(n, 1, device=device),
    ], dim=0)
    return phi1, phi2, y


def topk_idx(scores, k, mode='abs'):
    """
    Select top-k anchors by scoring criterion.

    Note: this function defaults to mode='abs' for safety when called
    directly from library code. The CLI flag --selection-mode defaults to
    'relu' (the theoretically justified positive-part rule). Explicit mode
    arguments at call sites in the experiment runners override this default.

    mode='abs'    : rank by |s_j|, take top-k. Treats positive and negative
                    magnitudes equivalently.
    mode='signed' : rank by s_j directly (largest positive first).
    mode='relu'   : rank by max(s_j, 0). Restricts selection to candidates
                    whose first-order energy contribution is monotonically
                    decreasing (s_j > 0 under the signed convention used in
                    score_gradient_alignment / score_hessian_aware). Candidates
                    with s_j <= 0 are tied at the bottom of the ranking and
                    only selected if k exceeds the number of positive-score
                    candidates.

    Note: PM has no first-order energy interpretation tied to its sign, so
    PM is always selected via mode='abs' regardless of this argument.
    """
    if mode == 'abs':
        key = scores.abs()
    elif mode == 'signed':
        key = scores
    elif mode == 'relu':
        key = scores.clamp(min=0)
    else:
        raise ValueError(f"unknown topk_idx mode: {mode!r}")
    return torch.topk(key, k=k).indices


def positive_fraction_selected(signed_scores, idx):
    """
    Diagnostic: among the selected indices, what fraction had a strictly
    positive signed score?

    For GA/HA selected via mode='abs', this measures whether magnitude-
    ranking is pulling in anti-suppressive (s_j <= 0) candidates. Under
    mode='relu' this should be 1.0 unless k exceeds the number of
    positive-score candidates. Useful for the abs-vs-relu ablation.

    Returns NaN if idx is empty.
    """
    if len(idx) == 0:
        return float('nan')
    sub = signed_scores[idx]
    return float((sub > 0).float().mean().item())


def run_e2_cell(rho, regime_label, args, out_dir):
    """One regime cell of E2 (feature-known counterfactual)."""
    print(f"\n{'='*70}\nE2 cell: {regime_label} (rho={rho})\n{'='*70}")
    cfg = regime_config(rho)
    rows = []

    for sd in args.seeds:
        print(f"\n--- seed {sd} ---")
        set_seed(5000 + sd)
        gen = ShortcutGenerator(cfg, device=args.device)
        phi1_s, phi2_s, y_s = sample_strict_pairs(gen, args.n_train,
                                                  beta_teacher=args.beta_teacher, rho_sign=+1.0)
        m_strict = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
        train_pairwise(m_strict, phi1_s, phi2_s, y_s, epochs=args.epochs, batch_size=512)
        ev0 = evaluate_all(m_strict, gen, n=10000, beta_teacher=args.beta_teacher)
        print(f"  strict-only: cf_margin={ev0['cf_margin']:.3f}, "
              f"id_acc={ev0['id_acc']:.3f}, adv_acc={ev0['adv_acc']:.3f}")
        rows.append({"seed": sd, "k": 0, "strategy": "strict_only", **ev0})

        anchors = gen.sample_phi(args.n_anchor_pool, rho_sign=+1.0)
        cf = FeatureKnownCF(gen)

        pm, ga, ha = score_all_three(m_strict, phi1_s, phi2_s, y_s,
                                     anchors, cf, n_val=2000, n_fisher=2000,
                                     gen_for_val=gen, fisher_damping=1e-3,
                                     cg_max_iter=50, cg_tol=1e-4,
                                     signed=True)

        for k in args.budgets:
            strategies = {
                "random": torch.randperm(args.n_anchor_pool, device=args.device)[:k],
                "PM": topk_idx(pm, k, mode='abs'),
                "GA": topk_idx(ga, k, mode=args.selection_mode),
                "HA": topk_idx(ha, k, mode=args.selection_mode),
            }
            # signed scores per strategy for the positive-fraction diagnostic
            # Note: PM is omitted from the positive-fraction diagnostic
            # because PM's sign is orientation-dependent (sign of the raw
            # margin r(phi_a) - r(phi_b)), not an energy-descent indicator.
            # PM's positive_fraction is reported as NaN in the row data.
            signed_for = {"random": None, "PM": None, "GA": ga, "HA": ha}
            for name, idx in strategies.items():
                anchors_sel = {key: val[idx] for key, val in anchors.items()}
                phi1_t, phi2_t, y_t = build_tie_packet(anchors_sel, cf, args.device)
                phi1_aug = torch.cat([phi1_s, phi1_t], dim=0)
                phi2_aug = torch.cat([phi2_s, phi2_t], dim=0)
                y_aug = torch.cat([y_s, y_t], dim=0)
                m_aug = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
                train_pairwise(m_aug, phi1_aug, phi2_aug, y_aug,
                               epochs=args.epochs, batch_size=512)
                ev = evaluate_all(m_aug, gen, n=10000, beta_teacher=args.beta_teacher)
                pos_frac = (positive_fraction_selected(signed_for[name], idx)
                            if signed_for[name] is not None else float('nan'))
                rows.append({"seed": sd, "k": k, "strategy": name,
                             "selection_mode": args.selection_mode,
                             "positive_fraction_selected": pos_frac, **ev})
                print(f"  k={k:5d}, {name:7s}: cf_margin={ev['cf_margin']:.3f}, "
                      f"id_acc={ev['id_acc']:.3f}, adv_acc={ev['adv_acc']:.3f}, "
                      f"pos_frac={pos_frac:.2f}")

    ensure_dir(out_dir)
    fname = f"e2_{regime_label}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def plot_e2_cell(rows, regime_label, budgets, out_dir):
    import matplotlib.pyplot as plt
    strategies = ["random", "PM", "GA", "HA"]

    # Aggregate (mean, std) per (strategy, budget) for each metric.
    def aggregate(metric_key):
        agg = {}
        for strat in strategies:
            means, stds = [], []
            for k in budgets:
                vs = [r[metric_key] for r in rows
                      if r["k"] == k and r["strategy"] == strat]
                means.append(np.mean(vs))
                stds.append(np.std(vs, ddof=1) if len(vs) > 1 else 0)
            agg[strat] = (means, stds)
        baseline = np.mean([r[metric_key] for r in rows
                            if r["strategy"] == "strict_only"])
        return agg, baseline

    agg_cm, cm0       = aggregate("cf_margin")
    agg_cf_sq, cf_sq0 = aggregate("cf_energy_sq")
    agg_ad, ad0       = aggregate("adv_acc")

    # Per-panel plots: 3 separate PNGs (cf_margin, cf_energy_sq, adv_acc).
    # cf_energy_sq is the theory-aligned energy whose gradient defines v_s;
    # cf_margin is the operational L1 proxy; adv_acc is the OOD robustness
    # metric.
    panel_specs = [
        ("cf_margin",    agg_cm,    "cf_margin (lower = more suppression)",                       cm0),
        ("cf_energy_sq", agg_cf_sq, r"$E_{\rm cf}^{(2)}$ (theory-aligned; lower = more suppression)", cf_sq0),
        ("adv_acc",      agg_ad,    "adversarial accuracy (higher = better OOD)",                 ad0),
    ]
    for metric_tag, agg, ylabel, baseline in panel_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for strat in strategies:
            means, stds = agg[strat]
            ax.errorbar(budgets, means, yerr=stds,
                        marker=DISPLAY_MARKER[strat], color=DISPLAY_COLOR[strat],
                        label=DISPLAY_LABEL[strat], capsize=3, lw=2)
        ax.axhline(baseline, color="black", ls="--", lw=1, alpha=0.5,
                   label=DISPLAY_LABEL["strict_only"])
        ax.set_xlabel("Tie budget k")
        ax.set_ylabel(ylabel)
        ax.set_title(f"E2 {metric_tag}: {regime_label} regime "
                     rf"($\rho$={'0.9' if regime_label=='coupled' else '0.0'})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = f"e2_{regime_label}_{metric_tag}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)
    print(f"[e2] wrote 3 per-panel plots to {out_dir}/e2_{regime_label}_*.png")


def run_e2(args):
    out_dir = os.path.join(args.out_root, "e2")
    ensure_dir(out_dir)
    if args.quick:
        args.n_train = 2000; args.n_anchor_pool = 200
        args.budgets = [50, 150]; args.seeds = [0]
    # Feature-known counterfactual only. The previous probe-based "agnostic"
    # operator did not preserve causal content per-anchor and has been removed.
    # Mixture mode (multiple known T_i) is the right next step but is deferred.
    res = {}
    for rho, label in [(0.0, "decoupled"), (0.9, "coupled")]:
        rows = run_e2_cell(rho, label, args, out_dir)
        plot_e2_cell(rows, label, args.budgets, out_dir)
        res[label] = rows
    return res


# =========================================================================
# === EXPERIMENT: E3 ======================================================
# Score-variant agreement vs rho. Necessity-of-HA argument.
# =========================================================================
def run_e3(args):
    from scipy.stats import spearmanr
    out_dir = os.path.join(args.out_root, "e3")
    ensure_dir(out_dir)
    if args.quick:
        args.rhos = [0.0, 0.5, 0.9]; args.seeds = [0]
        args.n_train = 3000; args.n_anchors = 100

    results = []
    for rho in args.rhos:
        print(f"\n=== rho={rho} ===")
        sp_pmga, sp_pmha, sp_gaha = [], [], []
        couplings, cms, id_accs = [], [], []
        for sd in args.seeds:
            set_seed(8000 + sd)
            gen = ShortcutGenerator(regime_config(rho), device=args.device)
            couplings.append(latent_coupling_ratio(gen, n=20000))
            phi1, phi2, y = sample_strict_pairs(gen, args.n_train,
                                                beta_teacher=args.beta_teacher, rho_sign=+1.0)
            model = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
            train_pairwise(model, phi1, phi2, y, epochs=args.epochs, batch_size=512)
            cms.append(counterfactual_margin(model, gen, n=5000))
            id_accs.append(in_distribution_accuracy(model, gen, n_pairs=5000,
                                                    beta_teacher=args.beta_teacher))

            anchors = gen.sample_phi(args.n_anchors, rho_sign=+1.0)
            cf = FeatureKnownCF(gen)
            pm, ga, ha = score_all_three(model, phi1, phi2, y, anchors, cf,
                                         n_val=2000, n_fisher=2000,
                                         gen_for_val=gen)
            pm_n, ga_n, ha_n = pm.cpu().numpy(), ga.cpu().numpy(), ha.cpu().numpy()
            sp_pmga.append(spearmanr(pm_n, ga_n).correlation)
            sp_pmha.append(spearmanr(pm_n, ha_n).correlation)
            sp_gaha.append(spearmanr(ga_n, ha_n).correlation)

        rec = {
            "rho": rho,
            "coupling_mean": float(np.mean(couplings)),
            "cf_margin_mean": float(np.mean(cms)),
            "id_acc_mean": float(np.mean(id_accs)),
            "spearman_pm_ga": [float(np.mean(sp_pmga)),
                               float(np.std(sp_pmga, ddof=1)) if len(sp_pmga) > 1 else 0.0],
            "spearman_pm_ha": [float(np.mean(sp_pmha)),
                               float(np.std(sp_pmha, ddof=1)) if len(sp_pmha) > 1 else 0.0],
            "spearman_ga_ha": [float(np.mean(sp_gaha)),
                               float(np.std(sp_gaha, ddof=1)) if len(sp_gaha) > 1 else 0.0],
        }
        results.append(rec)
        print(f"  coupling={rec['coupling_mean']:.3f}, "
              f"id_acc={rec['id_acc_mean']:.3f}, "
              f"PM-GA={rec['spearman_pm_ga'][0]:.3f}, "
              f"PM-HA={rec['spearman_pm_ha'][0]:.3f}, "
              f"GA-HA={rec['spearman_ga_ha'][0]:.3f}")

    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "e3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib.pyplot as plt
    rho_arr = np.array([r["rho"] for r in results])
    cou_arr = np.array([r["coupling_mean"] for r in results])
    cm_arr = np.array([r["cf_margin_mean"] for r in results])
    pm_ga = np.array([r["spearman_pm_ga"] for r in results])
    pm_ha = np.array([r["spearman_pm_ha"] for r in results])
    ga_ha = np.array([r["spearman_ga_ha"] for r in results])

    # Per-panel plots: 2 separate PNGs.
    # Panel 1: score-variant agreement (Spearman vs rho).
    # Panel 2: diagnostic (latent coupling and cf_margin vs rho).
    label_map = {"pm_ga": "STL-PM vs STL-GA",
                 "pm_ha": "STL-PM vs STL-HA",
                 "ga_ha": "STL-GA vs STL-HA"}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(rho_arr, pm_ga[:, 0], yerr=pm_ga[:, 1], marker="o",
                label=label_map["pm_ga"], capsize=3, lw=2)
    ax.errorbar(rho_arr, pm_ha[:, 0], yerr=pm_ha[:, 1], marker="s",
                label=label_map["pm_ha"], capsize=3, lw=2)
    ax.errorbar(rho_arr, ga_ha[:, 0], yerr=ga_ha[:, 1], marker="^",
                label=label_map["ga_ha"], capsize=3, lw=2)
    ax.set_xlabel(r"$\rho$ (vector spurious-causal coupling)")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("E3: Score-variant agreement vs coupling")
    ax.set_ylim(0.0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e3_score_agreement.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rho_arr, cou_arr, marker="o", color="tab:purple",
            label="latent coupling")
    axb = ax.twinx()
    axb.plot(rho_arr, cm_arr, marker="s", color="tab:orange",
             label="cf_margin (strict)")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("latent coupling", color="tab:purple")
    axb.set_ylabel("cf_margin (strict)", color="tab:orange")
    ax.set_title(r"Diagnostic: $\rho$ controls coupling and spurious learning")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e3_diagnostic.png"), dpi=120)
    plt.close(fig)
    print(f"[e3] wrote 2 per-panel plots to {out_dir}/e3_*.png")
    return results


# =========================================================================
# === EXPERIMENT: E4 ======================================================
# HA self-consistency at deployment budget. Tests claim 3 (deployability).
#
# E4a (always): score-level Spearman of HA-deployment vs HA-reference,
#               4-cell ablation (N_v, N_sub) for both regimes.
# E4b (--include-ood): retrain models under HA-reference and HA-deployment
#                       at multiple budgets; compare cf_margin and adv_acc.
#                       Overlay strict-only and UTL reference lines.
# =========================================================================
def compute_ha_with_budget(model, phi1, phi2, y, anchors, cf, gen,
                           n_val, n_fisher, fisher_damping, cg_max_iter, cg_tol):
    val_anchors = gen.sample_phi(n_val, rho_sign=+1.0)
    v_s = compute_v_s(model, cf, val_anchors)
    fisher = FisherOracle(model, phi1, phi2, y, beta=1.0,
                          damping=fisher_damping, hvp_subsample=n_fisher)
    return score_hessian_aware(model, anchors, cf, v_s, fisher,
                               cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                               signed=True)


def run_e4_score_level_regime(rho, regime_label, args, out_dir):
    """E4a: score-level self-consistency in one regime."""
    from scipy.stats import spearmanr
    print(f"\n{'='*70}\nE4a (score-level): {regime_label} (rho={rho})\n{'='*70}")
    cfg = regime_config(rho)

    # 4-cell ablation: (v_s budget, Fisher budget) in {large, small} x {large, small}
    cells = {
        "ref":         (args.nv_large, args.nsub_large, args.cg_tol_tight),
        "tight_v_loose_F": (args.nv_large, args.nsub_small, args.cg_tol_loose),
        "loose_v_tight_F": (args.nv_small, args.nsub_large, args.cg_tol_tight),
        "deploy":      (args.nv_small, args.nsub_small, args.cg_tol_loose),
    }
    fisher_damping = {"ref": args.damping_tight,
                      "tight_v_loose_F": args.damping_loose,
                      "loose_v_tight_F": args.damping_tight,
                      "deploy": args.damping_loose}

    seeds_all = {name: [] for name in cells if name != "ref"}
    for sd in args.seeds:
        print(f"\n--- seed {sd} ---")
        set_seed(9000 + sd)
        gen = ShortcutGenerator(cfg, device=args.device)
        phi1, phi2, y = sample_strict_pairs(gen, args.n_train_e4,
                                            beta_teacher=args.beta_teacher, rho_sign=+1.0)
        model = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
        train_pairwise(model, phi1, phi2, y, epochs=args.epochs, batch_size=512)
        anchors = gen.sample_phi(args.n_anchors_e4, rho_sign=+1.0)
        cf = FeatureKnownCF(gen)

        ranks = {}
        for name, (nv, nsub, cg_tol) in cells.items():
            print(f"  cell '{name}': N_v={nv}, N_sub={nsub}, cg_tol={cg_tol}")
            ha = compute_ha_with_budget(model, phi1, phi2, y, anchors, cf, gen,
                                        n_val=nv, n_fisher=nsub,
                                        fisher_damping=fisher_damping[name],
                                        cg_max_iter=args.cg_max_iter_e4,
                                        cg_tol=cg_tol)
            ranks[name] = ha.cpu().numpy()

        for name in seeds_all:
            sp = spearmanr(ranks["ref"], ranks[name]).correlation
            seeds_all[name].append(sp)
            print(f"    Spearman(ref, {name}) = {sp:.4f}")

    summary = {}
    for name, vals in seeds_all.items():
        summary[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }

    ensure_dir(out_dir)
    payload = {"regime": regime_label, "rho": rho, "summary": summary}
    with open(os.path.join(out_dir, f"e4_score_{regime_label}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    return summary


def plot_e4_score(summaries, out_dir):
    """Bar chart: Spearman(ref, cell) per cell, grouped by regime."""
    import matplotlib.pyplot as plt
    cells = ["tight_v_loose_F", "loose_v_tight_F", "deploy"]
    cell_labels = ["tight v_s\nloose F", "loose v_s\ntight F", "deployment\n(both loose)"]
    regimes = list(summaries.keys())
    width = 0.35
    x = np.arange(len(cells))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, reg in enumerate(regimes):
        means = [summaries[reg][c]["mean"] for c in cells]
        stds = [summaries[reg][c]["std"] for c in cells]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=reg, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(cell_labels)
    ax.set_ylabel("Spearman(ref, cell)")
    ax.set_title("E4a: HA self-consistency vs reference budget")
    ax.set_ylim(0.0, 1.05); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e4_score.png"), dpi=120)
    plt.close(fig)


def run_e4_ood_regime(rho, regime_label, args, out_dir):
    """E4b: OOD transfer comparing HA-deployment vs HA-reference at each budget,
            with strict-only and UTL overlays."""
    print(f"\n{'='*70}\nE4b (OOD): {regime_label} (rho={rho})\n{'='*70}")
    cfg = regime_config(rho)
    rows = []

    for sd in args.seeds_ood:
        print(f"\n--- seed {sd} ---")
        set_seed(11000 + sd)
        gen = ShortcutGenerator(cfg, device=args.device)
        phi1_s, phi2_s, y_s = sample_strict_pairs(gen, args.n_train_ood,
                                                  beta_teacher=args.beta_teacher, rho_sign=+1.0)
        m_strict = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
        train_pairwise(m_strict, phi1_s, phi2_s, y_s, epochs=args.epochs, batch_size=512)
        ev0 = evaluate_all(m_strict, gen, n=10000, beta_teacher=args.beta_teacher)
        rows.append({"seed": sd, "k": 0, "strategy": "strict_only", **ev0})
        print(f"  strict-only: cf_margin={ev0['cf_margin']:.3f}, "
              f"id_acc={ev0['id_acc']:.3f}, adv_acc={ev0['adv_acc']:.3f}")

        anchors = gen.sample_phi(args.n_anchor_pool_ood, rho_sign=+1.0)
        cf = FeatureKnownCF(gen)

        # HA at reference and deployment budgets
        print("  scoring HA-reference...")
        ha_ref = compute_ha_with_budget(m_strict, phi1_s, phi2_s, y_s,
                                        anchors, cf, gen,
                                        n_val=args.nv_large,
                                        n_fisher=args.nsub_large,
                                        fisher_damping=args.damping_tight,
                                        cg_max_iter=args.cg_max_iter_e4,
                                        cg_tol=args.cg_tol_tight)
        print("  scoring HA-deployment...")
        ha_dep = compute_ha_with_budget(m_strict, phi1_s, phi2_s, y_s,
                                        anchors, cf, gen,
                                        n_val=args.nv_small,
                                        n_fisher=args.nsub_small,
                                        fisher_damping=args.damping_loose,
                                        cg_max_iter=args.cg_max_iter_e4,
                                        cg_tol=args.cg_tol_loose)

        for k in args.budgets_ood:
            strategies = {
                "random": torch.randperm(args.n_anchor_pool_ood, device=args.device)[:k],
                "HA_ref": topk_idx(ha_ref, k, mode=args.selection_mode),
                "HA_deploy": topk_idx(ha_dep, k, mode=args.selection_mode),
            }
            signed_for = {"random": None, "HA_ref": ha_ref, "HA_deploy": ha_dep}
            for name, idx in strategies.items():
                anchors_sel = {key: val[idx] for key, val in anchors.items()}
                phi1_t, phi2_t, y_t = build_tie_packet(anchors_sel, cf, args.device)
                phi1_aug = torch.cat([phi1_s, phi1_t], dim=0)
                phi2_aug = torch.cat([phi2_s, phi2_t], dim=0)
                y_aug = torch.cat([y_s, y_t], dim=0)
                m_aug = ScorerMLP(in_dim=gen.feature_dim, hidden=64).to(args.device)
                train_pairwise(m_aug, phi1_aug, phi2_aug, y_aug,
                               epochs=args.epochs, batch_size=512)
                ev = evaluate_all(m_aug, gen, n=10000, beta_teacher=args.beta_teacher)
                pos_frac = (positive_fraction_selected(signed_for[name], idx)
                            if signed_for[name] is not None else float('nan'))
                rows.append({"seed": sd, "k": k, "strategy": name,
                             "selection_mode": args.selection_mode,
                             "positive_fraction_selected": pos_frac, **ev})
                print(f"  k={k:5d}, {name:9s}: cf_margin={ev['cf_margin']:.3f}, "
                      f"id_acc={ev['id_acc']:.3f}, adv_acc={ev['adv_acc']:.3f}, "
                      f"pos_frac={pos_frac:.2f}")

    ensure_dir(out_dir)
    with open(os.path.join(out_dir, f"e4_ood_{regime_label}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def plot_e4_ood(rows, regime_label, budgets, out_dir):
    import matplotlib.pyplot as plt
    strategies = ["random", "HA_ref", "HA_deploy"]

    def aggregate(metric_key):
        agg = {}
        for strat in strategies:
            means, stds = [], []
            for k in budgets:
                vs = [r[metric_key] for r in rows
                      if r["k"] == k and r["strategy"] == strat]
                means.append(np.mean(vs))
                stds.append(np.std(vs, ddof=1) if len(vs) > 1 else 0)
            agg[strat] = (means, stds)
        baseline = np.mean([r[metric_key] for r in rows
                            if r["strategy"] == "strict_only"])
        return agg, baseline

    agg_cm, cm0       = aggregate("cf_margin")
    agg_cf_sq, cf_sq0 = aggregate("cf_energy_sq")
    agg_ad, ad0       = aggregate("adv_acc")

    panel_specs = [
        ("cf_margin",    agg_cm,    "cf_margin",                            cm0),
        ("cf_energy_sq", agg_cf_sq, r"$E_{\rm cf}^{(2)}$ (theory-aligned)", cf_sq0),
        ("adv_acc",      agg_ad,    "adversarial accuracy",                 ad0),
    ]
    for metric_tag, agg, ylabel, baseline in panel_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for strat in strategies:
            means, stds = agg[strat]
            ax.errorbar(budgets, means, yerr=stds,
                        marker=DISPLAY_MARKER[strat], color=DISPLAY_COLOR[strat],
                        label=DISPLAY_LABEL[strat], capsize=3, lw=2)
        ax.axhline(baseline, color="black", ls="--", lw=1, alpha=0.5,
                   label=DISPLAY_LABEL["strict_only"])
        ax.set_xlabel("Tie budget k")
        ax.set_ylabel(ylabel)
        ax.set_title(f"E4b OOD {metric_tag}: {regime_label} regime")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = f"e4_ood_{regime_label}_{metric_tag}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)
    print(f"[e4-ood] wrote 3 per-panel plots to {out_dir}/e4_ood_{regime_label}_*.png")


def run_e4(args):
    out_dir = os.path.join(args.out_root, "e4")
    ensure_dir(out_dir)
    if args.quick:
        args.n_train_e4 = 2000; args.n_anchors_e4 = 100
        args.nv_large = 1000; args.nsub_large = 1000
        args.nv_small = 100;  args.nsub_small = 100
        args.seeds = [0, 1]
        args.n_train_ood = 2000; args.n_anchor_pool_ood = 200
        args.budgets_ood = [50, 150]; args.seeds_ood = [0]

    # E4a: score-level
    summaries = {}
    for rho, label in [(0.0, "decoupled"), (0.9, "coupled")]:
        summaries[label] = run_e4_score_level_regime(rho, label, args, out_dir)
    plot_e4_score(summaries, out_dir)
    print(f"\n[e4a] wrote {os.path.join(out_dir, 'e4_score.png')}")

    # E4b: OOD transfer (gated)
    ood_res = {}
    if args.include_ood:
        for rho, label in [(0.0, "decoupled"), (0.9, "coupled")]:
            rows = run_e4_ood_regime(rho, label, args, out_dir)
            plot_e4_ood(rows, label, args.budgets_ood, out_dir)
            ood_res[label] = rows
        print(f"\n[e4b] wrote OOD plots to {out_dir}")
    else:
        print("\n[e4] skipping OOD transfer (use --include-ood to enable)")

    return {"score_level": summaries, "ood": ood_res}


# =========================================================================
# === MAIN ================================================================
# =========================================================================
def build_argparser():
    p = argparse.ArgumentParser(
        description="STL nonlinear experiments (autonomous, single-file).")
    p.add_argument("--experiment", required=True,
                   choices=["smoke", "e1", "e2", "e3", "e4"])
    p.add_argument("--device", default=device_default())
    p.add_argument("--out-root", default="results")
    p.add_argument("--save_data", action="store_true",
                   help="Save the raw results to a pickle file.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny config for sanity-check runs.")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs for every model fit (strict and augmented).")
    p.add_argument("--beta-teacher", type=float, default=1.0,
                   help="BT teacher temperature. Higher = sharper labels, higher Bayes ceiling. "
                        "Bayes-optimal id_acc: beta=1.0 -> 0.725, beta=2.0 -> 0.828, beta=3.0 -> 0.878.")

    # E1 knobs
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-anchors", type=int, default=500)
    p.add_argument("--n-subsample", type=int, default=50)
    p.add_argument("--epsilon-packets", type=int, default=500)
    p.add_argument("--n-retrain-seeds", type=int, default=3)

    # E2 knobs
    p.add_argument("--n-anchor-pool", type=int, default=2000)
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[200, 500, 1000, 2000])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4])

    # selection mode (used by E2 and E4-OOD)
    p.add_argument("--selection-mode", default="relu",
                   choices=["abs", "signed", "relu"],
                   help="How to use signed scores at top-k selection. "
                        "'relu' (default): rank by max(s,0), restricting to "
                        "candidates with first-order energy-decreasing "
                        "influence (s > 0). This is the theoretically "
                        "justified default: the signed influence formula "
                        "$\\Delta E_{cf}^{(2)} \\approx -\\epsilon\\langle "
                        "v_s, F^{-1} g_j\\rangle$ guarantees energy decrease "
                        "only when the score is positive. "
                        "'abs': rank by |s|, magnitude regardless of sign. "
                        "Empirically equivalent to 'relu' when score "
                        "distributions are dominantly positive (typical at "
                        "strong coupling), but provides no energy-decrease "
                        "guarantee for negative-score candidates. "
                        "'signed': rank by s, largest positive first; "
                        "diagnostic only. "
                        "PM is always selected via 'abs' regardless of this "
                        "flag, since PM's sign has no energy-decrease "
                        "interpretation.")

    # E3 knobs
    p.add_argument("--rhos", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

    # E4 knobs (score-level)
    p.add_argument("--n-train-e4", type=int, default=5000)
    p.add_argument("--n-anchors-e4", type=int, default=500)
    p.add_argument("--nv-large", type=int, default=5000)
    p.add_argument("--nsub-large", type=int, default=5000)
    p.add_argument("--nv-small", type=int, default=200)
    p.add_argument("--nsub-small", type=int, default=200)
    p.add_argument("--damping-tight", type=float, default=1e-5)
    p.add_argument("--damping-loose", type=float, default=1e-3)
    p.add_argument("--cg-tol-tight", type=float, default=1e-5)
    p.add_argument("--cg-tol-loose", type=float, default=1e-3)
    p.add_argument("--cg-max-iter-e4", type=int, default=200)

    # E4 knobs (OOD transfer)
    p.add_argument("--include-ood", action="store_true",
                   help="E4: also run OOD transfer (slower, retrains models).")
    p.add_argument("--n-train-ood", type=int, default=10000)
    p.add_argument("--n-anchor-pool-ood", type=int, default=2000)
    p.add_argument("--budgets-ood", type=int, nargs="+",
                   default=[500, 1000, 2000])
    p.add_argument("--seeds-ood", type=int, nargs="+", default=[0, 1, 2])

    return p


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_root)

    print(f"[run_atl] experiment={args.experiment}, device={args.device}, "
          f"quick={args.quick}, out={args.out_root}")

    res = None
    if args.experiment == "smoke":
        res = run_smoke(args)
    elif args.experiment == "e1":
        res = run_e1(args)
    elif args.experiment == "e2":
        res = run_e2(args)
    elif args.experiment == "e3":
        res = run_e3(args)
    elif args.experiment == "e4":
        res = run_e4(args)
    else:
        raise ValueError(f"unknown experiment: {args.experiment}")

    if getattr(args, "save_data", False) and res is not None:
        import pickle
        exp_dir = os.path.join(args.out_root, args.experiment)
        ensure_dir(exp_dir)
        pkl_path = os.path.join(exp_dir, f"data_exp_{args.experiment}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({args.experiment: res}, f)
        print(f"\n    [data saved] {pkl_path}")


if __name__ == "__main__":
    main()
