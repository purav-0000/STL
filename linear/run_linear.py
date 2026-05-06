# -*- coding: utf-8 -*-
"""
STL Linear Theory Validation
============================

Linear / Gaussian-feature experiments validating
the STL theory. Each experiment can be run independently;
selection scores (FK / GA / HA) can use either absolute value or the
positive-part (signed -> max(0, .)) post-processing as predicted by
the signed influence theory. PM is always absolute (it carries no sign
information by construction).

Experiments:
  E1   Tanh closed form 
  E2   Per-anchor influence 
  E3   STL vs UTL aggregate suppression (HEADLINE)
  E4   Quadratic decontamination
  E5   Per-anchor monotone Delta E_j
  E5b  Full-DPO version of E5 with three predictions
  E6   Subspace-CMS ranking equivalence
  E7   Idealized influence vs simplified |u_j|
  E8   Deployment OOD accuracy under adversarial shift
  E6b  Five-way comparison FK/PM/GA/HA/Random-UTL under full DPO
       (this is where --mode {abs,signed} matters)

CLI usage:
  python run_linear.py --experiment 1
  python run_linear.py --experiment e6b --mode signed --num_seeds 5
  python run_linear.py --experiment all --mode abs --output_dir ./figs/
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm as scipy_norm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==================
# Defaults / palette
# ==================

DEFAULT_CONFIG = {
    # DGP & dimension
    "d_c": 5,
    "d_s": 5,
    "N": 1000,
    "beta": 0.5,
    "mu_scale": 0.05,
    "rho_cs": 0.15,
    "cond_max": 10.0,
    "seed": 42,
    # Multi-seed & rollout knobs
    "num_seeds": 10,
    "M_utl": 50,
    # E6b-specific (richer DGP for OOD evaluation)
    "e6b_mu_scale": 0.5,
    "e6b_rho_cs": 0.4,
    "e6b_k_values": [0, 25, 100, 400, 800],
    "e6b_M_rnd": 5,
    "e6b_N_v": 200,
    # Output
    "output_dir": ".",
    "show": False,
    "mode": "abs",
}

C_BLUE = "#1565C0"
C_RED = "#C62828"
C_GREEN = "#2E7D32"
C_GRAY = "#616161"
C_PURPLE = "#7B1FA2"
C_ORANGE = "#EF6C00"


# ===============
# Output handling
# ===============

def _outpath(cfg: dict, name: str) -> str:
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return os.path.join(cfg["output_dir"], name)


def _finalize(cfg: dict, fname: str):
    """Save current figure to output_dir/fname.pdf and (optionally) show."""
    path = _outpath(cfg, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if cfg.get("show", False):
        plt.show()
    plt.close()
    print(f"    [plot saved] {path}")


# ============
# Math helpers
# ============

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def block_split(M, d_c):
    return M[:d_c, :d_c], M[:d_c, d_c:], M[d_c:, :d_c], M[d_c:, d_c:]


def split_theta(theta, d_c):
    return theta[:d_c], theta[d_c:]


def schur_spurious(Sigma, d_c):
    Scc, Scs, Ssc, Sss = block_split(Sigma, d_c)
    return Sss - Ssc @ np.linalg.solve(Scc, Scs)


def linearized_optimum(mu, Sigma, beta):
    """theta^*_P = (2/beta) Sigma^{-1} mu."""
    return (2.0 / beta) * np.linalg.solve(Sigma, mu)


def spurious_energy(theta, Sigma_ss_deploy, d_c):
    theta_s = theta[d_c:]
    return float(theta_s @ Sigma_ss_deploy @ theta_s)


def empirical_dpo_hessian(anchors, theta, beta):
    """H = (beta^2/N) sum_i sigma(z_i)(1-sigma(z_i)) X_i X_i^T."""
    z = beta * (anchors @ theta)
    p = sigmoid(z)
    weights = p * (1.0 - p)
    return (beta ** 2 / len(anchors)) * (anchors.T @ (weights[:, None] * anchors))


# ===
# DGP
# ===

def generate_dgp(d_c, d_s, mu_scale, rho_cs, seed, cond_max=10.0):
    """Gaussian DGP for Delta phi with controlled cross-block coupling."""
    rng = np.random.default_rng(seed)
    d = d_c + d_s

    eigvals = np.linspace(1.0, cond_max, d)
    G = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(G)
    Cov_full = Q @ np.diag(eigvals) @ Q.T

    if rho_cs <= 0.0:
        Cov_cc = Cov_full[:d_c, :d_c]
        Cov_ss = Cov_full[d_c:, d_c:]
        Cov_cs = np.zeros((d_c, d_s))
    else:
        Cov_cc = Cov_full[:d_c, :d_c]
        Cov_ss = Cov_full[d_c:, d_c:]
        Cov_cs_raw = Cov_full[:d_c, d_c:]
        raw_norm = np.linalg.norm(Cov_cs_raw, "fro")
        target_norm = rho_cs * np.sqrt(
            np.linalg.norm(Cov_cc, "fro") * np.linalg.norm(Cov_ss, "fro"))
        scale = target_norm / (raw_norm + 1e-30)
        Cov_cs = scale * Cov_cs_raw

    Cov = np.block([[Cov_cc, Cov_cs], [Cov_cs.T, Cov_ss]])
    eigmin = np.linalg.eigvalsh(Cov).min()
    if eigmin < 1e-3:
        Cov = Cov + (1e-3 - eigmin) * np.eye(d)

    mu = mu_scale * rng.uniform(-1.0, 1.0, size=d)
    Sigma = Cov + np.outer(mu, mu)
    return {"mu": mu, "Cov": Cov, "Sigma": Sigma, "d_c": d_c, "d_s": d_s}


def sample_anchors(dgp, N, seed):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(dgp["mu"], dgp["Cov"], size=N)


def local_regime_check(theta, dgp, beta, anchors=None):
    if anchors is None:
        anchors = sample_anchors(dgp, 2000, seed=999)
    z = beta * (anchors @ theta)
    return {"max": np.abs(z).max(), "mean": np.abs(z).mean(),
            "p95": np.quantile(np.abs(z), 0.95)}


# =======
# Solvers
# =======

def compute_augmented_optimum(selected_anchors, dgp, beta, alpha):
    """Linearized augmented optimum under feature-known antisymmetric ties."""
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]

    delta_s = selected_anchors[:, d_c:]
    Sigma_tie_ss = (delta_s.T @ delta_s) / len(selected_anchors)

    Sigma_aug = alpha * Sigma_P.copy()
    Sigma_aug[d_c:, d_c:] += (1.0 - alpha) * Sigma_tie_ss

    theta_aug = (2.0 * alpha / beta) * np.linalg.solve(Sigma_aug, mu_P)
    return theta_aug, Sigma_aug


def solve_perturbed_foc(dgp, beta, eps, delta_phi_s_j,
                        max_iter=300, tol=1e-13):
    """Fixed-point for perturbed FOC under one antisymmetric tie packet."""
    d_c = dgp["d_c"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    theta = linearized_optimum(mu_P, Sigma_P, beta)
    rhs_strict = (2.0 / beta) * mu_P

    for it in range(max_iter):
        u = beta * (theta[d_c:] @ delta_phi_s_j)
        g_s = (beta / 2.0) * np.tanh(u / 2.0) * delta_phi_s_j
        rhs = rhs_strict.copy()
        rhs[d_c:] -= (4.0 * eps / beta ** 2) * g_s
        new_theta = np.linalg.solve(Sigma_P, rhs)
        if np.linalg.norm(new_theta - theta) < tol:
            return new_theta, it + 1
        theta = new_theta
    return theta, max_iter


def solve_full_dpo(anchors, beta, theta_init=None, lr=0.05,
                   max_iter=20000, tol=1e-10,
                   eps_aug=0.0, tie_anchor_s=None, d_c=None):
    """Full nonlinear DPO via gradient descent on empirical preference loss."""
    N, d = anchors.shape
    if d_c is None:
        d_c = d // 2
    theta = np.zeros(d) if theta_init is None else theta_init.copy()

    for step in range(max_iter):
        z = beta * (anchors @ theta)
        w = np.where(z >= 0,
                     np.exp(-z) / (1.0 + np.exp(-z)),
                     1.0 / (1.0 + np.exp(z)))
        grad = -(beta / N) * (anchors.T @ w)

        if eps_aug != 0.0 and tie_anchor_s is not None:
            u_local = beta * (theta[d_c:] @ tie_anchor_s)
            g_tie_s = (beta / 2.0) * np.tanh(u_local / 2.0) * tie_anchor_s
            grad[d_c:] += eps_aug * g_tie_s

        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        theta = theta - lr * grad
    return theta


def deployment_accuracy(theta, mu_Q, Sigma_Q):
    """Acc(Q; theta) = Phi( theta^T mu_Q / sqrt(theta^T Sigma_Q theta) )."""
    var = float(theta @ Sigma_Q @ theta)
    if var <= 0:
        return 0.5
    snr = float(theta @ mu_Q) / np.sqrt(var)
    return float(scipy_norm.cdf(snr))


def adversarial_shift_moments(mu_P, Sigma_P, d_c):
    """Q with cross-block flipped: Sigma_cs^Q = -Sigma_cs^P."""
    Sigma_Q = Sigma_P.copy()
    Sigma_Q[:d_c, d_c:] = -Sigma_P[:d_c, d_c:]
    Sigma_Q[d_c:, :d_c] = -Sigma_P[d_c:, :d_c]
    return mu_P.copy(), Sigma_Q


# ====================
# E1: Tanh closed form
# ====================

def experiment_E1(beta, n_test=400, d_s=5, seed=0):
    """Verify g_tie_s = (beta/2) tanh(u/2) Delta phi_s exactly."""
    print("\n" + "=" * 72)
    print("E1: Tanh closed form (Lemma B.6)  --  Exact regime")
    print("=" * 72)
    rng = np.random.default_rng(seed)

    u_targets = np.linspace(-10, 10, n_test)
    rel_errors = np.zeros(n_test)
    abs_errors = np.zeros(n_test)

    for i, u_target in enumerate(u_targets):
        delta_s = rng.standard_normal(d_s)
        theta_s = (u_target / beta) * delta_s / (delta_s @ delta_s)
        u1 = beta * theta_s @ delta_s
        u2 = beta * theta_s @ (-delta_s)
        g_direct = -(beta / 2.0) * (
            sigmoid(-u1) * delta_s + sigmoid(-u2) * (-delta_s))
        g_tanh = (beta / 2.0) * np.tanh(u_target / 2.0) * delta_s
        abs_errors[i] = np.linalg.norm(g_direct - g_tanh)
        rel_errors[i] = abs_errors[i] / (np.linalg.norm(g_direct) + 1e-30)

    print(f"  Sweep:  u in [-10, 10], n_test = {n_test}, beta = {beta}")
    print(f"  max rel error  : {rel_errors.max():.2e}")
    print(f"  mean rel error : {rel_errors.mean():.2e}")
    print(f"  max abs error  : {abs_errors.max():.2e}")
    pass_e1 = rel_errors.max() < 1e-10
    verdict = "PASS" if pass_e1 else "FAIL"
    print(f"  VERDICT: {verdict}")
    return {"u_targets": u_targets, "rel_errors": rel_errors,
            "abs_errors": abs_errors, "pass": pass_e1}


def plot_E1(res, cfg):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.semilogy(res["u_targets"], res["rel_errors"] + 1e-20,
                color=C_BLUE, lw=2)
    ax.axhline(1e-15, color=C_GRAY, ls="--", alpha=0.6,
               label="machine eps")
    ax.set_xlabel(r"$u = \beta\,\tilde\theta_s^\top \Delta\phi_s$",
                  fontsize=13)
    ax.set_ylabel("relative error", fontsize=13)
    ax.set_title(r"E1: Tanh closed form (Lemma B.6) -- exact across all $u$",
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize(cfg, "e1_tanh_form.pdf")


# ========================
# E2: Per-anchor influence
# ========================

def experiment_E2(dgp, beta, eps_values, n_anchors=20, seed=0, label=""):
    """Verify theta^*_eps = theta^*_P - eps H^{-1} g_j + O(eps^2)."""
    print("\n" + "=" * 72)
    print(f"E2: Per-anchor influence (Prop B.7) -- exact tanh, label={label}")
    print(f"    Tests deviation = O(eps^2)  =>  dev / eps^2 ~ const")
    print("=" * 72)
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    rng = np.random.default_rng(seed)

    theta_P = linearized_optimum(mu_P, Sigma_P, beta)
    H = (beta ** 2 / 4.0) * Sigma_P
    delta_s_set = rng.standard_normal((n_anchors, d_s))

    pred_norms = np.zeros((n_anchors, len(eps_values)))
    act_norms = np.zeros((n_anchors, len(eps_values)))
    devs = np.zeros((n_anchors, len(eps_values)))
    iters = np.zeros((n_anchors, len(eps_values)), dtype=int)

    for a, delta_s in enumerate(delta_s_set):
        u_at_P = beta * (theta_P[d_c:] @ delta_s)
        g_j_s = (beta / 2.0) * np.tanh(u_at_P / 2.0) * delta_s
        g_j = np.concatenate([np.zeros(d_c), g_j_s])
        H_inv_g = np.linalg.solve(H, g_j)

        for k, eps in enumerate(eps_values):
            pred_shift = -eps * H_inv_g
            theta_eps, n_iter = solve_perturbed_foc(dgp, beta, eps, delta_s)
            actual_shift = theta_eps - theta_P
            dev = actual_shift - pred_shift
            pred_norms[a, k] = np.linalg.norm(pred_shift)
            act_norms[a, k] = np.linalg.norm(actual_shift)
            devs[a, k] = np.linalg.norm(dev)
            iters[a, k] = n_iter

    print(f"  N anchors = {n_anchors}, averaged over anchor directions.")
    print(f"  {'eps':<10} {'mean ||pred||':<14} {'mean ||actual||':<16} "
          f"{'mean ||dev||':<14} {'mean dev/eps^2':<16} {'iters':<6}")
    print("  " + "-" * 76)
    for k, eps in enumerate(eps_values):
        print(f"  {eps:<10.0e} {pred_norms[:, k].mean():<14.4e} "
              f"{act_norms[:, k].mean():<16.4e} "
              f"{devs[:, k].mean():<14.4e} "
              f"{(devs[:, k] / eps ** 2).mean():<16.4e} "
              f"{iters[:, k].mean():<6.1f}")

    dev_per_eps2 = (devs / np.array(eps_values)[None, :] ** 2).mean(axis=0)
    pass_e2 = (dev_per_eps2.max() / dev_per_eps2.min()) < 20.0
    verdict = "PASS" if pass_e2 else "FAIL"
    print(f"  VERDICT: {verdict}  (max/min of dev/eps^2 = "
          f"{dev_per_eps2.max() / dev_per_eps2.min():.2f})")

    return {"eps_values": np.array(eps_values),
            "pred_norms": pred_norms, "act_norms": act_norms,
            "devs": devs, "dev_per_eps2": dev_per_eps2, "pass": pass_e2,
            "label": label}


def plot_E2(res_dec, res_coup, cfg):
    """Single-panel: deviation vs eps for both regimes (with ref line)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    eps = res_dec["eps_values"]
    ax.loglog(eps, res_dec["devs"].mean(axis=0), "o-",
              color=C_BLUE, lw=2, label="decoupled")
    ax.loglog(eps, res_coup["devs"].mean(axis=0), "s-",
              color=C_RED, lw=2, label="coupled")
    ref = eps ** 2 * res_dec["devs"][:, 0].mean() / eps[0] ** 2
    ax.loglog(eps, ref, ":", color=C_GRAY, lw=1.5, label=r"$\propto\epsilon^2$")
    ax.set_xlabel(r"$\epsilon$", fontsize=13)
    ax.set_ylabel(r"$\|$actual - predicted$\|$", fontsize=13)
    ax.set_title(r"E2: Per-anchor influence -- deviation $\sim \epsilon^2$",
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    _finalize(cfg, "e2_influence.pdf")


# ====================================
# E3: STL vs UTL aggregate suppression
# ====================================

def _experiment_E3_single(dgp, anchors, beta, k_values, M_utl=50,
                          seed=0, verbose=False):
    d_c = dgp["d_c"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Sigma_ss_P = Sigma_P[d_c:, d_c:]
    N = anchors.shape[0]
    rng = np.random.default_rng(seed)

    theta_P = linearized_optimum(mu_P, Sigma_P, beta)
    theta_P_s = theta_P[d_c:]
    delta_phi_s = anchors[:, d_c:]
    u_j = beta * (delta_phi_s @ theta_P_s)
    abs_u = np.abs(u_j)
    stl_order = np.argsort(-abs_u)
    E_P = spurious_energy(theta_P, Sigma_ss_P, d_c)

    rows = []
    for k in k_values:
        if k == 0:
            rows.append({"k": 0, "alpha": 1.0, "E_stl": E_P,
                         "E_utl_mean": E_P, "E_utl_std": 0.0,
                         "STL_wins": True})
            continue
        alpha = N / (N + k)
        stl_idx = stl_order[:k]
        theta_stl, _ = compute_augmented_optimum(
            anchors[stl_idx], dgp, beta, alpha)
        E_stl = spurious_energy(theta_stl, Sigma_ss_P, d_c)

        E_utls = np.zeros(M_utl)
        for m in range(M_utl):
            utl_idx = rng.choice(N, size=k, replace=False)
            theta_utl, _ = compute_augmented_optimum(
                anchors[utl_idx], dgp, beta, alpha)
            E_utls[m] = spurious_energy(theta_utl, Sigma_ss_P, d_c)

        rows.append({"k": k, "alpha": alpha, "E_stl": E_stl,
                     "E_utl_mean": E_utls.mean(),
                     "E_utl_std": E_utls.std(),
                     "STL_wins": E_stl < E_utls.mean()})

    if verbose:
        print(f"    [seed {seed}] E_P={E_P:.4f}, "
              f"|u_j|: mean={abs_u.mean():.3f} max={abs_u.max():.3f}")

    return {"rows": rows, "E_P": E_P, "abs_u": abs_u}


def experiment_E3(dgp_config, beta, k_values, n_dgp_seeds=10, M_utl=50,
                  base_seed=0, label=""):
    """Multi-seed E3: STL (top-k by |u|) vs UTL (random k) aggregate suppression."""
    print("\n" + "=" * 72)
    print(f"E3: STL vs UTL aggregate suppression -- HEADLINE, label={label}")
    print(f"    Multi-seed: n_dgp_seeds={n_dgp_seeds}, M_utl={M_utl}/seed")
    print("=" * 72)

    n_k = len(k_values)
    E_stl_arr = np.zeros((n_dgp_seeds, n_k))
    E_utl_arr = np.zeros((n_dgp_seeds, n_k))
    E_P_arr = np.zeros(n_dgp_seeds)
    rel_gap_arr = np.zeros((n_dgp_seeds, n_k))
    indiv_stl_wins = np.zeros((n_dgp_seeds, n_k), dtype=bool)
    abs_u_stats = []

    for s in range(n_dgp_seeds):
        seed_s = base_seed + 1000 * s
        dgp_s = generate_dgp(dgp_config["d_c"], dgp_config["d_s"],
                             dgp_config["mu_scale"],
                             rho_cs=dgp_config["rho_cs"],
                             seed=seed_s, cond_max=dgp_config["cond_max"])
        anchors_s = sample_anchors(dgp_s, dgp_config["N"], seed=seed_s + 1)
        single = _experiment_E3_single(
            dgp_s, anchors_s, beta, k_values, M_utl=M_utl,
            seed=seed_s + 2, verbose=True)
        for j, r in enumerate(single["rows"]):
            E_stl_arr[s, j] = r["E_stl"]
            E_utl_arr[s, j] = r["E_utl_mean"]
            indiv_stl_wins[s, j] = r["STL_wins"]
        E_P_arr[s] = single["E_P"]
        rel_gap_arr[s] = (E_utl_arr[s] - E_stl_arr[s]) / single["E_P"]
        abs_u_stats.append({"mean": single["abs_u"].mean(),
                            "max": single["abs_u"].max()})

    E_stl_mean = E_stl_arr.mean(axis=0); E_stl_std = E_stl_arr.std(axis=0)
    E_utl_mean = E_utl_arr.mean(axis=0); E_utl_std = E_utl_arr.std(axis=0)
    rel_gap_mean = rel_gap_arr.mean(axis=0)
    rel_gap_std = rel_gap_arr.std(axis=0)
    win_frac = indiv_stl_wins.mean(axis=0)

    print(f"\n  Aggregated across {n_dgp_seeds} DGP seeds (mean +/- std):")
    print(f"  {'k':<6} {'E_STL':<18} {'E_UTL':<18} "
          f"{'rel_gap':<18} {'wins':<10}")
    print("  " + "-" * 76)
    for j, k in enumerate(k_values):
        wins = (f"{int(win_frac[j] * n_dgp_seeds)}/{n_dgp_seeds}"
                if k > 0 else "-")
        print(f"  {k:<6} {E_stl_mean[j]:.4f} +/- {E_stl_std[j]:.4f}    "
              f"{E_utl_mean[j]:.4f} +/- {E_utl_std[j]:.4f}    "
              f"{rel_gap_mean[j]:+.4f} +/- {rel_gap_std[j]:.4f}    "
              f"{wins}")

    mean_dom = all(E_stl_mean[j] < E_utl_mean[j]
                   for j, k in enumerate(k_values) if k > 0)
    indiv_dom = all(win_frac[j] >= 0.8
                    for j, k in enumerate(k_values) if k > 0)
    pass_e3 = mean_dom and indiv_dom
    print(f"\n  VERDICT: {'PASS' if pass_e3 else 'FAIL'}")

    return {"k_values": np.array(k_values),
            "E_stl_arr": E_stl_arr, "E_utl_arr": E_utl_arr,
            "E_P_arr": E_P_arr,
            "E_stl_mean": E_stl_mean, "E_stl_std": E_stl_std,
            "E_utl_mean": E_utl_mean, "E_utl_std": E_utl_std,
            "rel_gap_mean": rel_gap_mean, "rel_gap_std": rel_gap_std,
            "win_frac": win_frac, "n_dgp_seeds": n_dgp_seeds,
            "M_utl": M_utl, "pass": pass_e3, "label": label}


def plot_E3_energies(res, cfg, label=""):
    """Single panel: absolute energy curves STL vs UTL with bands."""
    ks = res["k_values"]
    n_seeds = res["n_dgp_seeds"]
    E_P_mean = res["E_P_arr"].mean()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(ks, res["E_stl_mean"], "o-", color=C_BLUE, lw=2.5,
            label=f"STL (top-k by |u|), n={n_seeds} seeds")
    ax.fill_between(ks, res["E_stl_mean"] - res["E_stl_std"],
                    res["E_stl_mean"] + res["E_stl_std"],
                    color=C_BLUE, alpha=0.20)
    ax.plot(ks, res["E_utl_mean"], "s-", color=C_RED, lw=2.0,
            label="UTL (uniform random)")
    ax.fill_between(ks, res["E_utl_mean"] - res["E_utl_std"],
                    res["E_utl_mean"] + res["E_utl_std"],
                    color=C_RED, alpha=0.20)
    ax.axhline(E_P_mean, ls=":", color=C_GRAY, lw=1.5,
               label=f"strict $E_P$ = {E_P_mean:.3f}")
    ax.set_xlabel("budget $k$", fontsize=13)
    ax.set_ylabel(r"$E(\tilde\theta^*_{\rm aug})$", fontsize=13)
    ax.set_title(f"E3 ({label}): STL vs UTL energies "
                 f"(mean +/- 1 std, {n_seeds} seeds)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize(cfg, f"e3_energies_{label}.pdf")


def plot_E3_relgap(res, cfg, label=""):
    """Single panel: relative STL-UTL gap (paired per seed)."""
    ks = res["k_values"]
    n_seeds = res["n_dgp_seeds"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(ks, res["rel_gap_mean"], "d-", color=C_GREEN, lw=2.5)
    ax.fill_between(ks, res["rel_gap_mean"] - res["rel_gap_std"],
                    res["rel_gap_mean"] + res["rel_gap_std"],
                    color=C_GREEN, alpha=0.25,
                    label=f"+/- 1 std across {n_seeds} seeds")
    ax.axhline(0, color=C_GRAY, ls="--", alpha=0.6)
    ax.set_xlabel("budget $k$", fontsize=13)
    ax.set_ylabel(r"$(E_{\rm UTL} - E_{\rm STL})\,/\,E_P$", fontsize=13)
    ax.set_title(f"E3 ({label}): paired relative gap", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize(cfg, f"e3_relgap_{label}.pdf")


# =============================
# E4: Quadratic decontamination
# =============================

def experiment_E4(dgp, anchors, beta, k_values, seed=0, label=""):
    """Verify Sigma_sc^aug (Sigma_cc^aug)^{-1} Sigma_cs^aug = rho_k^2 * leakage_P."""
    print("\n" + "=" * 72)
    print(f"E4: Quadratic decontamination, label={label}")
    print("=" * 72)
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    Sigma_P = dgp["Sigma"]
    N = anchors.shape[0]
    rng = np.random.default_rng(seed)

    k_values = [k for k in k_values if k <= N]
    Scc_P, Scs_P, Ssc_P, Sss_P = block_split(Sigma_P, d_c)
    leakage_P = Ssc_P @ np.linalg.solve(Scc_P, Scs_P)
    leakage_P_norm = np.linalg.norm(leakage_P, "fro")
    Sss_P_norm = np.linalg.norm(Sss_P, "fro")
    Schur_P = Sss_P - leakage_P
    Schur_P_norm = np.linalg.norm(Schur_P, "fro")

    print(f"  Baselines: ||leakage_P||_F = {leakage_P_norm:.4f}, "
          f"||S_s^P||_F = {Schur_P_norm:.4f}")
    print(f"  {'k':<6} {'rho_k':<8} {'rho_k^2':<10} "
          f"{'leakage_aug':<14} {'pred':<14} {'rel_err':<10}")
    print("  " + "-" * 70)

    rows = []
    for k in k_values:
        if k == 0:
            rows.append({"k": 0, "rho_k": 1.0,
                         "leakage_aug": leakage_P_norm,
                         "pred": leakage_P_norm, "rel_err": 0.0,
                         "Schur_aug_norm": Schur_P_norm})
            print(f"  {0:<6} {1.0:<8.4f} {1.0:<10.4f} "
                  f"{leakage_P_norm:<14.4e} {leakage_P_norm:<14.4e} "
                  f"{0.0:<10.2e}")
            continue
        rho_k = N / (N + k)
        idx = rng.choice(N, size=k, replace=False)
        delta_s = anchors[idx, d_c:]
        Sigma_tie_ss = (delta_s.T @ delta_s) / k

        Sigma_aug_cc = Scc_P
        Sigma_aug_cs = rho_k * Scs_P
        Sigma_aug_sc = rho_k * Ssc_P
        Sigma_aug_ss = rho_k * Sss_P + (1.0 - rho_k) * Sigma_tie_ss

        leakage_aug = Sigma_aug_sc @ np.linalg.solve(Sigma_aug_cc, Sigma_aug_cs)
        leakage_aug_norm = np.linalg.norm(leakage_aug, "fro")
        pred = (rho_k ** 2) * leakage_P_norm
        rel_err = abs(leakage_aug_norm - pred) / (pred + 1e-30)
        Schur_aug = Sigma_aug_ss - leakage_aug
        rows.append({"k": k, "rho_k": rho_k,
                     "leakage_aug": leakage_aug_norm,
                     "pred": pred, "rel_err": rel_err,
                     "Schur_aug_norm": np.linalg.norm(Schur_aug, "fro")})
        print(f"  {k:<6} {rho_k:<8.4f} {rho_k**2:<10.4f} "
              f"{leakage_aug_norm:<14.4e} {pred:<14.4e} {rel_err:<10.2e}")

    pass_e4 = max(r["rel_err"] for r in rows) < 1e-10
    print(f"\n  VERDICT: {'PASS' if pass_e4 else 'FAIL'}")
    return {"rows": rows, "leakage_P_norm": leakage_P_norm,
            "Schur_P_norm": Schur_P_norm, "pass": pass_e4}


def plot_E4_shrinkage(res, cfg):
    """Single panel: rho_k^2 shrinkage."""
    rows = res["rows"]
    rho_k = np.array([r["rho_k"] for r in rows])
    leak = np.array([r["leakage_aug"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    mask = rho_k < 1.0
    ax.loglog(rho_k[mask], leak[mask], "o", color=C_BLUE, ms=8,
              label="measured")
    ax.loglog(rho_k[mask], pred[mask], "-", color=C_RED, lw=2,
              label=r"$\rho_k^2 \|$leakage$_P\|_F$")
    ax.set_xlabel(r"$\rho_k = N/(N+k)$", fontsize=13)
    ax.set_ylabel(r"$\|\Sigma_{sc}^{\rm aug}(\Sigma_{cc}^{\rm aug})^{-1}"
                  r"\Sigma_{cs}^{\rm aug}\|_F$", fontsize=12)
    ax.set_title("E4: Quadratic decontamination", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    _finalize(cfg, "e4_shrinkage.pdf")


def plot_E4_schur(res, cfg):
    """Single panel: augmented Schur norm."""
    rows = res["rows"]
    rho_k = np.array([r["rho_k"] for r in rows])
    Schur_norm = np.array([r["Schur_aug_norm"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.semilogx(rho_k, Schur_norm, "o-", color=C_GREEN, lw=2,
                label=r"$\|S_s^{\rm aug}\|_F$")
    ax.axhline(res["Schur_P_norm"], ls=":", color=C_GRAY,
               label=r"$\|S_s^P\|_F$ (strict-only)")
    ax.set_xlabel(r"$\rho_k = N/(N+k)$", fontsize=13)
    ax.set_ylabel(r"$\|S_s^{\rm aug}\|_F$", fontsize=12)
    ax.set_title("E4: Augmented Schur complement", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    _finalize(cfg, "e4_schur.pdf")


# =================================
# E5: Per-anchor monotone Delta E_j
# =================================

def experiment_E5(dgp, anchors, beta, eps=1e-3, seed=0, label=""):
    """Verify Delta E_j with both decoupled and coupled predictions."""
    print("\n" + "=" * 72)
    print(f"E5: Per-anchor monotone suppression, label={label}")
    print("=" * 72)
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Sigma_ss_P = Sigma_P[d_c:, d_c:]
    Scs_P_norm = np.linalg.norm(Sigma_P[:d_c, d_c:], "fro")
    coupled = Scs_P_norm > 1e-8

    theta_P = linearized_optimum(mu_P, Sigma_P, beta)
    theta_P_s = theta_P[d_c:]
    E_P = spurious_energy(theta_P, Sigma_ss_P, d_c)

    S_s_P = schur_spurious(Sigma_P, d_c)
    theta_eff = np.linalg.solve(S_s_P, Sigma_ss_P @ theta_P_s)
    delta_phi_s = anchors[:, d_c:]
    u_js = beta * (delta_phi_s @ theta_P_s)
    tilde_u_js = beta * (delta_phi_s @ theta_eff)

    n = len(anchors)
    actual_dE = np.zeros(n)
    pred_decoupled = np.zeros(n)
    pred_coupled = np.zeros(n)
    coef = -(4.0 * eps) / (beta ** 2)

    for j in range(n):
        theta_eps, _ = solve_perturbed_foc(dgp, beta, eps, delta_phi_s[j])
        actual_dE[j] = spurious_energy(theta_eps, Sigma_ss_P, d_c) - E_P
        pred_decoupled[j] = coef * u_js[j] * np.tanh(u_js[j] / 2.0)
        pred_coupled[j] = coef * np.tanh(u_js[j] / 2.0) * tilde_u_js[j]

    rel_dec = np.linalg.norm(actual_dE - pred_decoupled) / \
        (np.linalg.norm(actual_dE) + 1e-30)
    rel_coup = np.linalg.norm(actual_dE - pred_coupled) / \
        (np.linalg.norm(actual_dE) + 1e-30)

    print(f"  Regime: {'COUPLED' if coupled else 'DECOUPLED'}")
    print(f"  Sign-aligned: {(actual_dE <= 1e-12).sum()}/{n}")
    print(f"  Rel err decoupled = {rel_dec:.4e}, coupled = {rel_coup:.4e}")

    if coupled and Scs_P_norm >= 0.1:
        pass_e5 = rel_coup < 0.05 and rel_coup < 0.95 * rel_dec
    else:
        pass_e5 = rel_dec < 0.05 and rel_coup < 0.05
    print(f"  VERDICT: {'PASS' if pass_e5 else 'FAIL'}")

    return {"u_js": u_js, "tilde_u_js": tilde_u_js,
            "actual_dE": actual_dE,
            "pred_decoupled": pred_decoupled,
            "pred_coupled": pred_coupled,
            "coupled": coupled, "rel_dec": rel_dec,
            "rel_coup": rel_coup, "pass": pass_e5, "label": label}


def plot_E5_decoupled(res, cfg, label="decoupled"):
    actual = res["actual_dE"]; pred = res["pred_decoupled"]
    lim = max(abs(actual).max(), abs(pred).max()) * 1.1
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(pred, actual, s=18, alpha=0.6, color=C_BLUE)
    ax.plot([-lim, lim], [-lim, lim], "--", color=C_GRAY, lw=1.2,
            label="identity")
    ax.set_xlabel(r"predicted $\Delta E_j = -\frac{4\epsilon}{\beta^2}"
                  r" u_j \tanh(u_j/2)$", fontsize=11)
    ax.set_ylabel(r"actual $\Delta E_j$", fontsize=12)
    ax.set_title(f"E5 ({label}): per-anchor Delta E (decoupled formula)\n"
                 f"rel err = {res['rel_dec']:.2e}", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    _finalize(cfg, f"e5_decoupled_pred_{label}.pdf")


def plot_E5_coupled(res, cfg, label="coupled"):
    actual = res["actual_dE"]
    pred_c = res["pred_coupled"]; pred_d = res["pred_decoupled"]
    lim = max(abs(actual).max(), abs(pred_c).max()) * 1.1
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(pred_d, actual, s=18, alpha=0.4, color=C_GRAY,
               label=f"decoupled pred (rel err {res['rel_dec']:.2e})")
    ax.scatter(pred_c, actual, s=18, alpha=0.7, color=C_RED,
               label=f"coupled pred (rel err {res['rel_coup']:.2e})")
    ax.plot([-lim, lim], [-lim, lim], "--", color="k", lw=1.0,
            label="identity")
    ax.set_xlabel(r"predicted $\Delta E_j$", fontsize=12)
    ax.set_ylabel(r"actual $\Delta E_j$", fontsize=12)
    ax.set_title(f"E5 ({label}): coupled formula tracks; decoupled does not",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    _finalize(cfg, f"e5_coupled_pred_{label}.pdf")


# ===========================
# E5b: Full DPO version of E5
# ===========================

def _experiment_E5b_single(dgp, anchors, beta, eps, n_anchors_test, seed,
                           verbose=False):
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Sigma_ss_P = Sigma_P[d_c:, d_c:]
    Scs_P_norm = np.linalg.norm(Sigma_P[:d_c, d_c:], "fro")

    theta_P_lin = linearized_optimum(mu_P, Sigma_P, beta)
    theta_P_lin_s = theta_P_lin[d_c:]
    delta_phi_s = anchors[:, d_c:]
    n = min(n_anchors_test, len(anchors))
    u_js_lin = beta * (delta_phi_s[:n] @ theta_P_lin_s)
    pred_lin = -(4.0 * eps / beta ** 2) * u_js_lin * np.tanh(u_js_lin / 2.0)

    theta_P_full = solve_full_dpo(anchors, beta, lr=0.05,
                                  max_iter=20000, tol=1e-10)
    theta_P_full_s = theta_P_full[d_c:]
    E_P_full = spurious_energy(theta_P_full, Sigma_ss_P, d_c)
    margins_full = np.abs(beta * (anchors @ theta_P_full))
    theta_amp = (np.linalg.norm(theta_P_full) /
                 (np.linalg.norm(theta_P_lin) + 1e-30))

    z_full = beta * (anchors @ theta_P_full)
    p_full = sigmoid(z_full)
    kappa = float(np.mean(p_full * (1.0 - p_full)))

    scale_kc = 1.0 / (4.0 * kappa) if kappa > 1e-6 else 1.0
    theta_P_kc_s = scale_kc * theta_P_lin_s
    u_js_kc = beta * (delta_phi_s[:n] @ theta_P_kc_s)
    pred_kc = -(4.0 * eps / beta ** 2) * u_js_kc * np.tanh(u_js_kc / 2.0)

    H_full = empirical_dpo_hessian(anchors, theta_P_full, beta)
    grad_E = 2.0 * Sigma_ss_P @ theta_P_full_s
    u_js_full = beta * (delta_phi_s[:n] @ theta_P_full_s)
    pred_sc = np.zeros(n)
    for j in range(n):
        g_j_s = (beta / 2.0) * np.tanh(u_js_full[j] / 2.0) * delta_phi_s[j]
        g_j = np.concatenate([np.zeros(d_c), g_j_s])
        H_inv_g = np.linalg.solve(H_full, g_j)
        pred_sc[j] = -eps * grad_E @ H_inv_g[d_c:]

    actual_dE = np.zeros(n)
    for j in range(n):
        theta_aug = solve_full_dpo(anchors, beta,
                                   theta_init=theta_P_full,
                                   lr=0.02, max_iter=3000, tol=1e-10,
                                   eps_aug=eps,
                                   tie_anchor_s=delta_phi_s[j],
                                   d_c=d_c)
        actual_dE[j] = spurious_energy(theta_aug, Sigma_ss_P, d_c) - E_P_full

    def _stats(pred):
        rel = np.linalg.norm(actual_dE - pred) / \
            (np.linalg.norm(actual_dE) + 1e-30)
        sp = spearmanr(np.abs(pred), np.abs(actual_dE)).correlation
        if np.isnan(sp):
            sp = 0.0
        return float(rel), float(sp)

    rel_sc, sp_sc = _stats(pred_sc)
    rel_kc, sp_kc = _stats(pred_kc)
    rel_lin, sp_lin = _stats(pred_lin)
    sign_aligned = int((actual_dE <= 1e-12).sum())

    if verbose:
        print(f"    max margin={margins_full.max():.3f}, "
              f"kappa={kappa:.4f}, theta amp={theta_amp:.2f}")
        print(f"    rel_sc={rel_sc:.4e}, rel_kc={rel_kc:.4e}, "
              f"rel_lin={rel_lin:.4e}")

    return {"actual_dE": actual_dE,
            "pred_sc": pred_sc, "pred_kc": pred_kc, "pred_lin": pred_lin,
            "u_js_lin": u_js_lin, "u_js_full": u_js_full,
            "max_margin": float(margins_full.max()),
            "kappa": kappa, "E_P_full": float(E_P_full),
            "rel_sc": rel_sc, "rel_kc": rel_kc, "rel_lin": rel_lin,
            "spearman_sc": sp_sc, "spearman_kc": sp_kc,
            "spearman_lin": sp_lin,
            "sign_aligned": sign_aligned, "n": n}


def experiment_E5b(dgp_config, beta, eps=1e-3, n_anchors_test=50,
                   mu_scales=None, base_seed=0, label=""):
    print("\n" + "=" * 72)
    print(f"E5b: Full DPO version of E5, label={label}")
    print("=" * 72)
    print(f"\n  [default mu_scale={dgp_config['mu_scale']}]")
    dgp_def = generate_dgp(dgp_config["d_c"], dgp_config["d_s"],
                           dgp_config["mu_scale"],
                           rho_cs=dgp_config["rho_cs"],
                           seed=base_seed,
                           cond_max=dgp_config["cond_max"])
    anchors_def = sample_anchors(dgp_def, dgp_config["N"], seed=base_seed + 1)
    res_def = _experiment_E5b_single(dgp_def, anchors_def, beta, eps,
                                     n_anchors_test, seed=base_seed + 2,
                                     verbose=True)

    sweep_results = []
    if mu_scales is not None:
        print(f"\n  [mu_scale sweep: {[float(x) for x in mu_scales]}]")
        for mu_s in mu_scales:
            dgp_s = generate_dgp(dgp_config["d_c"], dgp_config["d_s"],
                                 mu_s, rho_cs=dgp_config["rho_cs"],
                                 seed=base_seed,
                                 cond_max=dgp_config["cond_max"])
            anchors_s = sample_anchors(dgp_s, dgp_config["N"],
                                       seed=base_seed + 1)
            r = _experiment_E5b_single(dgp_s, anchors_s, beta, eps,
                                       min(n_anchors_test, 30),
                                       seed=base_seed + 2)
            r["mu_scale"] = float(mu_s)
            sweep_results.append(r)
            print(f"    mu={mu_s:.3f}: max_marg={r['max_margin']:.3f}, "
                  f"rel_sc={r['rel_sc']:.4f}, rel_lin={r['rel_lin']:.4f}")

    sign_frac = res_def["sign_aligned"] / res_def["n"]
    pass_e5b = (res_def["rel_sc"] < 0.05 and sign_frac >= 0.90 and
                res_def["spearman_sc"] >= 0.95)
    print(f"\n  VERDICT: {'PASS' if pass_e5b else 'FAIL'}")
    return {"default": res_def, "sweep": sweep_results,
            "label": label, "pass": pass_e5b}


def plot_E5b_scatter(res, cfg, label=""):
    """Single panel: scatter of |actual| vs |predicted| for the three predictions."""
    default = res["default"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    actual = np.abs(default["actual_dE"])
    pred_sc = np.abs(default["pred_sc"])
    pred_kc = np.abs(default["pred_kc"])
    pred_lin = np.abs(default["pred_lin"])

    ax.loglog(pred_sc, actual, "o", color=C_BLUE, ms=6, alpha=0.7,
              label=f"self-consistent (rel={default['rel_sc']:.2e})")
    ax.loglog(pred_kc, actual, "s", color=C_GREEN, ms=5, alpha=0.6,
              label=f"curvature-corrected (rel={default['rel_kc']:.2e})")
    ax.loglog(pred_lin, actual, "^", color=C_RED, ms=5, alpha=0.5,
              label=f"linearized (rel={default['rel_lin']:.2e})")

    lo = min(pred_sc.min(), pred_kc.min(), pred_lin.min(),
             actual.min()) * 0.5 + 1e-30
    hi = max(pred_sc.max(), pred_kc.max(), pred_lin.max(),
             actual.max()) * 2.0
    ax.plot([lo, hi], [lo, hi], "--", color="k", lw=1.0, alpha=0.6,
            label="identity")
    ax.set_xlabel(r"$|$predicted $\Delta E_j|$", fontsize=12)
    ax.set_ylabel(r"$|$actual $\Delta E_j|$", fontsize=12)
    ax.set_title(f"E5b ({label}): three predictions vs full-DPO actual",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    _finalize(cfg, f"e5b_scatter_{label}.pdf")


def plot_E5b_sweep(res, cfg, label=""):
    """Single panel: mu-sweep of relative errors."""
    sweep = res["sweep"]
    if not sweep:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    mus = np.array([r["mu_scale"] for r in sweep])
    rel_sc = np.array([r["rel_sc"] for r in sweep])
    rel_kc = np.array([r["rel_kc"] for r in sweep])
    rel_lin = np.array([r["rel_lin"] for r in sweep])

    ax.plot(mus, rel_sc, "o-", color=C_BLUE, lw=2, label="self-consistent")
    ax.plot(mus, rel_kc, "s-", color=C_GREEN, lw=2,
            label=r"curvature-corrected ($1/4\kappa$)")
    ax.plot(mus, rel_lin, "^-", color=C_RED, lw=2, label="linearized")
    ax.axhline(0.05, color=C_GRAY, ls=":", alpha=0.6, label="5% target")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mu$-scale", fontsize=12)
    ax.set_ylabel("relative error", fontsize=12)
    ax.set_title(f"E5b ({label}): predictions vs $\\mu$-scale", fontsize=12)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3, which="both")
    _finalize(cfg, f"e5b_sweep_{label}.pdf")


# ====================================
# E6: Subspace-CMS ranking equivalence
# ====================================

def experiment_E6(dgp, anchors, beta, rho_values, k_for_energy=200,
                  seed=0, label=""):
    """Verify Spearman rank corr of subspace-CMS with oracle |u|."""
    print("\n" + "=" * 72)
    print(f"E6: Subspace-CMS ranking equivalence, label={label}")
    print("=" * 72)
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    assert d_c == d_s, "E6 probe construction assumes d_c == d_s"
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Sigma_ss_P = Sigma_P[d_c:, d_c:]
    N = anchors.shape[0]
    rng = np.random.default_rng(seed)

    theta_P = linearized_optimum(mu_P, Sigma_P, beta)
    theta_P_s = theta_P[d_c:]
    delta_phi_s = anchors[:, d_c:]
    abs_u_oracle = np.abs(beta * (delta_phi_s @ theta_P_s))

    E_P = spurious_energy(theta_P, Sigma_ss_P, d_c)
    alpha = N / (N + k_for_energy)
    stl_idx = np.argsort(-abs_u_oracle)[:k_for_energy]
    theta_stl, _ = compute_augmented_optimum(
        anchors[stl_idx], dgp, beta, alpha)
    E_stl = spurious_energy(theta_stl, Sigma_ss_P, d_c)

    M = 50
    E_utl = 0.0
    for m in range(M):
        idx = rng.choice(N, size=k_for_energy, replace=False)
        theta_utl, _ = compute_augmented_optimum(
            anchors[idx], dgp, beta, alpha)
        E_utl += spurious_energy(theta_utl, Sigma_ss_P, d_c) / M

    print(f"  E_P={E_P:.4f}, E_STL={E_stl:.4f}, E_UTL={E_utl:.4f}")
    print(f"  {'rho':<8} {'spearman':<12} {'E_subspace':<12} {'frac_closed':<12}")
    print("  " + "-" * 50)

    rows = []
    for rho in rho_values:
        U_hat = np.zeros((d_c + d_s, d_s))
        U_hat[:d_s, :] = np.sqrt(max(0.0, 1.0 - rho ** 2)) * np.eye(d_s)
        U_hat[d_c:, :] = rho * np.eye(d_s)
        P_hat = U_hat @ U_hat.T
        cms_sub = np.abs(beta * (anchors @ (P_hat @ theta_P)))
        sp = spearmanr(abs_u_oracle, cms_sub).correlation
        sub_idx = np.argsort(-cms_sub)[:k_for_energy]
        theta_sub, _ = compute_augmented_optimum(
            anchors[sub_idx], dgp, beta, alpha)
        E_sub = spurious_energy(theta_sub, Sigma_ss_P, d_c)
        gap = E_utl - E_stl
        frac = (E_utl - E_sub) / (gap + 1e-30) if gap > 1e-30 else 0.0
        rows.append({"rho": rho, "spearman": sp, "E_subspace": E_sub,
                     "frac_closed": frac})
        print(f"  {rho:<8.2f} {sp:<12.4f} {E_sub:<12.4f} {frac:<12.4f}")

    spearmans = np.array([r["spearman"] for r in rows])
    pass_mono = np.all(np.diff(spearmans) >= -0.05)
    pass_top = abs(spearmans[-1] - 1.0) < 1e-6
    pass_e6 = pass_mono and pass_top
    print(f"\n  VERDICT: {'PASS' if pass_e6 else 'FAIL'}")
    return {"rows": rows, "E_P": E_P, "E_stl": E_stl, "E_utl": E_utl,
            "pass": pass_e6}


def plot_E6_spearman(res, cfg):
    rows = res["rows"]
    rho = np.array([r["rho"] for r in rows])
    sp = np.array([r["spearman"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(rho, sp, "o-", color=C_BLUE, lw=2)
    ax.set_xlabel(r"probe-subspace alignment $\rho$", fontsize=13)
    ax.set_ylabel(r"Spearman$(|u_j|, {\rm CMS}_j^{\hat V})$", fontsize=13)
    ax.set_title(r"E6: Ranking equivalence vs. alignment $\rho$", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    _finalize(cfg, "e6_spearman.pdf")


def plot_E6_energy(res, cfg):
    rows = res["rows"]
    rho = np.array([r["rho"] for r in rows])
    E_sub = np.array([r["E_subspace"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(rho, E_sub, "o-", color=C_BLUE, lw=2.5,
            label="subspace top-$k$")
    ax.axhline(res["E_stl"], ls="--", color=C_GREEN,
               label=f"E_STL (oracle) = {res['E_stl']:.3f}")
    ax.axhline(res["E_utl"], ls="--", color=C_RED,
               label=f"E_UTL (random) = {res['E_utl']:.3f}")
    ax.axhline(res["E_P"], ls=":", color=C_GRAY,
               label=f"E_P (strict) = {res['E_P']:.3f}")
    ax.set_xlabel(r"probe-subspace alignment $\rho$", fontsize=13)
    ax.set_ylabel(r"$E(\tilde\theta^*_{\rm aug})$", fontsize=13)
    ax.set_title(r"E6: Subspace selection energy interpolates STL <-> UTL",
                 fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _finalize(cfg, "e6_energy.pdf")


# ===========================================
# E7: Idealized influence vs simplified |u_j|
# ===========================================

def _experiment_E7_single(dgp, anchors, beta, k_values, seed=0):
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Sigma_ss_P = Sigma_P[d_c:, d_c:]
    N = anchors.shape[0]
    theta_P = linearized_optimum(mu_P, Sigma_P, beta)
    theta_P_s = theta_P[d_c:]
    H = (beta ** 2 / 4.0) * Sigma_P
    delta_phi_s = anchors[:, d_c:]
    u_j = beta * (delta_phi_s @ theta_P_s)
    abs_u = np.abs(u_j)

    full_scores = np.zeros(N)
    for j in range(N):
        # u_j[j] is already beta * <theta_s, Delta phi_s>, so the tie
        # gradient is (beta/2) * tanh(u_j/2) * Delta phi_s -- NOT
        # tanh(beta*u_j/2) (that would double-scale by beta).
        tanh_factor = np.tanh(u_j[j] / 2.0)
        g_j_s_block = (beta / 2.0) * tanh_factor * delta_phi_s[j]
        g_j = np.concatenate([np.zeros(d_c), g_j_s_block])
        delta_theta_j = -np.linalg.solve(H, g_j)
        full_scores[j] = -2.0 * theta_P_s @ Sigma_ss_P @ delta_theta_j[d_c:]

    sp = spearmanr(full_scores, abs_u).correlation
    full_order = np.argsort(-full_scores)
    simp_order = np.argsort(-abs_u)
    E_P = spurious_energy(theta_P, Sigma_ss_P, d_c)

    rows = []
    for k in k_values:
        if k == 0:
            rows.append({"k": 0, "E_full": E_P, "E_simp": E_P,
                         "overlap": 1.0, "rel_diff": 0.0})
            continue
        rho_k = N / (N + k)
        full_idx = full_order[:k]; simp_idx = simp_order[:k]
        theta_full, _ = compute_augmented_optimum(
            anchors[full_idx], dgp, beta, rho_k)
        theta_simp, _ = compute_augmented_optimum(
            anchors[simp_idx], dgp, beta, rho_k)
        E_full = spurious_energy(theta_full, Sigma_ss_P, d_c)
        E_simp = spurious_energy(theta_simp, Sigma_ss_P, d_c)
        overlap = len(set(full_idx.tolist()) & set(simp_idx.tolist())) / k
        rel_diff = (E_simp - E_full) / E_P
        rows.append({"k": k, "E_full": E_full, "E_simp": E_simp,
                     "overlap": overlap, "rel_diff": rel_diff})
    return {"rows": rows, "spearman": sp, "E_P": E_P}


def experiment_E7(dgp_config, beta, k_values, n_dgp_seeds=10,
                  base_seed=0, label=""):
    print("\n" + "=" * 72)
    print(f"E7: Idealized influence vs simplified |u_j|, label={label}")
    print("=" * 72)
    n_k = len(k_values)
    spearmans = np.zeros(n_dgp_seeds)
    E_full_arr = np.zeros((n_dgp_seeds, n_k))
    E_simp_arr = np.zeros((n_dgp_seeds, n_k))
    overlap_arr = np.zeros((n_dgp_seeds, n_k))
    rel_diff_arr = np.zeros((n_dgp_seeds, n_k))
    E_P_arr = np.zeros(n_dgp_seeds)

    for s in range(n_dgp_seeds):
        seed_s = base_seed + 1000 * s
        dgp_s = generate_dgp(dgp_config["d_c"], dgp_config["d_s"],
                             dgp_config["mu_scale"],
                             rho_cs=dgp_config["rho_cs"],
                             seed=seed_s, cond_max=dgp_config["cond_max"])
        anchors_s = sample_anchors(dgp_s, dgp_config["N"], seed=seed_s + 1)
        single = _experiment_E7_single(
            dgp_s, anchors_s, beta, k_values, seed=seed_s + 2)
        spearmans[s] = single["spearman"]
        E_P_arr[s] = single["E_P"]
        for j, r in enumerate(single["rows"]):
            E_full_arr[s, j] = r["E_full"]
            E_simp_arr[s, j] = r["E_simp"]
            overlap_arr[s, j] = r["overlap"]
            rel_diff_arr[s, j] = r["rel_diff"]
        print(f"    [seed {seed_s}] sp={single['spearman']:.6f}")

    sp_min = spearmans.min()
    rel_diff_min = rel_diff_arr.min(axis=0)
    decoupled = (dgp_config["rho_cs"] < 1e-8)
    if decoupled:
        pass_e7 = sp_min >= 0.999 and rel_diff_arr.max() < 1e-6
    else:
        worst_loss = -rel_diff_min.min()
        pass_e7 = sp_min >= 0.95 and worst_loss < 0.01
    print(f"\n  VERDICT: {'PASS' if pass_e7 else 'FAIL'}")

    return {"k_values": np.array(k_values),
            "spearmans": spearmans,
            "E_full_arr": E_full_arr, "E_simp_arr": E_simp_arr,
            "overlap_arr": overlap_arr, "rel_diff_arr": rel_diff_arr,
            "E_P_arr": E_P_arr,
            "n_dgp_seeds": n_dgp_seeds, "pass": pass_e7, "label": label}


def plot_E7_energies(res, cfg, label=""):
    ks = res["k_values"]
    full_m = res["E_full_arr"].mean(axis=0)
    full_s = res["E_full_arr"].std(axis=0)
    simp_m = res["E_simp_arr"].mean(axis=0)
    simp_s = res["E_simp_arr"].std(axis=0)
    n_seeds = res["n_dgp_seeds"]; sp = res["spearmans"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(ks, full_m, "o-", color=C_BLUE, lw=2.5,
            label="idealized (Hessian-based)")
    ax.fill_between(ks, full_m - full_s, full_m + full_s,
                    color=C_BLUE, alpha=0.20)
    ax.plot(ks, simp_m, "s--", color=C_RED, lw=2.0,
            label=r"simplified ($|u_j|$)")
    ax.fill_between(ks, simp_m - simp_s, simp_m + simp_s,
                    color=C_RED, alpha=0.20)
    ax.set_xlabel("budget $k$", fontsize=13)
    ax.set_ylabel(r"$E(\tilde\theta^*_{\rm aug})$", fontsize=13)
    ax.set_title(f"E7 ({label}): idealized vs simplified ({n_seeds} seeds)\n"
                 f"Spearman = {sp.mean():.4f} +/- {sp.std():.2e}",
                 fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    _finalize(cfg, f"e7_energies_{label}.pdf")


def plot_E7_overlap(res, cfg, label=""):
    ks = res["k_values"]
    overlap_m = res["overlap_arr"].mean(axis=0)
    overlap_s = res["overlap_arr"].std(axis=0)
    n_seeds = res["n_dgp_seeds"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(ks, overlap_m, "d-", color=C_GREEN, lw=2.5)
    ax.fill_between(ks, overlap_m - overlap_s, overlap_m + overlap_s,
                    color=C_GREEN, alpha=0.25,
                    label=f"+/- 1 std across {n_seeds} seeds")
    ax.set_xlabel("budget $k$", fontsize=13)
    ax.set_ylabel("top-$k$ overlap fraction", fontsize=13)
    ax.set_title(f"E7 ({label}): top-$k$ selection overlap", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(1.0, color=C_GRAY, ls=":", alpha=0.6, label="identical")
    ax.legend(); ax.grid(True, alpha=0.3)
    _finalize(cfg, f"e7_overlap_{label}.pdf")


# ===========================
# E8: Deployment OOD accuracy
# ===========================

def _experiment_E8_single(dgp, anchors, beta, k_values, M_utl, seed,
                          verbose=False):
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    N = anchors.shape[0]
    rng = np.random.default_rng(seed)
    mu_Q, Sigma_Q = adversarial_shift_moments(mu_P, Sigma_P, d_c)

    theta_P_full = solve_full_dpo(anchors, beta, lr=0.05,
                                  max_iter=20000, tol=1e-10)
    theta_P_lin = linearized_optimum(mu_P, Sigma_P, beta)
    delta_phi_s = anchors[:, d_c:]
    abs_u = np.abs(beta * (delta_phi_s @ theta_P_lin[d_c:]))
    stl_order = np.argsort(-abs_u)

    acc_P_full = deployment_accuracy(theta_P_full, mu_Q, Sigma_Q)
    acc_P_in = deployment_accuracy(theta_P_full, mu_P, Sigma_P)

    rows = []
    for k in k_values:
        if k == 0:
            rows.append({"k": 0, "acc_stl": acc_P_full,
                         "acc_utl_mean": acc_P_full, "acc_utl_std": 0.0,
                         "acc_strict": acc_P_full})
            continue
        stl_idx = stl_order[:k]
        ties = []
        for j in stl_idx:
            ds = delta_phi_s[j]
            ties.append(np.concatenate([np.zeros(d_c), ds]))
            ties.append(np.concatenate([np.zeros(d_c), -ds]))
        ties = np.array(ties)
        aug_data = np.concatenate([anchors, ties], axis=0)
        theta_stl_full = solve_full_dpo(
            aug_data, beta, theta_init=theta_P_full,
            lr=0.03, max_iter=10000, tol=1e-9)
        acc_stl = deployment_accuracy(theta_stl_full, mu_Q, Sigma_Q)

        accs_utl = np.zeros(M_utl)
        for m in range(M_utl):
            utl_idx = rng.choice(N, size=k, replace=False)
            ties_p = []
            for j in utl_idx:
                ds = delta_phi_s[j]
                ties_p.append(np.concatenate([np.zeros(d_c), ds]))
                ties_p.append(np.concatenate([np.zeros(d_c), -ds]))
            ties_p = np.array(ties_p)
            aug_p = np.concatenate([anchors, ties_p], axis=0)
            theta_utl_full = solve_full_dpo(
                aug_p, beta, theta_init=theta_P_full,
                lr=0.03, max_iter=8000, tol=1e-9)
            accs_utl[m] = deployment_accuracy(theta_utl_full, mu_Q, Sigma_Q)

        rows.append({"k": k, "acc_stl": acc_stl,
                     "acc_utl_mean": float(accs_utl.mean()),
                     "acc_utl_std": float(accs_utl.std()),
                     "acc_strict": acc_P_full})
    if verbose:
        print(f"    acc_in={acc_P_in:.4f}, acc_OOD_strict={acc_P_full:.4f}")
    return {"rows": rows, "acc_P_full": acc_P_full, "acc_P_in": acc_P_in}


def experiment_E8(dgp_config, beta, k_values, n_dgp_seeds=5, M_utl=20,
                  base_seed=0, label=""):
    print("\n" + "=" * 72)
    print(f"E8: Deployment OOD accuracy under adversarial shift, label={label}")
    print("=" * 72)
    n_k = len(k_values)
    acc_stl_arr = np.zeros((n_dgp_seeds, n_k))
    acc_utl_arr = np.zeros((n_dgp_seeds, n_k))
    acc_P_arr = np.zeros(n_dgp_seeds)
    acc_P_in_arr = np.zeros(n_dgp_seeds)

    for s in range(n_dgp_seeds):
        seed_s = base_seed + 1000 * s
        dgp_s = generate_dgp(dgp_config["d_c"], dgp_config["d_s"],
                             dgp_config["mu_scale"],
                             rho_cs=dgp_config["rho_cs"],
                             seed=seed_s,
                             cond_max=dgp_config["cond_max"])
        anchors_s = sample_anchors(dgp_s, dgp_config["N"], seed=seed_s + 1)
        single = _experiment_E8_single(
            dgp_s, anchors_s, beta, k_values, M_utl=M_utl, seed=seed_s + 2,
            verbose=True)
        for j, r in enumerate(single["rows"]):
            acc_stl_arr[s, j] = r["acc_stl"]
            acc_utl_arr[s, j] = r["acc_utl_mean"]
        acc_P_arr[s] = single["acc_P_full"]
        acc_P_in_arr[s] = single["acc_P_in"]
        print(f"    [seed {seed_s}] in={single['acc_P_in']:.4f}, "
              f"OOD_strict={single['acc_P_full']:.4f}")

    stl_m = acc_stl_arr.mean(axis=0); stl_s = acc_stl_arr.std(axis=0)
    utl_m = acc_utl_arr.mean(axis=0); utl_s = acc_utl_arr.std(axis=0)

    print(f"\n  In-dist mean = {acc_P_in_arr.mean():.4f}, "
          f"OOD strict mean = {acc_P_arr.mean():.4f}")
    for j, k in enumerate(k_values):
        gap = stl_m[j] - utl_m[j]
        print(f"  k={k}: STL={stl_m[j]:.4f}+/-{stl_s[j]:.4f}, "
              f"UTL={utl_m[j]:.4f}+/-{utl_s[j]:.4f} ({gap:+.4f})")

    ks_arr = np.array(k_values)
    pos_idx = ks_arr > 0
    pass_e8 = (bool(np.all(stl_m[pos_idx] > utl_m[pos_idx])) and
               bool(np.all(stl_m[pos_idx] > acc_P_arr.mean())))
    print(f"\n  VERDICT: {'PASS' if pass_e8 else 'FAIL'}")
    return {"k_values": ks_arr,
            "acc_stl_arr": acc_stl_arr, "acc_utl_arr": acc_utl_arr,
            "acc_stl_mean": stl_m, "acc_stl_std": stl_s,
            "acc_utl_mean": utl_m, "acc_utl_std": utl_s,
            "acc_P_arr": acc_P_arr, "acc_P_in_arr": acc_P_in_arr,
            "n_dgp_seeds": n_dgp_seeds, "M_utl": M_utl, "pass": pass_e8,
            "label": label}


def plot_E8(res, cfg, label=""):
    ks = res["k_values"]
    stl_m, stl_s = res["acc_stl_mean"], res["acc_stl_std"]
    utl_m, utl_s = res["acc_utl_mean"], res["acc_utl_std"]
    acc_P_mean = res["acc_P_arr"].mean()
    acc_in_mean = res["acc_P_in_arr"].mean()
    n_seeds = res["n_dgp_seeds"]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    ax.plot(ks, stl_m, "o-", color=C_BLUE, lw=2.5, label="STL augmented")
    ax.fill_between(ks, stl_m - stl_s, stl_m + stl_s,
                    color=C_BLUE, alpha=0.20)
    ax.plot(ks, utl_m, "s-", color=C_RED, lw=2.0, label="UTL augmented")
    ax.fill_between(ks, utl_m - utl_s, utl_m + utl_s,
                    color=C_RED, alpha=0.20)
    ax.axhline(acc_P_mean, ls=":", color=C_GRAY, lw=1.5,
               label=f"strict OOD = {acc_P_mean:.3f}")
    ax.axhline(acc_in_mean, ls="--", color=C_GREEN, lw=1.0, alpha=0.7,
               label=f"in-dist = {acc_in_mean:.3f}")
    ax.set_xlabel("budget $k$", fontsize=13)
    ax.set_ylabel(r"deployment accuracy $\mathrm{Acc}(Q;\tilde\theta)$",
                  fontsize=13)
    ax.set_title(f"E8 ({label}): OOD accuracy under "
                 r"$\Sigma_{cs}^Q = -\Sigma_{cs}^P$", fontsize=12)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.3)
    _finalize(cfg, f"e8_deployment_{label}.pdf")


# ====================================================
# E6b: 5-way comparison FK / PM / GA / HA / Random-UTL
#      (score mode = abs or signed; PM is always abs)
# ====================================================

def build_antisym_FK(anchors, selected_idx, d_c):
    """Antisymmetric ties with Delta_phi_c = 0 enforced."""
    if len(selected_idx) == 0:
        return anchors
    delta_phi_s = anchors[selected_idx, d_c:]
    n = len(selected_idx)
    ties = np.zeros((2 * n, anchors.shape[1]))
    ties[0::2, d_c:] = delta_phi_s
    ties[1::2, d_c:] = -delta_phi_s
    return np.concatenate([anchors, ties], axis=0)


def estimate_v_s(theta_P_full, dgp, beta, N_v, rng, return_probes=False):
    """Spurious-sensitivity gradient via counterfactual probe.
    If return_probes=True, also returns (v_s_unnorm, probe_deltas)
    for use by sign_sanity_check."""
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    d = d_c + d_s
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Cov_P = Sigma_P - np.outer(mu_P, mu_P)
    L = np.linalg.cholesky(Cov_P + 1e-10 * np.eye(d))
    z1 = rng.standard_normal((N_v, d))
    winners = mu_P + z1 @ L.T
    z2 = rng.standard_normal((N_v, d))
    resampled = mu_P + z2 @ L.T
    deltas = np.zeros((N_v, d))
    deltas[:, d_c:] = winners[:, d_c:] - resampled[:, d_c:]
    proj = deltas @ theta_P_full
    # Note: omits beta^2 prefactor (constant > 0; ranking unaffected).
    v_s_unnorm = (proj[:, None] * deltas).mean(axis=0)
    n = np.linalg.norm(v_s_unnorm)
    v_s = v_s_unnorm / n if n > 1e-12 else v_s_unnorm
    if return_probes:
        return v_s, v_s_unnorm, deltas
    return v_s


def _ecf_energy(theta, probe_deltas):
    """Counterfactual energy proxy along the probe directions:
        E_cf(theta) = (1/2) E[ (Delta^T theta)^2 ].
    Used by sign_sanity_check below."""
    m = probe_deltas @ theta
    return 0.5 * float(np.mean(m ** 2))


def sign_sanity_check(theta, anchors, H, v_s_unnorm, probe_deltas, beta,
                      idxs, eta=1e-4, damping=1e-6):
    """For each anchor j in idxs, compare the predicted signed
    influence A_j (computed against the unnormalized v_s, i.e. the
    energy gradient itself) to a tiny-step empirical Delta E along
    the influence-direction.

    Returns rows of (A_j, Delta E_j, -Delta E_j / eta).
    Predictions:
      * Delta E_j ~ -eta * A_j  (so the ratio in column 3 ~ A_j).
      * sign(A_j) == sign(-Delta E_j) when sign convention is correct.
    """
    H_sym = 0.5 * (H + H.T) + damping * np.eye(H.shape[0])
    H_inv = np.linalg.inv(H_sym)
    E0 = _ecf_energy(theta, probe_deltas)
    rows = []
    for j in idxs:
        margin = beta * (anchors[j] @ theta)
        g_j = (beta / 2.0) * np.tanh(margin / 2.0) * anchors[j]
        A_j = float(v_s_unnorm @ (H_inv @ g_j))
        theta_new = theta - eta * (H_inv @ g_j)
        dE = _ecf_energy(theta_new, probe_deltas) - E0
        rows.append((A_j, dE, -dE / eta))
    return rows


# ----- Score functions with mode flag -----
#
# Mode semantics:
#   abs    : |raw signed influence|  (treats anti-suppressive == suppressive)
#   signed : max(0, raw signed influence)  (anti-suppressive get score 0,
#            matches r_j = [A_j]_+ from the signed-influence theory)
#
# PM is always 'abs' by construction (it has no sign information).
# FK is a magnitude score (|u_j|); for the local log-linear case the
# signed-influence A_j ~ u_j tanh(u_j/2) is non-negative, so signed
# and abs coincide ranking-wise. We still expose the flag for
# uniformity.

def _post_process(raw_score, mode):
    if mode == "abs":
        return np.abs(raw_score)
    elif mode == "signed":
        return np.maximum(raw_score, 0.0)
    raise ValueError(f"unknown mode: {mode}")


def score_PM(theta, anchors):
    """Pure margin: |theta^T Delta phi|.  Always abs."""
    return np.abs(anchors @ theta)


def score_FK(theta, anchors, beta, d_c, mode="abs"):
    """Simplified feature-known score (pure spurious-margin closed form):
        raw_j = u_j * tanh(u_j/2),    u_j = beta * <theta_s, Delta phi_s^(j)>.
    This is the FK score that holds exactly under decoupling
    (Sigma_cs^P = 0). In coupled regimes it is a first-order proxy for
    the true signed influence; for the Schur-rotated true oracle, use
    score_FK_oracle. Always non-negative -> abs and signed coincide
    ranking-wise; the mode flag is exposed only for uniformity.
    """
    u_j = beta * (anchors[:, d_c:] @ theta[d_c:])
    raw = u_j * np.tanh(u_j / 2.0)
    return _post_process(raw, mode)


def score_FK_oracle(theta, anchors, Sigma_ss, H, beta, d_c,
                    mode="abs", damping=1e-6):
    """True feature-known oracle: signed influence A_j computed exactly,
    using the energy gradient grad_E (causally clean, no probe needed)
    and the Hessian H.

        margin_j = beta * <theta_s, Delta phi_s^(j)>
        g_j      = [0; (beta/2) tanh(margin_j/2) Delta phi_s^(j)]
        grad_E   = [0; 2 Sigma_ss theta_s]
        A_j      = <grad_E, H^{-1} g_j>
                 = (beta/2) tanh(margin_j/2)
                   * (Delta phi_s^(j))^T (H^{-1} grad_E)_s.
    This is the FK score the appendix theorem refers to in coupled
    regimes (Schur-rotated). Reduces to score_FK in the decoupled
    case up to a positive scalar.
    """
    H_sym = 0.5 * (H + H.T)
    H_damped = H_sym + damping * np.eye(H_sym.shape[0])
    d = anchors.shape[1]
    grad_E = np.zeros(d)
    grad_E[d_c:] = 2.0 * Sigma_ss @ theta[d_c:]
    H_inv_gradE = np.linalg.solve(H_damped, grad_E)
    margin = beta * (anchors[:, d_c:] @ theta[d_c:])
    weights = (beta / 2.0) * np.tanh(margin / 2.0)
    raw = weights * (anchors[:, d_c:] @ H_inv_gradE[d_c:])
    return _post_process(raw, mode)


def score_GA(theta, anchors, v_s, beta, mode="abs"):
    """Gradient alignment: <v_s, g_j>.
        margin_j = beta * <theta, Delta phi_j>
        g_j      = (beta/2) tanh(margin_j/2) Delta phi_j
        raw_j    = <v_s, g_j>
                 = (beta/2) tanh(margin_j/2) (Delta phi_j^T v_s).
    Sign carries first-order suppressive vs anti-suppressive info.
    """
    margin = beta * (anchors @ theta)
    weights = (beta / 2.0) * np.tanh(margin / 2.0)
    raw = weights * (anchors @ v_s)
    return _post_process(raw, mode)


def score_HA(theta, anchors, v_s, H, beta, mode="abs", damping=1e-6):
    """Hessian-aware: <v_s, H^{-1} g_j>.
        margin_j = beta * <theta, Delta phi_j>
        raw_j    = <v_s, H^{-1} g_j>
                 = <H^{-T} v_s, g_j>     (H symmetric)
                 = (beta/2) tanh(margin_j/2) (Delta phi_j^T H^{-1} v_s).
    H is symmetrized (H + H^T)/2 and lightly damped for numerical
    stability; empirical DPO Hessians can be ill-conditioned.
    """
    H_sym = 0.5 * (H + H.T)
    H_damped = H_sym + damping * np.eye(H_sym.shape[0])
    margin = beta * (anchors @ theta)
    weights = (beta / 2.0) * np.tanh(margin / 2.0)
    H_inv_v = np.linalg.solve(H_damped, v_s)
    raw = weights * (anchors @ H_inv_v)
    return _post_process(raw, mode)


def _e6b_single_dgp(dgp, anchors, beta, k_values, M_rnd, N_v, seed,
                    score_mode="abs"):
    """Single-DGP run of E6b. Returns per-method arrays."""
    d_c, d_s = dgp["d_c"], dgp["d_s"]
    d = d_c + d_s
    mu_P, Sigma_P = dgp["mu"], dgp["Sigma"]
    Sigma_ss_P = Sigma_P[d_c:, d_c:]
    Sigma_cs_P = Sigma_P[:d_c, d_c:]
    rng = np.random.default_rng(seed)
    mu_Q, Sigma_Q = adversarial_shift_moments(mu_P, Sigma_P, d_c)

    theta_P_full = solve_full_dpo(anchors, beta, lr=0.05,
                                  max_iter=20000, tol=1e-10)
    H_full = empirical_dpo_hessian(anchors, theta_P_full, beta)
    E_strict = spurious_energy(theta_P_full, Sigma_ss_P, d_c)
    acc_strict = deployment_accuracy(theta_P_full, mu_Q, Sigma_Q)
    norm_strict = float(np.linalg.norm(theta_P_full))
    cross_strict = float(theta_P_full[:d_c] @ Sigma_cs_P @ theta_P_full[d_c:])

    v_s = estimate_v_s(theta_P_full, dgp, beta, N_v, rng)

    # Score arrays. PM is always abs; the others use score_mode.
    fk_scores = score_FK(theta_P_full, anchors, beta, d_c, mode=score_mode)
    pm_scores = score_PM(theta_P_full, anchors)
    ga_scores = score_GA(theta_P_full, anchors, v_s, beta, mode=score_mode)
    ha_scores = score_HA(theta_P_full, anchors, v_s, H_full, beta,
                         mode=score_mode)

    # Raw signed scores (no abs, no positive-part) for diagnostics.
    # margin = beta * <theta, Delta phi_j>  -- explicit scaling, no double beta
    margin = beta * (anchors @ theta_P_full)
    weights = (beta / 2.0) * np.tanh(margin / 2.0)
    raw_ga = weights * (anchors @ v_s)
    H_sym = 0.5 * (H_full + H_full.T) + 1e-6 * np.eye(d)
    H_inv_v = np.linalg.solve(H_sym, v_s)
    raw_ha = weights * (anchors @ H_inv_v)

    # Global frac+ over all N anchors.
    pos_frac_ga = float(np.mean(raw_ga > 0))
    pos_frac_ha = float(np.mean(raw_ha > 0))

    # Selected frac+: among the top-k anchors actually picked by the
    # current scoring rule (mode-aware), what fraction has positive
    # signed contribution? abs mode can pick large-|raw| candidates
    # with negative raw; signed mode should be 1.0 unless fewer than
    # k positive candidates exist.
    def _sel_pos(raw, scores, k):
        if k <= 0:
            return float("nan")
        idx = np.argsort(-scores)[:k]
        return float(np.mean(raw[idx] > 0))

    sel_pos_ga = np.array([_sel_pos(raw_ga, ga_scores, k) for k in k_values])
    sel_pos_ha = np.array([_sel_pos(raw_ha, ha_scores, k) for k in k_values])

    fk_order = np.argsort(-fk_scores)
    pm_order = np.argsort(-pm_scores)
    ga_order = np.argsort(-ga_scores)
    ha_order = np.argsort(-ha_scores)

    n_k = len(k_values)
    methods = ["fk", "pm", "ga", "ha"]
    out = {}
    for m in methods:
        for key in ("E", "acc", "norm", "cos", "cross"):
            out[f"{key}_{m}"] = np.zeros(n_k)
    for tag in ["rnd"]:
        for key in ("E", "acc", "norm", "cos", "cross"):
            out[f"{key}_{tag}"] = np.zeros((n_k, M_rnd))

    def diag(theta):
        n_t = float(np.linalg.norm(theta))
        cos = (float(theta @ theta_P_full) / (n_t * norm_strict)
               if n_t > 1e-12 else 0.0)
        cross = float(theta[:d_c] @ Sigma_cs_P @ theta[d_c:])
        return n_t, cos, cross

    orders = {"fk": fk_order, "pm": pm_order,
              "ga": ga_order, "ha": ha_order}

    for ik, k in enumerate(k_values):
        if k == 0:
            for m in methods:
                out[f"E_{m}"][ik] = E_strict
                out[f"acc_{m}"][ik] = acc_strict
                out[f"norm_{m}"][ik] = norm_strict
                out[f"cos_{m}"][ik] = 1.0
                out[f"cross_{m}"][ik] = cross_strict
            for tag in ["rnd"]:
                out[f"E_{tag}"][ik, :] = E_strict
                out[f"acc_{tag}"][ik, :] = acc_strict
                out[f"norm_{tag}"][ik, :] = norm_strict
                out[f"cos_{tag}"][ik, :] = 1.0
                out[f"cross_{tag}"][ik, :] = cross_strict
            continue

        for m in methods:
            sel = orders[m][:k]
            aug = build_antisym_FK(anchors, sel, d_c)
            theta = solve_full_dpo(aug, beta, theta_init=theta_P_full,
                                   lr=0.03, max_iter=8000, tol=1e-9)
            out[f"E_{m}"][ik] = spurious_energy(theta, Sigma_ss_P, d_c)
            out[f"acc_{m}"][ik] = deployment_accuracy(theta, mu_Q, Sigma_Q)
            n_t, cos, cross = diag(theta)
            out[f"norm_{m}"][ik] = n_t
            out[f"cos_{m}"][ik] = cos
            out[f"cross_{m}"][ik] = cross

        for j in range(M_rnd):
            sel = rng.choice(len(anchors), size=k, replace=False)
            aug = build_antisym_FK(anchors, sel, d_c)
            theta = solve_full_dpo(aug, beta, theta_init=theta_P_full,
                                   lr=0.03, max_iter=6000, tol=1e-9)
            out["E_rnd"][ik, j] = spurious_energy(theta, Sigma_ss_P, d_c)
            out["acc_rnd"][ik, j] = deployment_accuracy(theta, mu_Q, Sigma_Q)
            n_t, cos, cross = diag(theta)
            out["norm_rnd"][ik, j] = n_t
            out["cos_rnd"][ik, j] = cos
            out["cross_rnd"][ik, j] = cross

    out["E_strict"] = E_strict
    out["acc_strict"] = acc_strict
    out["norm_strict"] = norm_strict
    out["cross_strict"] = cross_strict
    out["pos_frac_ga"] = pos_frac_ga
    out["pos_frac_ha"] = pos_frac_ha
    out["sel_pos_ga"] = sel_pos_ga
    out["sel_pos_ha"] = sel_pos_ha
    return out


def experiment_E6b(cfg):
    """Multi-seed E6b. Score mode is read from cfg['mode']."""
    score_mode = cfg.get("mode", "abs")
    print("=" * 72)
    print(f"E6b: FK / HA / GA / PM / Random-UTL  --  score mode = {score_mode}")
    print(f"     PM always abs; FK/HA/GA use mode={score_mode}")
    print(f"     mu_scale = {cfg['e6b_mu_scale']}, "
          f"rho_cs = {cfg['e6b_rho_cs']}, "
          f"d = {cfg['d_c'] + cfg['d_s']}, N = {cfg['N']}")
    print(f"     k values: {cfg['e6b_k_values']}")
    print(f"     n_dgp_seeds = {cfg['num_seeds']}, "
          f"M_rnd = {cfg['e6b_M_rnd']}")
    print("=" * 72)

    n_seeds = cfg["num_seeds"]
    k_values = cfg["e6b_k_values"]
    n_k = len(k_values)
    M_rnd = cfg["e6b_M_rnd"]

    methods = ["fk", "pm", "ga", "ha"]
    storage = {}
    for m in methods:
        for key in ("E", "acc", "norm", "cos", "cross"):
            storage[f"{key}_{m}"] = np.zeros((n_seeds, n_k))
    for tag in ["rnd"]:
        for key in ("E", "acc", "norm", "cos", "cross"):
            storage[f"{key}_{tag}"] = np.zeros((n_seeds, n_k, M_rnd))
    E_strict_all = np.zeros(n_seeds)
    acc_strict_all = np.zeros(n_seeds)
    norm_strict_all = np.zeros(n_seeds)
    cross_strict_all = np.zeros(n_seeds)
    pos_frac_ga_all = np.zeros(n_seeds)
    pos_frac_ha_all = np.zeros(n_seeds)
    sel_pos_ga_all = np.zeros((n_seeds, n_k))
    sel_pos_ha_all = np.zeros((n_seeds, n_k))

    t0 = time.time()
    for s in range(n_seeds):
        seed_s = cfg["seed"] + 1000 * s
        dgp_s = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["e6b_mu_scale"],
                             rho_cs=cfg["e6b_rho_cs"], seed=seed_s,
                             cond_max=cfg["cond_max"])
        anchors_s = sample_anchors(dgp_s, cfg["N"], seed=seed_s + 1)
        t_s = time.time()
        res = _e6b_single_dgp(dgp_s, anchors_s, cfg["beta"], k_values,
                              M_rnd, cfg["e6b_N_v"], seed=seed_s + 2,
                              score_mode=score_mode)
        for key in storage:
            if storage[key].ndim == 2:
                storage[key][s] = res[key]
            else:
                storage[key][s] = res[key]
        E_strict_all[s] = res["E_strict"]
        acc_strict_all[s] = res["acc_strict"]
        norm_strict_all[s] = res["norm_strict"]
        cross_strict_all[s] = res["cross_strict"]
        pos_frac_ga_all[s] = res["pos_frac_ga"]
        pos_frac_ha_all[s] = res["pos_frac_ha"]
        sel_pos_ga_all[s] = res["sel_pos_ga"]
        sel_pos_ha_all[s] = res["sel_pos_ha"]
        print(f"  [seed {seed_s}] E_strict={res['E_strict']:.3f}, "
              f"acc_strict={res['acc_strict']:.4f}, "
              f"frac+(GA)={res['pos_frac_ga']:.3f}, "
              f"frac+(HA)={res['pos_frac_ha']:.3f}, "
              f"({time.time() - t_s:.1f}s)")

    print(f"\n  Total: {time.time() - t0:.1f}s")
    print(f"\n  Strict-only: E={E_strict_all.mean():.3f}, "
          f"acc={acc_strict_all.mean():.4f}, "
          f"||theta||={norm_strict_all.mean():.2f}")
    print(f"  Global frac+ (over all anchors): "
          f"GA={pos_frac_ga_all.mean():.3f}, "
          f"HA={pos_frac_ha_all.mean():.3f}")

    # Selected frac+ table: among the ties actually picked at each
    # budget k, what fraction has positive signed contribution?
    # Under signed mode, this should be 1.0 unless fewer than k
    # positive anchors exist; under abs mode, gaps from 1.0
    # quantify how often abs mode is selecting anti-suppressive ties.
    print(f"\n  Selected frac+ across seeds (mean) -- ties picked at top-k:")
    print(f"  {'k':<6} {'sel+(GA)':<12} {'sel+(HA)':<12}")
    print("  " + "-" * 32)
    for ik, k in enumerate(k_values):
        if k == 0:
            print(f"  {k:<6} {'(n/a)':<12} {'(n/a)':<12}")
            continue
        m_ga = float(np.nanmean(sel_pos_ga_all[:, ik]))
        m_ha = float(np.nanmean(sel_pos_ha_all[:, ik]))
        print(f"  {k:<6} {m_ga:<12.3f} {m_ha:<12.3f}")

    print(f"\n  OOD accuracy across seeds (mean):")
    print(f"  {'k':<6} {'FK':<8} {'HA':<8} {'GA':<8} {'PM':<8} {'rnd':<8}")
    print("  " + "-" * 48)
    for ik, k in enumerate(k_values):
        row = f"  {k:<6}"
        for m in ["fk", "ha", "ga", "pm"]:
            row += f"{storage[f'acc_{m}'][:, ik].mean():<8.4f}"
        row += f"{storage['acc_rnd'][:, ik].mean():<8.4f}"
        print(row)

    print(f"\n  Spurious energy across seeds (mean):")
    print(f"  {'k':<6} {'FK':<8} {'HA':<8} {'GA':<8} {'PM':<8} {'rnd':<8}")
    print("  " + "-" * 48)
    for ik, k in enumerate(k_values):
        row = f"  {k:<6}"
        for m in ["fk", "ha", "ga", "pm"]:
            row += f"{storage[f'E_{m}'][:, ik].mean():<8.3f}"
        row += f"{storage['E_rnd'][:, ik].mean():<8.3f}"
        print(row)

    return {"storage": storage,
            "k_values": np.array(k_values),
            "E_strict": E_strict_all.mean(),
            "acc_strict": acc_strict_all.mean(),
            "norm_strict": norm_strict_all.mean(),
            "cross_strict": cross_strict_all.mean(),
            "pos_frac_ga": pos_frac_ga_all.mean(),
            "pos_frac_ha": pos_frac_ha_all.mean(),
            "sel_pos_ga": sel_pos_ga_all,
            "sel_pos_ha": sel_pos_ha_all,
            "n_dgp_seeds": n_seeds,
            "score_mode": score_mode}


# Plotting helpers for E6b -- one panel per metric, individualized
# so each saves to its own PDF.

_E6B_PALETTE = {"fk": C_BLUE, "ha": C_GREEN, "ga": C_PURPLE,
                "pm": C_ORANGE, "rnd": C_GRAY}
_E6B_MARKERS = {"fk": "o", "ha": "s", "ga": "D", "pm": "^", "rnd": "v"}
_E6B_LABELS = {"fk": "FK-Oracle", "ha": "HA", "ga": "GA",
               "pm": "PM", "rnd": "Random-UTL"}


def _e6b_get_mean_std(storage, prefix, tag):
    if tag == "rnd":
        arr = storage[f"{prefix}_{tag}"]
        m = arr.mean(axis=(0, 2))
        s = arr.mean(axis=2).std(axis=0)
    else:
        arr = storage[f"{prefix}_{tag}"]
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
    return m, s


def _e6b_make_panel(res, prefix, ylabel, title, fname_template, cfg,
                    log_y=False, axhline=None, axhline_label=None):
    """Plot one metric (acc, E, norm, cos, cross) for FK/HA/GA/PM/rnd."""
    storage = res["storage"]; ks = res["k_values"]
    mode = res["score_mode"]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for tag in ["fk", "ha", "ga", "pm", "rnd"]:
        m, s = _e6b_get_mean_std(storage, prefix, tag)
        ls = "--" if tag == "rnd" else "-"
        if log_y:
            ax.semilogy(ks, m + 1e-6, marker=_E6B_MARKERS[tag], lw=2, ms=6,
                        ls=ls, color=_E6B_PALETTE[tag], label=_E6B_LABELS[tag])
        else:
            ax.plot(ks, m, marker=_E6B_MARKERS[tag], lw=2, ms=6, ls=ls,
                    color=_E6B_PALETTE[tag], label=_E6B_LABELS[tag])
            ax.fill_between(ks, m - s, m + s,
                            color=_E6B_PALETTE[tag], alpha=0.10)
    if axhline is not None:
        ax.axhline(axhline, ls=":", color="#999",
                   label=axhline_label or f"strict ({axhline:.3f})")
    ax.set_xlabel("budget $k$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"{title}  (mode={mode})", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both" if log_y else "major")
    fname = fname_template.format(mode=mode)
    _finalize(cfg, fname)


def plot_E6b_acc(res, cfg):
    _e6b_make_panel(res, "acc", "OOD accuracy",
                    r"E6b: OOD accuracy under $\Sigma_{cs}^Q = -\Sigma_{cs}^P$",
                    "e6b_{mode}_acc_OOD.pdf", cfg,
                    axhline=res["acc_strict"],
                    axhline_label=f"strict ({res['acc_strict']:.3f})")


def plot_E6b_energy(res, cfg):
    _e6b_make_panel(res, "E", "spurious energy (log)",
                    "E6b: Spurious energy at full-DPO optimum",
                    "e6b_{mode}_spurious_energy.pdf", cfg, log_y=True,
                    axhline=res["E_strict"],
                    axhline_label=f"strict ({res['E_strict']:.2f})")


def plot_E6b_norm(res, cfg):
    _e6b_make_panel(res, "norm", r"$\|\tilde\theta\|$",
                    r"E6b: Parameter norm $\|\tilde\theta\|$",
                    "e6b_{mode}_theta_norm.pdf", cfg,
                    axhline=res["norm_strict"],
                    axhline_label=f"strict ({res['norm_strict']:.2f})")


def plot_E6b_cross(res, cfg):
    _e6b_make_panel(res, "cross",
                    r"$\tilde\theta_c^\top \Sigma_{cs}^P \tilde\theta_s$",
                    "E6b: Cross-block term (deployment-cost driver)",
                    "e6b_{mode}_cross_term.pdf", cfg,
                    axhline=res["cross_strict"],
                    axhline_label=f"strict ({res['cross_strict']:+.3f})")


def plot_E6b_cos(res, cfg):
    _e6b_make_panel(res, "cos", r"$\cos(\tilde\theta, \tilde\theta_{\rm strict})$",
                    "E6b: Cosine similarity to strict optimum",
                    "e6b_{mode}_cos_strict.pdf", cfg)


# ===
# CLI
# ===

EXPERIMENTS = ["1", "2", "3", "4", "5", "5b", "6", "7", "8", "e6b", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="STL linear theory validation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiment", "-e", required=True,
                   choices=EXPERIMENTS,
                   help="Which experiment to run.")

    # Mode group: --signed XOR --abs (also accepts --mode for compat)
    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument("--signed", action="store_const",
                          dest="mode", const="signed",
                          help="Use positive-part [score]_+ for FK/GA/HA "
                               "(matches signed-influence theory).")
    mode_grp.add_argument("--abs", action="store_const",
                          dest="mode", const="abs",
                          help="Use |score| for FK/GA/HA (default).")
    mode_grp.add_argument("--mode", choices=["abs", "signed"],
                          help="Alias for --signed/--abs.")
    p.set_defaults(mode=None)

    # DGP & dimension
    p.add_argument("--d_c", type=int, default=DEFAULT_CONFIG["d_c"])
    p.add_argument("--d_s", type=int, default=DEFAULT_CONFIG["d_s"])
    p.add_argument("--N", type=int, default=DEFAULT_CONFIG["N"])
    p.add_argument("--beta", type=float, default=DEFAULT_CONFIG["beta"])
    p.add_argument("--mu_scale", type=float,
                   default=DEFAULT_CONFIG["mu_scale"],
                   help="DGP mu scale for E1-E8.")
    p.add_argument("--rho_cs", type=float,
                   default=DEFAULT_CONFIG["rho_cs"],
                   help="Cross-block correlation for coupled DGP (E1-E8).")
    p.add_argument("--cond_max", type=float,
                   default=DEFAULT_CONFIG["cond_max"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])

    # Multi-seed
    p.add_argument("--num_seeds", type=int,
                   default=DEFAULT_CONFIG["num_seeds"],
                   help="Number of DGP seeds (multi-seed experiments).")
    p.add_argument("--M_utl", type=int, default=DEFAULT_CONFIG["M_utl"],
                   help="Number of UTL random subsets per seed.")

    # E6b-specific overrides
    p.add_argument("--e6b_mu_scale", type=float,
                   default=DEFAULT_CONFIG["e6b_mu_scale"])
    p.add_argument("--e6b_rho_cs", type=float,
                   default=DEFAULT_CONFIG["e6b_rho_cs"])
    p.add_argument("--e6b_M_rnd", type=int,
                   default=DEFAULT_CONFIG["e6b_M_rnd"])
    p.add_argument("--e6b_N_v", type=int,
                   default=DEFAULT_CONFIG["e6b_N_v"])
    p.add_argument("--e6b_k_values", type=str, default=None,
                   help="Comma-separated list of budgets for E6b "
                        "(default: 0,25,100,400,800).")

    # k-values for headline experiments (E3/E7)
    p.add_argument("--k_values", type=str, default=None,
                   help="Comma-separated budgets for E3/E7 "
                        "(default: 0,10,25,50,100,200,400,600,800).")

    # Output
    p.add_argument("--output_dir", type=str,
                   default=DEFAULT_CONFIG["output_dir"],
                   help="Where to write PDF plots.")
    p.add_argument("--show", action="store_true",
                   help="Call plt.show() in addition to saving.")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip plotting entirely.")
    # [SALIENT COMMENT] Add argument to trigger saving raw data to pickle
    p.add_argument("--save_data", action="store_true",
                   help="Save the raw experiment data to a pickle file.")

    args = p.parse_args(argv)

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "d_c": args.d_c, "d_s": args.d_s, "N": args.N,
        "beta": args.beta, "mu_scale": args.mu_scale,
        "rho_cs": args.rho_cs, "cond_max": args.cond_max,
        "seed": args.seed, "num_seeds": args.num_seeds,
        "M_utl": args.M_utl,
        "e6b_mu_scale": args.e6b_mu_scale,
        "e6b_rho_cs": args.e6b_rho_cs,
        "e6b_M_rnd": args.e6b_M_rnd,
        "e6b_N_v": args.e6b_N_v,
        "output_dir": args.output_dir,
        "show": args.show, "no_plot": args.no_plot,
        "save_data": args.save_data,
        "mode": args.mode if args.mode is not None
        else DEFAULT_CONFIG["mode"],
    })
    if args.e6b_k_values is not None:
        cfg["e6b_k_values"] = [int(x) for x in args.e6b_k_values.split(",")]
    if args.k_values is not None:
        cfg["k_values"] = [int(x) for x in args.k_values.split(",")]
    else:
        cfg["k_values"] = [0, 10, 25, 50, 100, 200, 400, 600, 800]

    cfg["experiment"] = args.experiment
    return cfg


# ===========
# Dispatchers
# ===========

def _run_E1(cfg):
    res = experiment_E1(beta=cfg["beta"], n_test=400,
                        d_s=cfg["d_s"], seed=cfg["seed"])
    if not cfg["no_plot"]:
        plot_E1(res, cfg)
    return res


def _run_E2(cfg):
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    dgp_dec = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["mu_scale"],
                           rho_cs=0.0, seed=cfg["seed"],
                           cond_max=cfg["cond_max"])
    dgp_coup = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["mu_scale"],
                            rho_cs=cfg["rho_cs"], seed=cfg["seed"],
                            cond_max=cfg["cond_max"])
    res_dec = experiment_E2(dgp_dec, beta=cfg["beta"],
                            eps_values=eps_values, n_anchors=20,
                            seed=cfg["seed"] + 2, label="decoupled")
    res_coup = experiment_E2(dgp_coup, beta=cfg["beta"],
                             eps_values=eps_values, n_anchors=20,
                             seed=cfg["seed"] + 2, label="coupled")
    if not cfg["no_plot"]:
        plot_E2(res_dec, res_coup, cfg)
    return {"decoupled": res_dec, "coupled": res_coup}


def _run_E3(cfg):
    k_values = cfg["k_values"]
    dgp_cfg_dec = {**cfg, "rho_cs": 0.0}
    dgp_cfg_coup = {**cfg, "rho_cs": cfg["rho_cs"]}
    res_dec = experiment_E3(dgp_cfg_dec, beta=cfg["beta"],
                            k_values=k_values,
                            n_dgp_seeds=cfg["num_seeds"],
                            M_utl=cfg["M_utl"],
                            base_seed=cfg["seed"] + 3, label="decoupled")
    res_coup = experiment_E3(dgp_cfg_coup, beta=cfg["beta"],
                             k_values=k_values,
                             n_dgp_seeds=cfg["num_seeds"],
                             M_utl=cfg["M_utl"],
                             base_seed=cfg["seed"] + 3, label="coupled")
    if not cfg["no_plot"]:
        plot_E3_energies(res_dec, cfg, label="decoupled")
        plot_E3_relgap(res_dec, cfg, label="decoupled")
        plot_E3_energies(res_coup, cfg, label="coupled")
        plot_E3_relgap(res_coup, cfg, label="coupled")
    return {"decoupled": res_dec, "coupled": res_coup}


def _run_E4(cfg):
    dgp_coup = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["mu_scale"],
                            rho_cs=cfg["rho_cs"], seed=cfg["seed"],
                            cond_max=cfg["cond_max"])
    anchors = sample_anchors(dgp_coup, cfg["N"], seed=cfg["seed"] + 1)
    res = experiment_E4(dgp_coup, anchors, beta=cfg["beta"],
                        k_values=[0, 10, 50, 100, 250, 500, 800],
                        seed=cfg["seed"] + 4, label="coupled")
    if not cfg["no_plot"]:
        plot_E4_shrinkage(res, cfg)
        plot_E4_schur(res, cfg)
    return res


def _run_E5(cfg):
    dgp_dec = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["mu_scale"],
                           rho_cs=0.0, seed=cfg["seed"],
                           cond_max=cfg["cond_max"])
    dgp_coup = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["mu_scale"],
                            rho_cs=cfg["rho_cs"], seed=cfg["seed"],
                            cond_max=cfg["cond_max"])
    A_dec = sample_anchors(dgp_dec, cfg["N"], seed=cfg["seed"] + 1)
    A_coup = sample_anchors(dgp_coup, cfg["N"], seed=cfg["seed"] + 1)
    res_dec = experiment_E5(dgp_dec, A_dec[:300], beta=cfg["beta"],
                            eps=1e-3, seed=cfg["seed"] + 5,
                            label="decoupled")
    res_coup = experiment_E5(dgp_coup, A_coup[:300], beta=cfg["beta"],
                             eps=1e-3, seed=cfg["seed"] + 5,
                             label="coupled")
    if not cfg["no_plot"]:
        plot_E5_decoupled(res_dec, cfg, label="decoupled")
        plot_E5_coupled(res_coup, cfg, label="coupled")
    return {"decoupled": res_dec, "coupled": res_coup}


def _run_E5b(cfg):
    mu_sweep = [0.05, 0.10, 0.20, 0.40]
    dgp_cfg_dec = {**cfg, "rho_cs": 0.0}
    dgp_cfg_coup = {**cfg, "rho_cs": cfg["rho_cs"]}
    res_dec = experiment_E5b(dgp_cfg_dec, beta=cfg["beta"], eps=1e-3,
                             n_anchors_test=50, mu_scales=mu_sweep,
                             base_seed=cfg["seed"] + 50,
                             label="decoupled")
    res_coup = experiment_E5b(dgp_cfg_coup, beta=cfg["beta"], eps=1e-3,
                              n_anchors_test=50, mu_scales=None,
                              base_seed=cfg["seed"] + 50,
                              label="coupled")
    if not cfg["no_plot"]:
        plot_E5b_scatter(res_dec, cfg, label="decoupled")
        plot_E5b_sweep(res_dec, cfg, label="decoupled")
        plot_E5b_scatter(res_coup, cfg, label="coupled")
    return {"decoupled": res_dec, "coupled": res_coup}


def _run_E6(cfg):
    dgp_dec = generate_dgp(cfg["d_c"], cfg["d_s"], cfg["mu_scale"],
                           rho_cs=0.0, seed=cfg["seed"],
                           cond_max=cfg["cond_max"])
    A_dec = sample_anchors(dgp_dec, cfg["N"], seed=cfg["seed"] + 1)
    rho_values = np.linspace(0.0, 1.0, 11)
    res = experiment_E6(dgp_dec, A_dec, beta=cfg["beta"],
                        rho_values=rho_values, k_for_energy=200,
                        seed=cfg["seed"] + 6, label="decoupled")
    if not cfg["no_plot"]:
        plot_E6_spearman(res, cfg)
        plot_E6_energy(res, cfg)
    return res


def _run_E7(cfg):
    k_values = cfg["k_values"]
    dgp_cfg_dec = {**cfg, "rho_cs": 0.0}
    dgp_cfg_coup = {**cfg, "rho_cs": cfg["rho_cs"]}
    res_dec = experiment_E7(dgp_cfg_dec, beta=cfg["beta"],
                            k_values=k_values,
                            n_dgp_seeds=cfg["num_seeds"],
                            base_seed=cfg["seed"] + 7, label="decoupled")
    res_coup = experiment_E7(dgp_cfg_coup, beta=cfg["beta"],
                             k_values=k_values,
                             n_dgp_seeds=cfg["num_seeds"],
                             base_seed=cfg["seed"] + 7, label="coupled")
    if not cfg["no_plot"]:
        plot_E7_energies(res_dec, cfg, label="decoupled")
        plot_E7_overlap(res_dec, cfg, label="decoupled")
        plot_E7_energies(res_coup, cfg, label="coupled")
        plot_E7_overlap(res_coup, cfg, label="coupled")
    return {"decoupled": res_dec, "coupled": res_coup}


def _run_E8(cfg):
    dgp_cfg_e8 = {"d_c": cfg["d_c"], "d_s": cfg["d_s"],
                  "mu_scale": 0.5, "rho_cs": 0.4,
                  "cond_max": cfg["cond_max"], "N": cfg["N"]}
    k_e8 = [0, 25, 100, 400, 800]
    res = experiment_E8(dgp_cfg_e8, beta=cfg["beta"], k_values=k_e8,
                        n_dgp_seeds=min(cfg["num_seeds"], 5),
                        M_utl=min(cfg["M_utl"], 10),
                        base_seed=cfg["seed"] + 80, label="coupled")
    if not cfg["no_plot"]:
        plot_E8(res, cfg, label="coupled")
    return res


def _run_E6b(cfg):
    res = experiment_E6b(cfg)
    if not cfg["no_plot"]:
        plot_E6b_acc(res, cfg)
        plot_E6b_energy(res, cfg)
        plot_E6b_norm(res, cfg)
        plot_E6b_cross(res, cfg)
        plot_E6b_cos(res, cfg)
    return res


DISPATCH = {
    "1": _run_E1, "2": _run_E2, "3": _run_E3, "4": _run_E4,
    "5": _run_E5, "5b": _run_E5b, "6": _run_E6, "7": _run_E7,
    "8": _run_E8, "e6b": _run_E6b,
}


def main(argv=None):
    cfg = parse_args(argv)
    print(f"Config: experiment={cfg['experiment']}, mode={cfg['mode']}, "
          f"output_dir={cfg['output_dir']}")
    print(f"  d_c={cfg['d_c']}, d_s={cfg['d_s']}, N={cfg['N']}, "
          f"beta={cfg['beta']}, num_seeds={cfg['num_seeds']}")
    print(f"  mu_scale={cfg['mu_scale']}, rho_cs={cfg['rho_cs']}, "
          f"seed={cfg['seed']}\n")

    exp = cfg["experiment"]
    if exp == "all":
        results = {}
        for ek in ["1", "2", "3", "4", "5", "5b", "6", "7", "8", "e6b"]:
            print(f"\n\n###  RUNNING {ek}  ###")
            results[ek] = DISPATCH[ek](cfg)
        res_val = results
        res_to_save = results
    else:
        res_val = DISPATCH[exp](cfg)
        res_to_save = {exp: res_val}

    # [SALIENT COMMENT] Save the raw experiment data (nested dicts/arrays) as a pickle file if requested
    if cfg.get("save_data"):
        import pickle
        os.makedirs(cfg["output_dir"], exist_ok=True)
        mode_str = f"_{cfg['mode']}" if cfg.get('mode') else ""
        fname = f"data_exp_{exp}{mode_str}.pkl"
        path = os.path.join(cfg["output_dir"], fname)
        with open(path, "wb") as f:
            pickle.dump(res_to_save, f)
        print(f"    [data saved] {path}")

    return res_val


if __name__ == "__main__":
    main()
