# -*- coding: utf-8 -*-
"""
compute_scores.py

Score every candidate in a tie pool with one or more of the three CMS variants:

  PM (pure margin):       |r_pol(prompt, response_a) - r_pol(prompt, response_b)|
                          where r_pol is the implicit DPO reward at the
                          strict-only LoRA checkpoint.

  GA (gradient alignment): |v_s^T g_j| where g_j is the LoRA-param gradient
                          of the antisymmetric tie-packet loss for candidate j,
                          and v_s is the LoRA-param gradient of the aggregate
                          counterfactual energy at the strict-only checkpoint.

  HA (Hessian-aware):     |v_s^T F^{-1} g_j| where F is a LoRA-restricted
                          empirical Fisher precomputed from a subsample of
                          the strict training data, and F^{-1} g_j is solved
                          by conjugate gradients per candidate.

PM is essentially free (two forwards). GA pays one forward+backward per
candidate. HA reuses GA's per-candidate gradient as the CG right-hand side,
so it costs only the additional CG iterations on top of GA. With LoRA-r=16
on Llama-3.2-1B, d_LoRA ~ 11M and a Fisher cache of N_sub=200 samples in
fp16 fits in ~4 GB of GPU memory, giving sub-millisecond Fisher VPs.

Outputs an augmented JSONL with the original tie-pool fields plus
`pm_score`, `ga_score`, `ha_score` (latter two are null if not requested).
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# I/O
# ============================================================
def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ============================================================
# model loading
# ============================================================
def load_models(base_model: str, lora_path: str, device: str = "cuda"):
    """
    Load policy (LoRA-adapted, training mode for grad) and reference (frozen
    base, no grad). Returns (policy, ref, tokenizer).
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    base_pol = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map=device,
        dtype=torch.bfloat16,
    )
    policy = PeftModel.from_pretrained(base_pol, lora_path, is_trainable=True)
    # eval mode disables dropout so per-candidate gradients are deterministic.
    # The LoRA adapter parameters remain trainable; only the base remains frozen.
    policy.eval()

    ref = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map=device,
        dtype=torch.bfloat16,
    )
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    return policy, ref, tok


def lora_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


# ============================================================
# log-probs and DPO margin
# ============================================================
def sequence_logprob(model, tokenizer, prompt: str, response: str,
                     max_length: int = 1024,
                     device: str = "cuda",
                     create_graph: bool = False) -> torch.Tensor:
    """
    Compute sum log p(response | prompt) under `model`. Returns a scalar
    tensor (with autograd graph if create_graph=True).
    """
    full = prompt + response
    enc_full = tokenizer(full, return_tensors="pt", truncation=True,
                         max_length=max_length).to(device)
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
    n_prompt_toks = enc_prompt.input_ids.shape[1]

    out = model(input_ids=enc_full.input_ids,
                attention_mask=enc_full.attention_mask)
    logits = out.logits  # (1, T, V)
    # next-token prediction: logits at position t predict token at position t+1
    shift_logits = logits[:, :-1, :]
    shift_labels = enc_full.input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    # gather log p of each label token
    token_logp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # only sum over response-positions (after the prompt)
    response_logp = token_logp[:, n_prompt_toks - 1:].sum()
    return response_logp


def implicit_reward_gap(policy, ref, tokenizer, prompt: str,
                        response_a: str, response_b: str, beta: float,
                        device: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (m, m_no_grad) where:
      m         = beta * [(log pi_pol(a) - log pi_pol(b)) - (log pi_ref(a) - log pi_ref(b))]
                  with autograd graph attached to the pol terms
      m_no_grad = same numeric value, detached
    """
    # policy log-probs (with graph)
    lp_pol_a = sequence_logprob(policy, tokenizer, prompt, response_a,
                                max_length=max_length, device=device,
                                create_graph=True)
    lp_pol_b = sequence_logprob(policy, tokenizer, prompt, response_b,
                                max_length=max_length, device=device,
                                create_graph=True)
    # ref log-probs (no grad)
    with torch.no_grad():
        lp_ref_a = sequence_logprob(ref, tokenizer, prompt, response_a,
                                    max_length=max_length, device=device)
        lp_ref_b = sequence_logprob(ref, tokenizer, prompt, response_b,
                                    max_length=max_length, device=device)

    m = beta * ((lp_pol_a - lp_pol_b) - (lp_ref_a - lp_ref_b))
    return m, m.detach()


# ============================================================
# v_s: aggregate counterfactual energy gradient
# ============================================================
def compute_v_s(policy, ref, tokenizer, val_anchors: List[Dict],
                beta: float, device: str, max_length: int,
                normalize: bool = True) -> torch.Tensor:
    """
    v_s = grad_theta E_cf where E_cf = (1/2) * mean_n (m_n)^2
    aggregated over the validation anchor pool.

    For mixture mode you can either:
      (a) pre-mix at pool-build time (recommended): each val anchor uses
          a randomly sampled T_i, and this function processes them uniformly.
          The expectation over T then comes from the empirical mixture.
      (b) compute K terms per anchor and average them. More expensive,
          lower variance. Not implemented here; (a) is the default.
    """
    params = lora_params(policy)
    accum = [torch.zeros_like(p) for p in params]
    n = 0
    for a in val_anchors:
        m, _ = implicit_reward_gap(
            policy, ref, tokenizer,
            a["prompt"], a["response_a"], a["response_b"],
            beta=beta, device=device, max_length=max_length,
        )
        # gradient of (1/2) m^2 is m * grad(m), accumulate
        loss = 0.5 * (m * m)
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for acc, g in zip(accum, grads):
            acc.add_(g.detach())
        n += 1
    v_s = flatten([a / max(n, 1) for a in accum])
    if normalize:
        v_s = v_s / (v_s.norm() + 1e-12)
    return v_s.detach()


# ============================================================
# per-candidate tie-packet gradient
# ============================================================
def packet_gradient(policy, ref, tokenizer, candidate: Dict,
                    beta: float, device: str, max_length: int) -> torch.Tensor:
    """
    Antisymmetric tie packet loss for one candidate:
      L_Q = 0.5 * [softplus(-m) + softplus(m)]
    where m is the implicit reward gap from a to b.
    Returns flat LoRA-grad of L_Q, detached.
    """
    params = lora_params(policy)
    m, _ = implicit_reward_gap(
        policy, ref, tokenizer,
        candidate["prompt"], candidate["response_a"], candidate["response_b"],
        beta=beta, device=device, max_length=max_length,
    )
    L_Q = 0.5 * (F.softplus(-m) + F.softplus(m))
    grads = torch.autograd.grad(L_Q, params, retain_graph=False)
    return flatten(grads)


# ============================================================
# LoRA-restricted Fisher oracle
# ============================================================
class LoRAFisher:
    """
    Empirical Fisher F = (1/N_sub) G^T G + lambda * I, where each row of G is
    the per-sample gradient of the strict DPO loss at the strict-only theta.

    Memory footprint: N_sub * d_LoRA floats. With d_LoRA ~ 11M and N_sub=200
    in fp16 this is ~4 GB.
    """
    def __init__(self, policy, ref, tokenizer, strict_subsample: List[Dict],
                 beta: float, device: str, max_length: int,
                 damping: float = 1e-3, dtype: torch.dtype = torch.float16):
        self.damping = damping
        self.dtype = dtype
        self.device = device

        params = lora_params(policy)
        d_lora = sum(p.numel() for p in params)
        N = len(strict_subsample)
        print(f"  [Fisher] precomputing {N} per-sample gradients, "
              f"d_LoRA={d_lora}, dtype={dtype}, "
              f"~{N * d_lora * (2 if dtype == torch.float16 else 4) / 1e9:.2f} GB")

        # allocate G on GPU; if too big, stream to CPU instead
        G = torch.zeros((N, d_lora), dtype=dtype, device=device)
        for i, ex in enumerate(strict_subsample):
            # standard DPO loss: -log sigmoid(beta * margin) with margin chosen-rejected
            m, _ = implicit_reward_gap(
                policy, ref, tokenizer,
                ex["prompt"], ex["chosen"], ex["rejected"],
                beta=beta, device=device, max_length=max_length,
            )
            loss = -F.logsigmoid(m)
            grads = torch.autograd.grad(loss, params, retain_graph=False)
            G[i] = flatten(grads).to(dtype)
            if (i + 1) % 25 == 0:
                print(f"    Fisher row {i+1}/{N}")
        self.G = G
        self.N = N

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """F v = (1/N) G^T (G v) + damping * v."""
        v_d = v.to(self.dtype)
        Gv = self.G @ v_d
        FtGv = self.G.T @ Gv
        out = (FtGv / self.N).to(v.dtype)
        if self.damping > 0.0:
            out = out + self.damping * v
        return out


# ============================================================
# Conjugate Gradients
# ============================================================
def conjugate_gradients(hvp, b: torch.Tensor, max_iter: int = 30,
                        tol: float = 1e-3) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
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


# ============================================================
# main scoring routine
# ============================================================
def score_candidates(
    candidates: List[Dict],
    policy, ref, tokenizer,
    val_anchors: Optional[List[Dict]],
    strict_subsample: Optional[List[Dict]],
    beta: float, device: str, max_length: int,
    do_pm: bool, do_ga: bool, do_ha: bool,
    fisher_damping: float, fisher_dtype: torch.dtype,
    cg_max_iter: int, cg_tol: float,
    ha_top_from_ga: Optional[int],
) -> List[Dict]:
    # 1) PM (always cheap; compute even if not requested for sorting purposes)
    # Stores SIGNED PM = beta * ((logp_pol_a - logp_pol_b) - (logp_ref_a - logp_ref_b)).
    # The selector applies the abs/relu/signed mode at sort time. Note: PM's
    # sign is orientation-dependent (depends on which response is labeled A vs B
    # in the input file), so it has no energy-decrease interpretation. The
    # selector special-cases PM to always sort by |pm_score| regardless of mode.
    print("[score] PM ...")
    pm_scores = []
    for i, cand in enumerate(candidates):
        with torch.no_grad():
            lp_pol_a = sequence_logprob(policy, tokenizer, cand["prompt"],
                                        cand["response_a"], max_length, device)
            lp_pol_b = sequence_logprob(policy, tokenizer, cand["prompt"],
                                        cand["response_b"], max_length, device)
            lp_ref_a = sequence_logprob(ref, tokenizer, cand["prompt"],
                                        cand["response_a"], max_length, device)
            lp_ref_b = sequence_logprob(ref, tokenizer, cand["prompt"],
                                        cand["response_b"], max_length, device)
            m = beta * ((lp_pol_a - lp_pol_b) - (lp_ref_a - lp_ref_b))
        pm_scores.append(m.item())
        if (i + 1) % 250 == 0:
            print(f"  PM {i+1}/{len(candidates)}")

    if do_ga or do_ha:
        if val_anchors is None:
            raise ValueError("GA/HA require val_anchors for v_s computation")
        print("[score] computing v_s on validation anchors ...")
        v_s = compute_v_s(policy, ref, tokenizer, val_anchors,
                          beta=beta, device=device, max_length=max_length,
                          normalize=True)
        print(f"  v_s: shape={tuple(v_s.shape)}, norm={v_s.norm().item():.3f}")
    else:
        v_s = None

    fisher = None
    if do_ha:
        if strict_subsample is None:
            raise ValueError("HA requires strict_subsample for Fisher")
        print("[score] precomputing LoRA Fisher ...")
        fisher = LoRAFisher(policy, ref, tokenizer, strict_subsample,
                            beta=beta, device=device, max_length=max_length,
                            damping=fisher_damping, dtype=fisher_dtype)

    # decide which candidates get HA: optionally, top-X by GA only.
    # GA and HA are stored as SIGNED inner products. Under the influence formula
    # Delta E_cf^(2) ~ -epsilon * <v_s, F^-1 g_j>, a positive signed score
    # corresponds to first-order energy decrease. The selector applies the
    # abs/relu/signed mode at sort time. The HA-shortlist heuristic (when
    # ha_top_from_ga is set) uses |ga_score| as the shortlisting criterion
    # because it is a compute-saving pre-filter; the final ranking is still
    # determined by the selector's mode flag.
    ga_scores = [None] * len(candidates)
    ha_scores = [None] * len(candidates)

    if do_ga or do_ha:
        # if HA-restricted-to-top-from-GA is set, we need GA for everyone first,
        # then HA only for top-X. Otherwise compute g_j once and use it for both.
        if ha_top_from_ga is not None and do_ha:
            print(f"[score] GA pass (full pool); HA only on top-{ha_top_from_ga} by |GA|")
            # pass 1: GA only (signed)
            for i, cand in enumerate(candidates):
                g_j = packet_gradient(policy, ref, tokenizer, cand,
                                      beta=beta, device=device,
                                      max_length=max_length)
                ga_scores[i] = (v_s * g_j).sum().item()
                if (i + 1) % 100 == 0:
                    print(f"  GA {i+1}/{len(candidates)}")
            # pick top-X by |GA| (magnitude pre-filter), recompute g_j and run CG
            top_idx = sorted(range(len(candidates)),
                             key=lambda i: abs(ga_scores[i]), reverse=True
                             )[:ha_top_from_ga]
            print(f"[score] HA pass on {len(top_idx)} top-|GA| candidates")
            for k, i in enumerate(top_idx):
                g_j = packet_gradient(policy, ref, tokenizer, candidates[i],
                                      beta=beta, device=device,
                                      max_length=max_length)
                u_j = conjugate_gradients(fisher, g_j, max_iter=cg_max_iter,
                                          tol=cg_tol)
                ha_scores[i] = (v_s * u_j).sum().item()
                if (k + 1) % 25 == 0:
                    print(f"  HA {k+1}/{len(top_idx)}")
        else:
            # compute g_j once per candidate; do GA and (if requested) HA from same g_j
            print("[score] GA + HA pass (g_j shared)")
            for i, cand in enumerate(candidates):
                g_j = packet_gradient(policy, ref, tokenizer, cand,
                                      beta=beta, device=device,
                                      max_length=max_length)
                if do_ga:
                    ga_scores[i] = (v_s * g_j).sum().item()
                if do_ha:
                    u_j = conjugate_gradients(fisher, g_j,
                                              max_iter=cg_max_iter, tol=cg_tol)
                    ha_scores[i] = (v_s * u_j).sum().item()
                if (i + 1) % 100 == 0:
                    print(f"  GA/HA {i+1}/{len(candidates)}")

    # write back
    out = []
    for i, cand in enumerate(candidates):
        rec = dict(cand)
        rec["pm_score"] = pm_scores[i] if do_pm else None
        rec["ga_score"] = ga_scores[i]
        rec["ha_score"] = ha_scores[i]
        out.append(rec)
    return out


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tie-pool", required=True,
                    help="Input tie-pool JSONL from make_tie_pool.py.")
    ap.add_argument("--strict-source", required=True,
                    help="Strict-pair source: HF dataset name or JSONL. "
                         "Used for Fisher subsample (HA only).")
    ap.add_argument("--strict-split", default="train")
    ap.add_argument("--base-model", required=True,
                    help="HF base model id, e.g. meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--lora-path", required=True,
                    help="Path to strict-only LoRA checkpoint.")
    ap.add_argument("--output", required=True,
                    help="Output scored JSONL.")

    ap.add_argument("--score-pm", action="store_true")
    ap.add_argument("--score-ga", action="store_true")
    ap.add_argument("--score-ha", action="store_true")

    ap.add_argument("--n-val", type=int, default=200,
                    help="Validation anchors used for v_s (GA/HA).")
    ap.add_argument("--n-fisher", type=int, default=200,
                    help="Strict subsample size for LoRA Fisher (HA).")
    ap.add_argument("--ha-top-from-ga", type=int, default=None,
                    help="Optional: only run HA on the top-X by GA. "
                         "Saves CG cost when scoring large pools.")

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--fisher-damping", type=float, default=1e-3)
    ap.add_argument("--fisher-dtype", default="fp16",
                    choices=["fp16", "fp32"])
    ap.add_argument("--cg-max-iter", type=int, default=30)
    ap.add_argument("--cg-tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.score_pm or args.score_ga or args.score_ha):
        raise ValueError("specify at least one of --score-pm/--score-ga/--score-ha")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print("[scorer] loading models ...")
    policy, ref, tok = load_models(args.base_model, args.lora_path, device=device)

    print("[scorer] loading tie pool ...")
    candidates_all = load_jsonl(args.tie_pool)
    print(f"  -> {len(candidates_all)} ties total")

    val_anchors = None
    strict_subsample = None
    if args.score_ga or args.score_ha:
        # split the tie pool once into disjoint slices: V (validation, for v_s)
        # and A (candidate pool, for scoring and selection). this prevents the
        # same anchor contributing to both v_s and its own score.
        rng = torch.Generator().manual_seed(args.seed + 1)
        perm = torch.randperm(len(candidates_all), generator=rng).tolist()
        val_indices = perm[:args.n_val]
        a_indices = perm[args.n_val:]
        val_anchors = [candidates_all[i] for i in val_indices]
        candidates = [candidates_all[i] for i in a_indices]
        # tag each candidate with its index in the original tie pool so downstream
        # selection knows which ties to use even after the V/A split.
        for k, cand in enumerate(candidates):
            cand["original_pool_idx"] = a_indices[k]
        print(f"  -> split: |V|={len(val_anchors)}, |A|={len(candidates)}")
    else:
        # PM-only path: no v_s needed, no split required.
        candidates = candidates_all
        for k, cand in enumerate(candidates):
            cand["original_pool_idx"] = k

    if args.score_ha:
        # Load strict pairs for the Fisher cache. Supports two formats:
        #   - local JSONL (preferred for the hotel pipeline)
        #   - HuggingFace dataset name (legacy verbosity path)
        if os.path.isfile(args.strict_source) and args.strict_source.endswith(".jsonl"):
            strict_rows = []
            with open(args.strict_source) as f:
                for line in f:
                    strict_rows.append(json.loads(line))
        else:
            from datasets import load_dataset
            ds = load_dataset(args.strict_source, split=args.strict_split)
            strict_rows = [dict(r) for r in ds]
        rng = torch.Generator().manual_seed(args.seed + 2)
        idx = torch.randperm(len(strict_rows), generator=rng)[:args.n_fisher].tolist()
        strict_subsample = [strict_rows[i] for i in idx]

    fisher_dtype = torch.float16 if args.fisher_dtype == "fp16" else torch.float32

    scored = score_candidates(
        candidates, policy, ref, tok,
        val_anchors=val_anchors,
        strict_subsample=strict_subsample,
        beta=args.beta, device=device, max_length=args.max_length,
        do_pm=args.score_pm, do_ga=args.score_ga, do_ha=args.score_ha,
        fisher_damping=args.fisher_damping, fisher_dtype=fisher_dtype,
        cg_max_iter=args.cg_max_iter, cg_tol=args.cg_tol,
        ha_top_from_ga=args.ha_top_from_ga,
    )

    write_jsonl(scored, args.output)
    print(f"[scorer] wrote {args.output}")


if __name__ == "__main__":
    main()
