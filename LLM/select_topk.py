# -*- coding: utf-8 -*-
"""
select_topk.py

Select top-k candidates from a scored tie pool and emit a DPO-format JSONL
ready to concatenate with strict training data.

The scored pool is produced by compute_scores.py and contains SIGNED scores
in `pm_score`, `ga_score`, `ha_score`. This selector applies a chosen
selection mode at sort time:

  --selection-mode relu (default): rank by max(s, 0). Restricts selection
        to candidates whose first-order energy contribution is monotonically
        decreasing (s > 0). This is the theoretically justified default:
        the signed influence formula
            Delta E_cf^{(2)} ~ -epsilon * <v_s, F^{-1} g_j>
        guarantees energy decrease only when the score is positive.

  --selection-mode abs: rank by |s|. Magnitude regardless of sign.
        Empirically equivalent to 'relu' when score distributions are
        dominantly positive (typical at strong coupling), but provides no
        energy-decrease guarantee for negative-score candidates.

  --selection-mode signed: rank by s, largest positive first. Diagnostic.

PM (pure margin) is special-cased: PM's sign is orientation-dependent
(depends on which response is labeled A vs B in the input file), so it
has no energy-decrease interpretation. PM is ALWAYS sorted by |pm_score|
regardless of --selection-mode.

The antisymmetric tie packet is realized at the dataset level by writing
each selected candidate with a *random* assignment of (response_a,
response_b) to (chosen, rejected). Over the dataset, this implements the
symmetric tie measure Q_j = 0.5 delta_{a>b} + 0.5 delta_{b>a} per anchor.

Output schema is identical to your strict training data:
    {"prompt": str, "chosen": str, "rejected": str}
so it can be concatenated and consumed by your existing DPO trainer
without modification.
"""

import argparse
import json
import os
import random
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def selection_key(score: float, mode: str) -> float:
    """
    Map a signed score to a sortable key under the chosen selection mode.
    Higher key = higher rank.
    """
    if mode == "abs":
        return abs(score)
    elif mode == "signed":
        return score
    elif mode == "relu":
        return max(score, 0.0)
    else:
        raise ValueError(f"unknown selection mode: {mode!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-pool", required=True,
                    help="Output of compute_scores.py (with signed scores).")
    ap.add_argument("--score-key", required=True,
                    choices=["pm_score", "ga_score", "ha_score", "random"],
                    help="Which score to sort by. 'random' selects uniformly "
                         "(for the UTL baseline at matched budget).")
    ap.add_argument("--selection-mode", default="relu",
                    choices=["abs", "relu", "signed"],
                    help="How to rank by signed scores. 'relu' (default): "
                         "max(s, 0), conservative energy-decrease selection. "
                         "'abs': |s|, magnitude. 'signed': s, largest "
                         "positive first. PM is always sorted by |pm_score| "
                         "regardless of this flag, since PM's sign has no "
                         "energy-decrease interpretation.")
    ap.add_argument("--budget", type=int, required=True,
                    help="Number of ties to select (k).")
    ap.add_argument("--output", required=True,
                    help="Output JSONL in DPO format.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-fields", default="anchor_id,T_id,mode",
                    help="Comma-separated metadata fields to keep alongside "
                         "(prompt, chosen, rejected).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_jsonl(args.scored_pool)
    print(f"[select_topk] loaded {len(pool)} scored candidates")

    if args.score_key == "random":
        rng.shuffle(pool)
        selected = pool[:args.budget]
        print(f"[select_topk] UTL: uniform random selection")
    else:
        # filter out candidates whose chosen score is null
        scored = [r for r in pool if r.get(args.score_key) is not None]
        if len(scored) < args.budget:
            print(f"  WARNING: only {len(scored)} candidates have "
                  f"{args.score_key} scored; budget={args.budget}")

        # PM is special-cased: always sort by |pm_score| regardless of mode.
        effective_mode = "abs" if args.score_key == "pm_score" else args.selection_mode

        # Sort descending by selection key.
        scored.sort(key=lambda r: selection_key(r[args.score_key], effective_mode),
                    reverse=True)
        selected = scored[:args.budget]

        # Diagnostics: report fraction of selected with positive signed score.
        signed_vals = [r[args.score_key] for r in selected]
        n_pos = sum(1 for s in signed_vals if s > 0)
        pos_frac = n_pos / max(len(signed_vals), 1)
        print(f"[select_topk] {args.score_key} sorted via "
              f"selection_mode='{effective_mode}'")
        print(f"  positive_fraction_selected = {n_pos}/{len(selected)} "
              f"= {pos_frac:.3f}")

        # If relu mode and fewer positive candidates than budget, warn.
        if effective_mode == "relu":
            n_pos_pool = sum(1 for r in scored if r[args.score_key] > 0)
            if n_pos_pool < args.budget:
                print(f"  WARNING: only {n_pos_pool} candidates have "
                      f"positive {args.score_key}; budget={args.budget}. "
                      f"Selection includes {args.budget - n_pos_pool} "
                      f"non-positive candidates (no energy-decrease guarantee).")

    keep = [k.strip() for k in args.keep_fields.split(",") if k.strip()]
    out_rows = []
    for rec in selected:
        # random label assignment realizes the symmetric tie measure
        if rng.random() < 0.5:
            chosen, rejected = rec["response_a"], rec["response_b"]
        else:
            chosen, rejected = rec["response_b"], rec["response_a"]
        row = {
            "prompt":   rec["prompt"],
            "chosen":   chosen,
            "rejected": rejected,
        }
        for k in keep:
            if k in rec:
                row[k] = rec[k]
        out_rows.append(row)

    write_jsonl(out_rows, args.output)
    print(f"[select_topk] wrote {len(out_rows)} ties to {args.output}")


if __name__ == "__main__":
    main()
