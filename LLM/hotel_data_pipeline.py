# -*- coding: utf-8 -*-
"""
hotel_data_pipeline.py
======================

Strict-anchored hotel data generation for STL experiments.

Unlike the original generator (which produces strict and ties as independent
samples), this pipeline:

  1. Generates N strict pairs first.
  2. For each strict pair, takes the WINNING hotel and constructs tie pairs by
     cloning the winner and perturbing it.
  3. Each tie carries a stable `anchor_id` linking back to its strict example.

This anchoring is what makes selective tie learning operate correctly: the
scorer needs to associate per-anchor tie packets with their parent strict
examples, which the original two-stage independent sampling does not provide.

Two orthogonal axes control tie construction:

  --tie-construction
      exact-tie   : clone winner via deepcopy. dphi_c = 0 exactly.
      near-tie    : clone + modify_hotel_slightly (perturb 1-2 causal features).
                    |dphi_c| small but non-zero. Defends against the reviewer
                    objection that exact ties do not occur in real data.

  --spurious-strategy
      decorrelated_spurious   : one clone gets level=1.0 spurious, the other
                                level=0.0. Maximum spurious contrast.
      random_uniform          : both clones get independent uniform spurious
                                levels.
      suppressed              : both clones get fully randomized spurious.
      standard_monotonic      : both clones get spurious at level matching
                                their utility. Near-zero dphi_s "failure" mode.
      standard_monotonic_noisy: standard_monotonic + small Gaussian noise so
                                dphi_s is small but not exactly zero.

Cross-product gives 2 x 5 = 10 cells. Pass subsets to generate fewer cells.

  --label-mode
      bt     : BT teacher returns its actual preference (default; matches
               the rest of the hotel pipeline).
      random : symmetric 50/50 assignment, independent of utilities. Realizes
               the theoretical tie measure exactly.
      as-is  : hotel_a -> chosen, hotel_b -> rejected. The downstream tie
               packet construction in the scorer handles symmetrization.

Output schema:

  Strict file: standard DPO format with metadata.
    {
      "anchor_id": "strict_0042",
      "prompt":    str,
      "chosen":    str,
      "rejected":  str,
      "metadata":  {...}                 # utilities, true_label, spurious tracking
    }

  Tie files: one per (construction, spurious_strategy) cell, named
  ties_<construction>_<spurious_strategy>.jsonl.
    {
      "anchor_id":          "strict_0042",
      "prompt":             str,
      "chosen":             str,
      "rejected":           str,
      "construction":       "exact-tie" | "near-tie",
      "spurious_strategy":  "decorrelated_spurious" | ...,
      "label_mode":         "bt" | "random" | "as-is",
      "metadata":           {...}        # utilities, tracking
    }

USAGE:
    python hotel_data_pipeline.py \\
        --output-dir hotel_data/ \\
        --n 10000 \\
        --tie-construction exact-tie near-tie \\
        --spurious-strategy decorrelated_spurious random_uniform \\
                            suppressed standard_monotonic \\
                            standard_monotonic_noisy \\
        --correlation-strength 0.99 \\
        --label-mode bt \\
        --seed 7
"""

import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# We import primitives from the original generator file rather than duplicating
# its ~1500 lines. The original file is treated as a library here.
# By default we expect it next to this file or on PYTHONPATH; the path can be
# overridden via --generator-path on the CLI.

ORIGINAL_GENERATOR_DEFAULT = "icml_2026_llm_data_generation_original.py"


def _import_original_generator(path: str):
    """Import the original generator module from a path, return the module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hotel_orig", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hotel_orig"] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# strict generation
# ============================================================
def generate_strict_with_tracking(
    n: int,
    correlation_strength: float,
    correlation_mode: str,
    force_balance: bool,
    orig,
) -> List[Dict]:
    """
    Generate N strict pairs and KEEP the underlying Hotel objects alongside
    the rendered DPO record. The Hotel objects are needed downstream to build
    ties (the original generator does not retain them after rendering).
    """
    generator = orig.HotelDatasetGenerator(
        spurious_correlation_strength=correlation_strength
    )
    context = {
        "requirements": (
            "Trip for conference, want to be close to convention center "
            "and have low price while being at least 3 stars."
        )
    }
    context_prompt = (
        "You are helping someone choose the right hotel for their stay. "
        "Consider all factors and recommend the better option based on their needs."
    )

    target_each = n // 2
    a_better_count = 0
    b_better_count = 0
    records = []
    attempts = 0
    max_attempts = n * 10

    while len(records) < n and attempts < max_attempts:
        attempts += 1
        hotel_a, hotel_b, label, pair_tracking = generator.generate_hotel_pair(
            context, correlation_mode=correlation_mode
        )
        if force_balance:
            need_a = (label == orig.PreferenceLabel.A_BETTER and a_better_count < target_each)
            need_b = (label == orig.PreferenceLabel.B_BETTER and b_better_count < target_each)
            if not (need_a or need_b):
                continue

        util_a = generator.calculate_true_utility(hotel_a, context)
        util_b = generator.calculate_true_utility(hotel_b, context)

        example = orig.format_dpo_example(
            context_prompt=context_prompt,
            option_a_text=hotel_a.to_description(),
            option_b_text=hotel_b.to_description(),
            util_a=util_a, util_b=util_b, label=label,
        )
        if example is None:
            continue

        # winner is the hotel chosen by the BT teacher in the strict pair
        winner = hotel_a if label == orig.PreferenceLabel.A_BETTER else hotel_b
        winner_util = util_a if label == orig.PreferenceLabel.A_BETTER else util_b

        anchor_id = f"strict_{len(records):06d}"
        rec = {
            "anchor_id": anchor_id,
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "metadata": {
                **example["metadata"],
                "correlation_mode": correlation_mode,
                "correlation_strength": correlation_strength,
                "spurious_tracking": pair_tracking,
                # winner Hotel snapshot for tie construction; serialize via vars()
                "winner_hotel": _hotel_to_dict(winner),
                "winner_util": winner_util,
            },
        }
        records.append(rec)
        if force_balance:
            if label == orig.PreferenceLabel.A_BETTER:
                a_better_count += 1
            else:
                b_better_count += 1

    if len(records) < n:
        print(f"[warn] only generated {len(records)}/{n} (max attempts hit)")
    return records, context_prompt, generator, context


def _hotel_to_dict(hotel) -> Dict:
    """Serialize a Hotel dataclass instance to a dict for JSON storage."""
    return {f: getattr(hotel, f) for f in hotel.__dataclass_fields__}


def _hotel_from_dict(d: Dict, orig) -> "Hotel":
    """Reconstruct a Hotel dataclass instance from a dict."""
    return orig.Hotel(**d)


# ============================================================
# tie construction
# ============================================================
def construct_tie_pair(
    winner_hotel,
    construction: str,
    spurious_strategy: str,
    correlation_mode: str,
    orig,
    spurious_strength: float = 1.0,
):
    """
    Build a tie pair by cloning the winner and applying construction +
    spurious_strategy.

    Returns (hotel_a, hotel_b, util_a, util_b, tracking).

    `spurious_strength` is only consulted when `spurious_strategy ==
    'decorrelated_mixture'`. It interpolates between random_uniform
    (strength=0) and decorrelated_spurious (strength=1) by deciding,
    per tie, whether to apply the decorrelated rule (with probability
    spurious_strength) or the random_uniform rule (with probability
    1 - spurious_strength). Across the pool this realizes a mixture
    distribution; reducing to either endpoint recovers the corresponding
    single strategy exactly.
    """
    # Step 1: causal axis (construction).
    hotel_a = copy.deepcopy(winner_hotel)
    hotel_b = copy.deepcopy(winner_hotel)
    if construction == "near-tie":
        # modify_hotel_slightly takes a Hotel and returns a new one with 1-2
        # causal features perturbed within small bounds.
        hotel_b = orig.HotelDatasetGenerator(
            spurious_correlation_strength=0.0  # irrelevant for this call
        ).modify_hotel_slightly(hotel_b)
    elif construction == "exact-tie":
        pass  # hotel_b stays a true clone of hotel_a
    else:
        raise ValueError(f"unknown construction: {construction}")

    # Recompute utilities AFTER the causal-axis perturbation. For exact-tie
    # they will be identical; for near-tie they will be near-equal.
    context = {"requirements": "tie-construction-utility-context"}
    gen = orig.HotelDatasetGenerator(spurious_correlation_strength=1.0)
    util_a = gen.calculate_true_utility(hotel_a, context)
    util_b = gen.calculate_true_utility(hotel_b, context)

    # Step 2: spurious axis (strategy). Mutates hotel_a and hotel_b in place.
    if spurious_strategy == "decorrelated_mixture":
        # Per-tie mixture: with probability spurious_strength, apply the
        # decorrelated rule (extreme opposite levels); with probability
        # 1 - strength, apply random_uniform (independent uniform levels).
        if random.random() < spurious_strength:
            effective_strategy = "decorrelated_spurious"
        else:
            effective_strategy = "random_uniform"
    else:
        effective_strategy = spurious_strategy

    _, _, sp_tracking = orig.decorrelate_spurious_features_in_pair(
        hotel_a, hotel_b,
        correlation_mode=correlation_mode,
        strategy=effective_strategy,
        util_a=util_a, util_b=util_b,
    )
    # Tag the tracking with the requested strategy and the realized one,
    # plus the strength that controlled the mixture decision (if relevant).
    if spurious_strategy == "decorrelated_mixture":
        sp_tracking["mixture_strength"] = spurious_strength
        sp_tracking["realized_strategy"] = effective_strategy
        sp_tracking["requested_strategy"] = "decorrelated_mixture"
    return hotel_a, hotel_b, util_a, util_b, sp_tracking


def assign_tie_label(
    util_a: float, util_b: float, label_mode: str, beta: float = 1.0
) -> "PreferenceLabel":
    """
    Decide which of (hotel_a, hotel_b) becomes 'chosen' in the tie.

      bt     : BT teacher; A wins with prob sigmoid(beta * (util_a - util_b)).
               For exact-tie this is exactly 0.5; for near-tie it is near 0.5.
      random : symmetric 50/50 regardless of utilities.
      as-is  : hotel_a always becomes chosen.
    """
    if label_mode == "bt":
        # 1 / (1 + exp(-beta * (u_a - u_b))) without importing numpy
        import math
        p_a = 1.0 / (1.0 + math.exp(-beta * (util_a - util_b)))
        return "A" if random.random() < p_a else "B"
    elif label_mode == "random":
        return "A" if random.random() < 0.5 else "B"
    elif label_mode == "as-is":
        return "A"
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")


def render_tie_record(
    anchor_id: str,
    hotel_a, hotel_b,
    util_a: float, util_b: float,
    label_letter: str,            # "A" or "B"
    construction: str,
    spurious_strategy: str,
    label_mode: str,
    sp_tracking: Dict,
    context_prompt: str,
    orig,
) -> Dict:
    """
    Render the tie pair in the candidate-pool schema consumed by
    compute_scores.py: prompt + (response_a, response_b) + metadata.

    The two responses are always "Option ONE is the better choice." and
    "Option TWO is the better choice." (the hotel descriptions live inside
    the prompt). The label-mode decision is preserved in metadata as
    `tie_label_letter` ("A" or "B") and `tie_chosen` (which response string
    the label resolved to after the prompt's internal A/B -> ONE/TWO
    randomization).

    The downstream selector (select_topk.py) emits its own random
    chosen/rejected assignment per record, realizing the symmetric tie
    measure regardless of label_mode. This means label_mode primarily
    affects which utilities are stored in metadata, not the data the
    trainer ultimately sees.
    """
    label = (orig.PreferenceLabel.A_BETTER if label_letter == "A"
             else orig.PreferenceLabel.B_BETTER)
    example = orig.format_dpo_example(
        context_prompt=context_prompt,
        option_a_text=hotel_a.to_description(),
        option_b_text=hotel_b.to_description(),
        util_a=util_a, util_b=util_b, label=label,
    )

    response_one = "Option ONE is the better choice."
    response_two = "Option TWO is the better choice."
    rec = {
        "anchor_id": anchor_id,
        "prompt": example["prompt"],
        "response_a": response_one,
        "response_b": response_two,
        "construction": construction,
        "spurious_strategy": spurious_strategy,
        "label_mode": label_mode,
        "metadata": {
            **example["metadata"],
            "spurious_tracking": sp_tracking,
            "util_a": util_a,
            "util_b": util_b,
            "tie_label_letter": label_letter,         # which hotel was 'chosen'
            "tie_chosen": example["chosen"],          # which response string won
            "tie_rejected": example["rejected"],
        },
    }
    return rec


# ============================================================
# pipeline driver
# ============================================================
def run_train_mode(args, orig):
    """Generate strict + ties (the STL training data path)."""
    random.seed(args.seed)

    print(f"[strict] generating {args.n} pairs with correlation_mode="
          f"'{args.correlation_mode}', strength={args.correlation_strength}")
    strict_records, context_prompt, _, _ = generate_strict_with_tracking(
        n=args.n,
        correlation_strength=args.correlation_strength,
        correlation_mode=args.correlation_mode,
        force_balance=args.force_balance,
        orig=orig,
    )
    print(f"[strict] generated {len(strict_records)} records")

    os.makedirs(args.output_dir, exist_ok=True)
    strict_path = os.path.join(args.output_dir, "strict.jsonl")
    with open(strict_path, "w") as f:
        for r in strict_records:
            f.write(json.dumps(r) + "\n")
    print(f"[strict] wrote {strict_path}")

    # Build ties: cross-product of construction x spurious_strategy.
    # Reseed before tie generation so tie randomness is reproducible from
    # args.seed alone given a fixed strict file.
    random.seed(args.seed + 1)

    n_cells = len(args.tie_construction) * len(args.spurious_strategy)
    print(f"[ties] building {n_cells} (construction, spurious_strategy) cells")

    for construction in args.tie_construction:
        for spurious in args.spurious_strategy:
            tie_records = []
            for r in strict_records:
                winner = _hotel_from_dict(r["metadata"]["winner_hotel"], orig)
                hotel_a, hotel_b, util_a, util_b, sp_tracking = construct_tie_pair(
                    winner, construction, spurious,
                    correlation_mode=args.correlation_mode, orig=orig,
                    spurious_strength=args.spurious_strength,
                )
                label_letter = assign_tie_label(
                    util_a, util_b, label_mode=args.label_mode, beta=args.bt_beta
                )
                tie_rec = render_tie_record(
                    anchor_id=r["anchor_id"],
                    hotel_a=hotel_a, hotel_b=hotel_b,
                    util_a=util_a, util_b=util_b,
                    label_letter=label_letter,
                    construction=construction,
                    spurious_strategy=spurious,
                    label_mode=args.label_mode,
                    sp_tracking=sp_tracking,
                    context_prompt=context_prompt,
                    orig=orig,
                )
                tie_records.append(tie_rec)

            tie_path = os.path.join(
                args.output_dir, f"ties_{construction}_{spurious}.jsonl"
            )
            with open(tie_path, "w") as f:
                for rec in tie_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"[ties] {construction}/{spurious}: wrote {len(tie_records)} -> {tie_path}")


def run_test_mode(args, orig):
    """
    Generate test sets, mirroring the original generator's MAIN section:
      - test_p:    in-distribution (correlation_mode=normal)
      - test_qsup: spurious correlation suppressed
      - test_qadv: adversarial spurious correlation

    Uses --seed to vary across runs so different test seeds produce different
    test sets (the same way you'd want a held-out in-distribution test set
    independent of the training data).
    """
    os.makedirs(args.output_dir, exist_ok=True)
    requested = set(args.test_distributions)
    cells = []
    if "p" in requested:
        cells.append(("test_p", "normal"))
    if "qsup" in requested:
        cells.append(("test_qsup", "suppressed"))
    if "qadv" in requested:
        cells.append(("test_qadv", "adversarial"))

    print(f"[test] generating {len(cells)} test distribution(s) at n={args.n_test}")

    # Use seed for the first cell, advance per cell so each is independent.
    for i, (name, corr_mode) in enumerate(cells):
        random.seed(args.seed + 100 + i)
        print(f"[test] {name} (correlation_mode={corr_mode})")
        records, _, _, _ = generate_strict_with_tracking(
            n=args.n_test,
            correlation_strength=args.correlation_strength,
            correlation_mode=corr_mode,
            force_balance=args.force_balance,
            orig=orig,
        )
        path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[test] {name}: wrote {len(records)} -> {path}")


def run_pipeline(args):
    orig = _import_original_generator(args.generator_path)
    if args.mode == "train":
        run_train_mode(args, orig)
    elif args.mode == "test":
        run_test_mode(args, orig)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(
        description="Strict-anchored hotel data pipeline for STL.")
    p.add_argument("--mode", default="train", choices=["train", "test"],
                   help="train: generate strict + ties for STL training. "
                        "test: generate test distributions (P, Q_sup, Q_adv) "
                        "with no ties, mirroring the original generator's "
                        "MAIN section.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for output files. train mode writes "
                        "strict.jsonl + ties_*.jsonl; test mode writes "
                        "test_p.jsonl, test_qsup.jsonl, test_qadv.jsonl.")
    p.add_argument("--n", type=int, default=10000,
                   help="Number of strict pairs (train mode).")
    p.add_argument("--n-test", type=int, default=5000,
                   help="Number of pairs per test distribution (test mode).")
    p.add_argument("--test-distributions", nargs="+",
                   default=["p", "qsup", "qadv"],
                   choices=["p", "qsup", "qadv"],
                   help="Which test distributions to generate (test mode). "
                        "p=in-distribution, qsup=suppressed, qadv=adversarial.")
    p.add_argument("--correlation-mode", default="normal",
                   choices=["normal", "adversarial", "suppressed"],
                   help="Strict-pair generation mode (train mode only; "
                        "test mode iterates over normal/suppressed/adversarial "
                        "internally per --test-distributions).")
    p.add_argument("--correlation-strength", type=float, default=0.99,
                   help="Spurious-correlation strength for strict pairs.")
    p.add_argument("--force-balance", action="store_true", default=True,
                   help="Force balanced A-better / B-better.")
    p.add_argument("--tie-construction", nargs="+",
                   default=["exact-tie", "near-tie"],
                   choices=["exact-tie", "near-tie"],
                   help="Causal-axis construction modes (train mode only).")
    p.add_argument("--spurious-strategy", nargs="+",
                   default=["decorrelated_spurious", "random_uniform",
                            "suppressed", "standard_monotonic",
                            "standard_monotonic_noisy"],
                   choices=["decorrelated_spurious", "random_uniform",
                            "decorrelated_mixture",
                            "suppressed", "standard_monotonic",
                            "standard_monotonic_noisy"],
                   help="Spurious-axis strategies (train mode only). "
                        "decorrelated_mixture interpolates between "
                        "random_uniform (--spurious-strength=0) and "
                        "decorrelated_spurious (--spurious-strength=1) "
                        "via a per-tie Bernoulli mix.")
    p.add_argument("--spurious-strength", type=float, default=0.5,
                   help="Mixture weight for decorrelated_mixture strategy. "
                        "0.0 = fully random_uniform, 1.0 = fully decorrelated_spurious. "
                        "Default 0.5 (balanced mix). "
                        "Per tie, with probability spurious_strength the tie is "
                        "constructed via the decorrelated rule; otherwise via "
                        "random_uniform. Across the pool the realized distribution "
                        "is a convex combination of the two endpoint regimes. "
                        "Has no effect for other strategies.")
    p.add_argument("--label-mode", default="bt",
                   choices=["bt", "random", "as-is"],
                   help="Tie label assignment mode (train mode only).")
    p.add_argument("--bt-beta", type=float, default=1.0,
                   help="BT teacher inverse temperature for label-mode=bt.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--generator-path", default=ORIGINAL_GENERATOR_DEFAULT,
                   help="Path to icml_2026_llm_data_generation_original.py.")
    return p


def main():
    args = build_argparser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
