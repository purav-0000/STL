# -*- coding: utf-8 -*-
"""
concat_for_training.py
======================

Concatenate a strict file and one or more selected-tie files into a single
DPO-format JSONL ready for the training script.

Validates that all input files share the trainer-required schema
(prompt, chosen, rejected). Reports per-file row counts and a final summary.

USAGE
-----
Augmented STL training set (e.g., STL-HA at k=1000):
    python concat_for_training.py \\
        --strict   hotel_data/strict.jsonl \\
        --ties     hotel_data/ties_stl_ha_k1000.jsonl \\
        --output   hotel_data/augmented_stl_ha_k1000.jsonl

UTL baseline (random-selected ties at matched budget):
    python concat_for_training.py \\
        --strict   hotel_data/strict.jsonl \\
        --ties     hotel_data/ties_utl_k1000.jsonl \\
        --output   hotel_data/augmented_utl_k1000.jsonl

Strict-only baseline (no ties; just normalizes/copies the strict file):
    python concat_for_training.py \\
        --strict   hotel_data/strict.jsonl \\
        --output   hotel_data/strict_only.jsonl

DEFAULT BEHAVIOR
----------------
- Output schema is exactly {prompt, chosen, rejected}. Any extra fields in
  the input files (anchor_id, metadata, construction, spurious_strategy,
  ...) are dropped by default. Pass --keep-extra to preserve them.
- Order is strict-first, then ties in the order passed on the CLI. Pass
  --shuffle to interleave.
- A schema check runs before writing: every input file must have
  prompt, chosen, rejected as non-null fields. Files with response_a/
  response_b are flagged with a clear error pointing at select_topk.py
  as the missing intermediate step.
"""

import argparse
import json
import os
import random
from typing import Dict, List


REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def check_schema(rows: List[Dict], path: str) -> None:
    if not rows:
        raise ValueError(f"{path} is empty")

    first = rows[0]
    missing = [f for f in REQUIRED_FIELDS if f not in first]
    if missing:
        if "response_a" in first or "response_b" in first:
            raise ValueError(
                f"{path} is in tie-pool format (response_a / response_b). "
                "Run select_topk.py first to convert to DPO format "
                "(prompt / chosen / rejected) before passing to this script."
            )
        raise ValueError(
            f"{path} missing required field(s): {missing}. "
            f"Found fields: {sorted(first.keys())}"
        )

    for i, r in enumerate(rows[:50]):
        for f in REQUIRED_FIELDS:
            if r.get(f) is None or r.get(f) == "":
                raise ValueError(f"{path} row {i}: field '{f}' is null/empty")


def project_to_dpo_schema(row: Dict, keep_extra: bool) -> Dict:
    if keep_extra:
        return row
    return {f: row[f] for f in REQUIRED_FIELDS}


def main():
    ap = argparse.ArgumentParser(
        description="Concatenate strict + selected ties for DPO training."
    )
    ap.add_argument("--strict", required=True,
                    help="Path to strict.jsonl (DPO format).")
    ap.add_argument("--ties", nargs="*", default=[],
                    help="Zero or more tie JSONL files in DPO format from "
                         "select_topk.py. Pass none for a strict-only output.")
    ap.add_argument("--output", required=True,
                    help="Output JSONL path.")
    ap.add_argument("--keep-extra", action="store_true",
                    help="Keep extra fields beyond prompt/chosen/rejected.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the concatenated rows before writing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] strict: {args.strict}")
    strict_rows = load_jsonl(args.strict)
    check_schema(strict_rows, args.strict)
    print(f"  -> {len(strict_rows)} rows")

    tie_rows_all = []
    for tie_path in args.ties:
        print(f"[load] ties:   {tie_path}")
        rows = load_jsonl(tie_path)
        check_schema(rows, tie_path)
        print(f"  -> {len(rows)} rows")
        tie_rows_all.append(rows)

    combined = list(strict_rows)
    for rows in tie_rows_all:
        combined.extend(rows)
    n_strict = len(strict_rows)
    n_ties = sum(len(r) for r in tie_rows_all)
    print(f"[concat] {n_strict} strict + {n_ties} ties = {len(combined)} total")

    combined = [project_to_dpo_schema(r, args.keep_extra) for r in combined]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(combined)
        print(f"[shuffle] applied with seed={args.seed}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(combined)} rows -> {args.output}")

    one_count = sum(1 for r in combined if "Option ONE" in (r.get("chosen") or ""))
    two_count = len(combined) - one_count
    print(f"[balance] chose ONE: {one_count} ({one_count/len(combined)*100:.1f}%); "
          f"chose TWO: {two_count} ({two_count/len(combined)*100:.1f}%)")
    if abs(one_count - two_count) > len(combined) * 0.1:
        print(f"  WARNING: imbalance > 10%. Verify upstream label assignment.")


if __name__ == "__main__":
    main()
