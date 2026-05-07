# -*- coding: utf-8 -*-
"""
train_dpo.py
============

Command-line DPO trainer.

USAGE
-----
Strict-only baseline (local JSONL):

    python train_dpo.py \\
        --train-dataset hotel_data/strict_only.jsonl \\
        --experiment-name strict_only_p \\
        --output-base-dir ./dpo_experiments/ \\
        --train-samples 5000 \\
        --num-epochs 1

STL-HA augmented:

    python train_dpo.py \\
        --train-dataset hotel_data/augmented_stl_ha_k1000.jsonl \\
        --experiment-name stl_ha_k1000 \\
        --output-base-dir ./dpo_experiments/ \\
        --train-samples 5000 \\
        --num-epochs 1

UTL baseline:

    python train_dpo.py \\
        --train-dataset hotel_data/augmented_utl_k1000.jsonl \\
        --experiment-name utl_k1000 \\
        --output-base-dir ./dpo_experiments/

HuggingFace dataset (auto-detect by file extension):

    python train_dpo.py \\
        --train-dataset anonymous-author/hotel-aug-informative-a0p50-seed7 \\
        --experiment-name informative_a0p50 \\
        --output-base-dir ./dpo_experiments/

The script auto-detects local vs Hub paths by file extension. Anything ending
in .jsonl or .json is loaded as a local file; anything else is treated as a
HuggingFace dataset name.

OUTPUT
------
<output-base-dir>/<experiment-name>_<version>_<timestamp>/
    config.json              # full training config
    training_history.json    # final loss, runtime, samples/sec
    model/
        adapter_model.safetensors  # LoRA weights
        adapter_config.json
        model_info.json
        <tokenizer files>
"""

import argparse
import datetime
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig


def load_dataset_auto(path_or_name: str, split: str):
    """
    Auto-detect: local JSONL/JSON file vs HuggingFace dataset name.
    Local files always return the 'train' split.
    """
    if path_or_name.endswith(".jsonl") or path_or_name.endswith(".json"):
        return load_dataset("json", data_files=path_or_name, split="train")
    return load_dataset(path_or_name, split=split)


def verify_balance(dataset, label: str):
    """Sanity-check chose-ONE vs chose-TWO balance."""
    one_count = sum(1 for ex in dataset if "Option ONE" in ex["chosen"])
    two_count = len(dataset) - one_count
    pct_one = one_count / len(dataset) * 100
    print(f"  {label}: {one_count} ONE ({pct_one:.1f}%), {two_count} TWO "
          f"({100 - pct_one:.1f}%)")
    if abs(one_count - two_count) > len(dataset) * 0.1:
        print(f"    WARNING: imbalance > 10% in {label}")


def build_argparser():
    p = argparse.ArgumentParser(description="DPO training (CLI version).")

    # Experiment identification
    p.add_argument("--experiment-name", required=True,
                   help="Name for this run; used as the output subdirectory prefix.")
    p.add_argument("--version", default="v1")
    p.add_argument("--description", default="",
                   help="Free-form description stored in config.json.")
    p.add_argument("--notes", default="",
                   help="Free-form notes stored in config.json.")
    p.add_argument("--output-base-dir", required=True,
                   help="Parent directory for the experiment output. The script "
                        "creates <base>/<name>_<version>_<timestamp>/ under it.")

    # Dataset
    p.add_argument("--train-dataset", required=True,
                   help="HF dataset name OR local .jsonl/.json path.")
    p.add_argument("--train-dataset-split", default="train",
                   help="Split name for HF datasets (ignored for local files).")
    p.add_argument("--train-samples", type=int, default=None,
                   help="Truncate training data to this many samples after "
                        "shuffling. Default: use all.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of training data held out for eval during training.")

    # Model config
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Training hyperparameters
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4,
                   help="per_device_(train|eval)_batch_size.")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO KL penalty coefficient.")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=768)

    # Logging / saving
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-type", default="L4",
                   help="Free-form metadata for config.json.")
    p.add_argument("--report-to", default="none",
                   choices=["none", "wandb", "tensorboard"],
                   help="HuggingFace trainer report_to value.")
    return p


def main():
    args = build_argparser().parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{args.experiment_name}_{args.version}_{timestamp}"
    experiment_dir = os.path.join(args.output_base_dir, experiment_id)
    model_save_path = os.path.join(experiment_dir, "model")
    os.makedirs(model_save_path, exist_ok=True)

    config = {
        "experiment_name":          args.experiment_name,
        "experiment_id":            experiment_id,
        "version":                  args.version,
        "description":              args.description,
        "notes":                    args.notes,
        "train_dataset":            args.train_dataset,
        "train_dataset_split":      args.train_dataset_split,
        "train_samples":            args.train_samples,
        "val_fraction":             args.val_fraction,
        "base_model":               args.base_model,
        "lora_r":                   args.lora_r,
        "lora_alpha":               args.lora_alpha,
        "lora_dropout":             args.lora_dropout,
        "learning_rate":            args.learning_rate,
        "num_epochs":               args.num_epochs,
        "batch_size":               args.batch_size,
        "grad_accum":               args.grad_accum,
        "beta":                     args.beta,
        "max_length":               args.max_length,
        "max_prompt_length":        args.max_prompt_length,
        "seed":                     args.seed,
        "gpu_type":                 args.gpu_type,
        "timestamp":                timestamp,
    }
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print(f"EXPERIMENT: {experiment_id}")
    print("=" * 80)
    print(f"Description:   {args.description}")
    print(f"Training on:   {args.train_dataset}")
    print(f"Samples:       {args.train_samples}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta:          {args.beta}")
    print(f"Output dir:    {experiment_dir}")
    print("=" * 80)

    # --- 1. Load dataset ---
    print("\n[load] dataset")
    dataset = load_dataset_auto(args.train_dataset, args.train_dataset_split)
    print(f"  total size: {len(dataset)}")

    print("[shuffle] seed=42")
    dataset = dataset.shuffle(seed=42)

    # Spot-check shuffle quality.
    first_100 = dataset.select(range(min(100, len(dataset))))
    print("  shuffle spot-check (first 100):")
    verify_balance(first_100, "first 100")

    if args.train_samples:
        n = min(args.train_samples, len(dataset))
        print(f"[subset] taking {n} samples")
        dataset = dataset.select(range(n))
        verify_balance(dataset, "subset")

    print(f"[split] train/val with val_fraction={args.val_fraction}")
    dataset = dataset.train_test_split(test_size=args.val_fraction, seed=42)
    print(f"  train: {len(dataset['train'])}, val: {len(dataset['test'])}")
    verify_balance(dataset["train"], "training set")

    # --- 2. Model and tokenizer ---
    print("\n[model] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(f"[model] loading base model {args.base_model} in 4-bit")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    print("[model] adding LoRA adapters")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[model] loading reference model")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # --- 3. Training arguments ---
    print("\n[trainer] configuring DPO")
    training_args = DPOConfig(
        output_dir=os.path.join(experiment_dir, "_temp"),

        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",

        beta=args.beta,
        max_length=args.max_length,

        logging_steps=args.logging_steps,
        eval_strategy="no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        bf16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        seed=args.seed,
        report_to=args.report_to,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # --- 4. Train ---
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    train_output = dpo_trainer.train()
    print("\nTraining complete.")

    # --- 5. Save ---
    training_history = {
        "train_loss": getattr(train_output, "training_loss", None),
        "training_time": str(train_output.metrics.get("train_runtime", "N/A")),
        "samples_per_second": train_output.metrics.get("train_samples_per_second", "N/A"),
    }
    with open(os.path.join(experiment_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n[save] model -> {model_save_path}")
    dpo_trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    model_info = {
        "experiment_id": experiment_id,
        "base_model": args.base_model,
        "trained_on_dataset": args.train_dataset,
        "training_samples": len(dataset["train"]),
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "training_hyperparams": {
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "epochs": args.num_epochs,
        },
        "timestamp": timestamp,
    }
    with open(os.path.join(model_save_path, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    # Per-experiment log
    log_path = os.path.join(args.output_base_dir, "experiment_log.jsonl")
    log_entry = {
        "experiment_id": experiment_id,
        "name": args.experiment_name,
        "version": args.version,
        "timestamp": timestamp,
        "trained_on": args.train_dataset,
        "train_samples": len(dataset["train"]),
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "notes": args.notes,
        "model_path": model_save_path,
        "status": "trained",
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print("\n" + "=" * 80)
    print(f"DONE. Model at: {model_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
