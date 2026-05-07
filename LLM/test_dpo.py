# -*- coding: utf-8 -*-
"""
test_dpo.py
===========

Command-line evaluation for DPO-trained LoRA models.

Loads a base model + LoRA adapter, evaluates on a test dataset using log-prob
scoring of "Option ONE" vs "Option TWO", computes accuracy/precision/recall/F1,
and writes results JSON plus three PNG plots.

USAGE
-----
Adversarial test (the headline OOD metric):

    python test_dpo.py \\
        --model-path ./dpo_experiments/strict_only_p_v1_<ts>/model \\
        --test-dataset hotel_data_test/test_qadv.jsonl \\
        --test-name adversarial \\
        --output-dir ./dpo_eval/strict_only_qadv/

In-distribution test:

    python test_dpo.py \\
        --model-path ./dpo_experiments/stl_ha_k1000_v1_<ts>/model \\
        --test-dataset hotel_data_test/test_p.jsonl \\
        --test-name p \\
        --output-dir ./dpo_eval/stl_ha_p/

HuggingFace dataset (auto-detect by extension):

    python test_dpo.py \\
        --model-path ./dpo_experiments/.../model \\
        --test-dataset anonymous-author/hotel-strict-test-qadv-a0p50-seed7 \\
        --test-name adversarial

Headless (no inline plot display, just save PNGs):

    python test_dpo.py ... --no-show

OUTPUT
------
<output-dir>/
    results.json              # predictions, true_labels, margins, confidences
    metrics.json              # accuracy, precision, recall, F1, per-class
    confusion_matrix.png
    accuracy_breakdown.png
    per_class_metrics.png
"""

import argparse
import json
import math
import matplotlib
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Tuple


# ============================================================
# Below: helper functions
# ============================================================


# =====================================
# function: analyze unknown predictions
# =====================================
def analyze_unknown_predictions(results: Dict, n_examples: int = 10):
    """
    Show examples of unknown predictions.
    Note: With log-prob scoring, this should always be 0.
    """
    print("\n" + "="*80)
    print("ANALYZING UNKNOWN PREDICTIONS")
    print("="*80)

    unknown_indices = [
        i for i in range(len(results['predictions']))
        if results['predictions'][i] == "UNKNOWN"
    ]

    print(f"\nFound {len(unknown_indices)} unknown predictions ({len(unknown_indices)/len(results['predictions'])*100:.1f}%)")

    if not unknown_indices:
        print("✓ No unknown predictions!")
        return
    # ... (rest of function is fine but should not be triggered)

# ========================================
# function: get true label from metadata
# ========================================
def get_true_label_from_metadata(example: Dict) -> str:
    """
    Gets the true label directly from the 'metadata' field.
    'A' corresponds to "Option ONE"
    'B' corresponds to "Option TWO"
    """
    metadata = example.get('metadata')

    # Handle case where metadata is a JSON string
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            print("Warning: Could not parse metadata string.")
            return "UNKNOWN"

    if not isinstance(metadata, dict):
        print("Warning: 'metadata' field is missing or not a dict.")
        return "UNKNOWN"

    label = metadata.get('true_label')

    if label in {'A', 'B'}:
        return label

    print(f"Warning: Found unexpected label '{label}' in metadata.")
    return "UNKNOWN"


# =========================================
# function: score A/B candidates by logprob (RE-INTRODUCED)
# =========================================
@torch.no_grad()
def score_ab_candidates(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    # --- UPDATED CANDIDATE TEXTS ---
    candidate_texts: Tuple[str, str] = (
        "Option ONE is the better choice.",
        "Option TWO is the better choice."
    ),
    max_prompt_len: int = 2048, # Increased
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns two tensors [batch]: total log-likelihood of candidate ONE and TWO
    conditioned on each prompt. No free-form generation is used.
    """
    model.eval()

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # encode prompts
    enc = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_prompt_len
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    B = input_ids.size(0)

    # candidates (no specials)
    candA_ids = tokenizer.encode(candidate_texts[0], add_special_tokens=False)
    candB_ids = tokenizer.encode(candidate_texts[1], add_special_tokens=False)
    candA = torch.tensor(candA_ids, device=device, dtype=torch.long)
    candB = torch.tensor(candB_ids, device=device, dtype=torch.long)

    def concat_and_labels(cand: torch.Tensor):
        # concat [prompt || candidate]
        cand_rep = cand.unsqueeze(0).expand(B, -1)
        ids = torch.cat([input_ids, cand_rep], dim=1)
        att = torch.cat([attn, torch.ones((B, cand.numel()), dtype=attn.dtype, device=device)], dim=1)
        # labels: ignore prompt, supervise candidate
        labels = ids.clone()
        labels[:, :input_ids.size(1)] = -100
        return ids, att, labels

    idsA, attA, labA = concat_and_labels(candA)
    idsB, attB, labB = concat_and_labels(candB)

    # forward (we'll compute per-row token logprobs)
    # --- IMPORTANT: DO NOT PASS LABELS TO MODEL ---
    outA_logits = model(input_ids=idsA, attention_mask=attA).logits
    outB_logits = model(input_ids=idsB, attention_mask=attB).logits

    def per_row_sum_logprob(logits, labels):
        # shift for causal LM: predict token t from tokens < t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != -100)

        logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # gather logprob at true labels
        gathered = torch.gather(logprobs, -1, shift_labels.unsqueeze(-1).clamp_min(0)).squeeze(-1)
        gathered = torch.where(mask, gathered, torch.zeros_like(gathered))
        return gathered.sum(dim=1)  # [B]

    per_row_A = per_row_sum_logprob(outA_logits, labA)
    per_row_B = per_row_sum_logprob(outB_logits, labB)
    return per_row_A, per_row_B


# ===================================
# function: evaluate model on dataset (REPLACED)
# ===================================
@torch.no_grad()
def evaluate_model_on_dataset(
    model,
    tokenizer,
    dataset,
    batch_size: int = 8,
    max_prompt_length: int = 2048, # Increased
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Evaluate by scoring "Option ONE..." vs "Option TWO..."
    This matches the new DPO training format.
    """

    model.eval()
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions, true_labels, prompts, full_responses, confidences = [], [], [], [], []
    all_logpA, all_logpB, all_margins = [], [], []

    print(f"\nEvaluating model on {len(dataset)} examples (log-prob scoring of answers)…")
    print(f"Using device: {device}")

    for i in tqdm(range(0, len(dataset), batch_size)):
        # --- 1. Get Batch Data ---
        batch_data = dataset[i:i+batch_size]
        batch_prompts = batch_data['prompt']

        # Reconstruct full example dicts to pass to the label getter
        batch_example_dicts = []
        keys = batch_data.keys()
        num_in_batch = len(batch_prompts)
        for j in range(num_in_batch):
            example_dict = {key: batch_data[key][j] for key in keys}
            batch_example_dicts.append(example_dict)

        batch_true_labels = [get_true_label_from_metadata(ex) for ex in batch_example_dicts]

        # --- 2. Score both candidates ---
        logpA, logpB = score_ab_candidates(
            model, tokenizer, batch_prompts,
            device=device, max_prompt_len=max_prompt_length
            # Uses default candidate texts:
            # ("Option ONE is the better choice.", "Option TWO is the better choice.")
        )

        # --- 3. Get Predictions ---
        for pmt, la, lb, gt in zip(batch_prompts, logpA.tolist(), logpB.tolist(), batch_true_labels):
            if gt == "UNKNOWN":
                continue # Skip examples where we couldn't find a label

            pred = "A" if la > lb else "B" # 'A' = ONE, 'B' = TWO

            # confidence as softmax over {la, lb}
            m = max(la, lb)
            pa = math.exp(la - m) / (math.exp(la - m) + math.exp(lb - m))
            conf = pa if pred == "A" else (1.0 - pa)

            predictions.append(pred)
            true_labels.append(gt)
            prompts.append(pmt)
            full_responses.append("Option ONE is the better choice." if pred == "A" else "Option TWO is the better choice.")
            confidences.append(conf)
            all_logpA.append(la)
            all_logpB.append(lb)
            all_margins.append(la - lb)

    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'prompts': prompts,
        'full_responses': full_responses,
        'confidences': confidences,
        'logpA': all_logpA, # logp(ONE)
        'logpB': all_logpB, # logp(TWO)
        'margins': all_margins # logp(ONE) - logp(TWO)
    }

#@title Metrics

# ===========================
# function: calculate metrics
# ===========================
def calculate_metrics(predictions: List[str], true_labels: List[str]) -> Dict:
    """
    Calculate accuracy and other metrics
    """
    # Filter out UNKNOWN predictions
    valid_indices = [i for i, p in enumerate(predictions) if p in ['A', 'B']]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]

    if not valid_predictions:
        return {
            'accuracy': 0.0,
            'total_examples': len(predictions),
            'valid_predictions': 0,
            'unknown_predictions': len(predictions)
        }

    # Calculate accuracy
    correct = sum(p == t for p, t in zip(valid_predictions, valid_true_labels))
    accuracy = correct / len(valid_predictions)

    # Per-class accuracy
    a_correct = sum(1 for p, t in zip(valid_predictions, valid_true_labels) if p == t == 'A')
    b_correct = sum(1 for p, t in zip(valid_predictions, valid_true_labels) if p == t == 'B')
    a_total = sum(1 for t in valid_true_labels if t == 'A')
    b_total = sum(1 for t in valid_true_labels if t == 'B')

    return {
        'accuracy': accuracy,
        'total_examples': len(predictions),
        'valid_predictions': len(valid_predictions),
        'unknown_predictions': len(predictions) - len(valid_predictions),
        'correct_predictions': correct,
        'option_a_accuracy': a_correct / a_total if a_total > 0 else 0.0,
        'option_b_accuracy': b_correct / b_total if b_total > 0 else 0.0,
        'option_a_total': a_total,
        'option_b_total': b_total
    }

#@title Utils

# ===============================
# function: plot confusion matrix
# ===============================
def plot_confusion_matrix(predictions: List[str], true_labels: List[str], save_path: str = None):
    """
    Plot confusion matrix
    """
    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p in ['A', 'B']]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]

    # Create confusion matrix
    cm = confusion_matrix(valid_true_labels, valid_predictions, labels=['A', 'B'])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted ONE (A)', 'Predicted TWO (B)'],
        yticklabels=['True ONE (A)', 'True TWO (B)'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - DPO Model Predictions', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()

# =================================
# function: plot accuracy breakdown
# =================================
def plot_accuracy_breakdown(metrics: Dict, save_path: str = None):
    """
    Plot accuracy breakdown
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall accuracy
    ax1 = axes[0]
    categories = ['Overall\nAccuracy', 'Option ONE\nAccuracy', 'Option TWO\nAccuracy']
    accuracies = [
        metrics['accuracy'],
        metrics['option_a_accuracy'],
        metrics['option_b_accuracy']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Classification Accuracy Breakdown', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random Baseline')
    ax1.legend()

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontweight='bold'
        )

    # Prediction distribution
    ax2 = axes[1]
    labels = ['Valid\nPredictions', 'Unknown\nPredictions']
    sizes = [metrics['valid_predictions'], metrics['unknown_predictions']]
    colors = ['#2ecc71', '#e67e22']
    explode = (0.05, 0)

    ax2.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title('Prediction Quality Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy breakdown to {save_path}")

    plt.show()

# ================================
# function: plot per class metrics
# ================================
def plot_per_class_metrics(predictions: List[str], true_labels: List[str], save_path: str = None):
    """
    Plot precision, recall, F1 for each class
    """
    from sklearn.metrics import precision_recall_fscore_support

    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p in ['A', 'B']]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_true_labels,
        valid_predictions,
        labels=['A', 'B']
    )

    # Plot
    x = np.arange(2)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Option ONE (A)', 'Option TWO (B)'])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class metrics to {save_path}")

    plt.show()

# ===============================
# function: print detailed report
# ===============================
def print_detailed_report(results: Dict, metrics: Dict):
    """
    Print detailed classification report
    """
    from sklearn.metrics import classification_report

    print("\n" + "="*80)
    print("DETAILED EVALUATION REPORT")
    print("="*80)

    print(f"\nDataset Size: {metrics['total_examples']}")
    print(f"Valid Predictions: {metrics['valid_predictions']} ({metrics['valid_predictions']/metrics['total_examples']*100:.1f}%)")
    print(f"Unknown Predictions: {metrics['unknown_predictions']} ({metrics['unknown_predictions']/metrics['total_examples']*100:.1f}%)")

    print(f"\n{'OVERALL METRICS':^80}")
    print("-"*80)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Correct Predictions: {metrics['correct_predictions']} / {metrics['valid_predictions']}")

    print(f"\n{'PER-OPTION METRICS':^80}")
    print("-"*80)
    print(f"Option ONE (A) Accuracy: {metrics['option_a_accuracy']:.4f} ({metrics['option_a_accuracy']*100:.2f}%)")
    print(f"  - Correct: {int(metrics['option_a_accuracy'] * metrics['option_a_total'])} / {metrics['option_a_total']}")
    print(f"Option TWO (B) Accuracy: {metrics['option_b_accuracy']:.4f} ({metrics['option_b_accuracy']*100:.2f}%)")
    print(f"  - Correct: {int(metrics['option_b_accuracy'] * metrics['option_b_total'])} / {metrics['option_b_total']}")

    # Sklearn classification report
    valid_indices = [i for i, p in enumerate(results['predictions']) if p in ['A', 'B']]
    if valid_indices:
        valid_predictions = [results['predictions'][i] for i in valid_indices]
        valid_true_labels = [results['true_labels'][i] for i in valid_indices]

        print(f"\n{'SKLEARN CLASSIFICATION REPORT':^80}")
        print("-"*80)
        print(classification_report(
            valid_true_labels,
            valid_predictions,
            labels=['A', 'B'],
            target_names=['Option ONE (A)', 'Option TWO (B)'],
            digits=4
        ))

    print("="*80)

# ==================================
# function: show example predictions (UPDATED)
# ==================================
def show_example_predictions(results: Dict, n_correct: int = 3, n_incorrect: int = 3):
    """
    Show example correct and incorrect predictions
    """
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    # Get correct and incorrect examples
    correct_indices = [
        i for i in range(len(results['predictions']))
        if results['predictions'][i] == results['true_labels'][i] and results['predictions'][i] in ['A', 'B']
    ]
    incorrect_indices = [
        i for i in range(len(results['predictions']))
        if results['predictions'][i] != results['true_labels'][i] and results['predictions'][i] in ['A', 'B']
    ]

    # Show correct examples
    print(f"\n{'CORRECT PREDICTIONS':^80}")
    print("-"*80)
    for i, idx in enumerate(correct_indices[:n_correct], 1):
        print(f"\nExample {i}:")
        print(f"Prompt: {results['prompts'][idx][:500]}...") # Show more of the prompt
        print(f"True Label: {results['true_labels'][idx]} (ONE='A', TWO='B')")
        print(f"Prediction: {results['predictions'][idx]} ✓")
        print(f"Model chose: '{results['full_responses'][idx]}'")
        print(f"Scores (TotalLogL): ONE={results['logpA'][idx]:.4f}, TWO={results['logpB'][idx]:.4f}, Margin={results['margins'][idx]:.4f}")

    # Show incorrect examples
    if incorrect_indices:
        print(f"\n{'INCORRECT PREDICTIONS':^80}")
        print("-"*80)
        for i, idx in enumerate(incorrect_indices[:n_incorrect], 1):
            print(f"\nExample {i}:")
            print(f"Prompt: {results['prompts'][idx][:500]}...")
            print(f"True Label: {results['true_labels'][idx]} (ONE='A', TWO='B')")
            print(f"Prediction: {results['predictions'][idx]} ✗")
            print(f"Model chose: '{results['full_responses'][idx]}'")
            print(f"Scores (TotalLogL): ONE={results['logpA'][idx]:.4f}, TWO={results['logpB'][idx]:.4f}, Margin={results['margins'][idx]:.4f}")
    else:
        print("\n✓ No incorrect predictions found!")

    print("="*80)

# ===============================================
# function: plot preference strength distribution (UPDATED)
# ===============================================
def plot_preference_strength_distribution(
    preference_data: Dict,
    save_path: str = None
):
    """
    Plot distribution of preference strengths using margins = logp(ONE) - logp(TWO).
    """
    strengths = np.array(preference_data.get('margins',
                         preference_data.get('preference_strengths')))
    if 'is_correct' in preference_data:
        is_correct = np.array(preference_data['is_correct'], dtype=bool)
    else:
        preds = np.array(preference_data.get('predictions', []))
        trues = np.array(preference_data.get('true_labels', []))
        is_correct = (preds == trues) if len(preds) == len(strengths) else np.zeros_like(strengths, dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Overall distribution
    ax1 = axes[0, 0]
    ax1.hist(strengths, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (0)')
    ax1.axvline(x=float(np.mean(strengths)), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(strengths):.2f}')
    ax1.set_xlabel('Preference Strength (margin = log p(ONE) − log p(TWO))', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Preference Strengths (Margins)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Correct vs Incorrect
    ax2 = axes[0, 1]
    correct_strengths = strengths[is_correct]
    incorrect_strengths = strengths[~is_correct]

    ax2.hist(correct_strengths, bins=30, alpha=0.6, color='#2ecc71',
             label=f'Correct (n={len(correct_strengths)})', edgecolor='black')
    ax2.hist(incorrect_strengths, bins=30, alpha=0.6, color='#e74c3c',
             label=f'Incorrect (n={len(incorrect_strengths)})', edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Preference Strength (margin)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Preference Strength: Correct vs Incorrect', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_strengths = np.sort(strengths)
    cumulative = np.arange(1, len(sorted_strengths) + 1) / len(sorted_strengths)
    ax3.plot(sorted_strengths, cumulative, linewidth=2, color='#9b59b6')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Preference Strength (margin)', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Cumulative Distribution of Preference Strengths', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)

    # Add percentage preferring A/B
    pct_A = (strengths > 0).mean() * 100
    ax3.text(0.05, 0.95,
             f'{pct_A:.1f}% prefer ONE  |  {100-pct_A:.1f}% prefer TWO',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Box plot by correctness
    ax4 = axes[1, 1]
    box_data = [correct_strengths, incorrect_strengths]
    bp = ax4.boxplot(box_data, labels=['Correct', 'Incorrect'],
                     patch_artist=True, widths=0.6)

    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_ylabel('Preference Strength (margin)', fontsize=11)
    ax4.set_title('Preference Strength Distribution by Correctness', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved preference strength distribution to {save_path}")
    plt.show()

# =====================================
# function: plot DPO loss approximation (UPDATED)
# =====================================
def plot_dpo_loss_approximation(
    preference_data: Dict,
    beta: float = 0.1,
    save_path: str = None
):
    """
    Plot approximate DPO loss distribution using margins.
    DPO Loss ≈ -log(sigmoid(beta * margin))
    """
    strengths = np.array(preference_data.get('margins',
                             preference_data.get('preference_strengths')))
    if 'is_correct' in preference_data:
        is_correct = np.array(preference_data['is_correct'], dtype=bool)
    else:
        preds = np.array(preference_data.get('predictions', []))
        trues = np.array(preference_data.get('true_labels', []))
        is_correct = (preds == trues) if len(preds) == len(strengths) else np.zeros_like(strengths, dtype=bool)

    # Approximate DPO loss from margins
    dpo_losses = -np.log(1.0 / (1.0 + np.exp(-beta * strengths)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. DPO Loss distribution
    ax1 = axes[0]
    ax1.hist(dpo_losses, bins=50, alpha=0.7, color='#e67e22', edgecolor='black')
    ax1.axvline(x=float(np.mean(dpo_losses)), color='red', linestyle='--', linewidth=2,
                label=f'Mean Loss: {np.mean(dpo_losses):.3f}')
    ax1.set_xlabel('Approximate DPO Loss', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'DPO Loss Distribution (β={beta})', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Loss by correctness
    ax2 = axes[1]
    correct_losses = dpo_losses[is_correct]
    incorrect_losses = dpo_losses[~is_correct]

    ax2.hist(correct_losses, bins=30, alpha=0.6, color='#2ecc71',
             label=f'Correct (mean: {np.mean(correct_losses):.3f})', edgecolor='black')
    ax2.hist(incorrect_losses, bins=30, alpha=0.6, color='#e74c3c',
             label=f'Incorrect (mean: {np.mean(incorrect_losses):.3f})', edgecolor='black')
    ax2.set_xlabel('Approximate DPO Loss', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('DPO Loss: Correct vs Incorrect Predictions', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Relationship between margin and loss
    ax3 = axes[2]
    scatter = ax3.scatter(strengths, dpo_losses, c=is_correct, cmap='RdYlGn',
                          alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Preference Strength (margin = log p(ONE) − log p(TWO))', fontsize=11)
    ax3.set_ylabel('Approximate DPO Loss', fontsize=11)
    ax3.set_title('Preference Strength vs DPO Loss', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Correct Prediction', fontsize=10)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved DPO loss distribution to {save_path}")
    plt.show()

# =====================================
# function: print preference statistics (UPDATED)
# =====================================
def print_preference_statistics(preference_data: Dict):
    """
    Print detailed statistics about preference strengths (log-prob margins).
    """
    strengths = np.array(preference_data.get("margins", preference_data.get("preference_strengths")))
    preds = np.array(preference_data.get("predictions", []))
    trues = np.array(preference_data.get("true_labels", []))
    is_correct = (preds == trues)

    print("\n" + "="*80)
    print("PREFERENCE STRENGTH ANALYSIS (log p(ONE) - log p(TWO))")
    print("="*80)

    print(f"\n{'OVERALL STATISTICS':^80}")
    print("-"*80)
    print(f"Mean Preference Strength: {np.mean(strengths):.4f}")
    print(f"Median Preference Strength: {np.median(strengths):.4f}")
    print(f"Std Deviation: {np.std(strengths):.4f}")
    print(f"Min: {np.min(strengths):.4f}")
    print(f"Max: {np.max(strengths):.4f}")

    print(f"\n{'PREFERENCE DIRECTION':^80}")
    print("-"*80)
    pct_positive = (strengths > 0).mean() * 100
    print(f"Prefer ONE (A): {(strengths > 0).sum()} ({pct_positive:.1f}%)")
    print(f"Prefer TWO (B): {(strengths < 0).sum()} ({100 - pct_positive:.1f}%)")
    print(f"No Preference (|margin| < 0.01): {(np.abs(strengths) < 0.01).sum()}")

    print(f"\n{'STRENGTH BY CORRECTNESS':^80}")
    print("-"*80)
    correct_strengths = strengths[is_correct]
    incorrect_strengths = strengths[~is_correct]

    print(f"Correct Predictions (n={len(correct_strengths)}):")
    if len(correct_strengths) > 0:
        print(f"  Mean: {np.mean(correct_strengths):.4f}")
        print(f"  Median: {np.median(correct_strengths):.4f}")
        print(f"  Std: {np.std(correct_strengths):.4f}")
    else:
        print("  (No correct predictions)")

    if len(incorrect_strengths) > 0:
        print(f"\nIncorrect Predictions (n={len(incorrect_strengths)}):")
        print(f"  Mean: {np.mean(incorrect_strengths):.4f}")
        print(f"  Median: {np.median(incorrect_strengths):.4f}")
        print(f"  Std: {np.std(incorrect_strengths):.4f}")
    else:
        print("\n  (No incorrect predictions)")


    print(f"\n{'CONFIDENCE LEVELS':^80}")
    print("-"*80)
    abs_strengths = np.abs(strengths)
    # --- Confidence bands reset to reasonable values for total log-prob ---
    high_confidence = abs_strengths > 2.0
    medium_confidence = (abs_strengths > 0.5) & (abs_strengths <= 2.0)
    low_confidence = abs_strengths <= 0.5

    print(f"High Confidence (|margin| > 2.0): {high_confidence.sum()} ({high_confidence.mean()*100:.1f}%)")
    if high_confidence.sum() > 0:
        print(f"  Accuracy: {is_correct[high_confidence].mean()*100:.1f}%")

    print(f"Medium Confidence (0.5 < |margin| ≤ 2.0): {medium_confidence.sum()} ({medium_confidence.mean()*100:.1f}%)")
    if medium_confidence.sum() > 0:
        print(f"  Accuracy: {is_correct[medium_confidence].mean()*100:.1f}%")

    print(f"Low Confidence (|margin| ≤ 0.5): {low_confidence.sum()} ({low_confidence.mean()*100:.1f}%)")
    if low_confidence.sum() > 0:
        print(f"  Accuracy: {is_correct[low_confidence].mean()*100:.1f}%")

    print("="*80)

# =================================================
# function: plot empirical preference probabilities (UPDATED)
# =================================================
def plot_empirical_preference_probabilities(
    preference_data: Dict,
    save_path: str = None,
    beta: float = 1.0
):
    """
    Plot empirical distribution of preference probabilities using margins:
      P(ONE preferred) = sigmoid(beta * margin), where margin = log p(ONE) - log p(TWO)
    """

    strengths = np.array(preference_data.get('margins',
                         preference_data.get('preference_strengths')))
    if 'is_correct' in preference_data:
        is_correct = np.array(preference_data['is_correct'], dtype=bool)
    else:
        preds = np.array(preference_data.get('predictions', []))
        trues = np.array(preference_data.get('true_labels', []))
        is_correct = (preds == trues) if len(preds) == len(strengths) else np.zeros_like(strengths, dtype=bool)

    # Convert margins -> Bradley–Terry probabilities for "A preferred"
    bt_probs = 1.0 / (1.0 + np.exp(-beta * strengths))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: distribution split by correctness
    ax1 = axes[0]
    bins = np.linspace(0, 1, 50)
    correct_probs = bt_probs[is_correct]
    incorrect_probs = bt_probs[~is_correct]

    ax1.hist(correct_probs, bins=bins, alpha=0.7, color='#2ecc71',
             edgecolor='black', label=f'Correct (n={len(correct_probs)})')
    ax1.hist(incorrect_probs, bins=bins, alpha=0.7, color='#e74c3c',
             edgecolor='black', label=f'Incorrect (n={len(incorrect_probs)})')

    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Neutral (0.5)')
    ax1.set_xlabel('P_BT(ONE preferred) = sigmoid(β · (log p(ONE) − log p(TWO)))', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    title_beta = f" (β={beta})" if beta != 1.0 else ""
    ax1.set_title(f'Empirical Distribution of Preference Probabilities{title_beta}\n(Split by Correctness)',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: overall shape + summary stats
    ax2 = axes[1]
    ax2.hist(bt_probs, bins=bins, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Neutral (0.5)')
    ax2.axvline(x=float(np.mean(bt_probs)), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(bt_probs):.3f}')

    ax2.set_xlabel('P_BT(ONE preferred)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Overall Probability Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Text box with stats
    stats_text = (
        f'Statistics:\n'
        f'Mean: {np.mean(bt_probs):.3f}\n'
        f'Median: {np.median(bt_probs):.3f}\n'
        f'Std: {np.std(bt_probs):.3f}\n'
        f'% near 0.5 (0.4–0.6): {((bt_probs >= 0.4) & (bt_probs <= 0.6)).mean()*100:.1f}%\n'
        f'% confident (>0.8 or <0.2): {((bt_probs > 0.8) | (bt_probs < 0.2)).mean()*100:.1f}%'
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved empirical preference probabilities to {save_path}")
    plt.show()

# ==============================
# function: analyze tie behavior (UPDATED)
# ==============================
def analyze_tie_behavior(preference_data: Dict, beta: float = 1.0, tie_band: float = 0.2):
    """
    Analyze 'tie-like' behavior where the model's preference probability is ~0.5.
    Uses margins = log p(ONE) - log p(TWO).
    tie_band: width around 0.5 for BT prob, i.e., keep probs in [0.5 - tie_band, 0.5 + tie_band]
    """
    strengths = np.array(preference_data.get("margins",
                         preference_data.get("preference_strengths")))
    if strengths is None or strengths.size == 0:
        print("No margins/preference_strengths found.")
        return

    if "is_correct" in preference_data:
        is_correct = np.array(preference_data["is_correct"], dtype=bool)
    else:
        preds = np.array(preference_data.get("predictions", []))
        trues = np.array(preference_data.get("true_labels", []))
        is_correct = (preds == trues) if len(preds) == len(strengths) else np.zeros_like(strengths, dtype=bool)

    # Bradley–Terry probabilities from margins (option A is "success")
    bt_probs = 1.0 / (1.0 + np.exp(-beta * strengths))

    # Define "tie-like" region around 0.5
    lo, hi = 0.5 - tie_band, 0.5 + tie_band
    tie_mask = (bt_probs >= lo) & (bt_probs <= hi)
    non_tie_mask = ~tie_mask

    # Summary
    n = strengths.size
    n_tie = int(tie_mask.sum())
    n_non = int(non_tie_mask.sum())
    acc_tie = float(is_correct[tie_mask].mean()) if n_tie else None
    acc_non = float(is_correct[non_tie_mask].mean()) if n_non else None

    print("\n=== Tie Behavior Analysis ===")
    print(f"β used for probs: {beta}")
    print(f"Tie band: [{lo:.2f}, {hi:.2f}] around 0.0")
    print(f"Total: {n} | Tie-like: {n_tie} ({n_tie/n*100:.1f}%) | Non-tie: {n_non} ({n_non/n*100:.1f}%)")
    print(f"Accuracy in tie-like region: {acc_tie if acc_tie is not None else 'N/A'}")
    print(f"Accuracy outside tie region: {acc_non if acc_non is not None else 'N/A'}")

    # Optional small plot (hist of probs with tie band)
    bins = np.linspace(0, 1, 50)
    plt.figure(figsize=(7,4))
    plt.hist(bt_probs, bins=bins, alpha=0.7, color="#3498db", edgecolor="black")
    plt.axvline(0.5, color="red", linestyle="--", label="0.5")
    plt.axvspan(lo, hi, color="orange", alpha=0.2, label="tie band")
    plt.xlabel("P_BT(ONE preferred) = sigmoid(β · margin)")
    plt.ylabel("Count")
    plt.title("Empirical Preference Probabilities (tie-like region shaded)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# CLI entry point
# ============================================================
def load_dataset_auto(path_or_name: str, split: str):
    """Auto-detect: local JSONL/JSON file vs HuggingFace dataset name."""
    if path_or_name.endswith(".jsonl") or path_or_name.endswith(".json"):
        return load_dataset("json", data_files=path_or_name, split="train")
    return load_dataset(path_or_name, split=split)


def build_argparser():
    p = argparse.ArgumentParser(description="DPO eval (CLI version).")
    p.add_argument("--model-path", required=True,
                   help="Path to LoRA adapter directory (contains "
                        "adapter_model.safetensors + model_info.json).")
    p.add_argument("--test-dataset", required=True,
                   help="HF dataset name OR local .jsonl/.json path.")
    p.add_argument("--test-split", default="train",
                   help="Split for HF datasets (ignored for local files).")
    p.add_argument("--test-name", default="test",
                   help="Free-form label for this test (used in output filenames "
                        "and saved metadata).")
    p.add_argument("--description", default="",
                   help="Free-form description saved to results JSON.")

    # Test-time parameters
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--pilot-size", type=int, default=None,
                   help="If set, evaluate on the first N examples only. "
                        "Default: full dataset.")

    # Model fallback
    p.add_argument("--base-model", default=None,
                   help="Override base model name. Default: read from "
                        "model-path/model_info.json. Required if model_info.json "
                        "is missing.")

    # Output / plotting
    p.add_argument("--output-dir", required=True,
                   help="Directory for results.json, metrics.json, *.png.")
    p.add_argument("--no-show", action="store_true",
                   help="Headless mode: save plots to disk but skip plt.show(). "
                        "Set this when running on a remote machine without "
                        "a display.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation entirely.")
    return p


def main():
    args = build_argparser().parse_args()

    if args.no_show:
        matplotlib.use("Agg")  # headless backend before any plot is created

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model info ---
    model_info_path = os.path.join(args.model_path, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path) as f:
            model_info = json.load(f)
        base_model_name = args.base_model or model_info["base_model"]
        print("\nModel information:")
        print(f"  experiment_id: {model_info.get('experiment_id', 'n/a')}")
        print(f"  base_model:    {base_model_name}")
        print(f"  trained_on:    {model_info.get('trained_on_dataset', 'n/a')}")
        print(f"  samples:       {model_info.get('training_samples', 'n/a')}")
        print(f"  trained_at:    {model_info.get('timestamp', 'n/a')}")
    else:
        if args.base_model is None:
            print(f"ERROR: {model_info_path} not found and --base-model not given.",
                  file=sys.stderr)
            sys.exit(1)
        base_model_name = args.base_model
        model_info = {"base_model": base_model_name}
        print(f"[warn] model_info.json missing; using --base-model {base_model_name}")

    # --- Load tokenizer + base model + adapter ---
    print(f"\n[load] tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(f"[load] base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print(f"[load] LoRA adapter: {args.model_path}")
    eval_model = PeftModel.from_pretrained(base_model, args.model_path)
    eval_model.eval()

    # --- Load dataset ---
    print(f"\n[load] test dataset: {args.test_dataset}")
    test_dataset = load_dataset_auto(args.test_dataset, args.test_split)
    print(f"  {len(test_dataset)} examples")

    if args.pilot_size:
        n = min(args.pilot_size, len(test_dataset))
        print(f"  pilot mode: first {n}")
        test_dataset = test_dataset.select(range(n))

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print(f"EVALUATING: {args.test_name}")
    if args.description:
        print(f"  {args.description}")
    print("=" * 60)

    results = evaluate_model_on_dataset(
        model=eval_model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
    )
    analyze_unknown_predictions(results, n_examples=10)

    # --- Metrics ---
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    metrics = calculate_metrics(results["predictions"], results["true_labels"])
    print_detailed_report(results, metrics)
    show_example_predictions(results, n_correct=3, n_incorrect=3)

    # --- Save results / metrics ---
    results_path = os.path.join(args.output_dir, "results.json")
    metrics_path = os.path.join(args.output_dir,
                                f"metrics_test_{args.test_name}.json")
    summary = {
        "test_name":      args.test_name,
        "description":    args.description,
        "test_dataset":   args.test_dataset,
        "model_path":     args.model_path,
        "base_model":     base_model_name,
        "n_examples":     len(results["predictions"]),
        "predictions":    results["predictions"],
        "true_labels":    results["true_labels"],
        "confidences":    results["confidences"],
        "logpA":          results["logpA"],
        "logpB":          results["logpB"],
        "margins":        results["margins"],
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[write] {results_path}")
    print(f"[write] {metrics_path}")

    # --- Plots ---
    if not args.no_plots:
        print("\n[plots] generating")
        plot_confusion_matrix(
            results["predictions"], results["true_labels"],
            save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
        )
        plot_accuracy_breakdown(
            metrics,
            save_path=os.path.join(args.output_dir, "accuracy_breakdown.png"),
        )
        plot_per_class_metrics(
            results["predictions"], results["true_labels"],
            save_path=os.path.join(args.output_dir, "per_class_metrics.png"),
        )
        print(f"[plots] saved to {args.output_dir}")

    # --- Final headline numbers ---
    print("\n" + "=" * 60)
    print(f"HEADLINE: {args.test_name}")
    print("=" * 60)
    print(f"  accuracy:     {metrics.get('accuracy', 'n/a'):.4f}")
    print(f"  n_examples:   {len(results['predictions'])}")
    print(f"  model:        {args.model_path}")
    print(f"  test set:     {args.test_dataset}")
    print("=" * 60)


if __name__ == "__main__":
    main()
