"""
Compare action.pt files from different directories in train_out/real/
Performs pairwise comparison of action tensors
"""

import torch
import numpy as np
import os
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple


def load_action_file(file_path: str) -> torch.Tensor:
    """Load action tensor from pt file"""
    try:
        action = torch.load(file_path, map_location='cpu', weights_only=False)
        print(f"Loaded {file_path}: shape {action.shape}, dtype {action.dtype}")
        return action
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compare_actions(action1: torch.Tensor, action2: torch.Tensor, name1: str, name2: str) -> Dict:
    """Compare two action tensors and return metrics"""
    metrics = {
        'name1': name1,
        'name2': name2,
        'shape_match': action1.shape == action2.shape,
        'dtype_match': action1.dtype == action2.dtype,
    }

    if metrics['shape_match']:
        # Calculate various difference metrics
        diff = action1 - action2
        abs_diff = torch.abs(diff)

        metrics.update({
            'max_abs_diff': float(abs_diff.max().item()),
            'mean_abs_diff': float(abs_diff.mean().item()),
            'std_abs_diff': float(abs_diff.std().item()),
            'max_diff': float(diff.max().item()),
            'min_diff': float(diff.min().item()),
            'mean_diff': float(diff.mean().item()),
        })

        # Calculate relative error (L2 norm)
        l2_norm = torch.norm(action1 - action2).item()
        l1_norm = torch.norm(action1 - action2, p=1).item()
        metrics['l2_norm'] = l2_norm
        metrics['l1_norm'] = l1_norm

        # Check if exactly equal
        metrics['exactly_equal'] = torch.equal(action1, action2)

        # Check if approximately equal (with tolerance)
        metrics['approx_equal'] = torch.allclose(action1, action2, rtol=1e-5, atol=1e-8)

    return metrics


def print_comparison_result(metrics: Dict):
    """Print comparison result in a readable format"""
    print(f"\n{'='*80}")
    print(f"Comparing: {metrics['name1']} vs {metrics['name2']}")
    print(f"{'='*80}")

    if not metrics['shape_match']:
        print(f"❌ Shape mismatch - cannot compare")
        return

    print(f"Shape: {metrics['shape_match']} ✓")
    print(f"Dtype: {metrics['dtype_match']} ✓")

    if metrics['exactly_equal']:
        print(f"✅ Actions are EXACTLY equal")
        return

    if metrics['approx_equal']:
        print(f"✅ Actions are approximately equal (within tolerance)")
    else:
        print(f"❌ Actions are different")

    print(f"\nDifference Statistics:")
    print(f"  Max absolute difference: {metrics['max_abs_diff']:.6e}")
    print(f"  Mean absolute difference: {metrics['mean_abs_diff']:.6e}")
    print(f"  Std absolute difference: {metrics['std_abs_diff']:.6e}")
    print(f"  Max difference: {metrics['max_diff']:.6e}")
    print(f"  Min difference: {metrics['min_diff']:.6e}")
    print(f"  Mean difference: {metrics['mean_diff']:.6e}")
    print(f"\nNorm Measures:")
    print(f"  L1 norm: {metrics['l1_norm']:.6e}")
    print(f"  L2 norm: {metrics['l2_norm']:.6e}")


def compare_all_actions(base_dir: str = "train_out/real/"):
    """Compare all action.pt files in the given directory"""

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return

    # Find all action*.pt files
    action_files = list(base_path.rglob("action*.pt")) + list(base_path.rglob("actions*.pt"))

    if not action_files:
        print(f"No action*.pt or actions*.pt files found in {base_dir}")
        return

    print(f"Found {len(action_files)} action files:")
    for i, f in enumerate(action_files, 1):
        print(f"  {i}. {f}")

    # Load all action files
    actions = {}
    for file_path in action_files:
        action = load_action_file(str(file_path))
        if action is not None:
            # Use relative path as key
            rel_path = file_path.relative_to(base_path)
            actions[str(rel_path)] = action

    print(f"\nSuccessfully loaded {len(actions)}/{len(action_files)} action files")

    # Perform pairwise comparisons
    file_names = list(actions.keys())
    comparisons = []

    print(f"\n{'='*80}")
    print(f"Starting pairwise comparison ({len(list(combinations(file_names, 2)))} pairs)")
    print(f"{'='*80}")

    for name1, name2 in combinations(file_names, 2):
        metrics = compare_actions(actions[name1], actions[name2], name1, name2)
        comparisons.append(metrics)
        print_comparison_result(metrics)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    exact_matches = sum(1 for c in comparisons if c['exactly_equal'])
    approx_matches = sum(1 for c in comparisons if c['approx_equal'])
    total_pairs = len(comparisons)

    print(f"Total comparisons: {total_pairs}")
    print(f"Exactly equal: {exact_matches} ({100*exact_matches/total_pairs:.1f}%)")
    print(f"Approximately equal: {approx_matches} ({100*approx_matches/total_pairs:.1f}%)")
    print(f"Different: {total_pairs - approx_matches} ({100*(total_pairs-approx_matches)/total_pairs:.1f}%)")

    # Find most similar and most different pairs
    if comparisons:
        # Sort by L2 norm (lower = more similar)
        sorted_by_l2 = sorted(comparisons, key=lambda x: x['l2_norm'])
        most_similar = sorted_by_l2[0]
        most_different = sorted_by_l2[-1]

        print(f"\nMost similar pair: {most_similar['name1']} & {most_similar['name2']}")
        print(f"  L2 norm: {most_similar['l2_norm']:.6e}")

        print(f"\nMost different pair: {most_different['name1']} & {most_different['name2']}")
        print(f"  L2 norm: {most_different['l2_norm']:.6e}")


if __name__ == "__main__":
    import sys

    # Allow custom directory path
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "train_out/real/"

    compare_all_actions(base_dir)
