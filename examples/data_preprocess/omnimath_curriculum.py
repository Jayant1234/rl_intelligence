# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the Omni-MATH dataset to verl format.

This script ONLY converts raw Omni-MATH to the verl parquet format.
It does NOT handle curriculum learning - that's done by the custom dataset class.

The custom dataset (omnimath_curriculum_dataset.py) will dynamically:
- Sample random solution percentages (0-80%) for each example
- Split solutions by sentences
- Create prompts with partial solutions
- Re-randomize each epoch

Dataset structure:
- problem: The problem statement
- solution: Step-by-step solution (full, not partial)
- answer: Final answer (in \\boxed{} format)
- difficulty: Numerical rating
- domain: Problem classification
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def preprocess_omnimath(dataset: datasets.Dataset, split: str) -> datasets.Dataset:
    """
    Simple preprocessing: just convert to verl format.

    No curriculum logic here - just store the raw problem, solution, and answer.
    The dataset class will handle curriculum learning dynamically.

    Args:
        dataset: HuggingFace dataset
        split: Dataset split name ("train" or "val")

    Returns:
        Processed dataset ready for parquet export
    """
    data_source = "KbsdJames/Omni-MATH"

    def process_fn(example, idx):
        problem = example.get('problem', '')
        solution = example.get('solution', '')
        answer = example.get('answer', '')
        difficulty = example.get('difficulty', 0.0)
        domain = example.get('domain', '')
        source = example.get('source', '')

        # Simple format for verl - NO curriculum logic
        # The custom dataset class will add partial solutions dynamically
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": problem,  # Just the raw problem, no partial solution yet
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "custom",
                "ground_truth": answer,
                "full_solution": solution,  # Store full solution for dataset class to use
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "problem": problem,
                "difficulty": float(difficulty) if difficulty else 0.0,
                "domain": domain,
                "source": source,
            }
        }

        return data

    # Apply transformation
    processed = dataset.map(process_fn, with_indices=True)

    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Omni-MATH dataset to verl format (curriculum logic in dataset class)"
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="[DEPRECATED] Use --local_save_dir instead"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to copy the processed data to"
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/omnimath",
        help="Local directory to save the processed parquet files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        "--train_split_size",
        type=float,
        default=0.9,
        help="Proportion of data to use for training (default: 0.9)"
    )

    args = parser.parse_args()

    # Handle deprecated argument
    local_save_dir = args.local_dir if args.local_dir else args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)

    print("="*80)
    print("Omni-MATH Preprocessor (Simplified)")
    print("="*80)
    print("\nNote: This script ONLY converts to verl format.")
    print("Curriculum learning (0-80% solution sampling) is handled by:")
    print("  -> omnimath_curriculum_dataset.py (during training)\n")

    print(f"Loading Omni-MATH dataset from HuggingFace...")
    print("Note: Omni-MATH only has a 'test' split. Creating train/val splits...\n")

    dataset = datasets.load_dataset("KbsdJames/Omni-MATH", split="test")

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Available fields: {dataset.column_names}")
    print(f"\nCreating train/val split: {args.train_split_size:.1%} train, {1-args.train_split_size:.1%} val")

    # Split into train and validation using seed for reproducibility
    split_dataset = dataset.train_test_split(
        test_size=1.0 - args.train_split_size,
        seed=args.seed
    )
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Simple preprocessing (no curriculum)
    print(f"\n{'='*80}")
    print("Processing...")
    print(f"{'='*80}\n")

    train_processed = preprocess_omnimath(train_dataset, split="train")
    val_processed = preprocess_omnimath(val_dataset, split="val")

    # Create output directory
    os.makedirs(local_save_dir, exist_ok=True)

    # Save to parquet
    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "val.parquet")

    print(f"Saving processed datasets to: {local_save_dir}")
    train_processed.to_parquet(train_path)
    val_processed.to_parquet(val_path)

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"{'='*80}")
    print(f"\nFiles saved:")
    print(f"  Train: {train_path} ({len(train_processed)} examples)")
    print(f"  Val:   {val_path} ({len(val_processed)} examples)")

    # Sample example
    print(f"\n{'='*80}")
    print("Sample Example from Training Set")
    print(f"{'='*80}")
    sample = train_processed[0]
    print(f"\nProblem (first 200 chars):\n{sample['extra_info']['problem'][:200]}...")
    print(f"\nDifficulty: {sample['extra_info']['difficulty']}")
    print(f"Domain: {sample['extra_info']['domain']}")
    print(f"Has answer: {bool(sample['reward_model']['ground_truth'])}")
    print(f"Has solution: {bool(sample['reward_model']['full_solution'])}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"\n{'='*80}")
        print(f"Copying to HDFS: {args.hdfs_dir}")
        print(f"{'='*80}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print("HDFS copy completed")

    print(f"\n{'='*80}")
    print("Next Steps")
    print(f"{'='*80}")
    print("1. Configure curriculum learning in your training config:")
    print("   data:")
    print("     custom_cls:")
    print("       path: 'examples/data_preprocess/omnimath_curriculum_dataset.py'")
    print("       name: 'OmniMathCurriculumDataset'")
    print("     curriculum:")
    print("       min_percentage: 0.0")
    print("       max_percentage: 0.8")
    print("       base_seed: 42")
    print()
    print("2. Implement custom reward function (reward/omnimath_reward.py)")
    print("3. Run training!")
    print()
