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
Custom dataset class for Omni-MATH with per-epoch curriculum re-randomization.

This dataset does ALL the curriculum learning logic:
- Loads simple parquet files created by omnimath_curriculum.py
- Dynamically samples random solution percentages (0-80%) for each sample
- Splits solutions by sentences
- Creates prompts with partial solutions
- Re-randomizes everything each epoch via set_epoch()

Usage in config:
    data:
      train_files: ~/data/omnimath/train.parquet
      val_files: ~/data/omnimath/val.parquet
      custom_cls:
        path: "examples/data_preprocess/omnimath_curriculum_dataset.py"
        name: "OmniMathCurriculumDataset"
      curriculum:
        min_percentage: 0.0   # 0% to 80% of solution
        max_percentage: 0.8
        base_seed: 42         # For reproducibility
"""

import copy
import logging
import os
import re
from typing import List, Optional, Tuple

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while preserving mathematical expressions.

    Handles:
    - LaTeX expressions ($...$ and \\boxed{})
    - Numbered steps
    - Standard sentence boundaries

    Args:
        text: Solution text to split

    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []

    # Protect LaTeX expressions from being split
    text = re.sub(r'\$([^\$]+)\$', lambda m: m.group(0).replace('.', '<<<PERIOD>>>'), text)
    text = re.sub(r'\\boxed\{[^}]+\}', lambda m: m.group(0).replace('.', '<<<PERIOD>>>'), text)

    # Split on sentence boundaries
    # Matches: . ! ? followed by space/newline and capital letter or digit
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z]|$|\d+\.)'
    sentences = re.split(sentence_pattern, text)

    # Restore periods in math expressions
    sentences = [s.replace('<<<PERIOD>>>', '.').strip() for s in sentences if s.strip()]

    return sentences


def calculate_sentence_split(sentences: List[str], target_percentage: float) -> Tuple[List[str], List[str]]:
    """
    Split sentences based on target percentage of total character count.

    Args:
        sentences: List of sentence strings
        target_percentage: Target percentage (0.0 to 0.8) of solution to include

    Returns:
        Tuple of (given_sentences, remaining_sentences)
    """
    if not sentences:
        return [], []

    if target_percentage <= 0:
        return [], sentences

    if target_percentage >= 1.0:
        return sentences, []

    total_chars = sum(len(s) for s in sentences)
    target_chars = total_chars * target_percentage

    cumulative_chars = 0
    split_index = 0

    for i, sentence in enumerate(sentences):
        cumulative_chars += len(sentence)
        if cumulative_chars >= target_chars:
            split_index = i + 1  # Include this sentence
            break

    # Ensure we don't take all sentences if percentage < 1.0
    if target_percentage < 1.0 and split_index == len(sentences):
        split_index = max(0, len(sentences) - 1)

    given_sentences = sentences[:split_index]
    remaining_sentences = sentences[split_index:]

    return given_sentences, remaining_sentences


def create_curriculum_prompt(problem: str, given_solution: str) -> str:
    """
    Create the final prompt by combining problem and partial solution.

    Args:
        problem: The math problem statement
        given_solution: The partial solution to include (can be empty)

    Returns:
        Formatted prompt string
    """
    if given_solution.strip():
        # With partial solution: ask for reasoning first, then completion
        prompt = (
            f"{problem}\n\n"
            f"A partial solution is provided below:\n\n"
            f"{given_solution}\n\n"
            f"Task: Complete this solution by following these steps in order.\n\n"
            f"First, write your reasoning:\n"
            f"- Explain what the partial solution has established\n"
            f"- Identify what steps remain to be done\n\n"
            f"Then, complete the remaining solution:\n"
            f"- Show the remaining steps clearly\n"
            f"- Put your final answer in \\boxed{{}}\n\n"
            f"Format your response as:\n\n"
            f"**Reasoning:**\n"
            f"[Your understanding of the partial solution and plan for remaining steps]\n\n"
            f"**Remaining Solution:**\n"
            f"[Complete the remaining steps and final answer in \\boxed{{}}]"
        )
    else:
        # Without partial solution: ask for reasoning first, then full solution
        prompt = (
            f"{problem}\n\n"
            f"Task: Solve this problem by following these steps in order.\n\n"
            f"First, write your reasoning:\n"
            f"- Break down the problem\n"
            f"- Plan your solution approach\n\n"
            f"Then, write your complete solution:\n"
            f"- Show all steps clearly\n"
            f"- Put your final answer in \\boxed{{}}\n\n"
            f"Format your response as:\n\n"
            f"**Reasoning:**\n"
            f"[Your step-by-step reasoning and approach]\n\n"
            f"**Solution:**\n"
            f"[Complete solution with final answer in \\boxed{{}}]"
        )

    return prompt


class OmniMathCurriculumDataset(Dataset):
    """
    Custom dataset for Omni-MATH with dynamic per-epoch curriculum learning.

    Key responsibilities:
    - Load parquet files (created by omnimath_curriculum.py)
    - Dynamically generate partial solutions on-the-fly for each sample
    - Re-randomize solution percentages each epoch
    - Store both partial_solution_given and remaining_solution for reward calculation
    - Support reproducible curriculum via epoch-based seeding
    """

    def __init__(
        self,
        data_files: str | List[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        """
        Initialize the curriculum dataset.

        Args:
            data_files: Path(s) to parquet file(s) created by preprocessing
            tokenizer: Tokenizer for processing text
            config: Configuration with curriculum settings
            processor: Optional processor (not used for Omni-MATH)
        """
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # Curriculum learning parameters from config
        curriculum_config = config.get("curriculum", {})
        self.min_percentage = curriculum_config.get("min_percentage", 0.0)
        self.max_percentage = curriculum_config.get("max_percentage", 0.8)
        self.base_seed = curriculum_config.get("base_seed", 42)
        self.current_epoch = 0

        # Standard verl dataset parameters
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")

        # Load the raw data (problem + full_solution, no curriculum yet)
        self._download()
        self._read_files_and_tokenize()

        # Initialize RNG for current epoch
        self._initialize_epoch_rng()

        logger.info(f"OmniMathCurriculumDataset initialized with {len(self.dataframe)} examples")
        logger.info(f"Curriculum range: {self.min_percentage:.1%} to {self.max_percentage:.1%}")
        logger.info(f"Base seed: {self.base_seed}, Current epoch: {self.current_epoch}")

    def _download(self):
        """Download parquet files to local cache if needed."""
        from verl.utils.fs import copy_to_local

        for i, parquet_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(
                src=parquet_file,
                cache_dir=self.cache_dir,
                use_shm=self.config.get("use_shm", False)
            )

    def _read_files_and_tokenize(self):
        """Load parquet files into HuggingFace dataset."""
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"Loaded dataset with {len(self.dataframe)} examples")

    def _initialize_epoch_rng(self):
        """Initialize random number generator based on current epoch."""
        epoch_seed = self.base_seed + self.current_epoch
        self.rng = np.random.RandomState(epoch_seed)
        logger.info(f"Initialized RNG with epoch seed: {epoch_seed} (base: {self.base_seed} + epoch: {self.current_epoch})")

    def set_epoch(self, epoch: int):
        """
        Set the current epoch and re-initialize RNG for curriculum re-randomization.

        IMPORTANT: Call this at the start of each epoch to get different solution percentages!

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        self._initialize_epoch_rng()
        logger.info(f"Dataset epoch updated to {epoch} - curriculum will be re-randomized")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Get a single item with dynamically generated curriculum.

        This is where ALL the curriculum magic happens:
        1. Sample random percentage (0-80%) for this sample
        2. Split solution into sentences
        3. Take first N sentences to reach percentage
        4. Create prompt with partial solution
        5. Store both given and remaining solutions for reward
        """
        # Get raw data from parquet (problem, full_solution, answer, etc.)
        raw_item = self.dataframe[idx]

        # Extract fields
        problem = raw_item.get('extra_info', {}).get('problem', '')
        full_solution = raw_item.get('reward_model', {}).get('full_solution', '')
        ground_truth = raw_item.get('reward_model', {}).get('ground_truth', '')
        data_source = raw_item.get('data_source', 'KbsdJames/Omni-MATH')
        ability = raw_item.get('ability', 'math')

        # ===== CURRICULUM LOGIC STARTS HERE =====

        # 1. Sample random percentage for THIS sample in THIS epoch
        solution_percentage = self.rng.uniform(self.min_percentage, self.max_percentage)

        # 2. Split solution into sentences
        solution_sentences = split_into_sentences(full_solution)

        # 3. Calculate split point based on percentage
        given_sentences, remaining_sentences = calculate_sentence_split(
            solution_sentences,
            solution_percentage
        )

        # 4. Reconstruct partial solutions
        partial_solution_given = ' '.join(given_sentences).strip()
        remaining_solution = ' '.join(remaining_sentences).strip()

        # 5. Create prompt with partial solution incorporated
        prompt_content = create_curriculum_prompt(problem, partial_solution_given)

        # ===== CURRICULUM LOGIC ENDS HERE =====

        # Tokenize the prompt
        raw_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Pad/truncate to max length
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        # Calculate actual percentage achieved
        actual_percentage = (
            len(partial_solution_given) / len(full_solution)
            if full_solution else 0.0
        )

        # Build return dict - CRITICAL: include partial_solution_given and remaining_solution
        result = {
            # Standard verl fields
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],

            # Routing fields
            "data_source": data_source,
            "ability": ability,

            # Reward model fields - YOUR CUSTOM REWARD FUNCTION WILL ACCESS THESE
            "reward_model": {
                "style": "custom",
                "ground_truth": ground_truth,  # Final answer in \boxed{} format
                "partial_solution_given": partial_solution_given,  # What was in prompt
                "remaining_solution": remaining_solution,  # What model needs to generate
                "full_solution": full_solution,  # Complete solution (for reference)
            },

            # Extra metadata
            "extra_info": {
                **raw_item.get('extra_info', {}),
                "solution_percentage_target": float(solution_percentage),
                "solution_percentage_actual": float(actual_percentage),
                "num_solution_sentences": len(solution_sentences),
                "num_given_sentences": len(given_sentences),
                "num_remaining_sentences": len(remaining_sentences),
                "current_epoch": self.current_epoch,
                "epoch_seed": self.base_seed + self.current_epoch,
            },
        }

        return result

    def __getstate__(self):
        """Support for checkpointing - don't serialize large dataframe."""
        state = self.__dict__.copy()
        if 'dataframe' in state:
            del state['dataframe']
        return state

    def __setstate__(self, state):
        """Support for resuming from checkpoint."""
        self.__dict__.update(state)
        self._download()
        self._read_files_and_tokenize()
        self._initialize_epoch_rng()
