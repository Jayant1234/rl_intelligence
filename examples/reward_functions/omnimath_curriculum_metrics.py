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
Compute custom curriculum metrics using policy and reference models.

This function is called AFTER both policy and reference models have computed
their log probabilities, giving us access to model outputs for reward calculation.
"""

import torch
import numpy as np
from verl import DataProto


def compute_omnimath_curriculum_metrics(batch: DataProto, tokenizer) -> dict:
    """
    Compute custom metrics using policy/reference models and curriculum data.

    This function is called in ray_trainer.py after:
    1. old_log_probs (policy) are computed
    2. ref_log_prob (reference) are computed
    3. BEFORE reward function is called

    Args:
        batch: DataProto containing:
            batch.batch["old_log_probs"]: (batch_size, seq_len) - Policy log probs
            batch.batch["ref_log_prob"]: (batch_size, seq_len) - Reference log probs
            batch.batch["responses"]: (batch_size, response_len) - Generated tokens
            batch.batch["prompts"]: (batch_size, prompt_len) - Input prompts
            batch.batch["attention_mask"]: (batch_size, total_len) - Masks
            batch.non_tensor_batch["reward_model"]["partial_solution_given"]
            batch.non_tensor_batch["reward_model"]["remaining_solution"]
            batch.non_tensor_batch["extra_info"] - Curriculum metadata

        tokenizer: For decoding tokens to text

    Returns:
        dict with per-sample metrics to be used in reward function:
        {
            "policy_confidence": np.array([...]),  # Shape: (batch_size,)
            "kl_divergence": np.array([...]),
            "decoded_responses": np.array([...], dtype=object),
            "partial_solution_given": np.array([...], dtype=object),
            "remaining_solution": np.array([...], dtype=object),
        }
    """

    batch_size = len(batch)

    # Initialize metric arrays
    policy_confidence = []
    kl_divergence = []
    decoded_responses = []
    partial_solutions = []
    remaining_solutions = []

    # Extract model outputs (these are already computed by the trainer)
    old_log_probs = batch.batch["old_log_probs"]  # Policy
    ref_log_probs = batch.batch.get("ref_log_prob", None)  # Reference (may be None)
    responses = batch.batch["responses"]
    prompts = batch.batch["prompts"]
    attention_mask = batch.batch["attention_mask"]

    prompt_length = prompts.shape[-1]

    for i in range(batch_size):
        # ===== EXTRACT CURRICULUM DATA =====
        partial_given = batch.non_tensor_batch["reward_model"][i]["partial_solution_given"]
        remaining_sol = batch.non_tensor_batch["reward_model"][i]["remaining_solution"]

        partial_solutions.append(partial_given)
        remaining_solutions.append(remaining_sol)

        # ===== DECODE RESPONSE =====
        # Get valid response length
        valid_response_len = attention_mask[i, prompt_length:].sum().item()
        valid_response_ids = responses[i, :valid_response_len]

        # Decode to text
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        decoded_responses.append(response_str)

        # ===== COMPUTE MODEL-BASED METRICS =====

        # 1. Policy Confidence: Average probability over generated tokens
        #    Higher = model is more confident in this response
        valid_old_log_probs = old_log_probs[i, prompt_length:prompt_length + valid_response_len]
        policy_probs = torch.exp(valid_old_log_probs)  # Convert log prob to prob
        avg_confidence = policy_probs.mean().item()
        policy_confidence.append(avg_confidence)

        # 2. KL Divergence: How much policy deviates from reference
        #    Higher = more exploration, lower = more conservative
        if ref_log_probs is not None:
            valid_ref_log_probs = ref_log_probs[i, prompt_length:prompt_length + valid_response_len]
            kl = (valid_old_log_probs - valid_ref_log_probs).sum().item()
            kl_divergence.append(kl)
        else:
            kl_divergence.append(0.0)

    # Return as numpy arrays (compatible with batch.non_tensor_batch)
    return {
        "policy_confidence": np.array(policy_confidence, dtype=np.float32),
        "kl_divergence": np.array(kl_divergence, dtype=np.float32),
        "decoded_responses": np.array(decoded_responses, dtype=object),
        "partial_solution_given": np.array(partial_solutions, dtype=object),
        "remaining_solution": np.array(remaining_solutions, dtype=object),
    }
