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
OmniMath curriculum learning reward function with Information Gain + Format Reward.

This reward function combines:
1. Information Gain (IG): Measures if CoT reasoning helps the model
   IG = log P(answer | prompt + CoT) - log P(answer | prompt only)

2. Format Reward (RLVR-style): Rewards proper response structure
   - Has **Reasoning:** section
   - Has **Solution:** or **Remaining Solution:** section
   - Sections in correct order
   - Contains \\boxed{} with answer

Additional metrics for analysis:
- policy_confidence: Average token probability from policy model
- kl_divergence: KL(policy || reference)
- partial_solution_given: Solution shown in prompt
- remaining_solution: Solution model should generate
- solution_percentage: Percentage of solution given in prompt (0.0 to 0.8)
"""

import re


def check_format_compliance(response_text: str, has_partial_solution: bool) -> dict:
    """
    Check if response follows the required format (RLVR-style format reward).

    Args:
        response_text: The model's generated response
        has_partial_solution: Whether partial solution was provided in prompt

    Returns:
        dict with format compliance scores and breakdown
    """
    format_scores = {
        "has_reasoning_header": 0.0,
        "has_solution_header": 0.0,
        "has_boxed_answer": 0.0,
        "correct_order": 0.0,
        "reasoning_substantive": 0.0,
        "solution_substantive": 0.0,
        "total_format_score": 0.0,
    }

    # Check for **Reasoning:** header
    if re.search(r'\*\*Reasoning:\*\*', response_text, re.IGNORECASE):
        format_scores["has_reasoning_header"] = 1.0

    # Check for solution header (depends on prompt type)
    if has_partial_solution:
        # Should have "**Remaining Solution:**"
        if re.search(r'\*\*Remaining Solution:\*\*', response_text, re.IGNORECASE):
            format_scores["has_solution_header"] = 1.0
    else:
        # Should have "**Solution:**"
        if re.search(r'\*\*Solution:\*\*', response_text, re.IGNORECASE):
            format_scores["has_solution_header"] = 1.0

    # Check for \boxed{} answer
    if re.search(r'\\boxed\{[^}]+\}', response_text):
        format_scores["has_boxed_answer"] = 1.0

    # Check correct order: Reasoning should come before Solution
    reasoning_match = re.search(r'\*\*Reasoning:\*\*', response_text, re.IGNORECASE)
    solution_match = re.search(r'\*\*(Remaining )?Solution:\*\*', response_text, re.IGNORECASE)

    if reasoning_match and solution_match:
        if reasoning_match.start() < solution_match.start():
            format_scores["correct_order"] = 1.0

    # Check if reasoning section is substantive (at least 20 chars)
    if reasoning_match:
        reasoning_start = reasoning_match.end()
        # Find end of reasoning section (next header or end of text)
        solution_start = solution_match.start() if solution_match else len(response_text)
        reasoning_content = response_text[reasoning_start:solution_start].strip()

        if len(reasoning_content) >= 20:  # At least 20 characters
            format_scores["reasoning_substantive"] = 1.0

    # Check if solution section is substantive (at least 10 chars)
    if solution_match:
        solution_start = solution_match.end()
        solution_content = response_text[solution_start:].strip()

        if len(solution_content) >= 10:  # At least 10 characters
            format_scores["solution_substantive"] = 1.0

    # Calculate total format score (all components weighted equally)
    format_scores["total_format_score"] = sum([
        format_scores["has_reasoning_header"],
        format_scores["has_solution_header"],
        format_scores["has_boxed_answer"],
        format_scores["correct_order"],
        format_scores["reasoning_substantive"],
        format_scores["solution_substantive"],
    ]) / 6.0  # Normalize to [0, 1]

    return format_scores


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward combining Information Gain + Format Reward (RLVR-style).

    This reward function combines:
    1. Information Gain (IG): Measures if CoT reasoning helps the model
       IG = log P(answer | prompt + CoT) - log P(answer | prompt only)

    2. Format Reward: Rewards proper response structure (RLVR-style)
       - Has **Reasoning:** section
       - Has **Solution:** or **Remaining Solution:** section
       - Sections in correct order
       - Contains \\boxed{} with answer

    Args:
        data_source (str): Dataset identifier (e.g., 'KbsdJames/Omni-MATH')
        solution_str (str): Model's generated response (decoded text)
        ground_truth (str): Final answer from dataset
        extra_info (dict): Contains computed metrics:
            - information_gain: float, IG = log P(answer|CoT) - log P(answer|no CoT)
            - policy_confidence: float, model's average token probability (optional)
            - kl_divergence: float, KL(policy || reference) (optional)
            - partial_solution_given: str, solution in prompt
            - remaining_solution: str, solution model should generate
            - solution_percentage_actual: float, percentage of solution given (0.0-0.8)

    Returns:
        dict with "score" and other metrics for logging
    """

    if not extra_info:
        # Fallback if no custom metrics
        return {"score": 0.0, "warning": "No extra_info available"}

    # ===== EXTRACT METRICS =====
    information_gain = extra_info.get("information_gain", None)  # Normalized to [-1, 1]
    information_gain_raw = extra_info.get("information_gain_raw", None)  # Original scale
    policy_confidence = extra_info.get("policy_confidence", None)
    kl_divergence = extra_info.get("kl_divergence", None)
    partial_given = extra_info.get("partial_solution_given", "")
    remaining_sol = extra_info.get("remaining_solution", "")
    solution_percentage = extra_info.get("solution_percentage_actual", 0.0)

    # Check if information gain is available
    if information_gain is None:
        # Fallback: IG not computed (compute_information_gain=False in config)
        return {
            "score": 0.0,
            "warning": "information_gain not found in extra_info. Set compute_information_gain=True"
        }

    # ===== COMPONENT 1: INFORMATION GAIN REWARD =====

    # STRATEGY: Use Information Gain as primary reward signal
    # IG = log P(answer | prompt + CoT) - log P(answer | prompt only)
    #
    # The IG is NORMALIZED to [-1, 1] range using batch-level statistics:
    # normalized_IG = tanh((raw_IG - batch_mean) / batch_std)
    #
    # Interpretation of normalized IG:
    # - IG close to +1: CoT helps significantly (best reasoning in this batch)
    # - IG close to 0: CoT provides average/neutral information
    # - IG close to -1: CoT confuses the model (worst reasoning in this batch)
    #
    # Goal: Maximize normalized information gain = encourage useful reasoning

    # Use normalized IG directly as the IG reward component
    ig_reward = information_gain  # [-1, 1]

    # ===== COMPONENT 2: FORMAT REWARD (RLVR-style) =====

    # Check if response follows the required format
    has_partial_solution = len(partial_given.strip()) > 0
    format_scores = check_format_compliance(solution_str, has_partial_solution)

    # Extract format metrics
    format_reward = format_scores["total_format_score"]  # [0, 1]

    # Scale format reward to match IG reward magnitude
    # IG reward is roughly [-3, 3], so scale format reward to [-1, 1]
    # Give positive reward for good format, negative for bad format
    format_reward_scaled = (format_reward - 0.5) * 2.0  # Map [0, 1] to [-1, 1]

    # ===== COMBINE REWARDS =====

    # Weight for format reward (adjust based on importance)
    # Start with 0.3 to make format 30% as important as IG
    format_weight = 0.3

    # Final combined reward: simple sum of IG + weighted format reward
    final_reward = ig_reward + (format_weight * format_reward_scaled)

    # Note: We don't clip the reward because:
    # - Negative IG rewards are informative (tell model reasoning was harmful)
    # - Format reward adds differentiation between responses in same group
    # - This increases within-group variance for GRPO advantages

    # Return detailed dict for logging and analysis
    return {
        "score": float(final_reward),

        # Information Gain metrics
        "information_gain": float(information_gain),  # Normalized [-1, 1]
        "information_gain_raw": float(information_gain_raw) if information_gain_raw is not None else None,  # Original scale
        "ig_reward": float(ig_reward),

        # Format metrics
        "format_reward": float(format_reward),  # [0, 1]
        "format_reward_scaled": float(format_reward_scaled),  # [-1, 1]
        "has_reasoning_header": float(format_scores["has_reasoning_header"]),
        "has_solution_header": float(format_scores["has_solution_header"]),
        "has_boxed_answer": float(format_scores["has_boxed_answer"]),
        "correct_order": float(format_scores["correct_order"]),
        "reasoning_substantive": float(format_scores["reasoning_substantive"]),
        "solution_substantive": float(format_scores["solution_substantive"]),

        # Metadata for analysis
        "solution_percentage": float(solution_percentage),
        "partial_given_len": len(partial_given),
        "remaining_sol_len": len(remaining_sol),

        # Optional metrics (may be None if not computed)
        "policy_confidence": float(policy_confidence) if policy_confidence is not None else None,
        "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
    }
