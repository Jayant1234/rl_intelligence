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
OmniMath curriculum learning reward function with Information Gain.

This reward function uses information gain as the primary reward signal:
- information_gain: How much the partial solution (CoT) helps the model predict the answer
  IG = log P(answer | prompt + CoT) - log P(answer | prompt only)

Additional metrics for analysis:
- policy_confidence: Average token probability from policy model
- kl_divergence: KL(policy || reference)
- partial_solution_given: Solution shown in prompt
- remaining_solution: Solution model should generate
- solution_percentage: Percentage of solution given in prompt (0.0 to 0.8)
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward using information gain as the primary signal.

    Information Gain measures how much the Chain-of-Thought reasoning (partial solution)
    helps the model predict the correct answer. Higher IG = better reasoning.

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
    information_gain = extra_info.get("information_gain", None)
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

    # ===== INFORMATION GAIN BASED REWARD CALCULATION =====

    # STRATEGY: Use Information Gain as primary reward signal
    # IG = log P(answer | prompt + CoT) - log P(answer | prompt only)
    #
    # Interpretation:
    # - IG > 0: CoT helps the model (reasoning is useful)
    # - IG = 0: CoT provides no information
    # - IG < 0: CoT confuses the model (bad reasoning)
    #
    # Goal: Maximize information gain = encourage useful reasoning

    # 1. BASE REWARD: Use Information Gain directly
    # Scale IG to a reasonable reward range
    # Typical IG values might be in range [-10, 10], but can vary
    base_reward = information_gain

    # 2. CURRICULUM MULTIPLIER: Reward harder problems more
    # When solution_percentage is low (less help given), the task is harder
    # Give higher reward for achieving high IG on harder problems
    curriculum_multiplier = 1.0 + (1.0 - solution_percentage) * 0.5
    # Examples:
    # - solution_percentage = 0.0 (no help): multiplier = 1.5
    # - solution_percentage = 0.4 (40% help): multiplier = 1.3
    # - solution_percentage = 0.8 (80% help): multiplier = 1.1

    # 3. APPLY CURRICULUM SCALING
    final_reward = base_reward * curriculum_multiplier

    # 4. OPTIONAL: Add bonus for very high IG (exceptional reasoning)
    if information_gain > 5.0:
        # Very high IG = model is learning strong reasoning patterns
        final_reward *= 1.2

    # 5. OPTIONAL: Penalize negative IG more on easy problems
    # If we gave lots of help but model still has negative IG, penalize more
    if information_gain < 0 and solution_percentage > 0.5:
        final_reward *= 1.5  # Make negative reward worse

    # Note: We don't clip the reward because IG can naturally be negative
    # Negative rewards are informative - they tell the model its reasoning was harmful

    # Return detailed dict for logging and analysis
    return {
        "score": float(final_reward),
        "information_gain": float(information_gain),
        "base_reward": float(base_reward),
        "curriculum_multiplier": float(curriculum_multiplier),
        "solution_percentage": float(solution_percentage),
        "partial_given_len": len(partial_given),
        "remaining_sol_len": len(remaining_sol),
        # Optional metrics (may be None if not computed)
        "policy_confidence": float(policy_confidence) if policy_confidence is not None else None,
        "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
    }
