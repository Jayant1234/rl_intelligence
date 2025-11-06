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
OmniMath curriculum learning reward function.

This reward function receives custom metrics computed from policy and reference models:
- policy_confidence: Average token probability from policy model
- kl_divergence: KL(policy || reference)
- decoded_responses: Model's generated text
- partial_solution_given: Solution shown in prompt
- remaining_solution: Solution model should generate
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward using both text-based and model-based metrics.

    Args:
        data_source (str): Dataset identifier
        solution_str (str): Model's generated response (decoded text)
        ground_truth (str): Final answer from dataset
        extra_info (dict): Contains injected custom metrics:
            - policy_confidence: float, model's average token probability
            - kl_divergence: float, KL(policy || reference)
            - decoded_responses: str, same as solution_str
            - partial_solution_given: str, solution in prompt
            - remaining_solution: str, solution model should generate

    Returns:
        dict with "score" and other metrics for logging
    """

    if not extra_info:
        # Fallback if no custom metrics
        return {"score": 0.5, "warning": "No custom metrics available"}

    # ===== EXTRACT INJECTED METRICS =====
    policy_confidence = extra_info.get("policy_confidence", 0.5)
    kl_divergence = extra_info.get("kl_divergence", 0.0)
    partial_given = extra_info.get("partial_solution_given", "")
    remaining_sol = extra_info.get("remaining_solution", "")
    solution_percentage = extra_info.get("solution_percentage_actual", 0.0)

    # ===== PSEUDOCODE REWARD CALCULATION =====
    # TODO: Implement your actual logic here!

    # 1. TEXT-BASED: Check if model continued correctly
    # For now: simple substring check (you should use better similarity)
    if remaining_sol:
        # Simple check: does response contain keywords from remaining solution?
        remaining_words = set(remaining_sol.lower().split())
        response_words = set(solution_str.lower().split())
        overlap = len(remaining_words & response_words) / max(len(remaining_words), 1)
        continuation_score = min(overlap * 2.0, 1.0)  # Scale to [0, 1]
    else:
        # No remaining solution (100% was given)
        continuation_score = 1.0

    # 2. MODEL-BASED: Use policy confidence
    # High confidence + correct = bonus
    # High confidence + wrong = penalty
    if continuation_score > 0.7:
        # Correct continuation
        if policy_confidence > 0.6:
            confidence_bonus = 1.2  # Confident AND correct
        else:
            confidence_bonus = 1.0  # Uncertain but correct
    else:
        # Wrong continuation
        if policy_confidence > 0.7:
            confidence_bonus = 0.5  # Overconfident but wrong - penalty
        else:
            confidence_bonus = 0.8  # Uncertain and wrong - small penalty

    base_reward = continuation_score * confidence_bonus

    # 3. CURRICULUM: Harder curriculum (less help) = higher multiplier
    # When solution_percentage is low (less help given), reward success more
    curriculum_multiplier = 1.0 + (1.0 - solution_percentage)

    # 4. EXPLORATION BONUS: Reward exploration on hard problems
    # High KL = policy is exploring
    if solution_percentage < 0.3:  # Hard curriculum (little help)
        if kl_divergence > 5.0 and continuation_score > 0.7:
            # Exploring AND correct on hard problem = bonus
            curriculum_multiplier *= 1.3

    # 5. FINAL REWARD
    final_reward = base_reward * curriculum_multiplier

    # Clip to reasonable range
    final_reward = max(0.0, min(final_reward, 2.0))

    # Return detailed dict for logging
    return {
        "score": float(final_reward),
        "continuation_score": float(continuation_score),
        "policy_confidence": float(policy_confidence),
        "kl_divergence": float(kl_divergence),
        "confidence_bonus": float(confidence_bonus),
        "curriculum_multiplier": float(curriculum_multiplier),
        "solution_percentage": float(solution_percentage),
        "partial_given_len": len(partial_given),
        "remaining_sol_len": len(remaining_sol),
    }
