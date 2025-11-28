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


def check_format_compliance(response_text: str, has_partial_solution: bool = None) -> dict:
    """
    Check if response follows the required document continuation format.

    IMPORTANT: This uses BINARY scoring - the model must pass ALL checks to get format_score=1.0.
    If any check fails, format_score=0.0.

    Tags are CASE-SENSITIVE.

    Expected format:
        [some content before]
        <|startofprediction|>
        [continuation content]
        <|endofprediction|>

    Requirements:
    1. <|startofprediction|> tag appears exactly once
    2. <|endofprediction|> tag appears exactly once
    3. <|startofprediction|> comes before <|endofprediction|>

    Args:
        response_text: The model's generated response (does NOT include prompt or chat template tags)
        has_partial_solution: DEPRECATED - no longer used (format is same regardless)

    Returns:
        dict with format compliance scores and breakdown
    """
    format_scores = {
        "has_prediction_open": 0.0,
        "has_prediction_close": 0.0,
        "correct_order": 0.0,
        "prediction_open_count": 0,
        "prediction_close_count": 0,
        "total_format_score": 0.0,
    }

    # Check for <|startofprediction|> tag - must appear EXACTLY ONCE (case-sensitive)
    prediction_open_matches = list(re.finditer(r'<\|startofprediction\|>', response_text))
    format_scores["prediction_open_count"] = len(prediction_open_matches)
    if len(prediction_open_matches) == 1:
        format_scores["has_prediction_open"] = 1.0
        prediction_open_match = prediction_open_matches[0]
    elif len(prediction_open_matches) > 1:
        # Multiple occurrences - set total score to 0 and return immediately
        format_scores["total_format_score"] = 0.0
        return format_scores
    else:
        prediction_open_match = None

    # Check for <|endofprediction|> tag - must appear EXACTLY ONCE (case-sensitive)
    prediction_close_matches = list(re.finditer(r'<\|endofprediction\|>', response_text))
    format_scores["prediction_close_count"] = len(prediction_close_matches)
    if len(prediction_close_matches) == 1:
        format_scores["has_prediction_close"] = 1.0
        prediction_close_match = prediction_close_matches[0]
    elif len(prediction_close_matches) > 1:
        # Multiple occurrences - set total score to 0 and return immediately
        format_scores["total_format_score"] = 0.0
        return format_scores
    else:
        prediction_close_match = None

    # Check correct order: <|startofprediction|> → <|endofprediction|>
    if prediction_open_match and prediction_close_match:
        if prediction_open_match.start() < prediction_close_match.start():
            format_scores["correct_order"] = 1.0

    # BINARY ALL-OR-NOTHING SCORING
    # Model must pass ALL 3 checks to get format_score = 1.0
    all_checks_pass = all([
        format_scores["has_prediction_open"] == 1.0,
        format_scores["has_prediction_close"] == 1.0,
        format_scores["correct_order"] == 1.0,
    ])

    format_scores["total_format_score"] = 1.0 if all_checks_pass else 0.0

    return format_scores


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward combining Continuous Information Gain + Binary Format Reward.

    This reward function combines:
    1. Format Reward (BINARY - CHECKED FIRST): Rewards proper document continuation structure
       Model must have ALL of the following to get format_reward=1.0:
       - Exactly one <|startofprediction|> tag
       - Exactly one <|endofprediction|> tag
       - Correct order: <|startofprediction|> → <|endofprediction|>

       If ANY check fails, format_reward=0.0 (all-or-nothing)

    2. Information Gain (IG - CONTINUOUS, CONDITIONAL ON GROUP FORMAT): Measures if thinking helps
       IG = mean log P(answer | prompt + thinking) - mean log P(answer | prompt only)
       Uses per-token average to make IG independent of gold solution length (important for curriculum learning)
       CONTINUOUS REWARD: Uses normalized_IG directly (range: [0, 1])
       - Better thinking → higher IG reward (closer to 1)
       - Worse thinking → lower IG reward (closer to 0)

       CONDITIONAL ACTIVATION (GROUP-LEVEL - 3 CONDITIONS):
       - If format_reward = 0: ig_reward = 0 (individual format failed)
       - If ANY sample in GRPO group has format_reward = 0: ig_reward = 0 for ALL in group
       - If group max IG ≤ 0: ig_reward = 0 (no good examples in group)
       - If ALL conditions met: ig_reward = normalized_IG (continuous learning)

    Final Reward Range: [0.0, 1.3] (all rewards non-negative)
       - Best: format=1, all gates pass, IG=1.0 → 1.3 (perfect format + excellent thinking)
       - Medium: format=1, all gates pass, IG=0.5 → 0.8 (perfect format + average thinking)
       - Low: format=1, all gates pass, IG=0.0 → 0.3 (perfect format + poor thinking)
       - Worst: format=0 OR any gate fails → 0.0 (must learn format first)

    Args:
        data_source (str): Dataset identifier (e.g., 'KbsdJames/Omni-MATH')
        solution_str (str): Model's generated response (decoded text)
        ground_truth (str): Final answer from dataset (not used for scoring currently)
        extra_info (dict): Contains computed metrics:
            - information_gain: float, IG = log P(answer|thinking) - log P(answer|no thinking)
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
    information_gain = extra_info.get("information_gain", None)  # Normalized to [0, 1]
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

    # ===== COMPONENT 1: FORMAT REWARD (BINARY) - CHECKED FIRST =====

    # Check if response follows the required document continuation format
    # Note: has_partial_solution no longer needed - format is same regardless
    format_scores = check_format_compliance(solution_str)

    # Extract format metrics - now BINARY (0.0 or 1.0 only)
    format_reward = format_scores["total_format_score"]  # Binary: 0.0 or 1.0

    # ===== COMPONENT 2: INFORMATION GAIN REWARD (CONTINUOUS, CONDITIONAL ON FORMAT & VARIANCE) =====

    # STRATEGY: Use Information Gain as primary reward signal
    # IG = mean log P(answer | prompt + thinking) - mean log P(answer | prompt only)
    # Using per-token mean makes IG independent of gold solution length
    #
    # The IG is NORMALIZED to [0, 1] range using batch-level statistics:
    # normalized_IG = sigmoid((raw_IG - batch_mean) / batch_std)
    #
    # CONTINUOUS REWARD: Use normalized_IG directly (no binarization)
    # - This allows GRPO to learn from the continuous variance in thinking quality
    # - Better thinking gets higher rewards, worse thinking gets lower rewards
    #
    # CONDITIONAL ACTIVATION (GROUP-LEVEL): Only give IG reward if:
    # 1. All samples in the GRPO group have perfect format
    # 2. Group has at least one positive IG (max > 0)
    #
    # Conditions explained:
    # - format_reward = 1: Individual format must be perfect (format must be learned first)
    # - group_all_formats_passed = True: ALL group members have perfect format (no contamination)
    # - group_max_ig_positive = True: Group has at least one IG > 0 (at least one good example)
    #
    # Goal: Model learns format first (group-level), then learns from continuous IG variance

    # Get group-level gating conditions
    group_all_formats_passed = extra_info.get("group_all_formats_passed", True)
    group_max_ig_positive = extra_info.get("group_max_ig_positive", True)

    # Use continuous IG reward when all conditions are met
    # Requires:
    # 1. Individual format is perfect (format_reward = 1.0)
    # 2. ALL formats in the group are perfect (group_all_formats_passed = True)
    # 3. Group has at least one positive IG (group_max_ig_positive = True, max > 0)
    if (format_reward == 1.0 and
        group_all_formats_passed and
        group_max_ig_positive):
        ig_reward = information_gain  # Use continuous normalized IG [0, 1]
    else:
        ig_reward = 0.0  # No IG reward if any condition fails

    # No scaling needed - format reward is already binary {0, 1}
    # Bad format gives 0, good format gives 1
    format_reward_scaled = format_reward  # Binary: 0.0 or 1.0 (no scaling)

    # ===== COMBINE REWARDS =====

    # Weight for format reward (adjust based on importance)
    # Format is binary, IG is continuous, this controls format's contribution
    format_weight = 0.3

    # Final combined reward: continuous IG + weighted binary format
    # Reward range when conditions met (format=1, all group checks pass):
    # - Best case: IG=1.0 (top thinking) → 1.0 + 0.3 = 1.3
    # - Medium case: IG=0.5 (average thinking) → 0.5 + 0.3 = 0.8
    # - Low case: IG=0.0 (poor thinking) → 0.0 + 0.3 = 0.3
    # Reward when conditions fail (format=0 OR group checks fail):
    # - All cases: IG forced to 0 → 0.0 + 0.0 = 0.0
    final_reward = ig_reward + (format_weight * format_reward_scaled)

    # Note:
    # - Conditional IG (GROUP-LEVEL): Model must meet ALL conditions before getting IG rewards:
    #   1. Individual format is perfect (format_reward = 1.0)
    #   2. ALL group members have perfect format (group_all_formats_passed = True)
    #   3. Group has at least one positive IG (group_max_ig_positive = True, max > 0)
    # - This creates curriculum: format first (all in group), then learn from IG variance
    # - Format reward provides clear signal: perfect format (0.3) vs bad format (0.0)
    # - IG reward is CONTINUOUS (normalized_IG in [0, 1]) when conditions met
    # - GRPO learns from the continuous variance in thinking quality within qualified groups
    # - Lower IG rewards are possible (when thinking hurts, closer to 0), but only after passing all gates
    # - Group-level gating ensures format is learned first

    # Return detailed dict for logging and analysis
    return {
        "score": float(final_reward),

        # Information Gain metrics
        "information_gain": float(information_gain),  # Normalized [0, 1]
        "information_gain_raw": float(information_gain_raw) if information_gain_raw is not None else None,  # Original scale
        "ig_reward": float(ig_reward),

        # Format metrics (simplified tag-based format)
        "format_reward": float(format_reward),  # Binary: 0.0 or 1.0
        "format_reward_scaled": float(format_reward_scaled),  # Binary: 0.0 or 1.0 (no scaling)
        "has_prediction_open": float(format_scores["has_prediction_open"]),
        "has_prediction_close": float(format_scores["has_prediction_close"]),
        "correct_order": float(format_scores["correct_order"]),
        "prediction_open_count": int(format_scores["prediction_open_count"]),
        "prediction_close_count": int(format_scores["prediction_close_count"]),

        # Metadata for analysis
        "solution_percentage": float(solution_percentage),
        "partial_given_len": len(partial_given),
        "remaining_sol_len": len(remaining_sol),

        # Group-level gating conditions
        "group_all_formats_passed": bool(group_all_formats_passed),
        "group_max_ig_positive": bool(group_max_ig_positive),

        # Optional metrics (may be None if not computed)
        "policy_confidence": float(policy_confidence) if policy_confidence is not None else None,
        "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
    }
