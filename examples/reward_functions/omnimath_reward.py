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

    Tags are CASE-SENSITIVE and must be lowercase.

    Expected format:
        <think>
        [long, detailed planning - at least 100 chars]
        </think>
        <|startofprediction|>
        [continuation - at least 10 chars]
        <|endofprediction|>

    Args:
        response_text: The model's generated response
        has_partial_solution: DEPRECATED - no longer used (format is same regardless)

    Returns:
        dict with format compliance scores and breakdown
    """
    format_scores = {
        "has_think_open": 0.0,
        "has_think_close": 0.0,
        "has_prediction_open": 0.0,
        "has_prediction_close": 0.0,
        "correct_order": 0.0,
        "think_substantive": 0.0,
        "prediction_substantive": 0.0,
        "think_open_count": 0,
        "think_close_count": 0,
        "prediction_open_count": 0,
        "prediction_close_count": 0,
        "total_format_score": 0.0,
    }

    # Check for <think> opening tag - must appear EXACTLY ONCE (case-sensitive)
    think_open_matches = list(re.finditer(r'<think>', response_text))
    format_scores["think_open_count"] = len(think_open_matches)
    if len(think_open_matches) == 1:
        format_scores["has_think_open"] = 1.0
        think_open_match = think_open_matches[0]
    elif len(think_open_matches) > 1:
        # Multiple occurrences - set total score to 0 and return immediately
        format_scores["total_format_score"] = 0.0
        return format_scores
    else:
        think_open_match = None

    # Check for </think> closing tag - must appear EXACTLY ONCE (case-sensitive)
    think_close_matches = list(re.finditer(r'</think>', response_text))
    format_scores["think_close_count"] = len(think_close_matches)
    if len(think_close_matches) == 1:
        format_scores["has_think_close"] = 1.0
        think_close_match = think_close_matches[0]
    elif len(think_close_matches) > 1:
        # Multiple occurrences - set total score to 0 and return immediately
        format_scores["total_format_score"] = 0.0
        return format_scores
    else:
        think_close_match = None

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

    # Check correct order: <think> → </think> → <|startofprediction|> → <|endofprediction|>
    if think_open_match and think_close_match and prediction_open_match and prediction_close_match:
        if (think_open_match.start() < think_close_match.start() <
            prediction_open_match.start() < prediction_close_match.start()):
            format_scores["correct_order"] = 1.0

    # Check if think section is substantive (at least 100 chars - prompt says "long and detailed, 1000-2000 tokens")
    if think_open_match and think_close_match:
        think_start = think_open_match.end()
        think_end = think_close_match.start()
        think_content = response_text[think_start:think_end].strip()

        if len(think_content) >= 100:  # At least 100 characters
            format_scores["think_substantive"] = 1.0

    # Check if prediction section is substantive (at least 10 chars)
    if prediction_open_match and prediction_close_match:
        prediction_start = prediction_open_match.end()
        prediction_end = prediction_close_match.start()
        prediction_content = response_text[prediction_start:prediction_end].strip()

        if len(prediction_content) >= 10:  # At least 10 characters
            format_scores["prediction_substantive"] = 1.0

    # BINARY ALL-OR-NOTHING SCORING
    # Model must pass ALL 7 checks to get format_score = 1.0
    all_checks_pass = all([
        format_scores["has_think_open"] == 1.0,
        format_scores["has_think_close"] == 1.0,
        format_scores["has_prediction_open"] == 1.0,
        format_scores["has_prediction_close"] == 1.0,
        format_scores["correct_order"] == 1.0,
        format_scores["think_substantive"] == 1.0,
        format_scores["prediction_substantive"] == 1.0,
    ])

    format_scores["total_format_score"] = 1.0 if all_checks_pass else 0.0

    return format_scores


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward combining Binary Information Gain + Binary Format Reward.

    This reward function combines:
    1. Format Reward (BINARY - CHECKED FIRST): Rewards proper document continuation structure
       Model must have ALL of the following to get format_reward=1.0:
       - Exactly one <think> opening tag
       - Exactly one </think> closing tag
       - Exactly one <|startofprediction|> tag
       - Exactly one <|endofprediction|> tag
       - Correct order: <think> → </think> → <|startofprediction|> → <|endofprediction|>
       - Think section has ≥ 100 characters
       - Prediction section has ≥ 10 characters

       If ANY check fails, format_reward=0.0 (all-or-nothing)

    2. Information Gain (IG - BINARIZED, CONDITIONAL ON FORMAT): Measures if thinking helps
       IG = log P(answer | prompt + thinking) - log P(answer | prompt only)
       BINARY THRESHOLD:
       - If IG > 0: ig_reward = 1.0 (thinking helps)
       - If IG ≤ 0: ig_reward = 0.0 (thinking doesn't help)

       CONDITIONAL ACTIVATION:
       - If format_reward = 0: ig_reward = 0 (format must be learned first)
       - If format_reward = 1: ig_reward = binary IG (reward good thinking)

    Final Reward Range: [-0.3, 1.3]
       - Best: format=1, IG>0 → 1.3 (perfect format + helpful thinking)
       - Medium: format=1, IG≤0 → 0.3 (perfect format, thinking doesn't help)
       - Worst: format=0 → -0.3 (bad format, IG ignored)

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

    # ===== COMPONENT 1: FORMAT REWARD (BINARY) - CHECKED FIRST =====

    # Check if response follows the required document continuation format
    # Note: has_partial_solution no longer needed - format is same regardless
    format_scores = check_format_compliance(solution_str)

    # Extract format metrics - now BINARY (0.0 or 1.0 only)
    format_reward = format_scores["total_format_score"]  # Binary: 0.0 or 1.0

    # ===== COMPONENT 2: INFORMATION GAIN REWARD (BINARIZED, CONDITIONAL ON FORMAT) =====

    # STRATEGY: Use Information Gain as primary reward signal
    # IG = log P(answer | prompt + thinking) - log P(answer | prompt only)
    #
    # The IG is NORMALIZED to [-1, 1] range using batch-level statistics:
    # normalized_IG = tanh((raw_IG - batch_mean) / batch_std)
    #
    # BINARIZATION: Threshold at 0
    # - If IG > 0 (thinking helps): reward = +1
    # - If IG ≤ 0 (thinking hurts or neutral): reward = 0
    #
    # CONDITIONAL ACTIVATION: Only give IG reward if format is perfect
    # - If format_reward = 0: ig_reward = 0 (format must be learned first)
    # - If format_reward = 1: ig_reward = binary IG (reward good thinking)
    #
    # Goal: Model must first learn format, then learn good thinking

    # Binarize IG: positive → +1, negative/zero → 0
    # BUT only activate if format is perfect
    if format_reward == 1.0:
        ig_reward = 1.0 if information_gain > 0 else 0.0
    else:
        ig_reward = 0.0  # No IG reward if format is wrong

    # No scaling needed - format reward is already binary {0, 1}
    # Bad format gives 0, good format gives 1
    format_reward_scaled = format_reward  # Binary: 0.0 or 1.0 (no scaling)

    # ===== COMBINE REWARDS =====

    # Weight for format reward (adjust based on importance)
    # With both IG and format binary, this controls format's contribution
    format_weight = 0.3

    # Final combined reward: conditional binary IG + weighted binary format
    # Possible reward values (with conditional IG):
    # - format=1, IG>0: 1.0 + 0.3 = 1.3 (best: correct format + helpful thinking)
    # - format=1, IG≤0: 0.0 + 0.3 = 0.3 (correct format, thinking doesn't help)
    # - format=0: 0.0 - 0.3 = -0.3 (worst: bad format, IG forced to 0)
    final_reward = ig_reward + (format_weight * format_reward_scaled)

    # Note:
    # - Conditional IG: Model must learn format BEFORE getting IG rewards
    # - This creates curriculum: format first, then good thinking
    # - Format reward provides clear signal: perfect format (0.3) vs bad format (-0.3)
    # - IG reward adds bonus (+1.0) only when format is perfect AND thinking helps
    # - GRPO advantages come from variance in IG (within format=1 group)

    # Return detailed dict for logging and analysis
    return {
        "score": float(final_reward),

        # Information Gain metrics
        "information_gain": float(information_gain),  # Normalized [-1, 1]
        "information_gain_raw": float(information_gain_raw) if information_gain_raw is not None else None,  # Original scale
        "ig_reward": float(ig_reward),

        # Format metrics (new tag-based format)
        "format_reward": float(format_reward),  # Binary: 0.0 or 1.0
        "format_reward_scaled": float(format_reward_scaled),  # Binary: -1.0 or 1.0
        "has_think_open": float(format_scores["has_think_open"]),
        "has_think_close": float(format_scores["has_think_close"]),
        "has_prediction_open": float(format_scores["has_prediction_open"]),
        "has_prediction_close": float(format_scores["has_prediction_close"]),
        "correct_order": float(format_scores["correct_order"]),
        "think_substantive": float(format_scores["think_substantive"]),
        "prediction_substantive": float(format_scores["prediction_substantive"]),
        "think_open_count": int(format_scores["think_open_count"]),
        "think_close_count": int(format_scores["think_close_count"]),
        "prediction_open_count": int(format_scores["prediction_open_count"]),
        "prediction_close_count": int(format_scores["prediction_close_count"]),

        # Metadata for analysis
        "solution_percentage": float(solution_percentage),
        "partial_given_len": len(partial_given),
        "remaining_sol_len": len(remaining_sol),

        # Optional metrics (may be None if not computed)
        "policy_confidence": float(policy_confidence) if policy_confidence is not None else None,
        "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
    }
