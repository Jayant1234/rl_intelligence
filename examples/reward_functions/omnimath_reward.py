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

    Expected format (NOTE: <think> is added by chat template, NOT checked):
        [thinking content - at least 100 chars]
        </think>
        <|startofprediction|>
        [continuation - at least 10 chars]
        <|endofprediction|>

    The chat template adds <think> after the assistant tag, so we DON'T check for it.
    We only check that the model properly closes it and uses prediction tags.

    Args:
        response_text: The model's generated response (does NOT include prompt or chat template tags)
        has_partial_solution: DEPRECATED - no longer used (format is same regardless)

    Returns:
        dict with format compliance scores and breakdown
    """
    format_scores = {
        "has_think_open": 1.0,  # Always 1.0 since chat template adds it
        "has_think_close": 0.0,
        "has_prediction_open": 0.0,
        "has_prediction_close": 0.0,
        "correct_order": 0.0,
        "think_substantive": 0.0,
        "prediction_substantive": 0.0,
        "think_open_count": 0,  # Not checked (added by chat template)
        "think_close_count": 0,
        "prediction_open_count": 0,
        "prediction_close_count": 0,
        "total_format_score": 0.0,
    }

    # NOTE: We DON'T check for <think> opening tag since chat template adds it
    # Assume it exists at position 0 (added by chat template before response)
    think_open_match = None  # Not checked

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

    # Check correct order: </think> → <|startofprediction|> → <|endofprediction|>
    # (Note: <think> is in chat template at position 0, so we don't check it)
    if think_close_match and prediction_open_match and prediction_close_match:
        if (think_close_match.start() < prediction_open_match.start() < prediction_close_match.start()):
            format_scores["correct_order"] = 1.0

    # Check if think section is substantive (at least 100 chars)
    # Since <think> is in the chat template, thinking content is from start of response to </think>
    if think_close_match:
        think_start = 0  # Response starts with thinking content (after chat template's <think>)
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
    # Model must pass ALL 6 checks to get format_score = 1.0
    # (Note: has_think_open is always 1.0 since chat template adds it, so we don't check it)
    all_checks_pass = all([
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
       - Exactly one </think> closing tag (opening tag added by chat template)
       - Exactly one <|startofprediction|> tag
       - Exactly one <|endofprediction|> tag
       - Correct order: </think> → <|startofprediction|> → <|endofprediction|>
       - Think section (from start to </think>) has ≥ 100 characters
       - Prediction section (between prediction tags) has ≥ 10 characters

       If ANY check fails, format_reward=0.0 (all-or-nothing)

       NOTE: The <think> opening tag is added by the chat template, so we don't check for it.

    2. Information Gain (IG - BINARIZED, CONDITIONAL ON GROUP FORMAT): Measures if thinking helps
       IG = log P(answer | prompt + thinking) - log P(answer | prompt only)
       BINARY THRESHOLD:
       - If IG > 0.5 AND raw_IG > 0: ig_reward = 1.0 (thinking helps significantly)
       - Otherwise: ig_reward = 0.0 (thinking doesn't help enough)

       CONDITIONAL ACTIVATION (GROUP-LEVEL):
       - If format_reward = 0: ig_reward = 0 (individual format failed)
       - If ANY sample in GRPO group has format_reward = 0: ig_reward = 0 for ALL in group
       - If format_reward = 1 AND all group formats passed: ig_reward = binary IG (reward good thinking)

    Final Reward Range: [0.0, 1.3] (no negative rewards)
       - Best: format=1, group_all_passed, IG>0.5, raw_IG>0 → 1.3 (perfect format + top thinking)
       - Medium: format=1, group_all_passed, IG≤0.5 or raw_IG≤0 → 0.3 (perfect format, thinking not top tier)
       - Worst: format=0 OR group has format failure → 0.0 (bad format or group contamination)

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
    # BINARIZATION: Threshold at 0.5 (top performers)
    # - If IG > 0.5 AND raw_IG > 0 (thinking helps significantly): reward = +1
    # - Otherwise: reward = 0
    #
    # CONDITIONAL ACTIVATION (GROUP-LEVEL): Only give IG reward if ALL samples in the
    # GRPO group have perfect format
    # - If format_reward = 0: ig_reward = 0 (format must be learned first)
    # - If group has ANY format failure: ig_reward = 0 for ALL in group
    # - If format_reward = 1 AND group_all_formats_passed: ig_reward = binary IG (reward good thinking)
    #
    # Goal: Model must first learn format (group-level), then learn good thinking

    # Get group-level format status
    group_all_formats_passed = extra_info.get("group_all_formats_passed", True)

    # Binarize IG: only reward top performers with truly helpful thinking
    # Requires:
    # 1. Individual format is perfect (format_reward = 1.0)
    # 2. ALL formats in the group are perfect (group_all_formats_passed = True)
    # 3. Normalized IG > 0.5 (top ~30% of batch, significantly helpful)
    # 4. Raw IG > 0 (thinking actually helps, not just relatively better)
    if format_reward == 1.0 and group_all_formats_passed:
        ig_reward = 1.0 if (information_gain > 0.5 and information_gain_raw > 0) else 0.0
    else:
        ig_reward = 0.0  # No IG reward if format wrong or group has any format failure

    # No scaling needed - format reward is already binary {0, 1}
    # Bad format gives 0, good format gives 1
    format_reward_scaled = format_reward  # Binary: 0.0 or 1.0 (no scaling)

    # ===== COMBINE REWARDS =====

    # Weight for format reward (adjust based on importance)
    # With both IG and format binary, this controls format's contribution
    format_weight = 0.3

    # Final combined reward: conditional binary IG + weighted binary format
    # Possible reward values (with conditional IG, no negative rewards):
    # - format=1, IG>0: 1.0 + 0.3*1 = 1.3 (best: correct format + helpful thinking)
    # - format=1, IG≤0: 0.0 + 0.3*1 = 0.3 (correct format, thinking doesn't help)
    # - format=0: 0.0 + 0.3*0 = 0.0 (worst: bad format, IG forced to 0)
    final_reward = ig_reward + (format_weight * format_reward_scaled)

    # Note:
    # - Conditional IG (GROUP-LEVEL): Model must learn format for ENTIRE GROUP before getting IG rewards
    # - This creates curriculum: format first (all in group), then good thinking
    # - Format reward provides clear signal: perfect format (0.3) vs bad format (0.0)
    # - IG reward adds bonus (+1.0) only when:
    #   1. Individual format is perfect (format_reward = 1.0)
    #   2. ALL group members have perfect format (group_all_formats_passed = True)
    #   3. Thinking is significantly helpful (IG > 0.5, top ~30% of batch)
    #   4. Thinking actually helps (raw_IG > 0, not just relatively better)
    # - GRPO advantages come from variance in IG (within format=1 groups that fully passed)
    # - No negative rewards: worst case is 0.0, not negative
    # - Group-level gating prevents format failures from contaminating IG learning

    # Return detailed dict for logging and analysis
    return {
        "score": float(final_reward),

        # Information Gain metrics
        "information_gain": float(information_gain),  # Normalized [-1, 1]
        "information_gain_raw": float(information_gain_raw) if information_gain_raw is not None else None,  # Original scale
        "ig_reward": float(ig_reward),

        # Format metrics (new tag-based format)
        "format_reward": float(format_reward),  # Binary: 0.0 or 1.0
        "format_reward_scaled": float(format_reward_scaled),  # Binary: 0.0 or 1.0 (no scaling)
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

        # Group-level format gating (NEW)
        "group_all_formats_passed": bool(group_all_formats_passed),

        # Optional metrics (may be None if not computed)
        "policy_confidence": float(policy_confidence) if policy_confidence is not None else None,
        "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
    }
