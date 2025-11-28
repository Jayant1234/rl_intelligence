# OmniMath Curriculum Learning: Reward Logic Trace

## Overview
This document provides a detailed trace of the reward calculation logic in the OmniMath curriculum learning setup with GRPO trainer. The system combines format rewards (binary) with information gain (continuous, conditional), gated at the group level for GRPO.

---

## 1. Core Architecture: Data Flow

```
Dataset Generation (omnimath_curriculum_dataset.py)
    ↓ (provides: problem, partial_solution_given, remaining_solution, ground_truth)
    ↓
Rollout/Generation (vLLM)
    ↓ (generates response with thinking + <|startofprediction|>...<|endofprediction|>)
    ↓
Information Gain Computation (ray_trainer.py: lines 1500-1595)
    ↓ (computes: information_gain_raw, information_gain_normalized)
    ↓
Curriculum Metrics (omnimath_curriculum_metrics.py)
    ↓ (computes: format compliance, group-level gating conditions)
    ↓
Reward Function (omnimath_reward.py: compute_score)
    ↓ (combines: format_reward + conditional ig_reward)
    ↓
Token-Level Scores
    ↓
Advantage Estimation (GRPO: core_algos.py)
    ↓ (groups by uid, normalizes by group std)
    ↓
Policy Loss & Backprop
```

---

## 2. Dataset Layer: Information Preparation

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/examples/data_preprocess/omnimath_curriculum_dataset.py`

### Curriculum Learning Setup

The dataset performs **dynamic per-epoch curriculum randomization**:

```python
def __getitem__(self, idx):
    # 1. Sample random percentage for THIS sample in THIS epoch
    solution_percentage = self.rng.uniform(self.min_percentage, self.max_percentage)
    # Range: 0.0 to 0.8 (configurable)
    
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
```

### Prompt Format

**For training (curriculum prompt)**:
```
Complete the text provided under ### Context by continuing the mathematical solution from exactly where it stops.
Think carefully about the best way to continue. Then write the continuation of the solution and enclose it within <|startofprediction|> and <|endofprediction|> tags.
Do not restate the problem or summarize the context within this continuation.

### Context
[problem]
[partial_solution_given]  # (0-80% of full solution)
```

Expected model response:
```
[thinking/reasoning]
<|startofprediction|>
[continuation - should complete remaining_solution]
<|endofprediction|>
```

### Reward Model Data Structure

Stored in each sample's `batch.non_tensor_batch["reward_model"][i]`:
```python
{
    "style": "custom",
    "ground_truth": ground_truth,              # Final answer in \boxed{} format
    "partial_solution_given": str,             # What was in prompt (0-80%)
    "remaining_solution": str,                 # What model needs to generate
    "full_solution": str,                      # Complete solution (for reference)
}
```

Stored in each sample's `batch.non_tensor_batch["extra_info"][i]`:
```python
{
    "problem": str,
    "solution_percentage_target": float,       # Random target (0.0-0.8)
    "solution_percentage_actual": float,       # Actual percentage achieved
    "num_solution_sentences": int,
    "num_given_sentences": int,
    "num_remaining_sentences": int,
    "current_epoch": int,
    "epoch_seed": int,
}
```

---

## 3. Information Gain Computation

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/verl/trainer/ppo/ray_trainer.py` (lines 1500-1595)

### Why Information Gain?

IG measures whether the model's reasoning (thinking + document format) helps it better predict the correct answer:
```
IG = mean log P(gold_solution | prompt + thinking) - mean log P(gold_solution | baseline)
```

**Using per-token mean** (instead of sequence sum) makes IG independent of gold solution length, which is critical for curriculum learning where `remaining_solution` length varies.

### IG Calculation Process

#### Step 1: Parse Response to Extract Thinking

```python
def _parse_reasoning_from_response(response_text):
    # Find <|startofprediction|> and <|endofprediction|> tags (CASE-SENSITIVE)
    prediction_open_match = re.search(r'<\|startofprediction\|>', response_text)
    prediction_close_match = re.search(r'<\|endofprediction\|>', response_text)
    
    # Extract thinking: everything BEFORE <|startofprediction|>
    if prediction_open_match:
        thinking_text = response_text[:prediction_open_match.start()].strip()
    else:
        thinking_text = response_text.strip()  # No tags = entire response is thinking
    
    # Extract prediction: content between the tags
    if prediction_open_match and prediction_close_match:
        prediction_start = prediction_open_match.end()
        prediction_end = prediction_close_match.start()
        prediction_text = response_text[prediction_start:prediction_end].strip()
    else:
        prediction_text = ""  # No valid tags = empty prediction
    
    return thinking_text, prediction_text
```

#### Step 2: Construct Two Batches

**Batch WITH thinking** (document continuation format):
```
[original prompt with document continuation instructions]
### Context
[problem]
[partial_solution_given]

[thinking_text extracted from response]
<|startofprediction|>
[gold_remaining_solution from dataset]
<|endofprediction|>
```

**Batch WITHOUT thinking** (minimal baseline):
```
Solve the following math problem.

[problem]

[partial_solution_given]

[gold_remaining_solution from dataset]
```

#### Step 3: Compute Log Probabilities

```python
# Compute log probs for gold_solution in both scenarios
log_probs_with = actor_rollout_wg.compute_log_prob(batch_with_reasoning)  # [B, response_length]
log_probs_without = actor_rollout_wg.compute_log_prob(batch_without_reasoning)  # [B, response_length]

# Create response masks (only gold_remaining_solution tokens)
response_masks_with = (responses_with != pad_token_id).float()
response_masks_without = (responses_without != pad_token_id).float()
```

#### Step 4: Calculate Information Gain

```python
# Average log probs over gold solution tokens (per-token mean)
mean_with_reasoning = (log_probs_with * response_masks_with).sum(dim=1) / response_masks_with.sum(dim=1)
mean_without_reasoning = (log_probs_without * response_masks_without).sum(dim=1) / response_masks_without.sum(dim=1)

# Raw IG (original scale)
information_gain_raw = mean_with_reasoning - mean_without_reasoning  # [B]

# Example values:
# Positive IG: thinking helps → mean_with > mean_without
# Negative IG: thinking hurts → mean_with < mean_without
```

#### Step 5: Normalize IG to [0, 1]

```python
# Standardize using batch statistics
ig_std = information_gain_raw.std() + 1e-8
ig_mean = information_gain_raw.mean()

# Apply sigmoid to normalize to [0, 1]
information_gain_normalized = torch.sigmoid((information_gain_raw - ig_mean) / ig_std)

# Store both in batch for later use
batch.batch["information_gain_raw"] = information_gain_raw     # Original scale
batch.batch["information_gain"] = information_gain_normalized  # [0, 1]

# Inject into non_tensor_batch for reward function access
batch.non_tensor_batch["extra_info"][i]["information_gain"] = float(ig_numpy[i])
batch.non_tensor_batch["extra_info"][i]["information_gain_raw"] = float(ig_raw_numpy[i])
```

**Normalization Details**:
- Batch mean becomes 0.5 (sigmoid(0) = 0.5)
- Batch min/max become approximately 0.1-0.9
- Prevents outlier IG values from dominating
- All samples in batch use same normalization parameters

### Logged Metrics

```python
metrics = {
    "curriculum/information_gain_mean": information_gain.mean().item(),
    "curriculum/information_gain_std": information_gain.std().item(),
    "curriculum/information_gain_min": information_gain.min().item(),
    "curriculum/information_gain_max": information_gain.max().item(),
    "curriculum/information_gain_raw_mean": information_gain_raw.mean().item(),
    "curriculum/information_gain_raw_std": information_gain_raw.std().item(),
    "curriculum/information_gain_raw_min": information_gain_raw.min().item(),
    "curriculum/information_gain_raw_max": information_gain_raw.max().item(),
}
```

---

## 4. Format Rewards: startofprediction/endofprediction Tags

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/examples/reward_functions/omnimath_reward.py` (lines 39-116)

### What Are These Tags?

The model must wrap its prediction within special XML-like tags:
```xml
<|startofprediction|>
[model's generated solution continuation]
<|endofprediction|>
```

These tags mark the transition from "thinking/reasoning" to "formal solution" in document continuation format.

### Format Compliance Checking

**Binary All-or-Nothing Scoring**:

```python
def check_format_compliance(response_text: str) -> dict:
    """
    Check if response follows document continuation format.
    BINARY: Must pass ALL 3 checks to get format_score = 1.0
    """
    format_scores = {
        "has_prediction_open": 0.0,      # Check 1: <|startofprediction|> exists (exactly once)
        "has_prediction_close": 0.0,     # Check 2: <|endofprediction|> exists (exactly once)
        "correct_order": 0.0,            # Check 3: Open tag comes before close tag
        "prediction_open_count": 0,      # Diagnostic counter
        "prediction_close_count": 0,     # Diagnostic counter
        "total_format_score": 0.0,
    }
    
    # Check 1: <|startofprediction|> EXACTLY ONCE (case-sensitive)
    prediction_open_matches = list(re.finditer(r'<\|startofprediction\|>', response_text))
    format_scores["prediction_open_count"] = len(prediction_open_matches)
    if len(prediction_open_matches) == 1:
        format_scores["has_prediction_open"] = 1.0
        prediction_open_match = prediction_open_matches[0]
    elif len(prediction_open_matches) > 1:
        # Multiple occurrences: FAIL
        format_scores["total_format_score"] = 0.0
        return format_scores
    else:
        # Zero occurrences: FAIL
        prediction_open_match = None
    
    # Check 2: <|endofprediction|> EXACTLY ONCE (case-sensitive)
    prediction_close_matches = list(re.finditer(r'<\|endofprediction\|>', response_text))
    format_scores["prediction_close_count"] = len(prediction_close_matches)
    if len(prediction_close_matches) == 1:
        format_scores["has_prediction_close"] = 1.0
        prediction_close_match = prediction_close_matches[0]
    elif len(prediction_close_matches) > 1:
        # Multiple occurrences: FAIL
        format_scores["total_format_score"] = 0.0
        return format_scores
    else:
        # Zero occurrences: FAIL
        prediction_close_match = None
    
    # Check 3: Correct order: open before close
    if prediction_open_match and prediction_close_match:
        if prediction_open_match.start() < prediction_close_match.start():
            format_scores["correct_order"] = 1.0
    
    # BINARY: ALL-OR-NOTHING
    all_checks_pass = all([
        format_scores["has_prediction_open"] == 1.0,
        format_scores["has_prediction_close"] == 1.0,
        format_scores["correct_order"] == 1.0,
    ])
    
    format_scores["total_format_score"] = 1.0 if all_checks_pass else 0.0
    
    return format_scores
```

### Examples

**PASS** (format_score = 1.0):
```
[thinking content]
<|startofprediction|>
Therefore, the answer is 42.
<|endofprediction|>
```

**FAIL** (format_score = 0.0):
```
[thinking content]
<|startofprediction|>
The answer is 42.
```
(Missing close tag)

**FAIL** (format_score = 0.0):
```
[thinking content]
<|endofprediction|>
The answer is 42.
<|startofprediction|>
```
(Wrong order)

**FAIL** (format_score = 0.0):
```
[thinking content]
<|startofprediction|>
Part 1
<|endofprediction|>
<|startofprediction|>
Part 2
<|endofprediction|>
```
(Multiple occurrences)

---

## 5. Curriculum Metrics: Group-Level Gating

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/examples/reward_functions/omnimath_curriculum_metrics.py`

### Purpose

Compute group-level conditions to gate Information Gain rewards. This ensures GRPO (which learns from group variance) has meaningful signal.

### Computation Flow

#### 1. Model-Based Metrics (Per-Sample)

```python
for i in range(batch_size):
    # Policy Confidence: Average probability over generated tokens
    valid_old_log_probs = old_log_probs[i, prompt_length:prompt_length + valid_response_len]
    policy_probs = torch.exp(valid_old_log_probs)
    policy_confidence[i] = policy_probs.mean().item()
    
    # KL Divergence: Policy vs Reference
    if ref_log_probs is not None:
        valid_ref_log_probs = ref_log_probs[i, prompt_length:prompt_length + valid_response_len]
        kl_divergence[i] = (valid_old_log_probs - valid_ref_log_probs).sum().item()
```

#### 2. Group-Level Format Checking

For GRPO training, we group samples by `uid` (unique ID):

```python
if group_ids is not None:
    # Collect format scores by group
    group_format_status = defaultdict(list)  # group_id -> list of format scores
    
    for i in range(batch_size):
        group_id = group_ids[i]
        # Get or compute format score
        format_result = check_format_compliance(decoded_responses[i])
        format_score = format_result["total_format_score"]
        group_format_status[group_id].append(format_score)
    
    # Determine if ALL formats passed in each group
    group_all_passed = {
        gid: all(score == 1.0 for score in scores)
        for gid, scores in group_format_status.items()
    }
    
    # Create per-sample array (all samples in same group share status)
    group_all_formats_passed = np.array([
        group_all_passed[group_ids[i]] for i in range(batch_size)
    ], dtype=bool)
```

**Meaning**: 
- If ALL samples in a GRPO group have format_reward = 1.0 → `group_all_formats_passed = True`
- If ANY sample in the group has format_reward = 0.0 → `group_all_formats_passed = False` (for entire group)

#### 3. Group-Level IG Variance Checking

GRPO uses group advantage, so we need meaningful variance:

```python
if group_ids is not None:
    # Extract raw IG values from extra_info
    raw_ig_values = [
        batch.non_tensor_batch["extra_info"][i].get("information_gain_raw", 0.0)
        for i in range(batch_size)
    ]
    
    # Group raw IG values by group_id
    group_ig_values = defaultdict(list)
    for i in range(batch_size):
        group_id = group_ids[i]
        group_ig_values[group_id].append(raw_ig_values[i])
    
    # Compute statistics for each group
    group_ig_stats = {}
    for gid, ig_vals in group_ig_values.items():
        ig_min = min(ig_vals)
        ig_max = max(ig_vals)
        ig_range = ig_max - ig_min
        
        group_ig_stats[gid] = {
            "min": ig_min,
            "max": ig_max,
            "range": ig_range,
            "range_sufficient": ig_range >= 100.0,    # Require range >= 100
            "max_positive": ig_max > 0.0,             # Require at least one positive IG
        }
    
    # Create per-sample arrays (all samples in group share status)
    group_ig_range_sufficient = np.array([
        group_ig_stats[group_ids[i]]["range_sufficient"] for i in range(batch_size)
    ], dtype=bool)
    
    group_max_ig_positive = np.array([
        group_ig_stats[group_ids[i]]["max_positive"] for i in range(batch_size)
    ], dtype=bool)
```

**Criteria Explained**:
- `range >= 100.0`: Group has meaningful IG variance (difference between max and min IG)
- `max > 0.0`: At least one sample in group has positive IG (thinking helps)

### Return Values

```python
return {
    "policy_confidence": np.array([...]),           # Per-sample
    "kl_divergence": np.array([...]),               # Per-sample
    "decoded_responses": np.array([...], dtype=object),  # Per-sample
    "partial_solution_given": np.array([...], dtype=object),  # Per-sample
    "remaining_solution": np.array([...], dtype=object),  # Per-sample
    
    # GROUP-LEVEL GATING (all samples in group have same value)
    "group_all_formats_passed": np.array([bool, ...]),
    "group_ig_range_sufficient": np.array([bool, ...]),
    "group_max_ig_positive": np.array([bool, ...]),
}
```

---

## 6. Final Reward Calculation

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/examples/reward_functions/omnimath_reward.py` (lines 119-307)

### Algorithm: Conditional Reward Combination

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Combines:
    1. FORMAT REWARD (BINARY): Checked first, immediate gating
    2. INFORMATION GAIN REWARD (CONTINUOUS): Conditional on format + group variance
    
    Final Reward Range: [0.0, 1.3]
    """
    
    if not extra_info:
        return {"score": 0.0, "warning": "No extra_info available"}
    
    # ===== COMPONENT 1: FORMAT REWARD (BINARY) =====
    
    format_scores = check_format_compliance(solution_str)
    format_reward = format_scores["total_format_score"]  # 0.0 or 1.0
    
    # ===== COMPONENT 2: INFORMATION GAIN REWARD (CONTINUOUS, CONDITIONAL) =====
    
    information_gain = extra_info.get("information_gain", None)  # Normalized [0, 1]
    
    # Get group-level gating conditions
    group_all_formats_passed = extra_info.get("group_all_formats_passed", True)
    group_ig_range_sufficient = extra_info.get("group_ig_range_sufficient", True)
    group_max_ig_positive = extra_info.get("group_max_ig_positive", True)
    
    # Conditional IG reward: 4 conditions must ALL be met
    if (format_reward == 1.0 and                    # Condition 1: Individual format is perfect
        group_all_formats_passed and                # Condition 2: ALL group members have perfect format
        group_ig_range_sufficient and               # Condition 3: Group has sufficient IG variance (range >= 100)
        group_max_ig_positive):                     # Condition 4: Group has at least one positive IG
        
        ig_reward = information_gain  # Use continuous normalized IG [0, 1]
    else:
        ig_reward = 0.0  # No IG reward if any condition fails
    
    # ===== COMBINE REWARDS =====
    
    format_weight = 0.3
    format_reward_scaled = format_reward  # Binary: 0.0 or 1.0 (no scaling)
    
    final_reward = ig_reward + (format_weight * format_reward_scaled)
    
    return {
        "score": float(final_reward),
        
        # IG metrics
        "information_gain": float(information_gain),
        "information_gain_raw": float(information_gain_raw) if information_gain_raw is not None else None,
        "ig_reward": float(ig_reward),
        
        # Format metrics
        "format_reward": float(format_reward),
        "format_reward_scaled": float(format_reward_scaled),
        "has_prediction_open": float(format_scores["has_prediction_open"]),
        "has_prediction_close": float(format_scores["has_prediction_close"]),
        "correct_order": float(format_scores["correct_order"]),
        
        # Group-level gating
        "group_all_formats_passed": bool(group_all_formats_passed),
        "group_ig_range_sufficient": bool(group_ig_range_sufficient),
        "group_max_ig_positive": bool(group_max_ig_positive),
        
        # Optional metrics
        "policy_confidence": float(policy_confidence) if policy_confidence is not None else None,
        "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
    }
```

### Reward Ranges with Examples

**Best Case** (format=1, all group checks pass, IG=1.0):
```
final_reward = 1.0 + (0.3 * 1.0) = 1.3
```
Perfect format + excellent thinking (thinking helps most)

**Good Case** (format=1, all group checks pass, IG=0.5):
```
final_reward = 0.5 + (0.3 * 1.0) = 0.8
```
Perfect format + average thinking

**Acceptable Case** (format=1, all group checks pass, IG=0.0):
```
final_reward = 0.0 + (0.3 * 1.0) = 0.3
```
Perfect format + poor thinking (thinking doesn't help, but format is learned)

**Failure Case** (format=0 OR any group check fails):
```
final_reward = 0.0 + (0.3 * 0.0) = 0.0
```
Model doesn't get IG reward until format is learned

### Curriculum Learning Strategy

The reward structure enforces a **two-stage curriculum**:

1. **Stage 1: Learn Format (Groups at 0.3 reward)**
   - All samples in group get ig_reward = 0 until ALL format checks pass
   - Format reward alone (0.3) gives initial learning signal
   - Model learns to generate `<|startofprediction|>...<|endofprediction|>` tags

2. **Stage 2: Optimize Thinking with IG (Groups at 0.3-1.3 reward)**
   - Once all group members have perfect format, IG reward activates
   - Group must also have sufficient IG variance (range >= 100)
   - And at least one sample with positive IG (thinking helps)
   - Now model learns continuous IG signal: better thinking → higher reward

---

## 7. Integration with Training Loop

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/verl/trainer/ppo/ray_trainer.py` (lines 1500-1737)

### Execution Order

```
for batch in rollout_batches:
    # 1. Compute log probs (policy and reference)
    old_log_probs = actor_wg.compute_log_probs(batch)
    ref_log_probs = ref_wg.compute_log_probs(batch) if use_ref
    
    # 2. Compute Information Gain (if enabled)
    if compute_information_gain:
        batch_with_reasoning, batch_without_reasoning = _construct_ig_batches_with_gold_solution(batch)
        log_probs_with = actor_wg.compute_log_prob(batch_with_reasoning)
        log_probs_without = actor_wg.compute_log_prob(batch_without_reasoning)
        
        information_gain_raw = (mean_with - mean_without)
        information_gain_normalized = sigmoid((information_gain_raw - mean) / std)
        
        batch.batch["information_gain_raw"] = information_gain_raw
        batch.batch["information_gain"] = information_gain_normalized
        
        # CRITICAL: Inject into extra_info for reward function
        batch.non_tensor_batch["extra_info"][i]["information_gain"] = float(ig_numpy[i])
        batch.non_tensor_batch["extra_info"][i]["information_gain_raw"] = float(ig_raw_numpy[i])
    
    # 3. Compute Custom Curriculum Metrics
    if custom_curriculum_metrics:
        curriculum_metrics_dict = _curriculum_metrics_fn(batch, tokenizer)
        
        # Merge into extra_info (deep copy to avoid shared references)
        for key, values in curriculum_metrics_dict.items():
            for i in range(batch_size):
                extra_info_list[i][key] = values[i]
        
        batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)
    
    # 4. Compute Reward
    reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
    batch.batch["token_level_scores"] = reward_tensor
    
    # 5. Apply KL Penalty
    if use_kl_in_reward:
        data, metrics = apply_kl_penalty(batch, kl_ctrl)
    
    # 6. Compute Advantages (GRPO)
    if adv_estimator == GRPO:
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            index=batch.non_tensor_batch["uid"],           # Grouping key
            norm_adv_by_std_in_grpo=True,                  # Normalize by group std
        )
```

### Token-Level Scores to Advantages

**Important Note**: 
- `token_level_scores` = scalar reward per sample (same across all tokens in response)
- `token_level_rewards` = `token_level_scores * response_mask` (replicated across response tokens)
- GRPO advantage = group-normalized scalar reward
- Advantages are broadcast across response tokens for loss computation

```python
# In ray_trainer.py, line 1684
batch.batch["token_level_scores"] = reward_tensor  # Shape: [B] (scalar rewards)

# In core_algos.py, line 301
scores = token_level_rewards.sum(dim=-1)  # Sum across token dimension to get scalar
# [B, R] → [B]

# Group normalization
for i in range(batch_size):
    id2score[group_id].append(scores[i])

for group_id in id2score:
    id2mean[group_id] = scores_in_group.mean()
    id2std[group_id] = scores_in_group.std()

# Compute advantage (scalar, same for all tokens in response)
for i in range(batch_size):
    advantage[i] = (scores[i] - id2mean[group_id]) / (id2std[group_id] + epsilon)

# Broadcast advantage across response tokens
advantages = advantage.unsqueeze(-1) * response_mask  # [B, R]
```

---

## 8. GRPO-Specific Reward Logic

**File**: `/mnt/c/Users/jayan/Documents/Projects/verl/verl/trainer/ppo/core_algos.py` (lines 265-328)

### What is GRPO?

**Group-wise Relative Policy Optimization**: Instead of using critic values, GRPO compares rewards within groups.

### Group Advantage Computation

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # [B, R] - per-token rewards
    response_mask: torch.Tensor,         # [B, R] - response mask
    index: np.ndarray,                   # [B] - group IDs (uid)
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO by comparing within groups.
    
    For each group:
        a_i = (r_i - mean(group)) / std(group)  [if norm_adv_by_std_in_grpo=True]
        a_i = r_i - mean(group)                 [if norm_adv_by_std_in_grpo=False]
    """
    
    # Step 1: Sum token rewards to get scalar reward per sample
    scores = token_level_rewards.sum(dim=-1)  # [B] (scalar per sample)
    
    # Step 2: Build group statistics
    id2score = defaultdict(list)  # group_id -> list of scalar rewards
    id2mean = {}                  # group_id -> group mean reward
    id2std = {}                   # group_id -> group std reward
    
    for i in range(batch_size):
        group_id = index[i]  # uid of this sample
        id2score[group_id].append(scores[i])
    
    # Step 3: Compute group statistics
    for group_id in id2score:
        if len(id2score[group_id]) == 1:
            # Single sample in group: no variance
            id2mean[group_id] = torch.tensor(0.0)
            id2std[group_id] = torch.tensor(1.0)
        else:
            scores_tensor = torch.stack(id2score[group_id])
            id2mean[group_id] = scores_tensor.mean()
            id2std[group_id] = scores_tensor.std()
    
    # Step 4: Compute advantage (normalize by group)
    for i in range(batch_size):
        group_id = index[i]
        if norm_adv_by_std_in_grpo:
            # Standard GRPO: normalize by std
            scores[i] = (scores[i] - id2mean[group_id]) / (id2std[group_id] + epsilon)
        else:
            # Dr.GRPO: without std normalization
            scores[i] = scores[i] - id2mean[group_id]
    
    # Step 5: Broadcast scalar advantage across response tokens
    advantages = scores.unsqueeze(-1) * response_mask  # [B, R]
    
    return advantages, advantages  # Return same for both advantages and returns
```

### Example with Curriculum Rewards

Suppose we have a GRPO group with 5 samples:

```
Sample 1: format=1, ig=0.8 → reward = 0.8 + 0.3 = 1.1
Sample 2: format=1, ig=0.6 → reward = 0.6 + 0.3 = 0.9
Sample 3: format=0, ig=X  → reward = 0.0 + 0.0 = 0.0  (format failed)
Sample 4: format=1, ig=0.7 → reward = 0.7 + 0.3 = 1.0
Sample 5: format=1, ig=0.5 → reward = 0.5 + 0.3 = 0.8

Group all_formats_passed? NO (Sample 3 failed)
→ All samples get ig_reward = 0 regardless of IG
→ Adjusted rewards:
Sample 1: 0 + 0.3 = 0.3
Sample 2: 0 + 0.3 = 0.3
Sample 3: 0 + 0.0 = 0.0
Sample 4: 0 + 0.3 = 0.3
Sample 5: 0 + 0.3 = 0.3

Group mean = 0.24
Group std = 0.0952

Advantages (normalized):
Sample 1: (0.3 - 0.24) / 0.0952 = +0.63
Sample 2: (0.3 - 0.24) / 0.0952 = +0.63
Sample 3: (0.0 - 0.24) / 0.0952 = -2.52
Sample 4: (0.3 - 0.24) / 0.0952 = +0.63
Sample 5: (0.3 - 0.24) / 0.0952 = +0.63
```

**Curriculum Effect**: 
- Sample 3 is penalized harder (negative advantage)
- Other samples get positive advantage
- Model learns to fix format issues to unlock IG rewards

---

## 9. Flow Summary Diagram

```
Generation Response
      ↓
      Text: "Let me solve step by step... <|startofprediction|>42<|endofprediction|>"
      ↓
_parse_reasoning_from_response()
      ├→ thinking_text = "Let me solve step by step..."
      └→ prediction_text = "42"
      ↓
[INFORMATION GAIN]
_construct_ig_batches_with_gold_solution()
      ├→ batch_with_thinking: [instructions + context + thinking + gold]
      └→ batch_without_thinking: [simple instruction + problem + gold]
      ↓
compute_log_prob() on both batches
      ├→ mean log P(gold | with thinking)
      └→ mean log P(gold | without thinking)
      ↓
information_gain_raw = mean_with - mean_without
information_gain_normalized = sigmoid((raw - mean) / std)
      ↓
[CURRICULUM METRICS]
compute_omnimath_curriculum_metrics()
      ├→ decoded_responses = decode(response_tokens)
      ├→ check_format_compliance(decoded_responses)
      │   ├→ has_prediction_open? (exactly 1 × <|startofprediction|>)
      │   ├→ has_prediction_close? (exactly 1 × <|endofprediction|>)
      │   └→ correct_order? (open before close)
      │   └→ format_reward = 1.0 if ALL pass else 0.0
      │
      ├→ group_all_formats_passed = all([sample.format_reward for sample in group])
      ├→ group_ig_range_sufficient = (max(group.ig) - min(group.ig) >= 100)
      └→ group_max_ig_positive = max(group.ig) > 0.0
      ↓
[REWARD CALCULATION]
compute_score()
      ├→ Check condition 1: format_reward == 1.0
      ├→ Check condition 2: group_all_formats_passed
      ├→ Check condition 3: group_ig_range_sufficient
      ├→ Check condition 4: group_max_ig_positive
      │
      ├→ IF all conditions met:
      │   ig_reward = information_gain_normalized [0, 1]
      │ ELSE:
      │   ig_reward = 0.0
      │
      └→ final_reward = ig_reward + (0.3 × format_reward)
            Range: [0.0, 1.3]
      ↓
batch.batch["token_level_scores"] = reward_scalar
      ↓
[ADVANTAGE ESTIMATION - GRPO]
compute_grpo_outcome_advantage()
      ├→ For each group:
      │   group_mean = mean(group.rewards)
      │   group_std = std(group.rewards)
      │   
      ├→ For each sample in group:
      │   advantage = (reward - group_mean) / (group_std + eps)
      │   
      └→ advantage.broadcast(response_tokens) * response_mask
      ↓
Policy Loss & Backprop
```

---

## 10. Key Takeaways

### startofprediction/endofprediction Tags

1. **Purpose**: Mark the boundary between "thinking" and "formal solution" in document continuation format
2. **Format Check**: Must appear exactly once each, in correct order (open before close)
3. **Case-Sensitive**: `<|startofprediction|>` not `<|StartOfPrediction|>`
4. **Binary Reward**: Model either passes all 3 checks (1.0) or fails (0.0) - no partial credit

### Information Gain

1. **Definition**: Measures if thinking helps predict the gold solution compared to baseline
2. **Computation**: Uses per-token average, making it independent of solution length
3. **Normalization**: Batch-level sigmoid normalization to [0, 1]
4. **Conditional**: Only applied if format conditions and group variance conditions are met

### Group-Level Gating

1. **Format Gating**: If ANY group member has bad format, ALL get ig_reward = 0
2. **Variance Gating**: If group IG range < 100 or max IG ≤ 0, ALL get ig_reward = 0
3. **Curriculum Effect**: Forces format learning first, then IG optimization

### GRPO-Specific

1. **Advantage = Within-Group Comparison**: Not policy vs critic, but sample vs group mean
2. **Standard Deviation Normalization**: Scales advantage by group std for robustness
3. **Scalar Rewards**: Curriculum rewards are scalars (same across all response tokens)
4. **Broadcast**: Scalar advantage is broadcast across response tokens for loss computation

### Reward Flow

```
Curriculum Reward (scalar [0.0, 1.3])
    ↓ (summed across token dimension in GRPO)
Outcome Reward per Sample
    ↓ (normalized by group mean/std)
Advantage per Sample
    ↓ (broadcast across response tokens)
Token-Level Advantage [B, R]
    ↓ (multiplied by token log prob difference)
Policy Loss
```

