# OmniMath Curriculum Learning - Technical Details

This document contains comprehensive technical information about the OmniMath Curriculum Learning implementation, including architecture, development history, metrics interpretation, and troubleshooting.

**For a quick start guide, see [README.md](README.md)**

---

## Table of Contents

1. [Architecture](#architecture)
2. [Design Decisions](#design-decisions)
3. [Development History](#development-history)
4. [Metrics Guide](#metrics-guide)
5. [Troubleshooting](#troubleshooting)
6. [Extension Points](#extension-points)

---

# Architecture

## System Overview

The OmniMath Curriculum Learning system combines curriculum learning with Information Gain (IG) to train mathematical reasoning models. The key innovation is measuring **reasoning quality** by testing whether generated reasoning helps predict correct answers.

## Complete Data Flow

```
1. DATASET LOADING
   ├─ OmniMathCurriculumDataset loads parquet files
   ├─ set_epoch(epoch) called at start of each epoch
   ├─ RNG reseeded: base_seed + epoch * 1000 + worker_id
   └─ For each sample:
      ├─ Sample random % ∈ [0%, 80%]
      ├─ Split solution into sentences
      ├─ Take first N sentences (based on %)
      ├─ Create prompt with partial solution
      └─ Return: prompt, partial_given, remaining_solution

2. ROLLOUT PHASE
   ├─ vLLM/SGLang generates responses
   └─ Expected format:
      **Reasoning:**
      [model's step-by-step reasoning]
      **Remaining Solution:**
      [model's answer attempt with \boxed{}]

3. LOG PROBABILITY COMPUTATION
   ├─ Policy model: Compute old_log_probs [B, R]
   └─ Reference model: Compute ref_log_prob [B, R]

4. INFORMATION GAIN COMPUTATION
   ├─ Parse reasoning from generated responses
   ├─ Construct two batches:
   │  ├─ WITH reasoning: prompt + reasoning + gold_solution
   │  └─ WITHOUT reasoning: prompt + gold_solution
   ├─ Compute log P(gold | prompt + reasoning)
   ├─ Compute log P(gold | prompt)
   ├─ IG_raw = log P(gold | with) - log P(gold | without)
   ├─ Normalize: IG = tanh((IG_raw - mean) / std) → [-1, 1]
   └─ Inject into batch.non_tensor_batch["extra_info"]

5. REWARD COMPUTATION
   ├─ Extract IG from extra_info
   ├─ Check format compliance:
   │  ├─ Exactly one **Reasoning:** header?
   │  ├─ Exactly one **Solution:** header?
   │  ├─ Exactly one \boxed{} answer?
   │  ├─ Correct order (reasoning → solution → boxed)?
   │  └─ Substantive content (reasoning ≥20 chars, solution ≥10 chars)?
   ├─ Combine: final_reward = IG + (0.3 × format_reward)
   └─ Return detailed metrics

6. GRPO UPDATE
   ├─ Group samples by prompt
   ├─ Compute advantages relative to group mean
   └─ Update policy to maximize advantages
```

## Key Components

### 1. Curriculum Dataset
**File:** `examples/data_preprocess/omnimath_curriculum_dataset.py`

**Key Method:**
```python
def __getitem__(self, idx):
    # Sample percentage uniformly
    percentage = rng.uniform(min_percentage, max_percentage)

    # Split solution into sentences
    sentences = re.split(r'(?<=[.!?])\s+', full_solution)

    # Take first N sentences
    num_sentences = max(1, int(len(sentences) * percentage))
    partial = ' '.join(sentences[:num_sentences])
    remaining = ' '.join(sentences[num_sentences:])

    # Create prompt with partial solution
    prompt = create_curriculum_prompt(problem, partial, has_partial=percentage > 0)

    return {
        "prompt": prompt,
        "partial_solution_given": partial,
        "remaining_solution": remaining  # Used for IG computation
    }
```

### 2. Information Gain Computation
**File:** `verl/trainer/ppo/ray_trainer.py` (lines 1493-1584)

**Why IG Uses Gold Solution:**

**❌ Alternative Approach (WRONG):**
```
IG = log P(generated_response | prompt + CoT)
   - log P(generated_response | prompt)
```
- **Problem:** Rewards confidence in ANY response, even incorrect ones
- Model can be highly confident in wrong answers
- No learning signal for correctness

**✅ Our Approach (CORRECT):**
```
IG = log P(gold_solution | prompt + reasoning)
   - log P(gold_solution | prompt)
```
- **Measures:** "Does reasoning help predict the CORRECT answer?"
- Positive IG → reasoning moves toward correct solution
- Negative IG → reasoning confuses or misleads
- Strong learning signal for reasoning quality

**Implementation:**
```python
# 1. Parse reasoning from generated response
reasoning_text = _parse_reasoning_from_response(response)

# 2. Construct two prompts
prompt_with = problem + partial + "**Reasoning:**" + reasoning_text +
              "**Remaining Solution:**" + gold_solution

prompt_without = problem + partial + "**Remaining Solution:**" + gold_solution

# 3. Compute log probs
log_prob_with = policy.compute_log_prob(tokenize(prompt_with))
log_prob_without = policy.compute_log_prob(tokenize(prompt_without))

# 4. Sum over gold solution tokens
sum_with = (log_prob_with * response_mask).sum()
sum_without = (log_prob_without * response_mask).sum()

# 5. Calculate IG
IG_raw = sum_with - sum_without

# 6. Normalize to [-1, 1]
IG = tanh((IG_raw - mean(IG_raw)) / std(IG_raw))
```

**Why Batch Normalization:**
- Raw IG scale varies with model size, solution length, training progress
- Normalization ensures consistent scale → stable gradients
- Allows format reward to contribute meaningfully
- Using tanh: bounds to [-1, 1], preserves ordering, smooth gradients

### 3. Format Reward
**File:** `examples/reward_functions/omnimath_reward.py`

**Format Requirements:**
1. **Exactly ONE** `**Reasoning:**` header (multiple = fail)
2. **Exactly ONE** `**Solution:**` or `**Remaining Solution:**` header
3. **Exactly ONE** `\boxed{answer}`
4. Correct order: Reasoning → Solution → Boxed answer
5. Substantive content: reasoning ≥20 chars, solution ≥10 chars

**Scoring:**
```python
total_format_score = (
    has_reasoning_header +      # 1.0 if exactly 1, else 0.0
    has_solution_header +       # 1.0 if exactly 1, else 0.0
    has_boxed_answer +          # 1.0 if exactly 1, else 0.0
    correct_order +             # 1.0 if proper order, else 0.0
    reasoning_substantive +     # 1.0 if ≥20 chars, else 0.0
    solution_substantive        # 1.0 if ≥10 chars, else 0.0
) / 6.0  # Normalize to [0, 1]
```

**Why Format Reward Matters:**
- **Provides within-group variance:** GRPO needs variance within groups. If all samples have identical IG, advantages = 0 (no learning). Format reward differentiates samples.
- **Quick wins:** Format is easier to learn than reasoning → early training signal
- **Improves usability:** Structured outputs easier to parse and evaluate

### 4. Combined Reward

**Formula:**
```python
final_reward = IG_normalized + (0.3 × format_reward)
# Default range: [-1, 1.3]
```

**Why This Combination:**
- IG is primary signal (weight = 1.0) for reasoning quality
- Format is secondary (weight = 0.3) for structure
- Format learns quickly → early epochs
- IG improves gradually → long-term quality
- Combined provides continuous learning signal

---

# Design Decisions

## Why Curriculum in Dataset (Not Reward)?

**❌ Alternative: Curriculum in Reward Function**
```python
# Sample percentage in dataset (always 0-80%)
# Apply curriculum multiplier in reward
reward = base_reward * curriculum_multiplier(percentage)
```
**Problems:**
- Tight coupling between curriculum and reward
- Hard to modify curriculum strategy
- Curriculum logic spread across codebase

**✅ Our Approach: Curriculum in Dataset**
```python
# Dataset samples percentage and creates prompt
# Reward function is curriculum-agnostic
reward = f(response, gold_solution)  # No curriculum knowledge
```
**Benefits:**
- Clean separation of concerns
- Easy to modify curriculum (one file)
- Reward function reusable for non-curriculum tasks
- Simpler debugging

## Why Per-Epoch Re-randomization?

**❌ Alternative: Fixed Random Assignment**
```python
# Each sample always gets same percentage
percentage[idx] = fixed_random_value[idx]
```
**Problems:**
- Model memorizes percentage for each sample
- No exploration of different curriculum levels
- Overfitting risk

**✅ Our Approach: Per-Epoch Re-randomization**
```python
def set_epoch(epoch):
    self.rng.seed(base_seed + epoch * 1000 + worker_id)
    # Each epoch, re-sample percentage for every sample
```
**Benefits:**
- Same sample sees different curriculum each epoch
- Forces model to handle all curriculum levels
- Better generalization
- Maintains reproducibility (deterministic seed)

## Why GRPO Instead of PPO?

**GRPO (Group Relative Policy Optimization):**
- Groups samples by prompt
- Computes advantages relative to group mean
- No value network, no GAE (Generalized Advantage Estimation)

**Benefits for Math Reasoning:**
1. **Natural grouping:** Same problem with multiple reasoning attempts
2. **Relative comparison:** "Which reasoning is better for THIS problem?"
3. **Reduced variance:** Group statistics more stable than individual baselines
4. **Simpler implementation:** Fewer hyperparameters, no value network

**Why Format Reward is Critical for GRPO:**
- GRPO requires within-group variance for non-zero advantages
- If all samples in a group have identical IG → no learning signal
- Format reward provides differentiation even when IG is similar

---

# Development History

## Timeline

**Development Period:** November 6-7, 2025 (~22 hours)
**Total Commits:** 10 custom commits
**Lines Added:** 2,058 lines
**Files Created:** 8 files

## Commit History (Chronological)

### 1. [6aaaee1d] - Nov 6, 15:34 - "RDU changes first draft"
**Initial Implementation**

**Created:**
- Complete curriculum learning system
- Dataset with curriculum logic
- Format reward system
- OLD IG formula (later redesigned)
- Training configuration
- Documentation (RDU_update.md)

**Known Issues (fixed later):**
- Wrong IG formula
- No normalization
- Reward computed before IG

### 2. [de9acc49] - Nov 6, 16:14 - "some arguments added to yaml"
**Fixed Hydra Struct Mode Compatibility**

**Problem:** Trainer enables `OmegaConf.set_struct(config, True)` which prevents adding new keys. CLI overrides like `--data.curriculum.min_percentage=0.0` would fail with KeyError.

**Solution:** Added curriculum structure to base config with null defaults.

### 3. [68ab3910] - Nov 6, 16:22 - "compute information gain added to ppo yaml"
**Added IG Feature Flag**

Added `compute_information_gain: false` to base config, allowing experiments to enable IG via config override.

### 4. [828fb990] - Nov 6, 18:42 - "moved the reward after the IG..."
**Critical Architectural Change**

**Problem:** Reward function couldn't access IG because it was computed AFTER rewards.

**Solution:** Reordered computation:
1. Compute log probs
2. **Compute IG**
3. **Inject IG into extra_info**
4. Compute rewards (can now access IG)

### 5. [54915f2f] - Nov 6, 21:13 - "fixed the dtype for ig in batch to np from lst"
**Fixed DataProto Serialization**

**Problem:** `extra_info` was left as Python list, breaking DataProto's `.chunk()` method in distributed training.

**Solution:** Convert to numpy array after modifications:
```python
extra_info_list = list(batch.non_tensor_batch["extra_info"])
# ... modify list ...
batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)
```

### 6. [3fc646b3] - Nov 6, 23:02 - "IG normalized since it was off the charts"
**Added Batch Normalization**

**Problem:** Raw IG values extremely large ([-50, +50]), causing unstable training and dominating format reward.

**Solution:** Batch-level normalization with tanh:
```python
IG = tanh((IG_raw - mean(IG_raw)) / std(IG_raw))
```

### 7. [9802e6fd] - Nov 7, 02:01 - "improved prompts and debugged IG calculation"
**MAJOR REDESIGN - New IG Formula** ⭐

**Critical Change:** Complete redesign of IG formula

**OLD (WRONG):**
```
IG = log P(generated_response | prompt + CoT)
   - log P(generated_response | prompt)
```

**NEW (CORRECT):**
```
IG = log P(gold_solution | prompt + reasoning)
   - log P(gold_solution | prompt)
```

**Added:**
- 3 helper methods (parse reasoning, construct IG batches)
- 445 lines of new IG computation logic
- Improved prompts with explicit two-section format

### 8. [a8d817f4] - Nov 7, 02:08 - "responses were missing..."
**Fixed Missing Response Tensor**

**Problem:** IG batch construction wasn't creating the `responses` tensor required by `compute_log_prob()`.

**Solution:** Extract gold solution tokens and create proper response tensors with padding.

### 9. [3ddfe4b5] - Nov 7, 02:26 - "shape mismatch due to IG response calc and masks?"
**Fixed Shape Mismatch**

**Problem:** `log_probs` shape [B, response_length] didn't match `response_mask` shape [B, full_seq_length].

**Solution:** Derive response mask from responses tensor (correct shape):
```python
response_mask = (responses != tokenizer.pad_token_id).float()
```

### 10. [f42a171c] - Nov 7, 13:00 - "format rewards logged and improved"
**Enhanced Format Reward** ✅ Production Ready

**Improvements:**
- Enforce exactly ONE occurrence of each header
- Better order checking (reasoning → solution → boxed)
- Comprehensive metric logging (mean, std, min, max)
- Added header count metrics

---

# Metrics Guide

## Information Gain Metrics

### curriculum/information_gain_mean
**Range:** [-1, 1]
**Target:** Positive and increasing

**What it measures:** Average normalized IG across batch
**Interpretation:**
- **Positive:** Model's reasoning helps predict correct answers
- **Negative:** Model's reasoning confuses or misleads
- **Near zero:** Reasoning doesn't help (early training)
- **Increasing:** Model learning better reasoning

**Expected progression:**
- Epochs 0-2: -0.1 to 0.1 (random)
- Epochs 3-10: 0.2 to 0.4 (improving)
- Epochs 10+: 0.5 to 0.7 (high quality)

### curriculum/information_gain_std
**Range:** [0, ~0.7]
**Target:** 0.3 to 0.7

**What it measures:** Variance in IG across batch
**Interpretation:**
- **Higher:** More differentiation between samples (better for GRPO)
- **Lower:** All samples similar (weaker learning signal)
- **Too high (>0.8):** Possible instability

**Troubleshooting:**
- If too low (<0.2): Format reward may dominate, increase IG weight
- If too high (>0.8): Reduce batch size or check for outliers

### curriculum/information_gain_raw_mean
**Range:** Unbounded
**Use:** Analysis and debugging

**What it measures:** Raw IG before normalization
**Interpretation:**
- Scale varies with model size and solution length
- Positive trend still indicates improvement
- Use to verify normalization is working correctly

## Format Metrics

### format/format_reward_mean
**Range:** [0, 1]
**Target:** > 0.8

**What it measures:** Overall format compliance
**Interpretation:**
- **1.0:** Perfect format (all 6 checks pass)
- **0.5-0.8:** Partial compliance (some checks pass)
- **< 0.5:** Poor format (most checks fail)

**Expected progression:**
- Epoch 0-1: 0.3 to 0.5 (learning)
- Epoch 2-5: 0.7 to 0.9 (good)
- Epoch 5+: 0.9 to 1.0 (near-perfect)

### format/has_reasoning_header_mean
**Range:** [0, 1]
**Target:** 1.0

**What it measures:** Fraction of samples with exactly one `**Reasoning:**` header
**Interpretation:**
- **1.0:** All samples have correct reasoning header
- **< 0.8:** Model struggling with format

**Troubleshooting:**
- If stuck < 0.5: Check prompt clarity, increase format weight
- If > 1.0 in count metric: Model repeating headers (correctly scored 0)

### format/has_solution_header_mean
**Range:** [0, 1]
**Target:** 1.0

**What it measures:** Fraction with exactly one `**Solution:**` or `**Remaining Solution:**` header

### format/has_boxed_answer_mean
**Range:** [0, 1]
**Target:** 1.0

**What it measures:** Fraction with exactly one `\boxed{answer}`
**Interpretation:**
- Critical for answer extraction
- Usually learns quickly (epoch 2-3)

### format/correct_order_mean
**Range:** [0, 1]
**Target:** 1.0

**What it measures:** Fraction with correct section order (reasoning → solution → boxed)
**Interpretation:**
- **1.0:** All samples properly ordered
- **< 0.9:** Some samples have mixed-up sections

### format/reasoning_substantive_mean
**Range:** [0, 1]
**Target:** > 0.9

**What it measures:** Fraction with meaningful reasoning (≥20 characters)
**Interpretation:**
- **< 0.5:** Model writing trivial reasoning
- **> 0.9:** Model providing substantial reasoning

## Reward Metrics

### reward/rewards_mean
**Range:** [-1, 1.3]
**Target:** Increasing

**What it measures:** Combined final reward (IG + 0.3 × format)
**Interpretation:**
- **Increasing:** Overall training progress
- **Plateauing:** May need hyperparameter adjustment
- **Decreasing:** Check for issues (curriculum too hard, etc.)

---

# Troubleshooting

## Training Issues

### IG Mean Near Zero, Not Improving

**Symptoms:**
- `curriculum/information_gain_mean` stays around 0.0 after several epochs
- `curriculum/information_gain_std` very low (< 0.2)

**Possible Causes:**
1. **Model not learning from reasoning:**
   - Check if format_reward dominates
   - IG signal too weak compared to format

2. **Data issues:**
   - Gold solutions not accessible
   - Partial solutions too long/short

**Solutions:**
```python
# Increase IG weight
final_reward = 2.0 * IG + 0.3 * format_reward

# Or reduce format weight
final_reward = IG + 0.1 * format_reward

# Check curriculum range
data.curriculum.max_percentage = 0.5  # Reduce max to make task harder
```

### Format Reward High But IG Still Low

**Symptoms:**
- `format/format_reward_mean` > 0.9
- `curriculum/information_gain_mean` < 0.2

**Diagnosis:** Model "gaming" format without improving reasoning

**Solutions:**
1. **Increase IG weight:**
   ```python
   final_reward = 2.0 * IG + 0.3 * format
   ```

2. **Add content-based rewards** (not just structure):
   ```python
   # Check if reasoning mentions key problem elements
   # Check if solution steps are logically connected
   ```

3. **Gradually reduce format weight:**
   ```python
   # Epoch 0-5: weight = 0.3
   # Epoch 5-10: weight = 0.2
   # Epoch 10+: weight = 0.1
   ```

### IG Values Exploding

**Symptoms:**
- `curriculum/information_gain_raw_mean` > 100 or < -100
- Training unstable, loss spikes

**Diagnosis:** Normalization issues

**Check:**
```python
# Verify normalization is enabled
assert "information_gain" in batch.batch  # Normalized version
assert "information_gain_raw" in batch.batch  # Raw version

# Check batch size (too small → unstable stats)
# Minimum recommended: 64
```

**Solutions:**
- Increase batch size for more stable normalization
- Check for inf/nan in log probs
- Verify gold solutions are valid

### Shape Mismatch Errors

**Error:**
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (512)
```

**Location:** IG computation block

**Cause:** Response mask shape doesn't match log_probs shape

**Solution:**
```python
# Ensure response_mask is derived from responses tensor
responses = batch.batch["responses"]  # [B, R]
response_mask = (responses != tokenizer.pad_token_id).float()  # [B, R]

# NOT from input_ids (wrong length)
```

## Performance Issues

### IG Computation Very Slow

**Symptoms:**
- IG computation takes 2-3x longer than rollout
- Training throughput drops significantly

**Why:** IG requires 2 additional forward passes per batch

**Optimizations:**

**1. Cache WITHOUT reasoning batch:**
```python
# WITHOUT reasoning batch can be precomputed
# Only depends on prompt + gold_solution (no generated reasoning)
# Cache and reuse across training
```

**2. Compute IG on subset:**
```python
# Compute IG on random 50% of samples
# Use format reward only for others
if random.random() < 0.5:
    compute_ig(sample)
```

**3. Async IG computation:**
```python
# Overlap IG computation with next rollout
# Use background thread pool
```

**4. Reduce IG batch size:**
```python
# Compute IG in smaller sub-batches
# Trade speed for memory
```

### Out of Memory During IG

**Symptoms:**
- OOM error during IG computation
- Works with smaller batches

**Cause:** 2 additional batches (WITH and WITHOUT reasoning) in memory simultaneously

**Solutions:**

**1. Reduce batch size:**
```python
# IG batch size can be smaller than rollout batch
ig_batch_size = 32  # Even if rollout uses 128
```

**2. Gradient checkpointing:**
```python
# Enable for IG forward passes only
model.gradient_checkpointing_enable()
```

**3. Offload to CPU:**
```python
# Move batches to CPU between computations
batch_with = batch_with.to('cpu')
# Compute WITH
batch_with = batch_with.to('cuda')
```

**4. Mixed precision:**
```python
# Use fp16/bf16 for IG computation
with torch.autocast('cuda'):
    log_prob_with = model.compute_log_prob(batch_with)
```

## Data Issues

### Curriculum Not Varying

**Symptoms:**
- All samples seem to have 0% or 80% (no variation)
- IG doesn't improve

**Check:**
```python
# Verify set_epoch is called
dataset.set_epoch(current_epoch)

# Check curriculum config
print(config.data.curriculum.min_percentage)  # Should be 0.0
print(config.data.curriculum.max_percentage)  # Should be 0.8

# Verify RNG is working
print(dataset.rng.uniform(0, 1))  # Should vary each call
```

### Gold Solutions Missing

**Symptoms:**
- Error: KeyError 'remaining_solution'
- IG computation fails

**Check:**
```python
# Verify dataset returns remaining_solution
sample = dataset[0]
assert "remaining_solution" in sample.get("reward_model", {})

# Check parquet files have solution column
df = pd.read_parquet(train_file)
assert "solution" in df.columns
```

## Debugging Tools

### Enable Verbose Logging

```python
# In ray_trainer.py, add debug prints
if self.global_steps % 10 == 0:
    print(f"IG stats: mean={ig_mean:.4f}, std={ig_std:.4f}")
    print(f"Format stats: mean={format_mean:.4f}")
    print(f"Sample IG values: {information_gain[:5]}")
```

### Dump Sample Outputs

```python
# Save generated responses for inspection
if self.global_steps % 100 == 0:
    for i in range(min(5, len(batch))):
        with open(f"debug_output_{self.global_steps}_{i}.txt", "w") as f:
            f.write(f"Prompt: {batch.prompts[i]}\n\n")
            f.write(f"Response: {batch.responses[i]}\n\n")
            f.write(f"IG: {information_gain[i]}\n")
            f.write(f"Format: {format_scores[i]}\n")
```

### Validate IG Computation

```python
# Unit test for IG calculation
def test_ig_positive_for_good_reasoning():
    # Create sample with good reasoning → correct answer
    # IG should be positive
    assert information_gain > 0

def test_ig_negative_for_bad_reasoning():
    # Create sample with misleading reasoning
    # IG should be negative
    assert information_gain < 0
```

---

# Extension Points

## Adding Custom Curriculum Strategies

### Difficulty-Based Curriculum

```python
class DifficultyAwareCurriculumDataset(OmniMathCurriculumDataset):
    def __getitem__(self, idx):
        # Start with more support (higher %), decrease over time
        progress = self.current_epoch / self.total_epochs

        # Gradually reduce support
        min_pct = 0.8 * (1 - progress)  # 80% → 0%
        max_pct = 0.9 * (1 - progress) + 0.5 * progress  # 90% → 50%

        percentage = self.rng.uniform(min_pct, max_pct)
        # ... rest of logic
```

### Performance-Based Curriculum

```python
class AdaptiveCurriculumDataset(OmniMathCurriculumDataset):
    def adjust_curriculum(self, validation_accuracy):
        # If accuracy high → reduce support (harder)
        if validation_accuracy > 0.8:
            self.max_percentage *= 0.9
        # If accuracy low → increase support (easier)
        elif validation_accuracy < 0.5:
            self.max_percentage = min(0.8, self.max_percentage * 1.1)
```

## Alternative IG Metrics

### Token-Level IG Attribution

```python
def compute_token_level_ig(reasoning_tokens, gold_solution):
    """Compute IG contribution of each reasoning token."""
    ig_per_token = []

    for i in range(len(reasoning_tokens)):
        # Use first i tokens of reasoning
        partial_reasoning = reasoning_tokens[:i+1]

        # Compute IG with partial reasoning
        ig = compute_ig(partial_reasoning, gold_solution)
        ig_per_token.append(ig)

    # Can identify which tokens contribute most to IG
    return ig_per_token
```

### Step-Level IG

```python
def compute_step_level_ig(reasoning_steps, gold_solution):
    """Compute IG for each reasoning step."""
    # Split reasoning into steps (by newline or step markers)
    steps = reasoning.split('\n')

    ig_per_step = []
    for i in range(len(steps)):
        partial_steps = '\n'.join(steps[:i+1])
        ig = compute_ig(partial_steps, gold_solution)
        ig_per_step.append(ig)

    # Reward steps that increase IG
    return ig_per_step
```

## Content-Based Rewards

### Reward Key Concepts

```python
def reward_concept_coverage(reasoning, problem):
    """Reward reasoning that mentions key problem concepts."""
    # Extract key terms from problem (numbers, operations, concepts)
    key_terms = extract_key_terms(problem)

    # Check if reasoning mentions each term
    coverage = sum(term in reasoning for term in key_terms) / len(key_terms)

    return coverage  # [0, 1]
```

### Reward Logical Coherence

```python
def reward_logical_coherence(reasoning):
    """Reward logically connected reasoning steps."""
    steps = reasoning.split('\n')

    coherence_score = 0
    for i in range(len(steps) - 1):
        # Check if step i+1 follows from step i
        # Use simple heuristics or LLM-based checker
        if follows_logically(steps[i], steps[i+1]):
            coherence_score += 1

    return coherence_score / max(1, len(steps) - 1)
```

## Multi-Turn Extensions

### Interactive Reasoning

```python
class MultiTurnReasoningDataset:
    def create_turns(self, problem, solution):
        turns = []

        # Turn 1: Model attempts initial reasoning
        turns.append({
            "prompt": problem,
            "type": "initial_reasoning"
        })

        # Turn 2: Model can request hint
        turns.append({
            "prompt": "Request hint or continue?",
            "type": "hint_request"
        })

        # Turn 3: Provide hint if requested
        turns.append({
            "prompt": f"Hint: {get_hint(problem)}",
            "type": "hint"
        })

        # Turn 4: Final answer
        turns.append({
            "prompt": "Complete solution",
            "type": "final_answer"
        })

        return turns
```

### Curriculum Over Turns

```python
class TurnBasedCurriculum:
    def get_max_turns(self, epoch):
        """Gradually increase number of turns allowed."""
        # Early: 4 turns (more support)
        # Late: 2 turns (less support)
        return max(2, 4 - epoch // 10)
```

---

## Summary

This system represents a production-ready implementation of curriculum learning with novel IG-based reasoning quality measurement. Key innovations:

1. **IG measures reasoning quality** - Tests if reasoning helps predict correct answers
2. **Curriculum in dataset** - Clean separation of concerns
3. **Batch normalization** - Stable training with bounded rewards
4. **Format reward** - Provides within-group variance for GRPO
5. **Comprehensive logging** - Detailed metrics for debugging

For quick start instructions, see [README.md](README.md).
