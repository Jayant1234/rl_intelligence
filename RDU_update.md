# OmniMath Curriculum Learning - Changes Summary

## Overview
Implemented curriculum learning for OmniMath with GRPO, allowing 0-80% of solution to be randomly included in prompts with per-epoch re-randomization. Custom reward combines Information Gain (reasoning quality) + Format Reward (RLVR-style) to encourage both helpful reasoning and structured outputs.

## Latest Updates (Information Gain Redesign)

**Major Change**: Redesigned Information Gain to measure **reasoning quality** rather than response quality.

**New IG Formula**:
```
IG = log P(gold_solution | prompt + reasoning) - log P(gold_solution | prompt)
```

**Why**: This measures whether the model's generated reasoning helps predict the CORRECT answer, providing a stronger learning signal for reasoning improvement.

## File Structure

```
examples/
├── data_preprocess/
│   ├── omnimath_curriculum.py              [NEW] - Preprocessing
│   └── omnimath_curriculum_dataset.py      [NEW] - Custom dataset
├── reward_functions/
│   ├── omnimath_curriculum_metrics.py      [NEW] - Metrics computation
│   └── omnimath_reward.py                  [NEW] - Reward function
└── grpo_trainer/
    ├── config/
    │   └── omnimath_curriculum_grpo.yaml   [NEW] - Config
    └── run_omnimath_curriculum.sh          [NEW] - Training script

verl/trainer/
├── ppo/
│   └── ray_trainer.py                      [MODIFIED] - Lines 1141-1167
└── config/
    ├── ppo_trainer.yaml                    [MODIFIED] - Lines 62-64
    └── data/
        └── legacy_data.yaml                [MODIFIED] - Lines 114-124
```

## New Files

### 1. `examples/data_preprocess/omnimath_curriculum.py`
- Converts OmniMath HuggingFace dataset to verl parquet format
- Creates 90/10 train/val split
- Stores full solutions only (no curriculum logic)

### 2. `examples/data_preprocess/omnimath_curriculum_dataset.py`
- Custom dataset class handling ALL curriculum logic
- `set_epoch(epoch)`: Re-initializes RNG for new randomization
- `__getitem__()`: Dynamically samples 0-80% solution, splits by sentences, creates prompt
- Returns `partial_solution_given` and `remaining_solution` in `reward_model` dict

### 3. `examples/reward_functions/omnimath_curriculum_metrics.py`
- `compute_omnimath_curriculum_metrics(batch, tokenizer)`: Computes metrics from models
- Extracts `old_log_probs` (policy) and `ref_log_prob` (reference) from batch
- Returns: `policy_confidence`, `kl_divergence`, `decoded_responses`, `partial_solution_given`, `remaining_solution`

### 4. `examples/reward_functions/omnimath_reward.py`
- `check_format_compliance()`: RLVR-style format checker (lines 39-114)
  - Checks for **Reasoning:** header
  - Checks for **Solution:** or **Remaining Solution:** header
  - Verifies `\boxed{}` answer presence
  - Validates section order (reasoning before solution)
  - Checks substantive content (reasoning ≥20 chars, solution ≥10 chars)
  - Returns total_format_score in [0, 1]

- `compute_score()`: Reward function combining IG + Format Reward (lines 117-241)
  - **Simple formula**: `final_reward = normalized_IG + (0.3 × format_reward_scaled)`
  - Uses Information Gain from `extra_info` (injected by ray_trainer)
  - Applies format reward to encourage structured responses
  - Returns detailed metrics for logging

### 5. `examples/grpo_trainer/config/omnimath_curriculum_grpo.yaml`
- Training configuration linking all components
- Specifies custom dataset, metrics function, reward function
- GRPO settings with reference model enabled

### 6. `examples/grpo_trainer/run_omnimath_curriculum.sh`
- Shell script to launch training with config

## Modified Files

### 1. `verl/trainer/config/data/legacy_data.yaml`

**Lines 114-124**: Added curriculum learning configuration structure

**Purpose**: Enable curriculum parameters to be passed via command-line or config files without Hydra struct mode errors

**Added fields**:
```yaml
# Curriculum learning parameters (optional, used by custom datasets like OmniMathCurriculumDataset)
curriculum:

  # Minimum percentage of solution to include in prompt
  min_percentage: null

  # Maximum percentage of solution to include in prompt
  max_percentage: null

  # Base seed for reproducible curriculum randomization
  base_seed: null
```

**Why needed**:
- The trainer sets `OmegaConf.set_struct(config, True)` at `ray_trainer.py:409`
- Struct mode prevents adding new keys that don't exist in base config
- Adding `curriculum` to base config allows override via CLI: `data.curriculum.min_percentage=0.0`
- Values default to `null` so they're optional for non-curriculum experiments

### 2. `verl/trainer/ppo/ray_trainer.py`

**Major redesign of Information Gain computation**. Three new methods added and IG computation block completely rewritten.

#### **1. Helper Method: `_construct_empty_cot_batch()` (Lines 519-602)**

**Location**: After `_get_gen_batch()` method

**Purpose**: *(DEPRECATED - kept for reference but no longer used)* Originally reconstructed batch with empty CoT for baseline

**Status**: Not used by new IG calculation

---

#### **2. NEW Helper Method: `_parse_reasoning_from_response()` (Lines 604-644)**

**Location**: After `_construct_empty_cot_batch()` method

**Purpose**: Parse generated responses to extract reasoning and solution sections

**Key functionality**:
```python
def _parse_reasoning_from_response(self, response_text: str, has_partial_solution: bool) -> tuple[str, str]:
    """
    Returns: (reasoning_text, generated_remaining_solution_text)
    """
    # Find **Reasoning:** section
    # Find **Solution:** or **Remaining Solution:** section
    # Extract text between headers
    # Return tuple of (reasoning, solution)
```

**Returns**:
- `reasoning_text`: Content of **Reasoning:** section
- `generated_remaining_solution_text`: Content of **Solution:** or **Remaining Solution:** section

---

#### **3. NEW Helper Method: `_construct_ig_batches_with_gold_solution()` (Lines 646-839)**

**Location**: After `_parse_reasoning_from_response()` method

**Purpose**: Construct two batches for new IG formula

**Key functionality**:

For each sample in batch:
1. **Parse generated response** to extract reasoning
2. **Get gold solution** from `batch.non_tensor_batch["reward_model"][i]["remaining_solution"]`
3. **Construct TWO prompts**:

   **Prompt WITH reasoning**:
   ```
   problem + partial_solution + **Reasoning:** + model_reasoning + **Remaining Solution:** + gold_solution
   ```

   **Prompt WITHOUT reasoning**:
   ```
   problem + partial_solution + **Remaining Solution:** + gold_solution
   ```

4. **Tokenize both prompts** and create response masks for gold solution tokens
5. **Pad to max length** and compute position IDs

**Returns**: `tuple[DataProto, DataProto]`
- `batch_with_reasoning`: For computing `log P(gold_solution | prompt + reasoning)`
- `batch_without_reasoning`: For computing `log P(gold_solution | prompt)`

---

#### **4. REDESIGNED Information Gain Computation Block (Lines 1450-1534)**

**Location**: After `ref_log_prob` computation, before reward computation

**Complete rewrite of IG calculation**:

```python
# ===== COMPUTE INFORMATION GAIN BASED ON REASONING QUALITY =====
# New IG formula: IG = log P(gold_solution | prompt + reasoning) - log P(gold_solution | prompt)
# This measures: "Does the model's reasoning help predict the CORRECT answer?"
if self.config.get("compute_information_gain", False):
    with marked_timer("information_gain", timing_raw, color="purple"):
        # Step 1: Construct two batches for IG calculation
        batch_with_reasoning, batch_without_reasoning = self._construct_ig_batches_with_gold_solution(batch)

        # Step 2: Compute log probs for both batches
        # Log P(gold_solution | prompt + reasoning)
        log_prob_with_reasoning = self.actor_rollout_wg.compute_log_prob(batch_with_reasoning)
        log_probs_with = log_prob_with_reasoning.batch["old_log_probs"]  # [B, R]
        response_masks_with = batch_with_reasoning.batch["response_mask"]  # [B, R]

        # Log P(gold_solution | prompt only)
        log_prob_without_reasoning = self.actor_rollout_wg.compute_log_prob(batch_without_reasoning)
        log_probs_without = log_prob_without_reasoning.batch["old_log_probs"]  # [B, R]
        response_masks_without = batch_without_reasoning.batch["response_mask"]  # [B, R]

        # Step 3: Sum log probs over gold solution tokens
        sum_with_reasoning = (log_probs_with * response_masks_with).sum(dim=1)  # [B]
        sum_without_reasoning = (log_probs_without * response_masks_without).sum(dim=1)  # [B]

        # Step 4: Calculate Information Gain
        # IG = log P(gold | prompt + reasoning) - log P(gold | prompt)
        # Positive IG = reasoning helps predict correct answer
        # Negative IG = reasoning hurts (confuses model)
        information_gain_raw = sum_with_reasoning - sum_without_reasoning  # [B]

        # Step 5: Normalize IG to [-1, 1] using batch statistics
        ig_std = information_gain_raw.std() + 1e-8
        ig_mean = information_gain_raw.mean()
        information_gain_normalized = torch.tanh((information_gain_raw - ig_mean) / (ig_std + 1e-8))
        information_gain = information_gain_normalized  # [B], range: [-1, 1]

        # Step 6: Store both raw and normalized IG
        batch.batch["information_gain_raw"] = information_gain_raw
        batch.batch["information_gain"] = information_gain

        # Step 7: Inject into non_tensor_batch for reward function access
        ig_numpy = information_gain.cpu().numpy()
        ig_raw_numpy = information_gain_raw.cpu().numpy()

        # [Lines 1500-1518: Code to inject IG into extra_info dict]

        batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)

        # Step 8: Log metrics
        ig_metrics = {
            "curriculum/information_gain_mean": information_gain.mean().item(),
            "curriculum/information_gain_std": information_gain.std().item(),
            "curriculum/information_gain_min": information_gain.min().item(),
            "curriculum/information_gain_max": information_gain.max().item(),
            "curriculum/information_gain_raw_mean": information_gain_raw.mean().item(),
            "curriculum/information_gain_raw_std": information_gain_raw.std().item(),
            "curriculum/information_gain_raw_min": information_gain_raw.min().item(),
            "curriculum/information_gain_raw_max": information_gain_raw.max().item(),
        }
        metrics.update(ig_metrics)
# ===== END INFORMATION GAIN =====
```

**What it does**:
- **OLD**: Measured `IG = log P(generated_response | prompt + response) - log P(generated_response | prompt)`
- **NEW**: Measures `IG = log P(gold_solution | prompt + reasoning) - log P(gold_solution | prompt)`
- Extracts reasoning from generated response
- Tests if reasoning helps predict the CORRECT answer
- Normalizes to [-1, 1] for reward stability
- Injects into `extra_info` for reward function access
- Logs both raw and normalized metrics

**Key Insight**: This measures **reasoning quality** (does it help predict correct answer?) rather than just response likelihood.

## Data Flow

```
1. omnimath_curriculum.py
   └─> Creates train.parquet, val.parquet

2. Training starts → omnimath_curriculum_dataset.py
   └─> Loads parquet, dynamically generates partial solution prompts
   └─> Returns batch with:
       - prompt (problem + partial_solution)
       - partial_solution_given (in reward_model dict)
       - remaining_solution (gold, in reward_model dict)

3. Rollout phase
   └─> Model generates responses with format:
       **Reasoning:** [model's reasoning]
       **Remaining Solution:** [model's answer attempt]

4. ray_trainer.py - Information Gain Computation (lines 1450-1534)
   ├─> Parse generated responses to extract reasoning sections
   ├─> Construct TWO batches:
   │   ├─> Batch WITH reasoning: prompt + reasoning + gold_solution
   │   └─> Batch WITHOUT reasoning: prompt + gold_solution
   ├─> Compute log probs for both batches
   ├─> Calculate IG = log P(gold | prompt + reasoning) - log P(gold | prompt)
   ├─> Normalize IG to [-1, 1]
   └─> Inject IG into batch.non_tensor_batch["extra_info"]

5. ray_trainer.py - Reward Computation
   └─> Calls omnimath_reward.py
       ├─> Receives IG from extra_info
       ├─> Checks format compliance (RLVR-style)
       └─> Returns: final_reward = normalized_IG + (0.3 × format_reward)

6. GRPO updates policy
   └─> Uses final rewards to compute advantages
   └─> Updates policy to:
       - Generate better reasoning (higher IG)
       - Follow structured format (higher format reward)
```

## Usage

```bash
# 1. Preprocess data
python examples/data_preprocess/omnimath_curriculum.py --local_save_dir ~/data/omnimath

# 2. Train
bash examples/grpo_trainer/run_omnimath_curriculum.sh
```

## Key Config Parameters

```yaml
data.curriculum:
  min_percentage: 0.0  # 0% of solution
  max_percentage: 0.8  # 80% of solution
  base_seed: 42

custom_curriculum_metrics:
  path: examples/reward_functions/omnimath_curriculum_metrics.py
  name: compute_omnimath_curriculum_metrics

custom_reward_function:
  path: examples/reward_functions/omnimath_reward.py
  name: compute_score
```
