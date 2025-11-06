# OmniMath Curriculum Learning - Changes Summary

## Overview
Implemented curriculum learning for OmniMath with GRPO, allowing 0-80% of solution to be randomly included in prompts with per-epoch re-randomization. Custom reward uses policy + reference model metrics.

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
- `compute_score()`: Reward function using injected metrics
- Combines text-based (continuation quality) + model-based (confidence, KL) signals
- Applies curriculum scaling (less help = higher reward)

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

Two modifications to the `RayPPOTrainer` class:

#### **1. Helper Method: `_construct_empty_cot_batch()` (Lines 519-602)**

**Location**: After `_get_gen_batch()` method

**Purpose**: Reconstructs batch with empty CoT (prompt only) for baseline log probability computation

**Key functionality**:
- Extracts question text from `batch.non_tensor_batch["extra_info"][i]["problem"]`
- Creates prompt without CoT using `create_curriculum_prompt(problem, "")`
- Tokenizes and pads to match original batch dimensions
- Returns DataProto with: prompt + answer (no CoT), keeping same response tokens

#### **2. Information Gain Computation Block (Lines 1226-1257)**

**Location**: After `ref_log_prob` computation (line 1224), before value computation

**Added code**:
```python
# ===== COMPUTE INFORMATION GAIN WITH EMPTY CoT BASELINE =====
if self.config.get("compute_information_gain", False):
    with marked_timer("information_gain", timing_raw, color="purple"):
        # Step 1: Current log probs (with CoT) already in batch
        log_probs_with_cot = batch.batch["old_log_probs"]  # [B, R]

        # Step 2: Construct empty CoT batch
        empty_cot_batch = self._construct_empty_cot_batch(batch)

        # Step 3: Compute log probs without CoT
        empty_log_prob = self.actor_rollout_wg.compute_log_prob(empty_cot_batch)
        log_probs_without_cot = empty_log_prob.batch["old_log_probs"]  # [B, R]

        # Step 4: Calculate Information Gain per sample
        response_masks = batch.batch["response_mask"]  # [B, R]

        # Sum log probs over valid tokens
        sum_with_cot = (log_probs_with_cot * response_masks).sum(dim=1)      # [B]
        sum_without_cot = (log_probs_without_cot * response_masks).sum(dim=1) # [B]

        information_gain = sum_with_cot - sum_without_cot  # [B]

        # Store in batch for reward function
        batch.batch["information_gain"] = information_gain  # [B]

        # Log metrics
        ig_metrics = {
            "curriculum/information_gain_mean": information_gain.mean().item(),
            "curriculum/information_gain_std": information_gain.std().item(),
        }
        metrics.update(ig_metrics)
# ===== END INFORMATION GAIN =====
```

**What it does**:
- Computes Information Gain: `IG = log P(answer | prompt + CoT) - log P(answer | prompt)`
- Stores result in `batch.batch["information_gain"]` tensor [B]
- Logs mean and std metrics for monitoring
- Only runs when `compute_information_gain: true` in config

## Data Flow

```
1. omnimath_curriculum.py
   └─> Creates train.parquet, val.parquet

2. Training starts → omnimath_curriculum_dataset.py
   └─> Loads parquet, dynamically generates partial solution prompts
   └─> Returns batch with partial_solution_given, remaining_solution

3. ray_trainer.py (lines 1116-1167)
   ├─> Computes old_log_probs (policy)
   ├─> Computes ref_log_prob (reference)
   ├─> Calls omnimath_curriculum_metrics.py [NEW INJECTION]
   │   └─> Computes policy_confidence, kl_divergence
   │   └─> Injects into batch.non_tensor_batch
   └─> Calls omnimath_reward.py
       └─> Receives metrics via extra_info
       └─> Returns final reward

4. GRPO updates policy with rewards
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
