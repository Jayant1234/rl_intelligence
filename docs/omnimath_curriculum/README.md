# OmniMath Curriculum Learning - Quick Start Guide

## Overview

This implementation combines **curriculum learning** with **Information Gain (IG)** to train mathematical reasoning models. The system teaches models to generate high-quality reasoning by:

1. **Curriculum Learning**: Randomly providing 0-80% of solution steps in prompts (re-randomized each epoch)
2. **Reasoning Quality Measurement**: Using IG to measure whether generated reasoning helps predict correct answers
3. **Format Enforcement**: Rewarding structured outputs with clear reasoning and solution sections

**Key Innovation**: Our IG metric measures `log P(correct_answer | prompt + reasoning) - log P(correct_answer | prompt)`, which directly evaluates reasoning quality rather than just model confidence.

## Quick Start (5 Minutes)

### Prerequisites

```bash
# Ensure verl is installed
pip install -e .

# Verify you have access to:
# - A GPU with at least 24GB VRAM (for 7B models)
# - OmniMath dataset on HuggingFace
```

### Step 1: Preprocess Data

```bash
python examples/data_preprocess/omnimath_curriculum.py \
    --local_save_dir ~/data/omnimath \
    --dataset_name "HuggingFaceM4/OmniMath" \
    --split_ratio 0.9
```

**What this does:**
- Downloads OmniMath dataset from HuggingFace
- Creates 90/10 train/validation split
- Saves as `train.parquet` and `val.parquet` in `~/data/omnimath`

### Step 2: Launch Training

```bash
cd examples/grpo_trainer
bash run_omnimath_curriculum.sh
```

**Default configuration:**
- Model: Qwen2.5-3B (modify in config for other models)
- Curriculum: 0-80% solution steps
- Batch size: 128 rollout, 64 training
- GRPO algorithm with reference model

### Step 3: Monitor Training

```bash
tensorboard --logdir ./logs
```

**Key metrics to watch:**

| Metric | Target | Meaning |
|--------|--------|---------|
| `curriculum/information_gain_mean` | Positive, increasing | Average reasoning quality |
| `curriculum/information_gain_std` | 0.3-0.7 | Variance (higher = better GRPO signal) |
| `format/format_reward_mean` | > 0.8 | Format compliance |
| `reward/rewards_mean` | Increasing | Combined reward |

See [METRICS_GUIDE.md](METRICS_GUIDE.md) for detailed metric interpretation.

## Configuration

### Basic Configuration

Edit `examples/grpo_trainer/config/omnimath_curriculum_grpo.yaml`:

```yaml
# Model settings
model:
  path: Qwen/Qwen2.5-3B-Instruct

# Curriculum settings
data:
  curriculum:
    min_percentage: 0.0    # 0% of solution
    max_percentage: 0.8    # 80% of solution
    base_seed: 42          # Reproducibility

# Reward weights
reward:
  format_weight: 0.3      # Weight for format reward

# Enable IG computation
compute_information_gain: true
```

### Command-Line Overrides

```bash
# Use different curriculum range
python -m verl.trainer.main_ppo \
    --config examples/grpo_trainer/config/omnimath_curriculum_grpo.yaml \
    --data.curriculum.min_percentage 0.2 \
    --data.curriculum.max_percentage 0.5

# Use larger model
python -m verl.trainer.main_ppo \
    --config examples/grpo_trainer/config/omnimath_curriculum_grpo.yaml \
    --model.path Qwen/Qwen2.5-7B-Instruct

# Adjust format reward weight
python -m verl.trainer.main_ppo \
    --config examples/grpo_trainer/config/omnimath_curriculum_grpo.yaml \
    --reward.format_weight 0.5
```

## Expected Outputs

### Training Progress

**Early Training (Epoch 0-2):**
```
information_gain_mean: -0.1 to 0.1  (random reasoning quality)
format_reward_mean: 0.3 to 0.6      (learning format)
rewards_mean: 0.0 to 0.4            (low combined reward)
```

**Mid Training (Epoch 3-10):**
```
information_gain_mean: 0.2 to 0.4   (improving reasoning)
format_reward_mean: 0.7 to 0.9      (good format compliance)
rewards_mean: 0.5 to 0.8            (solid progress)
```

**Late Training (Epoch 10+):**
```
information_gain_mean: 0.5 to 0.7   (high-quality reasoning)
format_reward_mean: 0.9 to 1.0      (near-perfect format)
rewards_mean: 0.8 to 1.0            (strong performance)
```

### Example Generated Output

**Prompt (with 40% partial solution):**
```
Problem: Solve for x: 2x + 5 = 13

A partial solution is provided below:
Step 1: Subtract 5 from both sides
2x = 8

Task: Complete this solution...
```

**Model Output (after training):**
```
**Reasoning:**
The partial solution has correctly isolated the term with x by subtracting 5
from both sides, leaving us with 2x = 8. To find x, I need to divide both
sides by 2 to isolate x completely.

**Remaining Solution:**
Step 2: Divide both sides by 2
x = 8/2
x = 4

Therefore, the solution is \boxed{4}
```

## File Structure

```
examples/
├── data_preprocess/
│   ├── omnimath_curriculum.py              # Data preprocessing script
│   └── omnimath_curriculum_dataset.py      # Custom dataset with curriculum logic
├── reward_functions/
│   ├── omnimath_curriculum_metrics.py      # Metrics extraction
│   └── omnimath_reward.py                  # Combined IG + format reward
└── grpo_trainer/
    ├── config/
    │   └── omnimath_curriculum_grpo.yaml   # Training configuration
    └── run_omnimath_curriculum.sh          # Launch script

verl/trainer/ppo/
└── ray_trainer.py                          # Core IG computation logic
```

## Next Steps

- **Understand the architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
- **Interpret metrics**: See [METRICS_GUIDE.md](METRICS_GUIDE.md)
- **Debug issues**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **View changelog**: See [CHANGELOG.md](CHANGELOG.md) for development history

## Common Questions

**Q: Can I use a different dataset?**
A: Yes! Modify `omnimath_curriculum_dataset.py` to load your data. Key requirements:
- Dataset must have `problem` and `solution` fields
- Solutions should be splittable into steps
- Answers should be extractable for evaluation

**Q: What GPU memory do I need?**
A: Depends on model size:
- 3B model: 24GB (single GPU)
- 7B model: 40GB (single GPU) or 2x24GB (multi-GPU)
- 32B model: 4x40GB (multi-GPU)

**Q: How long does training take?**
A: Approximate times on A100:
- 3B model: 2-4 hours for 10 epochs
- 7B model: 4-8 hours for 10 epochs
- 32B model: 12-24 hours for 10 epochs

**Q: Can I disable Information Gain?**
A: Yes, set `compute_information_gain: false` in config. The system will fall back to format reward only. However, IG provides the key learning signal for reasoning quality.

**Q: Why does IG use the gold solution?**
A: This measures whether reasoning helps predict the **correct** answer. Alternative approaches (measuring confidence in generated response) reward any confident response, even incorrect ones. See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed explanation.

## Citation

If you use this implementation, please cite:

```bibtex
@software{omnimath_curriculum_verl,
  title={OmniMath Curriculum Learning with Information Gain},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/verl}
}
```

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first
- **Contributing**: Pull requests welcome!
