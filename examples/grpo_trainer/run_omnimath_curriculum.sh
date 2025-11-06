#!/bin/bash
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

# OmniMath Curriculum Learning with GRPO
#
# This script trains a model on OmniMath with curriculum learning:
# - 0-80% of solution randomly included in each prompt
# - Re-randomized each epoch for true curriculum learning
# - Custom reward using policy + reference model metrics

set -x

# Data paths (adjust to your preprocessed data location)
omnimath_train_path=$HOME/data/omnimath/train.parquet
omnimath_val_path=$HOME/data/omnimath/val.parquet

# Model path (change to your model)
model_path="Qwen/Qwen2.5-7B-Instruct"

# Run training with config overrides
python3 -m verl.trainer.main_ppo \
    --config-path examples/grpo_trainer/config \
    --config-name omnimath_curriculum_grpo \
    data.train_files="$omnimath_train_path" \
    data.val_files="$omnimath_val_path" \
    actor_rollout_ref.model.path="$model_path" \
    $@
