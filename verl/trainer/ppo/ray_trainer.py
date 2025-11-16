# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _construct_empty_cot_batch(self, batch: DataProto) -> DataProto:
        """
        Reconstruct batch with prompt only (no CoT), keeping same answer tokens.

        For Information Gain calculation:
        IG = log P(answer | prompt + CoT) - log P(answer | prompt)

        Args:
            batch: Original batch with CoT in prompts

        Returns:
            DataProto with empty CoT (prompt only) but same responses
        """
        import torch
        from verl import DataProto
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask

        batch_size = len(batch)
        responses = batch.batch["responses"]  # [B, R]

        new_input_ids = []
        new_attention_mask = []
        new_position_ids = []

        for i in range(batch_size):
            # Get question text (no CoT)
            problem = batch.non_tensor_batch["extra_info"][i]["problem"]

            # Create prompt without CoT (empty partial solution)
            from examples.data_preprocess.omnimath_curriculum_dataset import create_curriculum_prompt
            prompt_text = create_curriculum_prompt(problem, "")

            # Apply chat template
            raw_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=False
            )

            # Tokenize prompt
            prompt_encoding = self.tokenizer(raw_prompt, add_special_tokens=False, return_tensors="pt")
            prompt_ids = prompt_encoding["input_ids"][0]  # [P]

            # Get answer tokens (remove padding)
            answer_ids = responses[i]  # [R]
            valid_answer_len = batch.batch["response_mask"][i].sum().item()
            answer_ids = answer_ids[:valid_answer_len]  # [R']

            # Concatenate: prompt + answer (no CoT)
            full_ids = torch.cat([prompt_ids, answer_ids])  # [P + R']

            # Create masks
            full_mask = torch.ones_like(full_ids)

            # Pad to original max length
            max_len = batch.batch["input_ids"].shape[1]
            full_ids_padded, full_mask_padded = verl_F.postprocess_data(
                input_ids=full_ids.unsqueeze(0),
                attention_mask=full_mask.unsqueeze(0),
                max_length=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error"
            )

            pos_ids = compute_position_id_with_mask(full_mask_padded)

            new_input_ids.append(full_ids_padded[0])
            new_attention_mask.append(full_mask_padded[0])
            new_position_ids.append(pos_ids[0])

        # Create new batch
        empty_cot_dict = {
            "input_ids": torch.stack(new_input_ids),         # [B, S]
            "attention_mask": torch.stack(new_attention_mask), # [B, S]
            "position_ids": torch.stack(new_position_ids),    # [B, S]
            "responses": batch.batch["responses"],            # [B, R] - same
        }

        empty_cot_batch = DataProto.from_single_dict(empty_cot_dict)
        empty_cot_batch.meta_info = batch.meta_info.copy()

        return empty_cot_batch

    def _parse_reasoning_from_response(self, response_text: str, has_partial_solution: bool = None) -> tuple[str, str]:
        """
        Parse the generated response to extract thinking and prediction sections.

        Expected format (NOTE: <think> is in the prompt, not in response):
            [thinking content]
            </think>
            <|startofprediction|>
            [prediction content]
            <|endofprediction|>

        The opening <think> tag is provided in the prompt, so the response starts
        with thinking content directly.

        Args:
            response_text: Full generated response text (does NOT include prompt)
            has_partial_solution: DEPRECATED - no longer used (format is same regardless)

        Returns:
            tuple of (thinking_text, prediction_text)
        """
        import re

        # Find </think> tag (case-sensitive)
        # NOTE: We DON'T look for <think> since it's in the prompt
        think_close_match = re.search(r'</think>', response_text)

        # Find <|startofprediction|> and <|endofprediction|> tags (case-sensitive)
        prediction_open_match = re.search(r'<\|startofprediction\|>', response_text)
        prediction_close_match = re.search(r'<\|endofprediction\|>', response_text)

        # Extract thinking content from start of response to </think>
        if think_close_match:
            think_start = 0  # Response starts with thinking content
            think_end = think_close_match.start()
            thinking_text = response_text[think_start:think_end].strip()
        else:
            # No valid closing tag found - treat entire response as thinking
            thinking_text = response_text.strip()

        # Extract prediction content if tags exist
        if prediction_open_match and prediction_close_match:
            prediction_start = prediction_open_match.end()
            prediction_end = prediction_close_match.start()
            prediction_text = response_text[prediction_start:prediction_end].strip()
        else:
            # No valid prediction tags found - empty prediction
            prediction_text = ""

        return thinking_text, prediction_text

    def _construct_ig_batches_with_gold_solution(self, batch: DataProto) -> tuple[DataProto, DataProto]:
        """
        Construct two batches for Information Gain calculation:
        1. Batch WITH thinking: Simplified doc continuation prompt + <think>thinking</think> + <|startofprediction|>gold_solution<|endofprediction|>
        2. Batch WITHOUT thinking: Minimal baseline = "Solve the following math problem." + problem + partial_solution + gold_solution

        IG formula:
        IG = log P(gold_solution | doc_format + thinking) - log P(gold_solution | minimal_baseline)

        This measures: "Does our document continuation approach with thinking help compared to simple instruction?"

        NOTE: Baseline includes minimal instruction to follow DeepSeek/Qwen prompting best practices
        (avoids artificially bad log probs from completely raw text).

        Args:
            batch: Original batch with generated responses

        Returns:
            tuple of (batch_with_thinking, batch_without_thinking)
        """
        import torch
        from verl import DataProto
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask

        batch_size = len(batch)
        responses = batch.batch["responses"]  # [B, R] - generated responses

        # Lists to store batch components
        with_thinking_input_ids = []
        with_thinking_attention_mask = []
        with_thinking_position_ids = []
        with_thinking_response_mask = []

        without_thinking_input_ids = []
        without_thinking_attention_mask = []
        without_thinking_position_ids = []
        without_thinking_response_mask = []

        for i in range(batch_size):
            # Get original prompt components
            problem = batch.non_tensor_batch["extra_info"][i]["problem"]
            partial_solution_given = batch.non_tensor_batch["reward_model"][i]["partial_solution_given"]
            gold_remaining_solution = batch.non_tensor_batch["reward_model"][i]["remaining_solution"]

            # Decode the generated response
            response_tokens = responses[i]  # [R]
            valid_response_len = batch.batch["response_mask"][i].sum().item()
            response_tokens = response_tokens[:valid_response_len]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Parse response to extract thinking (content inside <think>...</think>)
            thinking_text, _ = self._parse_reasoning_from_response(response_text)

            # Build context for document continuation format
            if partial_solution_given.strip():
                context = f"{problem}\n\n{partial_solution_given}"
            else:
                context = problem

            # ===== CONSTRUCT PROMPT WITH THINKING (Simplified Document Continuation Format) =====
            prompt_with_thinking = (
                "You are reading a mathematical document that contains problems and fully worked solutions.\n\n"
                "The text under ### Context is the beginning of one such solution, possibly cut off mid-argument.\n"
                "Your task is to continue this solution so that the argument is completed in a way that is "
                "mathematically correct and stylistically consistent with the context.\n\n"
                "First, reason step by step between <think> and </think>. In <think>, you should:\n"
                "- Understand what has already been proved in the Context.\n"
                "- Figure out what the author is trying to do next.\n"
                "- Plan how to continue from the last line to reach a clean conclusion.\n\n"
                "After </think>, write only the predicted continuation of the document, starting "
                "exactly from where the Context stops. Do NOT restate the problem and do NOT "
                "summarize the context; just continue the solution in the same style.\n\n"
                "Enclose this predicted continuation between <|startofprediction|> and <|endofprediction|>.\n\n"
                f"### Context\n{context}\n\n"
                f"<think>\n{thinking_text}\n</think>\n"
                f"<|startofprediction|>\n{gold_remaining_solution}\n<|endofprediction|>"
            )

            # ===== CONSTRUCT BASELINE WITHOUT THINKING (Simple Direct Instruction) =====
            # Minimal baseline that follows DeepSeek/Qwen prompting best practices
            # Simple instruction + problem context, no reasoning structure
            if partial_solution_given.strip():
                prompt_without_thinking = (
                    "Solve the following math problem.\n\n"
                    f"{problem}\n\n"
                    f"{partial_solution_given}\n\n"
                    f"{gold_remaining_solution}"
                )
            else:
                prompt_without_thinking = (
                    "Solve the following math problem.\n\n"
                    f"{problem}\n\n"
                    f"{gold_remaining_solution}"
                )

            # Apply chat template to both prompts
            prompt_with_thinking_chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_with_thinking}],
                add_generation_prompt=False,  # We're measuring P(text), not generating
                tokenize=False
            )

            prompt_without_thinking_chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_without_thinking}],
                add_generation_prompt=False,
                tokenize=False
            )

            # Tokenize both prompts
            encoding_with = self.tokenizer(prompt_with_thinking_chat, add_special_tokens=False, return_tensors="pt")
            encoding_without = self.tokenizer(prompt_without_thinking_chat, add_special_tokens=False, return_tensors="pt")

            ids_with = encoding_with["input_ids"][0]  # [L1]
            ids_without = encoding_without["input_ids"][0]  # [L2]

            # Tokenize gold solution separately to create response mask
            gold_solution_encoding = self.tokenizer(gold_remaining_solution, add_special_tokens=False, return_tensors="pt")
            gold_solution_ids = gold_solution_encoding["input_ids"][0]  # [G]
            gold_solution_len = len(gold_solution_ids)

            # Create attention masks (all 1s)
            mask_with = torch.ones_like(ids_with)
            mask_without = torch.ones_like(ids_without)

            # Create response masks (last gold_solution_len tokens are the response)
            response_mask_with = torch.zeros_like(ids_with)
            response_mask_with[-gold_solution_len:] = 1

            response_mask_without = torch.zeros_like(ids_without)
            response_mask_without[-gold_solution_len:] = 1

            # Pad to max length
            max_len = batch.batch["input_ids"].shape[1]

            ids_with_padded, mask_with_padded = verl_F.postprocess_data(
                input_ids=ids_with.unsqueeze(0),
                attention_mask=mask_with.unsqueeze(0),
                max_length=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error"
            )

            ids_without_padded, mask_without_padded = verl_F.postprocess_data(
                input_ids=ids_without.unsqueeze(0),
                attention_mask=mask_without.unsqueeze(0),
                max_length=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error"
            )

            # Pad response masks
            response_mask_with_padded = torch.zeros(max_len)
            response_mask_with_padded[-len(response_mask_with):] = response_mask_with

            response_mask_without_padded = torch.zeros(max_len)
            response_mask_without_padded[-len(response_mask_without):] = response_mask_without

            # Compute position IDs
            pos_ids_with = compute_position_id_with_mask(mask_with_padded)
            pos_ids_without = compute_position_id_with_mask(mask_without_padded)

            # Append to lists
            with_thinking_input_ids.append(ids_with_padded[0])
            with_thinking_attention_mask.append(mask_with_padded[0])
            with_thinking_position_ids.append(pos_ids_with[0])
            with_thinking_response_mask.append(response_mask_with_padded)

            without_thinking_input_ids.append(ids_without_padded[0])
            without_thinking_attention_mask.append(mask_without_padded[0])
            without_thinking_position_ids.append(pos_ids_without[0])
            without_thinking_response_mask.append(response_mask_without_padded)

        # Stack all components
        stacked_with_input_ids = torch.stack(with_thinking_input_ids)
        stacked_with_attention_mask = torch.stack(with_thinking_attention_mask)
        stacked_with_position_ids = torch.stack(with_thinking_position_ids)
        stacked_with_response_mask = torch.stack(with_thinking_response_mask)

        stacked_without_input_ids = torch.stack(without_thinking_input_ids)
        stacked_without_attention_mask = torch.stack(without_thinking_attention_mask)
        stacked_without_position_ids = torch.stack(without_thinking_position_ids)
        stacked_without_response_mask = torch.stack(without_thinking_response_mask)

        # Extract "responses" by finding where response_mask == 1 in input_ids
        # For "with thinking" batch
        with_thinking_responses = []
        for i in range(batch_size):
            response_indices = (stacked_with_response_mask[i] == 1).nonzero(as_tuple=True)[0]
            response_tokens = stacked_with_input_ids[i][response_indices]
            # Pad to max response length in batch
            with_thinking_responses.append(response_tokens)

        # For "without thinking" batch
        without_thinking_responses = []
        for i in range(batch_size):
            response_indices = (stacked_without_response_mask[i] == 1).nonzero(as_tuple=True)[0]
            response_tokens = stacked_without_input_ids[i][response_indices]
            without_thinking_responses.append(response_tokens)

        # Pad responses to same length within each batch
        max_response_len_with = max(len(r) for r in with_thinking_responses)
        max_response_len_without = max(len(r) for r in without_thinking_responses)

        padded_responses_with = []
        for r in with_thinking_responses:
            padded = torch.nn.functional.pad(r, (0, max_response_len_with - len(r)), value=self.tokenizer.pad_token_id)
            padded_responses_with.append(padded)

        padded_responses_without = []
        for r in without_thinking_responses:
            padded = torch.nn.functional.pad(r, (0, max_response_len_without - len(r)), value=self.tokenizer.pad_token_id)
            padded_responses_without.append(padded)

        # Create batch with thinking
        batch_with_thinking_dict = {
            "input_ids": stacked_with_input_ids,
            "attention_mask": stacked_with_attention_mask,
            "position_ids": stacked_with_position_ids,
            "response_mask": stacked_with_response_mask,
            "responses": torch.stack(padded_responses_with),  # [B, R]
        }

        # Create batch without thinking
        batch_without_thinking_dict = {
            "input_ids": stacked_without_input_ids,
            "attention_mask": stacked_without_attention_mask,
            "position_ids": stacked_without_position_ids,
            "response_mask": stacked_without_response_mask,
            "responses": torch.stack(padded_responses_without),  # [B, R]
        }

        batch_with_thinking = DataProto.from_single_dict(batch_with_thinking_dict)
        batch_without_thinking = DataProto.from_single_dict(batch_without_thinking_dict)

        batch_with_thinking.meta_info = batch.meta_info.copy()
        batch_without_thinking.meta_info = batch.meta_info.copy()

        return batch_with_thinking, batch_without_thinking

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ===== COMPUTE INFORMATION GAIN BASED ON REASONING QUALITY =====
                    # New IG formula: IG = log P(gold_solution | prompt + reasoning) - log P(gold_solution | prompt)
                    # This measures: "Does the model's reasoning help predict the CORRECT answer?"
                    if self.config.get("compute_information_gain", False):
                        with marked_timer("information_gain", timing_raw, color="purple"):
                            # Step 1: Construct two batches for IG calculation
                            # - batch_with_reasoning: prompt + reasoning + gold_solution
                            # - batch_without_reasoning: prompt + gold_solution (no reasoning)
                            batch_with_reasoning, batch_without_reasoning = self._construct_ig_batches_with_gold_solution(batch)

                            # Step 2: Compute log probs for both batches
                            # Log P(gold_solution | prompt + reasoning)
                            log_prob_with_reasoning = self.actor_rollout_wg.compute_log_prob(batch_with_reasoning)
                            log_probs_with = log_prob_with_reasoning.batch["old_log_probs"]  # [B, response_length]

                            # Create response mask from the responses tensor (non-padding tokens)
                            # response_mask has shape [B, response_length] matching log_probs
                            responses_with = batch_with_reasoning.batch["responses"]  # [B, response_length]
                            response_masks_with = (responses_with != self.tokenizer.pad_token_id).float()  # [B, response_length]

                            # Log P(gold_solution | prompt only)
                            log_prob_without_reasoning = self.actor_rollout_wg.compute_log_prob(batch_without_reasoning)
                            log_probs_without = log_prob_without_reasoning.batch["old_log_probs"]  # [B, response_length]

                            # Create response mask from the responses tensor (non-padding tokens)
                            responses_without = batch_without_reasoning.batch["responses"]  # [B, response_length]
                            response_masks_without = (responses_without != self.tokenizer.pad_token_id).float()  # [B, response_length]

                            # Step 3: Sum log probs over gold solution tokens
                            sum_with_reasoning = (log_probs_with * response_masks_with).sum(dim=1)  # [B]
                            sum_without_reasoning = (log_probs_without * response_masks_without).sum(dim=1)  # [B]

                            # Step 4: Calculate Information Gain
                            # IG = log P(gold | prompt + reasoning) - log P(gold | prompt)
                            # Positive IG = reasoning helps predict correct answer
                            # Negative IG = reasoning hurts (confuses model)
                            information_gain_raw = sum_with_reasoning - sum_without_reasoning  # [B]

                            # NORMALIZE INFORMATION GAIN to [-1, 1] range using batch statistics
                            # This prevents extremely large IG values from dominating the reward
                            ig_std = information_gain_raw.std() + 1e-8
                            ig_mean = information_gain_raw.mean()

                            # Standardize: (IG - mean) / std, then apply tanh to map to [-1, 1]
                            information_gain_normalized = torch.tanh((information_gain_raw - ig_mean) / (ig_std + 1e-8))

                            # Use normalized IG as the primary metric
                            information_gain = information_gain_normalized  # [B], range: [-1, 1]

                            # Store both raw and normalized IG in batch for analysis
                            batch.batch["information_gain_raw"] = information_gain_raw  # [B], raw values
                            batch.batch["information_gain"] = information_gain  # [B], normalized to [-1, 1]

                            # CRITICAL: Inject information gain into non_tensor_batch so reward function can access it
                            ig_numpy = information_gain.cpu().numpy()  # Normalized values
                            ig_raw_numpy = information_gain_raw.cpu().numpy()  # Raw values for logging

                            # Get or initialize extra_info as a list first (for manipulation)
                            # CRITICAL: Deep copy each dict to avoid shared references after batch.repeat()
                            import copy
                            if "extra_info" not in batch.non_tensor_batch:
                                extra_info_list = [{} for _ in range(len(batch))]
                            elif isinstance(batch.non_tensor_batch["extra_info"], np.ndarray):
                                # DEEP COPY to avoid shared dict references within GRPO groups
                                extra_info_list = [copy.deepcopy(item) if isinstance(item, dict) else {}
                                                   for item in batch.non_tensor_batch["extra_info"]]
                            else:
                                # DEEP COPY to avoid shared dict references within GRPO groups
                                extra_info_list = [copy.deepcopy(item) if isinstance(item, dict) else {}
                                                   for item in batch.non_tensor_batch["extra_info"]]

                            # Inject information gain values for each sample
                            for i in range(len(batch)):
                                if not isinstance(extra_info_list[i], dict):
                                    extra_info_list[i] = {}
                                # Normalized IG (primary, used by reward function)
                                extra_info_list[i]["information_gain"] = float(ig_numpy[i])
                                # Raw IG (for analysis/logging)
                                extra_info_list[i]["information_gain_raw"] = float(ig_raw_numpy[i])

                            # IMPORTANT: Convert back to numpy array (required by DataProto.chunk())
                            batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)

                            # Log metrics (both raw and normalized)
                            ig_metrics = {
                                # Normalized IG ([-1, 1] range)
                                "curriculum/information_gain_mean": information_gain.mean().item(),
                                "curriculum/information_gain_std": information_gain.std().item(),
                                "curriculum/information_gain_min": information_gain.min().item(),
                                "curriculum/information_gain_max": information_gain.max().item(),
                                # Raw IG (original scale, for comparison)
                                "curriculum/information_gain_raw_mean": information_gain_raw.mean().item(),
                                "curriculum/information_gain_raw_std": information_gain_raw.std().item(),
                                "curriculum/information_gain_raw_min": information_gain_raw.min().item(),
                                "curriculum/information_gain_raw_max": information_gain_raw.max().item(),
                            }
                            metrics.update(ig_metrics)
                    # ===== END INFORMATION GAIN =====

                    # ===== COMPUTE CUSTOM CURRICULUM METRICS =====
                    # This is called AFTER IG computation but BEFORE reward computation
                    # Results are injected into batch.non_tensor_batch["extra_info"] for reward function
                    if self.config.get("custom_curriculum_metrics"):
                        with marked_timer("curriculum_metrics", timing_raw, color="magenta"):
                            curriculum_metrics_config = self.config.custom_curriculum_metrics

                            # Load the custom metrics function (same pattern as custom reward function)
                            if not hasattr(self, "_curriculum_metrics_fn"):
                                import sys
                                import os
                                from importlib.util import spec_from_file_location, module_from_spec

                                file_path = curriculum_metrics_config.get("path")
                                function_name = curriculum_metrics_config.get("name")

                                if file_path and function_name:
                                    module_name = "custom_curriculum_metrics_module"
                                    module = sys.modules.get(module_name, None)

                                    if module is None:
                                        if not os.path.exists(file_path):
                                            raise FileNotFoundError(f"Curriculum metrics file not found: {file_path}")

                                        spec = spec_from_file_location(module_name, file_path)
                                        module = module_from_spec(spec)
                                        sys.modules[module_name] = module
                                        spec.loader.exec_module(module)

                                    self._curriculum_metrics_fn = getattr(module, function_name)
                                else:
                                    self._curriculum_metrics_fn = None

                            # Call the curriculum metrics function
                            if hasattr(self, "_curriculum_metrics_fn") and self._curriculum_metrics_fn is not None:
                                curriculum_metrics_dict = self._curriculum_metrics_fn(batch, self.tokenizer)

                                # Inject results into batch.non_tensor_batch["extra_info"]
                                # Deep copy extra_info if it exists to avoid shared references
                                import copy
                                if "extra_info" not in batch.non_tensor_batch:
                                    extra_info_list = [{} for _ in range(len(batch))]
                                elif isinstance(batch.non_tensor_batch["extra_info"], np.ndarray):
                                    extra_info_list = [copy.deepcopy(item) if isinstance(item, dict) else {}
                                                       for item in batch.non_tensor_batch["extra_info"]]
                                else:
                                    extra_info_list = [copy.deepcopy(item) if isinstance(item, dict) else {}
                                                       for item in batch.non_tensor_batch["extra_info"]]

                                # Merge curriculum metrics into each sample's extra_info
                                for key, values in curriculum_metrics_dict.items():
                                    if len(values) == len(batch):
                                        for i in range(len(batch)):
                                            if not isinstance(extra_info_list[i], dict):
                                                extra_info_list[i] = {}
                                            extra_info_list[i][key] = values[i]

                                # Update batch with merged extra_info
                                batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)
                    # ===== END CUSTOM CURRICULUM METRICS =====

                    # ===== COMPUTE REWARD =====
                    # Reward computation moved here (after information gain) so reward function can access IG
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Aggregate format reward metrics for logging
                            # Note: IG metrics (raw and normalized) are already logged separately at lines 1590-1602
                            format_metric_keys = [
                                "format_reward", "format_reward_scaled",
                                "has_reasoning_header", "has_solution_header", "has_boxed_answer",
                                "correct_order", "reasoning_substantive", "solution_substantive",
                                "reasoning_header_count", "solution_header_count", "boxed_answer_count"
                            ]

                            format_metrics = {}
                            for key in format_metric_keys:
                                if key in reward_extra_infos_dict:
                                    values = reward_extra_infos_dict[key]
                                    # Only aggregate numeric values, keep NaN values (don't filter)
                                    if values and isinstance(values[0], (int, float, np.integer, np.floating)):
                                        values_array = np.array(values, dtype=np.float32)
                                        format_metrics[f"format/{key}_mean"] = float(np.mean(values_array))
                                        # Add std, min, max for non-count metrics
                                        if not key.endswith("_count"):
                                            format_metrics[f"format/{key}_std"] = float(np.std(values_array))
                                            format_metrics[f"format/{key}_min"] = float(np.min(values_array))
                                            format_metrics[f"format/{key}_max"] = float(np.max(values_array))

                            metrics.update(format_metrics)

                            # Aggregate IG reward metrics (final binary reward after all gating)
                            if "ig_reward" in reward_extra_infos_dict:
                                ig_reward_values = reward_extra_infos_dict["ig_reward"]
                                if ig_reward_values and isinstance(ig_reward_values[0], (int, float, np.integer, np.floating)):
                                    ig_reward_array = np.array(ig_reward_values, dtype=np.float32)
                                    ig_reward_metrics = {
                                        "curriculum/information_gain_reward_mean": float(np.mean(ig_reward_array)),
                                        "curriculum/information_gain_reward_std": float(np.std(ig_reward_array)),
                                        "curriculum/information_gain_reward_min": float(np.min(ig_reward_array)),
                                        "curriculum/information_gain_reward_max": float(np.max(ig_reward_array)),
                                    }
                                    metrics.update(ig_reward_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
