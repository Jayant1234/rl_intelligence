# RL Intelligence: OmniMath curriculum + information-gain experiments

This repository is no longer presented as a generic mirror of the upstream `verl` README. It documents the custom work added here around **OmniMath curriculum learning**, **information gain (IG)** as the main reasoning-quality signal, and the sequence of experiments reflected in the commit history.

## What this repo changes relative to upstream verl

The work in this branch focuses on a custom RL setup for mathematical reasoning:

- **Curriculum dataset construction** for OmniMath, where each training sample can expose only a fraction of the solution and the hidden remainder is used as the verification target.
- **Information gain (IG) computation** inside the PPO/GRPO training flow to measure whether the model's generated reasoning helps predict the gold remaining solution.
- **Reward shaping and gating logic** to make IG usable with GRPO, including format rewards, group-level checks, logging, and several iterations on prompt design.
- **Documentation/debugging traces** explaining the reward pipeline and the rationale behind each redesign.

The main files that carry these changes are:

- `examples/data_preprocess/omnimath_curriculum_dataset.py`
- `examples/reward_functions/omnimath_curriculum_metrics.py`
- `examples/reward_functions/omnimath_reward.py`
- `verl/trainer/ppo/ray_trainer.py`
- `REWARD_LOGIC_TRACE.md`
- `docs/omnimath_curriculum/README.md`
- `docs/omnimath_curriculum/TECHNICAL_DETAILS.md`

## Core idea: information gain as the verification signal

The central idea explored here is:

> Does the model's reasoning make the correct continuation of the solution more predictable?

In this repo, IG is defined conceptually as:

- `log P(gold_solution | prompt + reasoning)`
- minus `log P(gold_solution | prompt)`

That makes IG a **verification-style signal**: the reasoning is rewarded when it increases the likelihood of the hidden gold continuation, not merely when it looks plausible or is stated confidently.

### What “IG as the sole verification signal” means here

Across the commit history, the experiments move toward using IG as the **main content/verification reward**, while format rewards are used mainly to:

- enforce structured outputs,
- create useful variance for GRPO,
- and gate IG until outputs become parseable/stable enough.

So the final direction is best described as:

- **IG is the primary reasoning-quality / verification signal**.
- **Format reward is auxiliary scaffolding**, not the substantive correctness signal.

This is also consistent with the technical docs, which explicitly say IG is the primary signal and format reward exists partly to provide within-group variance for GRPO.

## Experiment progression across commits

Below is the experiment story reconstructed from the commit messages plus the OmniMath technical docs in this repo.

### 1. Initial curriculum-learning draft

Early commits introduced the first full draft of the OmniMath curriculum setup:

- curriculum-aware dataset generation,
- reward hooks,
- initial IG integration,
- and supporting docs.

This phase established the overall workflow but still had an older IG formula and ordering issues in the trainer.

### 2. Enable IG in config and wire it into training

The next step was making IG configurable and then moving reward computation to happen **after** IG was computed, so the reward function could actually consume IG.

This appears to have been a key plumbing fix: before that, the reward function could not reliably use the IG values.

### 3. Fix batch/type bugs in IG handling

Several commits then addressed implementation bugs:

- IG dtype conversion problems,
- missing `responses` tensors in the IG path,
- shape mismatches between response tensors and masks,
- and what looks like a bug where IG values could be identical or misaligned across samples/groups.

These commits suggest the early experiments were mostly about getting IG computation to be **correct, aligned, and debuggable** before drawing conclusions from it.

### 4. Normalize IG because raw values were unstable

A major practical issue showed up quickly: the raw IG values were described as being “off the charts.”

That led to a normalization step so IG would become bounded and comparable across batches, instead of overwhelming the rest of the reward signal. This indicates one concrete experimental result seen during development:

- **raw IG magnitudes were too large / unstable to use directly**.

### 5. Redesign the IG formula

A major redesign then changed the IG objective from a weaker formulation to the more meaningful one based on the **gold solution continuation**.

This is the most important conceptual experiment in the branch:

- earlier formulations could reward confidence in generated content,
- the redesigned version instead checks whether reasoning helps predict the correct hidden solution.

That is the clearest implementation of using IG as a true verification signal.

### 6. Improve prompts and response format

Several prompt and output-format experiments followed:

- prompt simplification,
- adding/removing `think` tags,
- changing chat-template handling,
- adding a baseline sentence,
- and simplifying the prompt again in later RDU versions.

These commits imply that stable IG training depended heavily on the model producing reasoning in a consistent structure that the trainer and reward code could parse.

### 7. Add and improve format reward logging

Format rewards were then logged more explicitly and refined. This seems to have been important for understanding whether the model was failing because:

- reasoning quality was poor,
- or because the output format was too noisy for IG to be applied cleanly.

### 8. Gate IG until format is learned

A later experiment explicitly delayed IG rewards until format rewards were already kicking in.

This reveals an important conclusion from the experiments:

- **using IG alone from the very beginning was apparently too brittle in practice**,
- so the training process was staged to learn format first and unlock IG later.

### 9. Remove negative format rewards

Negative format penalties were removed in two adjacent commits.

That suggests the earlier penalty scheme was not helping optimization, and the setup converged toward a cleaner reward structure where:

- format mostly acts as a non-negative gate / bonus,
- IG carries the main content-learning burden.

### 10. Tight group gating for GRPO, then relax it

Another experiment added strict group-level conditions:

- perfect format across the group,
- sufficient IG spread,
- positive IG in at least one sample,
- only then allowing IG reward through.

After that, a later commit replaced a strict binary group rule with a **standard-deviation-based** group check.

This suggests a concrete learning from the experiments:

- the strict binary gating was likely too harsh or too sparse,
- and a variance-aware criterion gave a smoother signal for GRPO.

### 11. RDU 0.4 simplification

The latest visible stage changes the prompt again and simplifies format reward logic, indicating a continuing trend toward:

- simpler prompting,
- simpler auxiliary reward structure,
- IG retained as the central verification objective.

## What results are actually shown in the repo

I do **not** see hard experiment result tables, benchmark scores, or logged run outputs in the current repository snapshot.

So I cannot honestly claim numeric end results such as:

- final OmniMath accuracy,
- exact reward curves from a completed run,
- or a comparison proving one version beat another on held-out evaluation.

### What the repo does show clearly

The repo does show several **qualitative and implementation-level results**:

1. **Raw IG was unstable at first** and had to be normalized.
2. **The IG formulation was redesigned** to use the gold continuation rather than generated continuation, making it a much better verification signal.
3. **GRPO needed variance-aware reward structure**, so format and group-level checks were introduced to avoid zero-advantage groups.
4. **Format-only warmup / gating was needed** before IG became useful.
5. **Strict binary group gating was later loosened** into a standard-deviation-based group check.
6. **Prompt structure mattered a lot**, leading to repeated prompt iterations.

### Metric ranges documented here

The docs include **target or expected ranges**, not confirmed final run results. For example, the OmniMath docs describe metrics such as:

- `curriculum/information_gain_mean` should be positive and ideally increasing,
- `curriculum/information_gain_std` should stay in a healthy range for GRPO signal,
- `format/format_reward_mean` should rise as format compliance improves,
- `reward/rewards_mean` should increase as the combined signal improves.

Those are useful operating expectations, but they are not the same thing as completed experiment results.

## My best reconstruction of what you tried

Based on the commit sequence and local docs, the experimental path looks like this:

1. Build a curriculum-learning dataset that reveals only part of the solution.
2. Use the hidden remainder as the target for a verification-style IG score.
3. Inject IG into the PPO/GRPO trainer as the main reward for reasoning quality.
4. Debug the trainer plumbing until IG values are computed correctly.
5. Normalize IG once raw scores prove too unstable.
6. Improve prompts so the reasoning is easier to parse and evaluate.
7. Add format reward to make outputs structured enough for IG.
8. Use format gating so IG is only active once generations are well-formed.
9. Add group-level checks because GRPO needs within-group variance.
10. Replace overly strict binary gating with a softer variance/std-based criterion.
11. Continue simplifying prompt and format logic while keeping IG as the main substantive signal.

## Practical takeaway

The branch does **not** read like “IG alone worked cleanly from day one.” Instead, it reads like:

- **IG was the intended sole verification/content signal**,
- but it needed **substantial support infrastructure** to be trainable in practice:
  - curriculum construction,
  - prompt engineering,
  - normalization,
  - format rewards,
  - group-level gating,
  - logging,
  - and repeated debugging of tensor construction.

That is a meaningful result in itself: the repo shows both the ambition to use IG as the core verifier and the engineering needed to make that idea usable in RL training.

## Relevant references in this repo

If you want the implementation details behind this README, start with:

- `docs/omnimath_curriculum/README.md`
- `docs/omnimath_curriculum/TECHNICAL_DETAILS.md`
- `REWARD_LOGIC_TRACE.md`
- `examples/reward_functions/omnimath_reward.py`
- `examples/reward_functions/omnimath_curriculum_metrics.py`
- `examples/data_preprocess/omnimath_curriculum_dataset.py`
- `verl/trainer/ppo/ray_trainer.py`

## Note on the old README

The old top-level README was the generic upstream `verl` project README. It has been replaced so the repository front page now describes the actual custom OmniMath/IG work present in this branch.
