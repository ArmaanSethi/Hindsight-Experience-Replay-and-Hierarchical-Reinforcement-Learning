# Hierarchical RL with Hindsight Experience Replay

[![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org)
[![TensorFlow 1.x](https://img.shields.io/badge/TensorFlow-1.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Combining temporal abstraction with sample-efficient learning for robotic manipulation.

> **Note**: Built on TensorFlow 1.x / OpenAI Baselines (2017-2018 era). See [Setup](#setup) for environment requirements.

---

## Overview

This project extends the OpenAI Baselines HER implementation with a two-level hierarchical structure. The key insight is that HER can be applied at both levels of the hierarchy—relabeling not just final goals, but also the intermediate subgoals.

**Core idea**: A meta-controller sets subgoals every k steps, while a sub-controller takes primitive actions to achieve those subgoals. Both controllers benefit from hindsight relabeling.

---

## What I Modified

The HRL extension is implemented in [`src/hrl/`](src/hrl/):

| File | Purpose |
|------|---------|
| [`herhrl.py`](src/hrl/herhrl.py) | Sampling functions that apply HER at both hierarchy levels |
| [`meta_controller.py`](src/hrl/meta_controller.py) | High-level policy that generates subgoals |
| [`sub_policy.py`](src/hrl/sub_policy.py) | Low-level policy + intrinsic reward computation |

The key modification is in `herhrl.py:_sample_herhrl_transitions()`:
- Lines 58-62: Identify subgoal period boundaries
- Lines 65-73: Relabel subgoals with achieved intermediate states (the HRL extension)
- Lines 75-78: Compute intrinsic rewards for subgoal achievement

For integration with the training loop, I modified files in [`baselines/baselines/herhrl/`](baselines/baselines/herhrl/):
- `ddpg.py` — Added subgoal input to the value function
- `rollout.py` — Track subgoals during episode collection
- `replay_buffer.py` — Store subgoal transitions

---

## Results

| Environment | HER | HER+HRL | Outcome |
|-------------|-----|---------|---------|
| FetchReach | ~100% | ~100% | ✓ Baseline |
| FetchPush | ~100% | ~100% | ✓ No degradation |
| FetchPickAndPlace | ~100% | ~100% | ✓ No degradation |
| **FetchSlide** | 60% | **70%** | **+17% relative** |

**Why HRL only helps on FetchSlide**: Hierarchical methods provide benefits proportional to task horizon length ([Nachum et al., 2018](https://arxiv.org/abs/1805.08296)). Short-horizon tasks like FetchReach and FetchPush are solvable within a single subgoal period—temporal abstraction provides no advantage. FetchSlide requires hitting a puck to a *distant* target with *indirect contact*, creating a longer effective horizon where subgoal decomposition becomes valuable.

This aligns with the theoretical result that HRL's sample complexity advantage scales with O(H/k), where H is horizon length and k is the subgoal interval. When H ≈ k (simple tasks), the ratio approaches 1 and HRL reduces to flat RL. When H >> k (FetchSlide), hierarchical decomposition provides meaningful gains.

*Training curves in [`results/`](results/).*

**Demo videos**: [FetchPush](https://youtu.be/pPzTOkPKF2o) | [PickAndPlace](https://youtu.be/PcBb0IYE4F0) | [Slide](https://youtu.be/7k19-bpJLTA)

---

## Project Structure

```
src/hrl/                    # HRL implementation (my code)
├── herhrl.py               # HER+HRL sampling
├── meta_controller.py      # Subgoal generation
└── sub_policy.py           # Low-level policy

baselines/baselines/herhrl/ # Modified OpenAI Baselines
├── ddpg.py                 # DDPG with subgoal conditioning
├── rollout.py              # Episode collection
└── experiment/             # Training scripts

results/                    # Experiment outputs
docs/                       # Academic reports
```

---

## Usage

```bash
# Train on FetchPush
python -m baselines.herhrl.experiment.train \
    --env_name FetchPush-v0 \
    --n_epochs 200 \
    --num_cpu 4

# Visualize trained policy
python -m baselines.herhrl.experiment.play results/push200herhrl/policy_best.pkl
```

---

## Setup

Requires TensorFlow 1.x, MuJoCo, and mpi4py:

```bash
pip install tensorflow==1.15 mujoco-py gym[robotics] mpi4py
cd baselines && pip install -e .
```

---

## References

- Andrychowicz et al., ["Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495) (NeurIPS 2017)
- Nachum et al., ["Data-Efficient Hierarchical RL"](https://arxiv.org/abs/1805.08296) (NeurIPS 2018)

See [`docs/`](docs/) for the full project report and proposal.

---

*COMP 781: Robotics — UNC Chapel Hill*
