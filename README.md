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

| Environment | HER Only | HER+HRL | Notes |
|-------------|----------|---------|-------|
| FetchReach | ~100% | ~100% | Solved in <3 epochs |
| FetchPush | ~100% | ~100% | Both methods effective |
| FetchPickAndPlace | ~100% | ~100% | ~200 epochs |
| **FetchSlide** | **60%** | **70%** | HRL advantage on hardest task |

*FetchSlide is the most challenging environment—the agent must hit a puck to a distant target. HER+HRL shows a 10 percentage point improvement on this task.*

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
