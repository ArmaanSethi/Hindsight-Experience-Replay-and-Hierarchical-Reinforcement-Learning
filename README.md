# Hierarchical Reinforcement Learning with Hindsight Experience Replay

[![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org)
[![TensorFlow 1.x](https://img.shields.io/badge/TensorFlow-1.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Combining temporal abstraction with sample-efficient learning for robotic manipulation.**

This project implements a two-level hierarchical reinforcement learning (HRL) system integrated with Hindsight Experience Replay (HER) for learning goal-conditioned policies in sparse-reward robotic environments.

> ⚠️ **Note**: This codebase uses TensorFlow 1.x and MuJoCo, which require specific environment setup. See [Setup](#setup) for details.

---

## 🎯 Overview

Learning robotic manipulation from sparse rewards is challenging because:
1. **Sparse feedback** — The agent only knows if it succeeded or failed
2. **Long horizons** — Many steps between action and reward
3. **Sample inefficiency** — Random exploration rarely finds success

This project addresses all three issues by combining:

| Technique | Problem Solved | Key Idea |
|-----------|---------------|----------|
| **HER** | Sparse rewards | Learn from failures by relabeling goals |
| **HRL** | Long horizons | Decompose tasks into subgoals |
| **HER+HRL** | Sample efficiency | Apply hindsight at both hierarchy levels |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    META-CONTROLLER                          │
│         π_high(subgoal | state, final_goal)                 │
│         Updates every k=10 environment steps                │
└───────────────────────┬─────────────────────────────────────┘
                        │ subgoal
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    SUB-CONTROLLER                           │
│         π_low(action | state, subgoal)                      │
│         + intrinsic reward for subgoal achievement          │
│         + HER relabeling of subgoals                        │
└───────────────────────┬─────────────────────────────────────┘
                        │ primitive action
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT                              │
│         OpenAI Gym Fetch tasks (MuJoCo)                     │
│         Sparse reward: 0 if goal achieved, -1 otherwise    │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation**: HER is applied at **both levels** of the hierarchy:
- **Meta-level**: Relabel final goals with actually achieved goals
- **Sub-level**: Relabel subgoals with achieved intermediate states

This dramatically improves sample efficiency by extracting learning signal from every trajectory.

---

## 📊 Results

Experiments on OpenAI Gym Fetch environments:

| Environment | HER Only | HER+HRL | Improvement |
|-------------|----------|---------|-------------|
| FetchPush-v0 | 85% | **94%** | +9% |
| FetchPickAndPlace-v0 | 72% | **89%** | +17% |
| FetchSlide-v0 | 31% | **48%** | +17% |

*Success rate after 200 epochs. See [results/](results/) for training curves.*

### Training Curves

Training curves and detailed metrics are available in:
- `results/push200_herhrl/` — FetchPush with HER+HRL
- `results/pp200_herhrl/` — FetchPickAndPlace with HER+HRL  
- `results/slide500herhrl/` — FetchSlide with HER+HRL (500 epochs)

---

## 📁 Project Structure

```
├── src/
│   └── hrl/                          # Core HRL implementation
│       ├── meta_controller.py        # High-level subgoal policy
│       ├── sub_policy.py             # Low-level action policy
│       ├── herhrl.py                 # Combined HER+HRL sampling
│       └── __init__.py
│
├── baselines/                        # OpenAI Baselines (modified)
│   └── baselines/
│       ├── her/                      # Original HER implementation
│       └── herhrl/                   # HER+HRL integration
│           ├── ddpg.py               # DDPG agent
│           ├── rollout.py            # Experience collection
│           └── experiment/           # Training scripts
│
├── results/                          # Experiment results
│   ├── push200_her/                  # Baseline HER
│   ├── push200herhrl/                # HER+HRL
│   └── ...
│
└── docs/                             # Academic reports
    ├── Armaan-Final-Report.pdf       # Full project report
    ├── Armaan Sethi - Project Progress Report.pdf
    └── Armaan Sethi - Robotics Project Proposal.pdf
```

---

## 🚀 Usage

### Training

```bash
# Train HER+HRL agent on FetchPush
python -m baselines.herhrl.experiment.train \
    --env_name FetchPush-v0 \
    --n_epochs 200 \
    --num_cpu 4 \
    --replay_strategy future

# Train on FetchPickAndPlace (harder task)
python -m baselines.herhrl.experiment.train \
    --env_name FetchPickAndPlace-v0 \
    --n_epochs 200 \
    --replay_strategy future
```

### Evaluation

```bash
# Visualize trained policy
python -m baselines.herhrl.experiment.play \
    results/push200herhrl/policy_best.pkl
```

### Demo Videos

- [FetchPush with HER+HRL](https://youtu.be/pPzTOkPKF2o)
- [FetchPickAndPlace](https://youtu.be/PcBb0IYE4F0)
- [FetchSlide](https://youtu.be/7k19-bpJLTA)

---

## 🔧 Setup

### Requirements

- Python 3.6+
- TensorFlow 1.x (1.12-1.15 recommended)
- MuJoCo 1.50 or 2.0 with license
- OpenAI Gym with MuJoCo support
- mpi4py for parallel training

### Installation

```bash
# Clone repository
git clone https://github.com/ArmaanSethi/Hindsight-Experience-Replay-and-Hierarchical-Reinforcement-Learning.git
cd Hindsight-Experience-Replay-and-Hierarchical-Reinforcement-Learning

# Install dependencies
pip install tensorflow==1.15
pip install mujoco-py gym[robotics] mpi4py

# Install baselines
cd baselines
pip install -e .
```

> **Note**: MuJoCo requires a license. Academic licenses are free at [mujoco.org](https://mujoco.org).

---

## 📚 References

1. Andrychowicz et al., ["Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495) (NeurIPS 2017)
2. Nachum et al., ["Data-Efficient Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1805.08296) (NeurIPS 2018)
3. Levy et al., ["Learning Multi-Level Hierarchies with Hindsight"](https://arxiv.org/abs/1712.00948) (ICLR 2019)

---

## 📄 Academic Reports

For detailed methodology and analysis, see the full reports in [docs/](docs/):

- **Final Report**: Complete methodology, experiments, and results analysis
- **Progress Report**: Mid-project status and preliminary findings
- **Project Proposal**: Initial motivation and planned approach

---

## 📝 License

MIT License - See [LICENSE](baselines/LICENSE) for details.

---

*Built for COMP 781: Robotics at Rice University*
