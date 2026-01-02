"""
HER+HRL: Hindsight Experience Replay with Hierarchical Reinforcement Learning.

This module combines two powerful techniques for sample-efficient learning
in sparse-reward robotic manipulation tasks:

1. **Hindsight Experience Replay (HER)**:
   - Relabels failed trajectories with achieved goals
   - Enables learning from every experience, even failures
   - Crucial for sparse reward environments

2. **Hierarchical Reinforcement Learning (HRL)**:
   - Meta-controller sets intermediate subgoals
   - Sub-controller executes primitive actions to achieve subgoals
   - Intrinsic rewards provide dense signal for low-level learning
   - Temporal abstraction enables long-horizon planning

Combined Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Meta-Controller                       │
    │         (High-level policy, updates every k steps)       │
    │                  π_high(g_sub | s, g_final)              │
    └───────────────────────┬─────────────────────────────────┘
                            │ subgoal (g_sub)
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    Sub-Controller                        │
    │         (Low-level policy, updates every step)           │
    │                  π_low(a | s, g_sub)                     │
    │              + intrinsic reward r_int                    │
    └───────────────────────┬─────────────────────────────────┘
                            │ action
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     Environment                          │
    │              s', r_ext (sparse reward)                   │
    └─────────────────────────────────────────────────────────┘

Key Innovation:
    HER is applied at BOTH levels of the hierarchy:
    - Meta-level: Relabel final goals with actually achieved goals
    - Sub-level: Relabel subgoals with actually achieved intermediate states
    
    This dramatically improves sample efficiency by learning from
    every trajectory segment, not just successful ones.

References:
    - Andrychowicz et al., "Hindsight Experience Replay" (2017)
    - Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning" (2018)
    - Levy et al., "Learning Multi-Level Hierarchies with Hindsight" (2019)
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple


def make_sample_herhrl_transitions(
    replay_strategy: str,
    replay_k: int,
    reward_fun: Callable,
    subgoal_horizon: int = 10,
    intrinsic_reward_scale: float = 1.0
) -> Callable:
    """
    Creates a sample function for HER+HRL experience replay.
    
    This sampling strategy applies hindsight relabeling at both levels
    of the hierarchy, enabling efficient learning from failed trajectories.
    
    Args:
        replay_strategy: HER strategy ('future', 'final', 'episode', 'none')
            - 'future': Sample goals from future timesteps (recommended)
            - 'final': Use final achieved goal
            - 'episode': Sample from anywhere in episode
            - 'none': No hindsight relabeling (standard DDPG)
        replay_k: Ratio of HER replays to regular replays
            (e.g., k=4 means 80% HER, 20% regular)
        reward_fun: Function to recompute rewards after goal relabeling
        subgoal_horizon: Number of low-level steps per subgoal (k in HRL)
        intrinsic_reward_scale: Scaling factor for intrinsic subgoal rewards
        
    Returns:
        sample_transitions: Function that samples from the replay buffer
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    elif replay_strategy == 'none':
        future_p = 0
    else:
        future_p = 1 - (1. / (1 + replay_k))  # Same as 'future' for other strategies

    def _sample_herhrl_transitions(
        episode_batch: Dict[str, np.ndarray],
        batch_size_in_transitions: int
    ) -> Dict[str, np.ndarray]:
        """
        Sample transitions with hierarchical hindsight relabeling.
        
        Args:
            episode_batch: Dict of arrays with shape (buffer_size, T, dim)
                Keys: 'o' (obs), 'u' (actions), 'g' (goals), 'ag' (achieved goals)
            batch_size_in_transitions: Number of transitions to sample
            
        Returns:
            transitions: Dict of sampled and relabeled transitions
        """
        T = episode_batch['u'].shape[1]  # Timesteps per episode
        rollout_batch_size = episode_batch['u'].shape[0]  # Number of episodes
        batch_size = batch_size_in_transitions
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Sample episodes and timesteps
        # ═══════════════════════════════════════════════════════════════════
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        
        # Extract transitions at sampled time indices
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: HRL - Identify subgoal boundaries
        # ═══════════════════════════════════════════════════════════════════
        # Determine which subgoal period each transition belongs to
        subgoal_period = t_samples // subgoal_horizon
        subgoal_start = subgoal_period * subgoal_horizon
        subgoal_end = np.minimum(subgoal_start + subgoal_horizon, T)
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Apply HER relabeling for FINAL GOALS
        # ═══════════════════════════════════════════════════════════════════
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        
        # Sample future timesteps for goal relabeling
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        
        # Relabel goals with future achieved goals (standard HER)
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Apply HER relabeling for SUBGOALS (HRL-specific)
        # ═══════════════════════════════════════════════════════════════════
        # For the sub-controller, we also relabel subgoals using achieved
        # intermediate states. This is the key HRL+HER combination.
        
        if 'sg' in episode_batch:
            # Subgoal HER: relabel with achieved goal at end of subgoal period
            subgoal_her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
            
            # Use achieved goal at the end of the current subgoal period
            subgoal_future_t = np.minimum(
                subgoal_end[subgoal_her_indexes],
                T - 1
            )
            
            future_sg = episode_batch['ag'][
                episode_idxs[subgoal_her_indexes],
                subgoal_future_t
            ]
            transitions['sg'][subgoal_her_indexes] = future_sg
            
            # Compute intrinsic rewards for subgoal achievement
            sg_distances = np.linalg.norm(
                transitions['ag_2'] - transitions['sg'],
                axis=-1
            )
            intrinsic_rewards = np.where(
                sg_distances < 0.05,  # Subgoal achievement threshold
                0.0,
                -1.0
            ) * intrinsic_reward_scale
            
            # Store intrinsic rewards (can be combined with extrinsic)
            transitions['r_intrinsic'] = intrinsic_rewards
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: Recompute extrinsic rewards after relabeling
        # ═══════════════════════════════════════════════════════════════════
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        
        reward_params = {
            'ag_2': transitions['ag_2'],
            'g': transitions['g'],
            'info': info
        }
        transitions['r'] = reward_fun(**reward_params)
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: Combine intrinsic and extrinsic rewards
        # ═══════════════════════════════════════════════════════════════════
        if 'r_intrinsic' in transitions:
            # Total reward = extrinsic + scaled intrinsic
            transitions['r_total'] = (
                transitions['r'] + 
                intrinsic_reward_scale * transitions['r_intrinsic']
            )
        
        # Reshape transitions
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        
        assert transitions['u'].shape[0] == batch_size_in_transitions
        
        return transitions

    return _sample_herhrl_transitions


def make_sample_meta_transitions(
    replay_strategy: str,
    replay_k: int,
    reward_fun: Callable,
    subgoal_horizon: int = 10
) -> Callable:
    """
    Creates a sample function for meta-controller (high-level) experience replay.
    
    The meta-controller operates on a slower timescale, so its transitions
    span multiple low-level steps. HER is applied to relabel the final goals.
    
    Args:
        replay_strategy: HER strategy for goal relabeling
        replay_k: Ratio of HER to regular replays
        reward_fun: Function to recompute rewards
        subgoal_horizon: Number of low-level steps per meta-transition
        
    Returns:
        sample_transitions: Function for meta-controller replay
    """
    future_p = 1 - (1. / (1 + replay_k)) if replay_strategy == 'future' else 0

    def _sample_meta_transitions(
        episode_batch: Dict[str, np.ndarray],
        batch_size_in_transitions: int
    ) -> Dict[str, np.ndarray]:
        """
        Sample meta-controller transitions (one per subgoal_horizon steps).
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        
        # Sample at subgoal boundaries
        num_subgoal_periods = T // subgoal_horizon
        if num_subgoal_periods == 0:
            num_subgoal_periods = 1
            
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        period_samples = np.random.randint(num_subgoal_periods, size=batch_size)
        t_samples = period_samples * subgoal_horizon
        
        # Get state at start of subgoal period
        transitions = {
            'o': episode_batch['o'][episode_idxs, t_samples].copy(),
            'g': episode_batch['g'][episode_idxs, t_samples].copy(),
            'ag': episode_batch['ag'][episode_idxs, t_samples].copy(),
        }
        
        # Get state at end of subgoal period
        t_next = np.minimum(t_samples + subgoal_horizon, T - 1)
        transitions['o_2'] = episode_batch['o'][episode_idxs, t_next].copy()
        transitions['ag_2'] = episode_batch['ag'][episode_idxs, t_next].copy()
        
        # Get subgoal that was set (if stored)
        if 'sg' in episode_batch:
            transitions['sg'] = episode_batch['sg'][episode_idxs, t_samples].copy()
        
        # Compute cumulative reward over subgoal period
        rewards = np.zeros(batch_size)
        for dt in range(subgoal_horizon):
            t_cur = np.minimum(t_samples + dt, T - 1)
            rewards += episode_batch.get('r', np.zeros_like(rewards))[episode_idxs, t_cur]
        transitions['r'] = rewards
        
        # HER: Relabel final goals
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_t = (t_samples + 1 + future_offset.astype(int))[her_indexes]
        future_t = np.minimum(future_t, T - 1)
        
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        
        # Recompute rewards
        info = {}
        reward_params = {'ag_2': transitions['ag_2'], 'g': transitions['g'], 'info': info}
        transitions['r'] = reward_fun(**reward_params)
        
        return transitions

    return _sample_meta_transitions


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_goal_distance(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    """Compute L2 distance between achieved and desired goals."""
    return np.linalg.norm(achieved_goal - desired_goal, axis=-1)


def check_goal_achieved(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    threshold: float = 0.05
) -> np.ndarray:
    """Check if goal is achieved within threshold."""
    distance = compute_goal_distance(achieved_goal, desired_goal)
    return distance < threshold
