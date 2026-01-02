"""
HER+HRL sampling functions.

Combines hindsight experience replay with hierarchical RL by applying
goal relabeling at both the meta-level (final goals) and sub-level (subgoals).

Based on:
- Andrychowicz et al., "Hindsight Experience Replay" (2017)
- Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning" (2018)
"""

import numpy as np
from typing import Dict, Callable


def make_sample_herhrl_transitions(replay_strategy, replay_k, reward_fun,
                                   subgoal_horizon=10, intrinsic_reward_scale=1.0):
    """
    Creates a sampling function that applies HER at both hierarchy levels.
    
    The key idea is that we relabel not just the final goal (standard HER),
    but also the subgoals that the meta-controller set. This lets the
    sub-controller learn from failed subgoal attempts.
    
    Args:
        replay_strategy: 'future', 'final', 'episode', or 'none'
        replay_k: ratio of HER replays to regular (k=4 means 80% HER)
        reward_fun: function to recompute rewards after relabeling
        subgoal_horizon: steps between subgoal updates
        intrinsic_reward_scale: weight for subgoal-achievement rewards
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    def _sample_herhrl_transitions(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        
        # Sample random episodes and timesteps
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }
        
        # Figure out which subgoal period each sample is in
        subgoal_period = t_samples // subgoal_horizon
        subgoal_start = subgoal_period * subgoal_horizon
        subgoal_end = np.minimum(subgoal_start + subgoal_horizon, T)
        
        # Standard HER: relabel final goals with future achieved goals
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = (np.random.uniform(size=batch_size) * (T - t_samples)).astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        
        # HRL extension: also relabel subgoals with achieved intermediate states
        if 'sg' in episode_batch:
            subgoal_her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
            subgoal_future_t = np.minimum(subgoal_end[subgoal_her_indexes], T - 1)
            
            future_sg = episode_batch['ag'][episode_idxs[subgoal_her_indexes], subgoal_future_t]
            transitions['sg'][subgoal_her_indexes] = future_sg
            
            # Intrinsic reward: did we get close to the subgoal?
            sg_distances = np.linalg.norm(transitions['ag_2'] - transitions['sg'], axis=-1)
            intrinsic_rewards = np.where(sg_distances < 0.05, 0.0, -1.0)
            transitions['r_intrinsic'] = intrinsic_rewards * intrinsic_reward_scale
        
        # Recompute extrinsic rewards with new goals
        info = {k.replace('info_', ''): v for k, v in transitions.items() if k.startswith('info_')}
        transitions['r'] = reward_fun(ag_2=transitions['ag_2'], g=transitions['g'], info=info)
        
        # Combine rewards if we have intrinsic
        if 'r_intrinsic' in transitions:
            transitions['r_total'] = transitions['r'] + transitions['r_intrinsic']
        
        transitions = {k: v.reshape(batch_size, *v.shape[1:]) for k, v in transitions.items()}
        return transitions

    return _sample_herhrl_transitions


def make_sample_meta_transitions(replay_strategy, replay_k, reward_fun, subgoal_horizon=10):
    """
    Sampling function for the meta-controller's replay buffer.
    
    Meta-controller transitions span multiple low-level steps, so we sample
    at subgoal boundaries and accumulate rewards over the subgoal period.
    """
    future_p = 1 - (1. / (1 + replay_k)) if replay_strategy == 'future' else 0

    def _sample_meta_transitions(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        
        num_subgoal_periods = max(1, T // subgoal_horizon)
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        period_samples = np.random.randint(num_subgoal_periods, size=batch_size)
        t_samples = period_samples * subgoal_horizon
        
        # State at start of subgoal period
        transitions = {
            'o': episode_batch['o'][episode_idxs, t_samples].copy(),
            'g': episode_batch['g'][episode_idxs, t_samples].copy(),
            'ag': episode_batch['ag'][episode_idxs, t_samples].copy(),
        }
        
        # State at end of subgoal period
        t_next = np.minimum(t_samples + subgoal_horizon, T - 1)
        transitions['o_2'] = episode_batch['o'][episode_idxs, t_next].copy()
        transitions['ag_2'] = episode_batch['ag'][episode_idxs, t_next].copy()
        
        if 'sg' in episode_batch:
            transitions['sg'] = episode_batch['sg'][episode_idxs, t_samples].copy()
        
        # Sum rewards over the subgoal period
        rewards = np.zeros(batch_size)
        for dt in range(subgoal_horizon):
            t_cur = np.minimum(t_samples + dt, T - 1)
            if 'r' in episode_batch:
                rewards += episode_batch['r'][episode_idxs, t_cur]
        transitions['r'] = rewards
        
        # HER for final goals
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = (np.random.uniform(size=batch_size) * (T - t_samples)).astype(int)
        future_t = np.minimum((t_samples + 1 + future_offset)[her_indexes], T - 1)
        
        transitions['g'][her_indexes] = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['r'] = reward_fun(ag_2=transitions['ag_2'], g=transitions['g'], info={})
        
        return transitions

    return _sample_meta_transitions


def compute_goal_distance(achieved, desired):
    """L2 distance between achieved and desired goals."""
    return np.linalg.norm(achieved - desired, axis=-1)


def check_goal_achieved(achieved, desired, threshold=0.05):
    """Check if goal is achieved (within threshold)."""
    return compute_goal_distance(achieved, desired) < threshold
