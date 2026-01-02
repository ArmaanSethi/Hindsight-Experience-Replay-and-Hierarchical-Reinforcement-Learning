"""
Hierarchical Reinforcement Learning with Hindsight Experience Replay.

This package implements a two-level hierarchical RL system combined with
hindsight experience replay for sample-efficient learning in sparse-reward
robotic manipulation tasks.

Modules:
    meta_controller: High-level policy that sets subgoals
    sub_policy: Low-level policy that achieves subgoals with primitive actions
    herhrl: Combined HER+HRL sampling and training utilities

Example:
    from hrl import MetaController, SubPolicy, make_sample_herhrl_transitions
    
    meta = MetaController(input_dims)
    sub = SubPolicy(input_dims)
    sample_fn = make_sample_herhrl_transitions('future', 4, reward_fun)
"""

from .meta_controller import MetaController, SubgoalBuffer
from .sub_policy import SubPolicy, IntrinsicRewardCalculator
from .herhrl import make_sample_herhrl_transitions, make_sample_meta_transitions

__all__ = [
    'MetaController',
    'SubgoalBuffer', 
    'SubPolicy',
    'IntrinsicRewardCalculator',
    'make_sample_herhrl_transitions',
    'make_sample_meta_transitions',
]
