"""
Sub-Policy (Lower-Level Controller) for Hierarchical Reinforcement Learning.

This module implements the low-level controller that takes primitive actions
to achieve subgoals set by the meta-controller. It receives intrinsic rewards
for reaching subgoals, enabling dense reward signal even in sparse-reward environments.

Architecture:
    Meta-Controller (meta_controller.py)
        ↓ subgoals (abstract goals)
    Sub-Controller (this module) ← intrinsic rewards for subgoal completion
        ↓ primitive actions
    Environment → extrinsic rewards (sparse)

Key Features:
    1. Goal-conditioned policy: pi(a | s, subgoal)
    2. Intrinsic reward shaping based on subgoal proximity
    3. Seamless integration with HER for sample-efficient learning
    4. DDPG-based continuous control

References:
    - Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning" (2018)
    - Vezhnevets et al., "FeUdal Networks for Hierarchical Reinforcement Learning" (2017)
"""

import numpy as np
import tensorflow as tf

from baselines.her.util import store_args, nn
from baselines.her.normalizer import Normalizer


class SubPolicy:
    """
    Low-level policy that executes primitive actions to achieve subgoals.
    
    The sub-policy is conditioned on subgoals from the meta-controller and
    receives intrinsic rewards for making progress toward those subgoals.
    This enables learning in environments with sparse extrinsic rewards.
    
    Attributes:
        dimo (int): Observation dimension
        dimu (int): Action dimension
        dimsg (int): Subgoal dimension
        max_u (float): Maximum action magnitude
    """
    
    @store_args
    def __init__(self, input_dims, hidden=256, layers=3, max_u=1.0,
                 polyak=0.95, Q_lr=0.001, pi_lr=0.001, 
                 norm_eps=0.01, norm_clip=5, action_l2=1.0,
                 scope='sub_policy', reuse=False, **kwargs):
        """
        Initialize the Sub-Policy.
        
        Args:
            input_dims (dict): Dimensions for observation (o), subgoal (sg), action (u)
            hidden (int): Number of hidden units per layer
            layers (int): Number of hidden layers
            max_u (float): Maximum action magnitude (actions in [-max_u, max_u])
            polyak (float): Polyak averaging coefficient for target networks
            Q_lr (float): Learning rate for Q-function
            pi_lr (float): Learning rate for policy
            norm_eps (float): Epsilon for normalization stability
            norm_clip (float): Clipping value for normalized inputs
            action_l2 (float): L2 regularization on actions
            scope (str): TensorFlow variable scope
            reuse (bool): Whether to reuse variables
        """
        self.dimo = input_dims['o']
        self.dimu = input_dims['u']
        self.dimsg = input_dims.get('sg', input_dims['g'])
        
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        with tf.variable_scope(self.scope):
            self._create_network(reuse=reuse)
    
    def _create_network(self, reuse=False):
        """
        Create the sub-policy actor-critic networks.
        
        The actor maps (observation, subgoal) -> action
        The critic evaluates Q(observation, subgoal, action)
        """
        # Placeholders
        self.o_tf = tf.placeholder(tf.float32, shape=(None, self.dimo), name='observation')
        self.sg_tf = tf.placeholder(tf.float32, shape=(None, self.dimsg), name='subgoal')
        self.u_tf = tf.placeholder(tf.float32, shape=(None, self.dimu), name='action')
        self.r_tf = tf.placeholder(tf.float32, shape=(None,), name='reward')
        
        # Normalizers
        with tf.variable_scope('o_stats'):
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('sg_stats'):
            self.sg_stats = Normalizer(self.dimsg, self.norm_eps, self.norm_clip, sess=self.sess)
        
        # Normalize inputs
        o_norm = self.o_stats.normalize(self.o_tf)
        sg_norm = self.sg_stats.normalize(self.sg_tf)
        
        # Actor: maps (o, sg) -> action
        actor_input = tf.concat([o_norm, sg_norm], axis=1)
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(
                nn(actor_input, [self.hidden] * self.layers + [self.dimu])
            )
        
        # Critic: evaluates Q(o, sg, u)
        critic_input = tf.concat([o_norm, sg_norm, self.u_tf / self.max_u], axis=1)
        with tf.variable_scope('Q'):
            self.Q_tf = nn(critic_input, [self.hidden] * self.layers + [1])
        
        # Critic with actor's action (for policy gradient)
        critic_input_pi = tf.concat([o_norm, sg_norm, self.pi_tf / self.max_u], axis=1)
        with tf.variable_scope('Q', reuse=True):
            self.Q_pi_tf = nn(critic_input_pi, [self.hidden] * self.layers + [1], reuse=True)
    
    def get_action(self, observation, subgoal, noise_eps=0.0, random_eps=0.0):
        """
        Select an action given current observation and subgoal.
        
        Args:
            observation: Current environment observation
            subgoal: Target subgoal from meta-controller
            noise_eps: Scale of Gaussian exploration noise
            random_eps: Probability of taking completely random action
            
        Returns:
            action: Selected action (clipped to [-max_u, max_u])
        """
        o = np.array(observation).reshape(1, -1)
        sg = np.array(subgoal).reshape(1, -1)
        
        action = self.sess.run(self.pi_tf, feed_dict={
            self.o_tf: o,
            self.sg_tf: sg
        })
        
        # Add Gaussian exploration noise
        if noise_eps > 0:
            noise = noise_eps * self.max_u * np.random.randn(*action.shape)
            action = action + noise
        
        # Epsilon-greedy random actions
        if random_eps > 0 and np.random.random() < random_eps:
            action = np.random.uniform(-self.max_u, self.max_u, action.shape)
        
        # Clip to valid range
        action = np.clip(action, -self.max_u, self.max_u)
        
        return action.flatten()


class IntrinsicRewardCalculator:
    """
    Computes intrinsic rewards for subgoal achievement.
    
    The intrinsic reward provides a dense signal for the sub-policy,
    rewarding progress toward the subgoal set by the meta-controller.
    This is crucial for learning in sparse-reward environments.
    
    Reward Structure:
        - Sparse: 0 if subgoal achieved, -1 otherwise (default)
        - Dense: Negative L2 distance to subgoal
        - Shaped: Hybrid with achievement bonus
    """
    
    def __init__(self, reward_type='sparse', achievement_threshold=0.05,
                 distance_scale=1.0, achievement_bonus=1.0):
        """
        Initialize the intrinsic reward calculator.
        
        Args:
            reward_type: One of 'sparse', 'dense', 'shaped'
            achievement_threshold: Distance threshold for considering subgoal achieved
            distance_scale: Scaling factor for distance-based rewards
            achievement_bonus: Bonus reward for achieving subgoal (shaped mode)
        """
        self.reward_type = reward_type
        self.threshold = achievement_threshold
        self.distance_scale = distance_scale
        self.bonus = achievement_bonus
    
    def compute(self, achieved_goal, subgoal):
        """
        Compute intrinsic reward based on subgoal proximity.
        
        Args:
            achieved_goal: What the agent actually achieved (e.g., gripper position)
            subgoal: Target subgoal from meta-controller
            
        Returns:
            reward: Intrinsic reward value
            achieved: Boolean indicating if subgoal was achieved
        """
        distance = np.linalg.norm(np.array(achieved_goal) - np.array(subgoal))
        achieved = distance < self.threshold
        
        if self.reward_type == 'sparse':
            # Match HER's reward structure for consistency
            reward = 0.0 if achieved else -1.0
            
        elif self.reward_type == 'dense':
            # Continuous reward based on distance
            reward = -self.distance_scale * distance
            
        elif self.reward_type == 'shaped':
            # Combination: distance-based + achievement bonus
            reward = -self.distance_scale * distance
            if achieved:
                reward += self.bonus
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        return reward, achieved
    
    def compute_batch(self, achieved_goals, subgoals):
        """
        Compute intrinsic rewards for a batch of transitions.
        
        Args:
            achieved_goals: Array of achieved goals [batch, dim]
            subgoals: Array of subgoals [batch, dim]
            
        Returns:
            rewards: Array of intrinsic rewards [batch]
            achieved: Array of achievement flags [batch]
        """
        distances = np.linalg.norm(achieved_goals - subgoals, axis=1)
        achieved = distances < self.threshold
        
        if self.reward_type == 'sparse':
            rewards = np.where(achieved, 0.0, -1.0)
        elif self.reward_type == 'dense':
            rewards = -self.distance_scale * distances
        elif self.reward_type == 'shaped':
            rewards = -self.distance_scale * distances
            rewards = rewards + self.bonus * achieved.astype(float)
        
        return rewards, achieved
