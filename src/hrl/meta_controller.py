"""
Hierarchical Meta-Controller for Goal-Conditioned Reinforcement Learning.

This module implements a high-level meta-controller that operates on a slower timescale,
setting subgoals for a lower-level controller to achieve. This creates a two-level
hierarchy that enables efficient exploration and credit assignment in sparse reward settings.

Architecture:
    Meta-Controller (this module)
        ↓ subgoals every k steps
    Sub-Controller (sub_policy.py)
        ↓ primitive actions
    Environment

Key Innovation:
    By combining Hindsight Experience Replay (HER) with Hierarchical RL, we get:
    1. HER enables learning from failed trajectories by relabeling goals
    2. HRL provides temporal abstraction for long-horizon tasks
    3. The combination allows efficient learning in sparse-reward robotic manipulation

References:
    - Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning" (2018)
    - Levy et al., "Hierarchical Actor-Critic" (2017)
    - Andrychowicz et al., "Hindsight Experience Replay" (2017)
"""

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from baselines.her.util import store_args, nn
from baselines.her.normalizer import Normalizer


class MetaController:
    """
    High-level policy that sets subgoals for the sub-controller.
    
    The meta-controller observes the current state and final goal, then proposes
    intermediate subgoals that decompose the task into manageable steps.
    
    Attributes:
        subgoal_dim (int): Dimension of the subgoal space (typically same as goal space)
        subgoal_horizon (int): Number of low-level steps between subgoal updates
        intrinsic_reward_scale (float): Scaling factor for intrinsic rewards
    """
    
    @store_args
    def __init__(self, input_dims, hidden=256, layers=3, subgoal_horizon=10,
                 intrinsic_reward_scale=1.0, max_subgoal=1.0, polyak=0.95,
                 Q_lr=0.001, pi_lr=0.001, norm_eps=0.01, norm_clip=5,
                 scope='meta', reuse=False, **kwargs):
        """
        Initialize the Meta-Controller.
        
        Args:
            input_dims (dict): Dimensions for observation (o), goal (g), subgoal (sg)
            hidden (int): Number of hidden units per layer
            layers (int): Number of hidden layers
            subgoal_horizon (int): Steps between subgoal updates (k in the paper)
            intrinsic_reward_scale (float): Scale for intrinsic subgoal-reaching rewards
            max_subgoal (float): Maximum magnitude of subgoal outputs
            polyak (float): Polyak averaging coefficient for target network
            Q_lr (float): Learning rate for Q-function (critic)
            pi_lr (float): Learning rate for policy (actor)
            norm_eps (float): Epsilon for numerical stability in normalization
            norm_clip (float): Clipping value for normalized inputs
            scope (str): TensorFlow variable scope
            reuse (bool): Whether to reuse variables
        """
        self.dimo = input_dims['o']
        self.dimg = input_dims['g']
        self.dimsg = input_dims.get('sg', input_dims['g'])  # Subgoal dim defaults to goal dim
        
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        # Create networks
        with tf.variable_scope(self.scope):
            self._create_network(reuse=reuse)
    
    def _create_network(self, reuse=False):
        """
        Create the meta-controller actor-critic networks.
        
        The actor maps (observation, goal) -> subgoal
        The critic evaluates Q(observation, goal, subgoal)
        """
        # Placeholders
        self.o_tf = tf.placeholder(tf.float32, shape=(None, self.dimo), name='observation')
        self.g_tf = tf.placeholder(tf.float32, shape=(None, self.dimg), name='goal')
        self.sg_tf = tf.placeholder(tf.float32, shape=(None, self.dimsg), name='subgoal')
        self.r_tf = tf.placeholder(tf.float32, shape=(None,), name='reward')
        
        # Normalizers for stable training
        with tf.variable_scope('o_stats'):
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats'):
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        
        # Normalize inputs
        o_norm = self.o_stats.normalize(self.o_tf)
        g_norm = self.g_stats.normalize(self.g_tf)
        
        # Actor: maps (o, g) -> subgoal
        actor_input = tf.concat([o_norm, g_norm], axis=1)
        with tf.variable_scope('pi'):
            self.subgoal_tf = self.max_subgoal * tf.tanh(
                nn(actor_input, [self.hidden] * self.layers + [self.dimsg])
            )
        
        # Critic: evaluates Q(o, g, sg)
        critic_input = tf.concat([o_norm, g_norm, self.sg_tf / self.max_subgoal], axis=1)
        with tf.variable_scope('Q'):
            self.Q_tf = nn(critic_input, [self.hidden] * self.layers + [1])
        
        # Critic with actor's subgoal (for policy gradient)
        critic_input_pi = tf.concat([o_norm, g_norm, self.subgoal_tf / self.max_subgoal], axis=1)
        with tf.variable_scope('Q', reuse=True):
            self.Q_pi_tf = nn(critic_input_pi, [self.hidden] * self.layers + [1], reuse=True)
        
        # Target networks (for stable learning)
        self._create_target_network()
    
    def _create_target_network(self):
        """Create target networks for stable Q-learning."""
        # Target actor
        o_norm = self.o_stats.normalize(self.o_tf)
        g_norm = self.g_stats.normalize(self.g_tf)
        actor_input = tf.concat([o_norm, g_norm], axis=1)
        
        with tf.variable_scope('target_pi'):
            self.target_subgoal_tf = self.max_subgoal * tf.tanh(
                nn(actor_input, [self.hidden] * self.layers + [self.dimsg])
            )
        
        # Target critic
        critic_input = tf.concat([o_norm, g_norm, self.target_subgoal_tf / self.max_subgoal], axis=1)
        with tf.variable_scope('target_Q'):
            self.target_Q_tf = nn(critic_input, [self.hidden] * self.layers + [1])
    
    def get_subgoal(self, observation, goal, noise_eps=0.0):
        """
        Generate a subgoal for the current state and goal.
        
        Args:
            observation: Current environment observation
            goal: Final goal to achieve
            noise_eps: Exploration noise scale
            
        Returns:
            subgoal: Intermediate goal for the sub-controller
        """
        o = np.array(observation).reshape(1, -1)
        g = np.array(goal).reshape(1, -1)
        
        subgoal = self.sess.run(self.subgoal_tf, feed_dict={
            self.o_tf: o,
            self.g_tf: g
        })
        
        # Add exploration noise
        if noise_eps > 0:
            noise = noise_eps * self.max_subgoal * np.random.randn(*subgoal.shape)
            subgoal = np.clip(subgoal + noise, -self.max_subgoal, self.max_subgoal)
        
        return subgoal.flatten()
    
    def compute_intrinsic_reward(self, achieved_goal, subgoal, threshold=0.05):
        """
        Compute intrinsic reward for subgoal achievement.
        
        The sub-controller receives intrinsic reward based on how close
        it gets to the subgoal set by the meta-controller.
        
        Args:
            achieved_goal: What the agent actually achieved
            subgoal: Target subgoal from meta-controller
            threshold: Distance threshold for binary reward
            
        Returns:
            reward: Intrinsic reward (0 or -1 for sparse, or shaped)
        """
        distance = np.linalg.norm(achieved_goal - subgoal)
        
        # Sparse intrinsic reward (matches HER's reward structure)
        if distance < threshold:
            return 0.0  # Success
        else:
            return -1.0  # Not yet achieved
    
    def should_update_subgoal(self, steps_since_update):
        """
        Determine if it's time to generate a new subgoal.
        
        Args:
            steps_since_update: Number of low-level steps since last subgoal
            
        Returns:
            bool: True if subgoal should be updated
        """
        return steps_since_update >= self.subgoal_horizon


class SubgoalBuffer:
    """
    Experience replay buffer for meta-controller transitions.
    
    Stores transitions of the form (o, g, sg, r, o', g') where:
    - o: observation when subgoal was set
    - g: final goal
    - sg: subgoal that was set
    - r: cumulative extrinsic reward over subgoal_horizon steps
    - o': observation after subgoal_horizon steps
    """
    
    def __init__(self, buffer_size, obs_dim, goal_dim, subgoal_dim):
        """
        Initialize the subgoal buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            obs_dim: Dimension of observations
            goal_dim: Dimension of goals
            subgoal_dim: Dimension of subgoals
        """
        self.buffer_size = buffer_size
        self.current_size = 0
        self.pointer = 0
        
        # Pre-allocate arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.goals = np.zeros((buffer_size, goal_dim), dtype=np.float32)
        self.subgoals = np.zeros((buffer_size, subgoal_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
    
    def store(self, observation, goal, subgoal, reward, next_observation):
        """Store a meta-controller transition."""
        idx = self.pointer
        
        self.observations[idx] = observation
        self.goals[idx] = goal
        self.subgoals[idx] = subgoal
        self.rewards[idx] = reward
        self.next_observations[idx] = next_observation
        
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.current_size, size=batch_size)
        
        return {
            'o': self.observations[indices],
            'g': self.goals[indices],
            'sg': self.subgoals[indices],
            'r': self.rewards[indices],
            'o_next': self.next_observations[indices]
        }
