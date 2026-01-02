"""
Meta-controller for hierarchical RL.

The meta-controller operates on a slower timescale, setting subgoals
for the sub-controller to achieve. Updates every k environment steps.
"""

import numpy as np
import tensorflow as tf

from baselines.her.util import store_args, nn
from baselines.her.normalizer import Normalizer


class MetaController:
    """High-level policy that sets subgoals for the sub-controller."""
    
    @store_args
    def __init__(self, input_dims, hidden=256, layers=3, subgoal_horizon=10,
                 intrinsic_reward_scale=1.0, max_subgoal=1.0, polyak=0.95,
                 Q_lr=0.001, pi_lr=0.001, norm_eps=0.01, norm_clip=5,
                 scope='meta', reuse=False, **kwargs):
        
        self.dimo = input_dims['o']
        self.dimg = input_dims['g']
        self.dimsg = input_dims.get('sg', input_dims['g'])
        
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        with tf.variable_scope(self.scope):
            self._create_network(reuse=reuse)
    
    def _create_network(self, reuse=False):
        # Placeholders
        self.o_tf = tf.placeholder(tf.float32, shape=(None, self.dimo), name='obs')
        self.g_tf = tf.placeholder(tf.float32, shape=(None, self.dimg), name='goal')
        self.sg_tf = tf.placeholder(tf.float32, shape=(None, self.dimsg), name='subgoal')
        self.r_tf = tf.placeholder(tf.float32, shape=(None,), name='reward')
        
        # Normalizers
        with tf.variable_scope('o_stats'):
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats'):
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        
        o_norm = self.o_stats.normalize(self.o_tf)
        g_norm = self.g_stats.normalize(self.g_tf)
        
        # Actor: (obs, goal) -> subgoal
        actor_input = tf.concat([o_norm, g_norm], axis=1)
        with tf.variable_scope('pi'):
            self.subgoal_tf = self.max_subgoal * tf.tanh(
                nn(actor_input, [self.hidden] * self.layers + [self.dimsg]))
        
        # Critic: Q(obs, goal, subgoal)
        critic_input = tf.concat([o_norm, g_norm, self.sg_tf / self.max_subgoal], axis=1)
        with tf.variable_scope('Q'):
            self.Q_tf = nn(critic_input, [self.hidden] * self.layers + [1])
        
        # Q with actor's subgoal (for policy gradient)
        critic_input_pi = tf.concat([o_norm, g_norm, self.subgoal_tf / self.max_subgoal], axis=1)
        with tf.variable_scope('Q', reuse=True):
            self.Q_pi_tf = nn(critic_input_pi, [self.hidden] * self.layers + [1], reuse=True)
    
    def get_subgoal(self, observation, goal, noise_eps=0.0):
        """Generate a subgoal given current state and final goal."""
        o = np.array(observation).reshape(1, -1)
        g = np.array(goal).reshape(1, -1)
        
        subgoal = self.sess.run(self.subgoal_tf, feed_dict={self.o_tf: o, self.g_tf: g})
        
        if noise_eps > 0:
            noise = noise_eps * self.max_subgoal * np.random.randn(*subgoal.shape)
            subgoal = np.clip(subgoal + noise, -self.max_subgoal, self.max_subgoal)
        
        return subgoal.flatten()
    
    def compute_intrinsic_reward(self, achieved_goal, subgoal, threshold=0.05):
        """Reward for sub-controller based on subgoal proximity."""
        distance = np.linalg.norm(achieved_goal - subgoal)
        return 0.0 if distance < threshold else -1.0
    
    def should_update_subgoal(self, steps_since_update):
        """Check if it's time to set a new subgoal."""
        return steps_since_update >= self.subgoal_horizon


class SubgoalBuffer:
    """Replay buffer for meta-controller transitions."""
    
    def __init__(self, buffer_size, obs_dim, goal_dim, subgoal_dim):
        self.buffer_size = buffer_size
        self.current_size = 0
        self.pointer = 0
        
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.goals = np.zeros((buffer_size, goal_dim), dtype=np.float32)
        self.subgoals = np.zeros((buffer_size, subgoal_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
    
    def store(self, obs, goal, subgoal, reward, next_obs):
        idx = self.pointer
        self.observations[idx] = obs
        self.goals[idx] = goal
        self.subgoals[idx] = subgoal
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.current_size, size=batch_size)
        return {
            'o': self.observations[indices],
            'g': self.goals[indices],
            'sg': self.subgoals[indices],
            'r': self.rewards[indices],
            'o_next': self.next_observations[indices]
        }
