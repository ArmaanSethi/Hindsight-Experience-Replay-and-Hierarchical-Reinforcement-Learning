"""
Sub-controller (low-level policy) for hierarchical RL.

Takes primitive actions to achieve subgoals set by the meta-controller.
Receives intrinsic rewards for making progress toward subgoals.
"""

import numpy as np
import tensorflow as tf

from baselines.her.util import store_args, nn
from baselines.her.normalizer import Normalizer


class SubPolicy:
    """Low-level policy conditioned on subgoals."""
    
    @store_args
    def __init__(self, input_dims, hidden=256, layers=3, max_u=1.0,
                 polyak=0.95, Q_lr=0.001, pi_lr=0.001, 
                 norm_eps=0.01, norm_clip=5, action_l2=1.0,
                 scope='sub_policy', reuse=False, **kwargs):
        
        self.dimo = input_dims['o']
        self.dimu = input_dims['u']
        self.dimsg = input_dims.get('sg', input_dims['g'])
        
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        with tf.variable_scope(self.scope):
            self._create_network(reuse=reuse)
    
    def _create_network(self, reuse=False):
        self.o_tf = tf.placeholder(tf.float32, shape=(None, self.dimo), name='obs')
        self.sg_tf = tf.placeholder(tf.float32, shape=(None, self.dimsg), name='subgoal')
        self.u_tf = tf.placeholder(tf.float32, shape=(None, self.dimu), name='action')
        self.r_tf = tf.placeholder(tf.float32, shape=(None,), name='reward')
        
        with tf.variable_scope('o_stats'):
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('sg_stats'):
            self.sg_stats = Normalizer(self.dimsg, self.norm_eps, self.norm_clip, sess=self.sess)
        
        o_norm = self.o_stats.normalize(self.o_tf)
        sg_norm = self.sg_stats.normalize(self.sg_tf)
        
        # Actor: (obs, subgoal) -> action
        actor_input = tf.concat([o_norm, sg_norm], axis=1)
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(
                nn(actor_input, [self.hidden] * self.layers + [self.dimu]))
        
        # Critic: Q(obs, subgoal, action)
        critic_input = tf.concat([o_norm, sg_norm, self.u_tf / self.max_u], axis=1)
        with tf.variable_scope('Q'):
            self.Q_tf = nn(critic_input, [self.hidden] * self.layers + [1])
        
        critic_input_pi = tf.concat([o_norm, sg_norm, self.pi_tf / self.max_u], axis=1)
        with tf.variable_scope('Q', reuse=True):
            self.Q_pi_tf = nn(critic_input_pi, [self.hidden] * self.layers + [1], reuse=True)
    
    def get_action(self, observation, subgoal, noise_eps=0.0, random_eps=0.0):
        """Select action given current state and subgoal."""
        o = np.array(observation).reshape(1, -1)
        sg = np.array(subgoal).reshape(1, -1)
        
        action = self.sess.run(self.pi_tf, feed_dict={self.o_tf: o, self.sg_tf: sg})
        
        if noise_eps > 0:
            action += noise_eps * self.max_u * np.random.randn(*action.shape)
        
        if random_eps > 0 and np.random.random() < random_eps:
            action = np.random.uniform(-self.max_u, self.max_u, action.shape)
        
        return np.clip(action, -self.max_u, self.max_u).flatten()


class IntrinsicRewardCalculator:
    """Computes intrinsic rewards for subgoal achievement."""
    
    def __init__(self, reward_type='sparse', threshold=0.05, scale=1.0):
        self.reward_type = reward_type
        self.threshold = threshold
        self.scale = scale
    
    def compute(self, achieved_goal, subgoal):
        """Returns (reward, achieved_flag)."""
        distance = np.linalg.norm(np.array(achieved_goal) - np.array(subgoal))
        achieved = distance < self.threshold
        
        if self.reward_type == 'sparse':
            reward = 0.0 if achieved else -1.0
        elif self.reward_type == 'dense':
            reward = -self.scale * distance
        else:
            reward = -self.scale * distance + (1.0 if achieved else 0.0)
        
        return reward, achieved
    
    def compute_batch(self, achieved_goals, subgoals):
        """Batch version."""
        distances = np.linalg.norm(achieved_goals - subgoals, axis=1)
        achieved = distances < self.threshold
        
        if self.reward_type == 'sparse':
            rewards = np.where(achieved, 0.0, -1.0)
        else:
            rewards = -self.scale * distances
        
        return rewards, achieved
