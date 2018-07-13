# -*- coding: utf-8 -*-
"""
# @Time    : 07/07/18 10:30 PM
# @Author  : ZHIMIN HOU
# @FileName: Classfication_model.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import gym
import gym.spaces
from sklearn.neural_network import MLPClassifier
import numpy as np
import tensorflow as tf
import pandas as pd

env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space.n)
print(env.observation_space.shape[0])
MAX_EPISODE = 10000


class EnvModel:

    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, memory_size, n_features, actions):

        # the simplest case is to think about the model is a memory which has all past transition information
        self.actions = actions
        self.n_features = n_features
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + self.actions + 1))
        self.memory_counter = 0
        self.batch_size = 64

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, a, r, s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_s_a(self):

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        return batch_memory

    def one_hot(self, a):
        action = np.zeros((self.actions))
        action[a] = 1.
        return action


class Actor_model(object):

    def __init__(self, sess, n_features, n_actions, lr=0.001):

        self.sess = sess
        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.target_a = tf.placeholder(tf.float32, [None, n_actions], "target_a")
        self.learning_rate = lr
        self.behavior_policy = tf.placeholder(tf.float32, [1, n_actions], "behavior_policy")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=50,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.target_a * tf.log(self.acts_prob), reduction_indices=[1]))
            self.loss_prob = tf.reduce_mean(tf.squared_difference(self.acts_prob, self.behavior_policy))
            self.behavior = tf.reduce_mean(self.acts_prob, reduction_indices=[1])

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, b):
        s = s[np.newaxis, :][0]
        a = a[np.newaxis, :][0]
        b = b[np.newaxis, :]
        feed_dict = {self.s: s, self.target_a: a, self.behavior_policy: b}
        _, exp_v, loss_prob, acts_prob = self.sess.run([self.train_op, self.loss, self.loss_prob, self.acts_prob], feed_dict)
        return exp_v, loss_prob, acts_prob.ravel()


env_model = EnvModel(100000, env.observation_space.shape[0], env.action_space.n)
sess = tf.Session()
actor = Actor_model(sess, env.observation_space.shape[0], env.action_space.n, 1e-5)
sess.run(tf.global_variables_initializer())


"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
observation = env.reset()
for i_espisode in range(MAX_EPISODE):

    t = 0
    track_r = []
    behavior_policy = np.array([0.4, 0.3, 0.3])

    if env_model.memory_counter > env_model.batch_size:
        batch = env_model.sample_s_a()
        batch_state = batch[:, :env_model.n_features]
        batch_action = batch[:, env_model.n_features:env_model.n_features + env_model.actions]
        loss, loss_prob, acts_prob = actor.learn(batch_state, batch_action, behavior_policy)
        acts_mean = acts_prob.reshape(env_model.batch_size, 3)

        if i_espisode % 50 == 0:
            print('episode %d, loss %f, loss_prb %f' % (i_espisode, loss, loss_prob))
            print('actprobs:', np.mean(acts_mean, axis=0))

    action = np.random.choice(3, p=behavior_policy)
    one_hot_a = env_model.one_hot(action)
    observation_, reward, done, info = env.step(action)
    env_model.store_transition(observation, one_hot_a, reward, observation_)
    observation = observation_

    if done:
        env.reset()


# """+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
# clf = MLPClassifier(solver='adam', alpha=1e-5, batch_size=64,
#                     hidden_layer_sizes=(50, 30), random_state=0)
# observation = env.reset()
#
# for i_espisode in range(MAX_EPISODE):
#
#     t = 0
#     track_r = []
#     behavior_policy = np.array([0.4, 0.3, 0.3])
#
#     if env_model.memory_counter > env_model.batch_size:
#         batch = env_model.sample_s_a()
#         batch_state = batch[:, :env_model.n_features]
#         batch_action = batch[:, env_model.n_features:env_model.n_features + env_model.actions]
#         clf.fit(batch_state, batch_action)
#         if i_espisode % 50 == 0:
#             test_eval = clf.predict_proba(batch_state)
#             loss = np.square(test_eval - behavior_policy)
#             loss = np.mean(loss)
#             print('episode %d, loss %f, loss_prb %f' % (i_espisode, clf.loss_, loss))
#
#     action = np.random.choice(3, p=behavior_policy)
#     one_hot_a = env_model.one_hot(action)
#     observation_, reward, done, info = env.step(action)
#     env_model.store_transition(observation, one_hot_a, reward, observation_)
#     observation = observation_
#
#     if done:
#         env.reset()