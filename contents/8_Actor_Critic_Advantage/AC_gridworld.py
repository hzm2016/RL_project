# -*- coding: utf-8 -*-
"""
# @Time    : 03/07/18 4:09 PM
# @Author  : ZHIMIN HOU
# @FileName: AC_gridworld.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import numpy as np
import tensorflow as tf
from gridworld import gameEnv
slim = tf.contrib.layers
np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 30000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 50   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001     # learning rate for critic

env = gameEnv(8)
# env.seed(1)  # reproducible
# env = env.unwrapped

N_F = 21168
N_A = 4

def processState(states):
    return np.reshape(states, [21168])


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001, h_size =512):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            imageIn = tf.reshape(self.s, shape=[-1, 84, 84, 3])
            conv1 = slim.conv2d( \
                inputs=imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                biases_initializer=None)
            conv2 = slim.conv2d( \
                inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                biases_initializer=None)
            conv3 = slim.conv2d( \
                inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                biases_initializer=None)
            conv4 = slim.conv2d( \
                inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
                biases_initializer=None)
            l1 = tf.layers.dense(
                inputs=conv4,
                units=30,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
            self.acts_prob = tf.squeeze(self.acts_prob, [1, 2])
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, h_size=512):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            imageIn = tf.reshape(self.s, shape=[-1, 84, 84, 3])
            conv1 = slim.conv2d( \
                inputs=imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                biases_initializer=None)
            conv2 = slim.conv2d( \
                inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                biases_initializer=None)
            conv3 = slim.conv2d( \
                inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                biases_initializer=None)
            conv4 = slim.conv2d( \
                inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
                biases_initializer=None)
            l1 = tf.layers.dense(
                inputs=conv4,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
            self.v =tf.squeeze(self.v, [1,2])

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A, h_size=512)
critic = Critic(sess, n_features=N_F, lr=LR_C, h_size =512)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        # if RENDER: env.render()
        s = processState(s)
        a = actor.choose_action(s)

        s_, r, done = env.step(a)
        s_ = processState(s_)
        # if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            if OUTPUT_GRAPH:
                summary_writer = tf.summary.FileWriter("train_AC")
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(running_reward))
                summary_writer.add_summary(summary, i_episode)
                summary_writer.flush()
            break
