# -*- coding: utf-8 -*-
"""
# @Time    : 23/06/18 10:41 AM
# @Author  : ZHIMIN HOU
# @FileName: NonLinearActorCr.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import numpy as np
import tensorflow as tf
import gym


def cat_entropy_softmax(p0):
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis = 1)

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 4000
DISPLAY_REWARD_THRESHOLD = 4001  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 5000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')

env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
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

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error) + 0. * cat_entropy_softmax(self.acts_prob) # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        # if __debug__: print('policy gradient {}'.format(exp_v)),
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,  # number of hidden units
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

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
            # self.train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5).minimize(self.loss)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, dummy_a, dummy_done, dummy_ap):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _, loss = self.sess.run([self.td_error, self.train_op, self.loss],
                                          {self.s: s, self.v_: v_, self.r: r})
        # if __debug__:  print('critic loss {0}'.format(loss)),
        return td_error


class ActorAA(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.S = tf.placeholder(tf.float32, [1, n_features], "state")
        self.TD_ERROR_AA = tf.placeholder(tf.float32, [n_actions], "td_error_aa")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.S,
                units=75,    # number of hidden units
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

        with tf.variable_scope('exp_vaa'):
            prob = self.acts_prob[0, :]
            self.exp_v_aa = tf.reduce_mean(prob * self.TD_ERROR_AA)

        with tf.variable_scope('train_vaa'):
            self.train_op_aa = tf.train.AdamOptimizer(lr).minimize(-self.exp_v_aa)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td_error_aa):
        s = s[np.newaxis, :]
        feed_dict = {self.S: s, self.TD_ERROR_AA: td_error_aa}
        _, exp_v_aa = self.sess.run([self.train_op_aa, self.exp_v_aa], feed_dict)
        # if __debug__:  print('policy gradient {}'.format(exp_v_aa)),
        return exp_v_aa

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.S: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class CriticAA(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.sess = sess

        self.S = tf.placeholder(tf.float32, [1, n_features], "state")

        self.VP = tf.placeholder(tf.float32, None, "v_next")
        self.R = tf.placeholder(tf.float32, None, 'r')
        self.A = tf.placeholder(tf.int32, None, 'action')
        self.action_one_hot = tf.one_hot(self.A, n_actions, 1.0, 0.0, name='action_one_hot')
        self.DONE = tf.placeholder(tf.float32, None, 'done')

        with tf.variable_scope('CriticAA'):
            l1 = tf.layers.dense(
                inputs=self.S,
                units=50,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=100,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.q = tf.layers.dense(
                inputs=l2,
                units=n_actions,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Q'
            )

        with tf.variable_scope('loss'):
            self.qa = tf.reduce_sum(self.q * self.action_one_hot, reduction_indices=1)
            self.loss = tf.square(self.R + (1.-self.DONE)*self.VP - self.qa)
        with tf.variable_scope('all_action_td_error'):
            self.all_action_td_error = self.q
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, sp, a, done, ap):
        in_s, in_sp = s[np.newaxis, :], sp[np.newaxis, :]
        in_vp = self.sess.run(self.qa, {self.S: in_sp, self.A: ap})
        # in_vp = np.sum(in_qp, axis=1)
        all_action_td_error, _ , loss = self.sess.run([self.all_action_td_error, self.train_op, self.loss],
                                          {self.S: in_s, self.VP: in_vp, self.R: r, self.A: a, self.DONE:done})
        # print(loss)
        # print(all_action_td_error[0])
        # if __debug__:  print('vp:{0}, r:{2}, critic loss {1}'.format(in_vp, loss, r)),
        return all_action_td_error[0]



sess = tf.Session()
type = 'SIG'

if type == 'AA':
    actor = ActorAA(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = CriticAA(sess, n_features=N_F, n_actions=N_A, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
elif type == 'SIG':
    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, n_actions=N_A, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    a = actor.choose_action(s)

    while True:

        # RENDER = True
        if RENDER: env.render()

        s_, r, done, info = env.step(a)

        ap = actor.choose_action(s)

        # if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_, a, int(done), ap)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        a = ap
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
            break
