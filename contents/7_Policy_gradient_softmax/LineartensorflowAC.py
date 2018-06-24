# -*- coding: utf-8 -*-
"""
# @Time    : 23/06/18 10:23 PM
# @Author  : ZHIMIN HOU
# @FileName: LineartensorflowAC.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import gym
from Tile_coding import Tilecoder

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 30001  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

"""Hyperparameters for tilecoding"""
NUMBER_OF_TILINGS = 8
TILING_CARDINALITY = 10

env = gym.make('CartPole-v0')
env._max_episode_steps = 1000
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

tile = Tilecoder(env, 4, 8)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.gamma = tf.placeholder(tf.float32, None, "gamma")
        self.onehot_q = tf.placeholder(tf.float32, [1, n_actions], "onehot_value")
        self.lr = tf.Variable(tf.constant(lr), dtype=tf.float32, name="leraning_rate")

        with tf.variable_scope('Actor'):

            self.w_actor = tf.Variable(tf.random_uniform([n_features, n_actions]), dtype=tf.float32, name="w_actor")

            self.l1 = tf.matmul(self.s, self.w_actor)

            self.acts_prob = tf.layers.dense(
                inputs=self.l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                # kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                # bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('gradient'):
            gra = self.lr * tf.gradients(self.acts_prob, self.w_actor)
            print(gra[0].shape)
            self.all_gradient = tf.reduce_sum(gra * self.onehot_q, axis=0)
            print(self.all_gradient.shape)

        with tf.variable_scope('update'):
            self.update = tf.assign_add(self.w_actor, self.all_gradient)

    def learn(self, s, a, td_error, all_q):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td_error, self.onehot_q: all_q}
        _ = self.sess.run([self.update], feed_dict)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        a = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int
        return a, probs.ravel()


class Critic(object):

    def __init__(self, sess, n_features, n_action, lr=0.01):
        self.sess = sess

        self.x = tf.placeholder(tf.float32, [1, n_features], "current_state")
        self.x_ = tf.placeholder(tf.float32, [1, n_features], "next_state")
        self.q_next = tf.placeholder(tf.float32, [1, 1], "q_next")
        self.q = tf.placeholder(tf.float32, [1, 1], "q")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.gamma = tf.placeholder(tf.float32, None, 'gamma')
        self.a = tf.placeholder(tf.float32, None, "a")
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')
        self.all_actions = tf.placeholder(tf.float32, [1, n_action], "all_actions")
        self.done = tf.placeholder(tf.float32, None, 'done')
        self.lr = tf.Variable(tf.constant(lr), dtype=tf.float32, name="lr")
        self.n_action = n_action

        with tf.variable_scope('Critic'):

            self.w_critic = tf.Variable(tf.random_uniform([n_features, 1]), dtype=tf.float32, name= "w_critic")
            self.q = tf.matmul(self.x, self.w_critic)
            print(self.q.shape)

        with tf.variable_scope('gradient'):
            self.td_error = self.r + GAMMA * (1 - self.done) * self.q_next - self.q
            print(self.td_error.shape)
            # gra = self.lr * tf.gradients(self.q, self.w_critic) * self.td_error
            self.w_critic += self.lr * self.td_error * self.x
            # self.update = tf.assign_add(self.w_critic, gra)

    def learn(self, s, x, r, s_, a_, done, all_a):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        all_q = np.zeros(self.n_action)
        for i in range(self.n_action):
            x_new = tile.oneHotFeature(np.append(x, all_a[i]))
            x_new = x_new[np.newaxis, :]
            print(x_new.shape)
            all_q[i] = self.sess.run(self.q, {self.x: x_new, self.a: all_a[i]})[0]
            print(all_q[i].shape)

        q_ = self.sess.run(self.q, {self.x: s_, self.a: a_, self.done: done})
        td_error, _ = self.sess.run([self.td_error, self.update],
                                          {self.x: s, self.q_next: q_, self.done: done, self.r: r})
        return td_error, all_q


if __name__ == '__main__':


    sess = tf.Session()
    actor = Actor(sess, n_features=tile.numTiles, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=tile.numTiles, n_action=N_A, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        initial_s = tile.oneHotFeature(s)
        t = 0
        track_r = []

        a, _ = actor.choose_action(initial_s)
        a_last = a

        while True:

            RENDER = True
            if RENDER:
                env.render()

            s_, r, done, info = env.step(a)

            a, ap = actor.choose_action(tile.oneHotFeature(s))

            contact_s = np.append(s, a)
            contact_s_ = np.append(s, a_last)
            f = tile.oneHotFeature(contact_s)
            f_next = tile.oneHotFeature(contact_s_)

            td_error, all_q = critic.learn(f, s, r, f_next, a, int(done), ap)
            actor.learn(tile.oneHotFeature(s), a, td_error, ap)

            track_r.append(r)
            a_last = a
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

# class Actor(object):
#     def __init__(self, sess, n_features, n_actions, lr=0.001):
#         self.sess = sess
#
#         self.s = tf.placeholder(tf.float32, [1, n_features], "state")
#         self.a = tf.placeholder(tf.int32, None, "act")
#         self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
#
#         with tf.variable_scope('Actor'):
#             l1 = tf.layers.dense(
#                 inputs=self.s,
#                 units=30,    # number of hidden units
#                 activation=tf.nn.relu,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )
#
#             self.acts_prob = tf.layers.dense(
#                 inputs=l1,
#                 units=n_actions,    # output units
#                 activation=tf.nn.softmax,   # get action probabilities
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='acts_prob'
#             )
#
#         with tf.variable_scope('exp_v'):
#             log_prob = tf.log(self.acts_prob[0, self.a])
#             self.exp_v = tf.reduce_mean(log_prob * self.td_error) + 0. * cat_entropy_softmax(self.acts_prob) # advantage (TD_error) guided loss
#
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
#             # self.train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
#             # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
#
#     def learn(self, s, a, td):
#         s = s[np.newaxis, :]
#         feed_dict = {self.s: s, self.a: a, self.td_error: td}
#         _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
#         if __debug__:  print('policy gradient {}'.format(exp_v)),
#         return exp_v
#
#     def choose_action(self, s):
#         s = s[np.newaxis, :]
#         probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
#         return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
#
#
# class Critic(object):
#     def __init__(self, sess, n_features, n_actions, lr=0.01):
#         self.sess = sess
#
#         self.s = tf.placeholder(tf.float32, [1, n_features], "state")
#         self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
#         self.r = tf.placeholder(tf.float32, None, 'r')
#
#         with tf.variable_scope('Critic'):
#             l1 = tf.layers.dense(
#                 inputs=self.s,
#                 units=30,  # number of hidden units
#                 activation=tf.nn.relu,  # None
#                 # have to be linear to make sure the convergence of actor.
#                 # But linear approximator seems hardly learns the correct Q.
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )
#
#             self.v = tf.layers.dense(
#                 inputs=l1,
#                 units=1,  # output units
#                 activation=None,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='V'
#             )
#
#         with tf.variable_scope('squared_TD_error'):
#             self.td_error = self.r + GAMMA * self.v_ - self.v
#             self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
#             # self.train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5).minimize(self.loss)
#             # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
#
#     def learn(self, s, r, s_, dummy_a, dummy_done, dummy_ap):
#         s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
#
#         v_ = self.sess.run(self.v, {self.s: s_})
#         td_error, _ , loss = self.sess.run([self.td_error, self.train_op, self.loss],
#                                           {self.s: s, self.v_: v_, self.r: r})
#         if __debug__:  print('critic loss {0}'.format(loss)),
#         return td_error
#
#
# class ActorAA(object):
#     def __init__(self, sess, n_features, n_actions, lr=0.001):
#         self.sess = sess
#
#         self.S = tf.placeholder(tf.float32, [1, n_features], "state")
#         self.TD_ERROR_AA = tf.placeholder(tf.float32, [n_actions], "td_error_aa")
#
#         with tf.variable_scope('Actor'):
#             l1 = tf.layers.dense(
#                 inputs=self.S,
#                 units=20,    # number of hidden units
#                 activation=tf.nn.relu,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )
#
#             self.acts_prob = tf.layers.dense(
#                 inputs=l1,
#                 units=n_actions,    # output units
#                 activation=tf.nn.softmax,   # get action probabilities
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='acts_prob'
#             )
#
#         with tf.variable_scope('exp_vaa'):
#             prob = self.acts_prob[0, :]
#             self.exp_v_aa = tf.reduce_mean(prob * self.TD_ERROR_AA)
#
#         with tf.variable_scope('train_vaa'):
#             self.train_op_aa = tf.train.AdamOptimizer(lr).minimize(-self.exp_v_aa)  # minimize(-exp_v) = maximize(exp_v)
#
#     def learn(self, s, a, td_error_aa):
#         s = s[np.newaxis, :]
#         feed_dict = {self.S: s, self.TD_ERROR_AA: td_error_aa}
#         _, exp_v_aa = self.sess.run([self.train_op_aa, self.exp_v_aa], feed_dict)
#         if __debug__:  print('policy gradient {}'.format(exp_v_aa)),
#         return exp_v_aa
#
#     def choose_action(self, s):
#         s = s[np.newaxis, :]
#         probs = self.sess.run(self.acts_prob, {self.S: s})   # get probabilities for all actions
#         return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
#
#
# class CriticAA(object):
#     def __init__(self, sess, n_features, n_actions, lr=0.01):
#         self.sess = sess
#
#         self.S = tf.placeholder(tf.float32, [1, n_features], "state")
#
#         self.VP = tf.placeholder(tf.float32, None, "v_next")
#         self.R = tf.placeholder(tf.float32, None, 'r')
#         self.A = tf.placeholder(tf.int32, None, 'action')
#         self.action_one_hot = tf.one_hot(self.A, n_actions, 1.0, 0.0, name='action_one_hot')
#         self.DONE = tf.placeholder(tf.float32, None, 'done')
#
#         with tf.variable_scope('CriticAA'):
#             l1 = tf.layers.dense(
#                 inputs=self.S,
#                 units=50,  # number of hidden units
#                 activation=tf.nn.relu,  # None
#                 # have to be linear to make sure the convergence of actor.
#                 # But linear approximator seems hardly learns the correct Q.
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )
#             l2 = tf.layers.dense(
#                 inputs=l1,
#                 units=100,  # number of hidden units
#                 activation=tf.nn.relu,  # None
#                 # have to be linear to make sure the convergence of actor.
#                 # But linear approximator seems hardly learns the correct Q.
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l2'
#             )
#
#             self.q = tf.layers.dense(
#                 inputs=l2,
#                 units=n_actions,  # output units
#                 activation=None,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='Q'
#             )
#
#         with tf.variable_scope('loss'):
#             self.qa = tf.reduce_sum(self.q * self.action_one_hot, reduction_indices=1)
#             self.loss = tf.square(self.R + (1.-self.DONE)*self.VP - self.qa)
#         with tf.variable_scope('all_action_td_error'):
#             self.all_action_td_error = self.q # - tf.reduce_sum(self.q, reduction_indices=1)
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
#
#     def learn(self, s, r, sp, a, done, ap):
#         in_s, in_sp = s[np.newaxis, :], sp[np.newaxis, :]
#         in_vp = self.sess.run(self.qa, {self.S: in_sp, self.A: ap})
#         # in_vp = np.sum(in_qp, axis=1)
#         all_action_td_error, _ , loss = self.sess.run([self.all_action_td_error, self.train_op, self.loss],
#                                           {self.S: in_s, self.VP: in_vp, self.R: r, self.A: a, self.DONE:done})
#         # print(loss)
#         # print(all_action_td_error[0])
#         if __debug__:  print('vp:{0}, r:{2}, critic loss {1}'.format(in_vp, loss, r)),
#         return all_action_td_error[0]
#
#
# class All_Actor_critic:
#
#     def __init__(self, stepSize, numOfTilings=8, numTile=4):
#         self.numTile = numTile
#         self.num0fTilings = numOfTilings
#
#         # divide step size equally to each tiling
#         self.actor_step_size = stepSize / numOfTilings
#         self.critic_step_size = stepSize / numOfTilings
#
#         self.tile_coder = Tilecoder(env.observation_space.high, env.observation_space.low, \
#                                     self.num0fTilings, self.numTile, env.action_space.n)
#
#         # critic weights and actor weights
#         self.critic_weights = np.zeros(self.tile_coder.numTiles)
#         self.actor_weights = np.zeros([self.tile_coder.numTiles, env.action_space.n])
#         self.discount_rate = 0.9
#
#     def actor_learning(self, td_error, state_feature):
#
#         self.actor_weights += self.actor_step_size * td_error * state_feature
#
#     def get_action(self, state):
#
#         np.dot(state, self.actor_weights)
#
#         return
#
#     def critic_learning(self, td_error, state_features):
#
#         self.critic_weights += self.critic_step_size * td_error * state_features
#
#     def get_value(self, state):
#
#         return np.dot(self.critic_weights, state)
#
#
#     def update(self, td_error, state_next_feature, state_feature):
#
#         self.critic_learning(td_error, state_next_feature)
#         self.actor_learning(td_error, state_feature)
#
#
# sess = tf.Session()
#
# actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
# critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
#
# sess.run(tf.global_variables_initializer())
#
# if OUTPUT_GRAPH:
#     summary_writer = tf.summary.FileWriter("logs/", sess.graph)
#
# for i_episode in range(MAX_EPISODE):
#
#     s = env.reset()
#     t = 0
#     track_r = []
#     while True:
#         if RENDER: env.render()
#
#         a = actor.choose_action(s)
#
#         s_, r, done, info = env.step(a)
#
#         if done: r = -20
#
#         track_r.append(r)
#
#         td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
#         actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
#
#         s = s_
#         t += 1
#
#         if done or t >= MAX_EP_STEPS:
#             ep_rs_sum = sum(track_r)
#
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
#             print("episode:", i_episode, "  reward:", int(running_reward))
#             record = summary_pb2.Summary.Value(tag='reward', simple_value=running_reward)
#             record_value = summary_pb2.Summary(value=[record])
#             summary_writer.add_summary(record_value, i_episode)
#             break

