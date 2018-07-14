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
tf.set_random_seed(2)

__all__ = ['PolicyGradient', 'DiscreteActorCritic', 'DisAllActions']


"""REINFORCE"""
class PolicyGradient(object):

    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.0001,
            reward_decay=0.99,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_actor = learning_rate
        self.lr_critic = 10 * learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.actions = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.dis_return = tf.placeholder(tf.float32, [None], name="return")
            # self.prediction = tf.placeholder(tf.float32, [None, ], name="actions_value")

        with tf.name_scope('Actor'):
            self.w_u = tf.Variable(tf.random_uniform([self.n_features, self.n_actions]), dtype=tf.float32, name="w_u")
            self.action = tf.matmul(self.obs, self.w_u)

        with tf.name_scope('Critic'):
            self.w_v = tf.Variable(tf.random_uniform([self.n_features, 1]), dtype=tf.float32, name="w_v")
            self.prediction = tf.matmul(self.obs, self.w_v)
        # # fc1
        # layer = tf.layers.dense(
        #     inputs=self.tf_obs,
        #     units=self.n_features,
        #     activation=None,  # tanh activation
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.),
        #     bias_initializer=tf.constant_initializer(0.),
        #     name='fc1'
        # )
        # # fc2
        # all_act = tf.layers.dense(
        #     inputs=layer,
        #     units=self.n_actions,
        #     activation=None,
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.),
        #     bias_initializer=tf.constant_initializer(0.),
        #     name='fc2'
        # )
        self.all_act_prob = tf.nn.softmax(self.action, name='act_prob')

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            self.neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.actions, self.n_actions), axis=1) # this is negative log of chosen action
            delta = self.dis_return - self.prediction
            loss_v = tf.reduce_mean(tf.square(delta))
            # loss_u = tf.reduce_mean(self.neg_log_prob * delta)  # reward guided loss

        with tf.name_scope('update'):
            self.w_u = tf.assign_add(self.w_u, self.lr_actor * delta * tf.gradient(self.neg_log_prob, self.w_u))
            self.w_v = tf.assign_add(self.w_v, self.lr_critic * tf.gradients(loss_v, self.w_v))

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        for i in range(len(discounted_ep_rs_norm)):
            ng, _ = self.sess.run([self.neg_log_prob, self.w_u, self.w_v], feed_dict={
                 self.obs: np.array(self.ep_obs)[i, :],  # shape=[None, n_obs]
                 self.actions: np.array(self.ep_as)[i, :],  # shape=[None, ]
                 self.dis_return: discounted_ep_rs_norm[i],  # shape=[None, ]
            })

        # train on episode
        # ng, _ = self.sess.run([self.neg_log_prob, self.train_op], feed_dict={
        #      self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        #      self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        #      self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        # })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


class DiscreteActorCritic:

    def __init__(self, sess, n: int, num_actions: int,
                 gamma,
                 eta,
                 alpha_v,
                 alpha_u,
                 lamda_v,
                 lamda_u):
        assert (n > 0)
        assert (num_actions > 0)
        self.sess = sess
        self.num_actions = num_actions
        self.prediction = tf.placeholder(tf.float32, [1, 1], name='prediction')
        self.next_prediction = tf.placeholder(tf.float32, [1, 1], name='next_prediction')
        self.a = tf.placeholder(tf.int32, None, "a")
        self.s = tf.placeholder(tf.float32, [1, n], name='s')
        self.s_ = tf.placeholder(tf.float32, [1, n], name='s_')
        self.r = tf.placeholder(tf.float32, None, name='r')
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

        with tf.variable_scope('Prediction'):
            self.w_v = tf.Variable(tf.zeros([n, 1]), dtype=tf.float32, name='w_v')
            self.e_v = tf.Variable(tf.zeros([n, 1]), dtype=tf.float32, name='e_v')
            self.prediction = tf.matmul(self.s, self.w_v)

        with tf.variable_scope('Action'):
            self.w_u = tf.Variable(tf.zeros([n, num_actions]), dtype=tf.float32, name='w_u')
            self.e_u = tf.Variable(tf.zeros([n, num_actions]), dtype=tf.float32, name='e_u')
            self.action = tf.matmul(self.s, self.w_u)
            self.acts_prob = tf.nn.softmax(self.action)

        with tf.variable_scope('Update_Critic'):
            self.r_bar = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32, name='r_bar')
            self.delta = self.r - self.r_bar + self.gamma * self.next_prediction - self.prediction
            self.r_bar = tf.assign_add(self.r_bar, self.eta * self.delta)
            # gra_v = tf.gradients(tf.reduce_mean(tf.square(self.delta)), self.w_v)
            # self.e_v = self.lamda_v * self.gamma * self.e_v + gra_v[0]  # update trace_v
            self.e_v = self.lamda_v * self.gamma * self.e_v + tf.transpose(self.s)  # update trace_v
            self.w_v = tf.assign_add(self.w_v, self.alpha_v * self.delta * self.e_v)  # update w_v

        with tf.variable_scope('Update_Actor'):
            # self.mid = tf.Variable(tf.zeros([num_actions, n]), dtype=tf.float32, name='mid')
            # value = tf.scatter_update(self.mid, [self.a], self.s)
            log_prob = tf.reduce_mean(tf.log(self.acts_prob[0, self.a]))
            gra_u = tf.gradients(log_prob, self.w_u)
            self.e_u = self.lamda_u * self.gamma * self.e_u + gra_u[0]  # update trace_u
            # self.e_u = self.lamda_u * self.gamma * self.e_u
            # self.e_u = self.e_u + tf.transpose(value)
            # for other in range(self.num_actions):
            #     value_1 = tf.scatter_update(self.mid, [other], self.s)
            #     self.e_u = self.e_u - tf.transpose(value_1) * self.acts_prob[0, other]
            self.w_u = tf.assign_add(self.w_u, self.alpha_u * self.delta * self.e_u)  # update w_u

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def update(self, s, r, s_, a):
        s_ = s_[np.newaxis, :]
        next_p = self.sess.run(self.prediction, {self.s: s_})
        s = s[np.newaxis, :]
        delta, _, e_v, _, _, _ = self.sess.run([self.delta, self.r_bar, self.e_v, self.w_v, self.e_u, self.w_v], \
                                    {self.s: s, self.r: r, self.a: a, self.s_: s_, self.next_prediction: next_p})

        return delta, e_v


class DisAllActions:

    def __init__(self, sess, n: int,  # number of feature
                 num_actions: int,
                 gamma,
                 eta,
                 alpha_v,
                 alpha_u,
                 lamda_v,
                 lamda_u):
        assert (n > 0)
        assert (num_actions > 0)
        self.sess = sess
        self.num_actions = num_actions

        self.prediction = tf.placeholder(tf.float32, [1, num_actions], name='prediction')
        self.next_prediction = tf.placeholder(tf.float32, [1, num_actions], name='next_prediction')
        self.a = tf.placeholder(tf.int32, None, "a")
        self.a_ = tf.placeholder(tf.int32, None, "a_")
        self.s = tf.placeholder(tf.float32, [1, n], name='s')
        self.s_ = tf.placeholder(tf.float32, [1, n], name='s_')
        self.r = tf.placeholder(tf.float32, None, name='r')
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

        with tf.variable_scope('Evaluation'):
            self.w_q = tf.Variable(tf.zeros([n, num_actions]), dtype=tf.float32, name='w_q')
            self.e_q = tf.Variable(tf.random_uniform([n, num_actions]), dtype=tf.float32, name='e_q')
            self.prediction = tf.matmul(self.s, self.w_q)  # compute Q

        with tf.variable_scope('Policy'):
            self.w_u = tf.Variable(tf.zeros([n, num_actions]), dtype=tf.float32, name='w_u')
            self.e_u = tf.Variable(tf.random_uniform([n, num_actions]), dtype=tf.float32, name='e_u')
            self.action = tf.matmul(self.s, self.w_u)
            self.acts_prob = tf.nn.softmax(self.action)

        with tf.variable_scope('Update_Evaluation'):
            self.r_bar = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32, name='r_bar')
            self.delta = self.r - self.r_bar + self.gamma * self.next_prediction[0, self.a_] \
                         - self.prediction[0, self.a]   # sarsa update q value
            self.r_bar_update = tf.assign_add(self.r_bar, self.eta * self.delta)
            gra_v = tf.gradients(tf.reduce_mean(tf.square(self.delta)), self.w_q)
            self.e_q_update = tf.assign(self.e_q, self.lamda_v * self.gamma * self.e_q + gra_v[0])  # update trace_v
            self.w_q_update = tf.assign_add(self.w_q, self.alpha_v * self.e_q)  # update w_v

        with tf.variable_scope('Update_policy'):
            self.prob_prediction = tf.reduce_sum(tf.multiply(self.acts_prob, self.prediction))   # all actions
            gra_u = tf.gradients(self.prob_prediction, self.w_u)
            self.e_u_update = tf.assign(self.e_u, self.lamda_u * self.gamma * self.e_u + gra_u[0])  # update trace_u
            self.w_u_update = tf.assign_add(self.w_u, self.alpha_u * self.e_u)  # update w_u

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def update(self, s, r, s_, a, a_):
        s_ = s_[np.newaxis, :]
        next_p = self.sess.run(self.prediction, {self.s: s_})
        s = s[np.newaxis, :]
        delta, _, r_bar, _, e_q, _, w_q, _, e_u, _, w_u = self.sess.run([self.delta,
                                                                         self.r_bar_update, self.r_bar,
                                                                         self.e_q_update, self.e_q,
                                                                         self.w_q_update, self.w_q,
                                                                         self.e_u_update, self.e_u,
                                                                         self.w_u_update, self.w_u], \
                                    {self.s: s, self.r: r, self.a: a, self.a_: a_, self.s_: s_, self.next_prediction: next_p})

        return delta, r_bar, e_q, w_q, e_u, w_u


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

            self.acts_prob = tf.nn.softmax(self.l1)
            # self.acts_prob = tf.layers.dense(
            #     inputs=self.l1,
            #     units=n_actions,    # output units
            #     activation=tf.nn.softmax,   # get action probabilities
            #     # kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     # bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='acts_prob'
            # )

        with tf.variable_scope('gradient'):
            gra = self.lr * tf.gradients(self.acts_prob, self.w_actor)
            print(gra[0].shape)
            self.all_gradient = tf.reduce_sum(gra * self.onehot_q, axis=0)
            print('a', self.all_gradient.shape)

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
            print('qa', tf.gradients(self.q, self.w_critic))
            print('q', self.q.shape)

        with tf.variable_scope('gradient'):
            self.td_error = self.r + GAMMA * self.q_next - self.q
            print(self.td_error.shape)
            # gra = self.lr * tf.gradients(self.q, self.w_critic) * self.td_error
            # self.w_critic += self.lr * self.td_error * self.x

            gra = tf.gradients(tf.reduce_mean(tf.square(self.td_error)), self.w_critic)
            self.update = tf.assign_add(self.w_critic, gra[0])

            print(self.update.shape)

    def learn(self, s, x, r, s_, a_, done, all_a):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        all_q = np.zeros(self.n_action)
        for i in range(self.n_action):
            x_new = tile.oneHotFeature(np.append(x, all_a[i]))
            x_new = x_new[np.newaxis, :]

            all_q[i] = self.sess.run(self.q, {self.x: x_new, self.a: all_a[i]})[0]


        q_ = self.sess.run(self.q, {self.x: s_, self.a: a_, self.done: done})
        td_error, _ = self.sess.run([self.td_error, self.update],
                                          {self.x: s, self.q_next: q_, self.done: done, self.r: r})
        return td_error, all_q


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
        if __debug__:  print('policy gradient {}'.format(exp_v)),
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
        td_error, _ , loss = self.sess.run([self.td_error, self.train_op, self.loss],
                                          {self.s: s, self.v_: v_, self.r: r})
        if __debug__:
            print('critic loss {0}'.format(loss))
        return td_error


class ActorAA(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.S = tf.placeholder(tf.float32, [1, n_features], "state")
        self.TD_ERROR_AA = tf.placeholder(tf.float32, [n_actions], "td_error_aa")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.S,
                units=20,    # number of hidden units
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
        if __debug__:
            print('policy gradient {}'.format(exp_v_aa))
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
            self.all_action_td_error = self.q # - tf.reduce_sum(self.q, reduction_indices=1)
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
        if __debug__:
            print('vp:{0}, r:{2}, critic loss {1}'.format(in_vp, loss, r)),
        return all_action_td_error[0]


if __name__ == '__main__':


    sess = tf.Session()

    actor = Actor(sess, n_features=tile.numTiles, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=tile.numTiles, n_action=N_A, lr=LR_C)

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


