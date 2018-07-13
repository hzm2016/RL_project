# -*- coding: utf-8 -*-
"""
# @Time    : 23/06/18 12:41 PM
# @Author  : ZHIMIN HOU
# @FileName: LinearActorCritic.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

from typing import List, Tuple, Union
from Tile_coding import *
import numpy as np

__all__ = ['Reinforce', 'Allactions', 'ContinuousActorCritic', 'DiscreteActorCritic', 'AdvantageActorCritic']

# """Superparameters"""
# OUTPUT_GRAPH = True
# MAX_EPISODE = 3000
# DISPLAY_REWARD_THRESHOLD = 4001  # renders environment if total episode reward is greater then this threshold
# MAX_EP_STEPS = 10000   # maximum time step in one episode
# RENDER = False  # rendering wastes time
# GAMMA = 0.99     # reward discount in TD error
# LR_A = 0.005    # learning rate for actor
# LR_C = 0.01     # learning rate for critic
# EPSILON = 0
# load = False
#
# env = gym.make('MountainCar-v0')
# env._max_episode_steps = 10000
# # env = gym.make('CartPole-v0')
# env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped
#
# N_F = env.observation_space.shape[0]
# N_A = env.action_space.n
#
# print("Environments information:")
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
#
# """"Tile coding"""
# NumOfTilings = 10
# MaxSize = 10000
# HashTable = IHT(MaxSize)
#
# """position and velocity needs scaling to satisfy the tile software"""
# PositionScale = NumOfTilings / (env.observation_space.high[0] - env.observation_space.low[0])
# VelocityScale = NumOfTilings / (env.observation_space.high[1] - env.observation_space.low[1])
#
# def getQvalueFeature(obv, action):
#     activeTiles = tiles(HashTable, NumOfTilings, [PositionScale * obv[0], VelocityScale * obv[1]], [action])
#     return activeTiles
#
# def getValueFeature(obv):
#     activeTiles = tiles(HashTable, NumOfTilings, [PositionScale * obv[0], VelocityScale * obv[1]])
#     return activeTiles

class Reinforce:
    def __init__(self, n: int,
                 num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 lamda_v: float,
                 lamda_u: float):

        assert (n > 0)
        assert (num_actions > 0)
        self.num_actions = num_actions
        self.random_generator = np.random
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.last_action = 0
        self.last_prediction = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u
        self.ep_s = []
        self.ep_a = []
        self.ep_r = []

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def store_trasition(self, s, a, r):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)

    def choose_action(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        return action

    def get_return(self):
        """discount episode rewards"""
        ep_return = np.zeros_like(self.ep_r)
        running_add = 0
        for t in reversed(range(0, len(self.ep_r))):
            running_add = running_add * self.gamma + self.ep_r[t]
            ep_return[t] = running_add

        """normalize episode rewards"""
        ep_return -= np.mean(ep_return)
        ep_return /= np.std(ep_return)
        return ep_return

    def update(self):
        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v >= 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)

        ep_return = self.get_return()
        for i in range(len(self.ep_r)):
            x = np.asarray(self.ep_s[i], dtype=float)
            delta = ep_return[i] - self.predict(x)
            self.w_v += self.alpha_v * delta * x
            self.w_u[:, self.ep_a[i]] += self.alpha_u * delta * x
            pi = self.softmax(x)
            for other in range(self.num_actions):
                self.w_u[:, other] -= self.alpha_u * delta * x * pi[other]

        self.ep_s = []
        self.ep_a = []
        self.ep_r = []
        return float(delta)


class Allactions:
    """allactions"""
    def __init__(self, n: int,
                 num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 lamda_v: float,
                 lamda_u: float):

        assert (n > 0)
        assert (num_actions > 0)
        self.num_actions = num_actions
        self.random_generator = np.random
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.last_action = 0
        self.last_prediction = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def choose_action(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        return action

    def predict(self, x: Union[List[float], np.ndarray]):  # prediction Q value
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray],
             x_a: Union[List[float], np.ndarray],
             feature: Union[List[float], np.ndarray]) -> Tuple[int, float]:

        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)

        """compute value"""
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        q_value = np.dot(self.w_v, np.transpose(np.array(feature)))  # current value

        """action value update using sarsa learning"""
        prediction = np.dot(self.w_v, x_a)  # q_value # last_prediction = np.dot(self.w_v, x_a)
        delta = reward - self.reward_bar + self.gamma * prediction - self.last_prediction  # sarsa
        self.w_v += self.alpha_v * delta * self.e_v
        self.reward_bar += self.eta * delta
        self.e_v *= self.lamda_v * self.gamma
        self.e_v += x_a

        """policy update"""
        self.w_u += self.alpha_u * self.e_u
        self.e_u *= self.lamda_u * self.gamma
        for i in range(self.num_actions):
            self.e_u[:, i] += q_value[i] * x * pi[i]
            for other in range(self.num_actions):
                self.e_u[:, other] -= q_value[i] * x * pi[i] * pi[other]

        # for i in range(self.num_actions):
        #     self.w_u[:, i] += self.lamda_u * self.gamma * self.alpha_u * q_value[i] * x * pi[i]
        #     for other in range(self.num_actions):
        #         self.e_u[:, other] -= self.lamda_u * self.gamma * self.alpha_u * q_value[i] * x * pi[i] * pi[other]

        self.last_prediction = prediction
        return action, float(delta)


class DiscreteActorCritic:

    def __init__(self, n: int, num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 lamda_v: float,
                 lamda_u: float):

        assert (n > 0)
        assert (num_actions > 0)
        self.num_actions = num_actions
        self.random_generator = np.random
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.last_action = 0
        self.last_prediction = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def choose_action(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        return action

    def start(self, initial_x: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)
        x = np.asarray(x, dtype=float)
        prediction = np.dot(self.w_v, x)
        delta = reward - self.reward_bar + self.gamma * prediction - self.last_prediction
        self.w_v += self.alpha_v * delta * self.e_v
        self.w_u += self.alpha_u * delta * self.e_u
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.reward_bar += self.eta * delta
        self.e_v *= self.lamda_v * self.gamma
        self.e_v += x
        self.e_u *= self.lamda_u * self.gamma
        self.e_u[:, action] += x
        for other in range(self.num_actions):
            self.e_u[:, other] -= x * pi[other]
        self.last_action = action
        self.last_prediction = prediction
        return action, float(delta)


class AdvantageActorCritic:

    def __init__(self, n: int,
                 num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 lamda_v: float,
                 lamda_u: float):

        assert (n > 0)
        assert (num_actions > 0)
        self.num_actions = num_actions
        self.random_generator = np.random
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.last_action = 0
        self.last_prediction = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def choose_action(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        return action

    def predict(self, x: Union[List[float], np.ndarray]):  # prediction Q value
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray],
             x_a: Union[List[float], np.ndarray],
             feature: Union[List[float], np.ndarray]) -> Tuple[int, float]:

        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)

        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        q_value = np.dot(self.w_v, np.transpose(np.array(feature)))  # current value

        """action value update using sarsa learning"""
        prediction = np.dot(self.w_v, x_a)  # q_value # last_prediction = np.dot(self.w_v, x_a)
        # delta = reward - self.reward_bar + self.gamma * prediction - self.last_prediction  # sarsa
        delta = reward - self.reward_bar + self.gamma * np.mean(q_value) - self.last_prediction  # expected sarsa
        self.w_v += self.alpha_v * delta * self.e_v
        self.reward_bar += self.eta * delta
        self.e_v *= self.lamda_v * self.gamma
        self.e_v += x_a

        """policy update"""
        state_value = sum(pi * q_value)
        advantage = q_value - state_value * np.ones(len(q_value))
        self.w_u += self.alpha_u * self.e_u
        self.e_u *= self.lamda_u * self.gamma
        for i in range(self.num_actions):
            self.e_u[:, i] += advantage[i] * pi[i] * x
            for other in range(self.num_actions):
                self.e_u[:, other] -= advantage[i] * pi[i] * x * pi[other]

        self.last_action = action
        self.last_prediction = prediction
        return action, float(delta)


class OffDiscreteActorCritic:  # """Emphatic Actor Critic"""

    def __init__(self, n: int, num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 lamda_v: float,
                 lamda_u: float):

        assert (n > 0)
        assert (num_actions > 0)
        self.num_actions = num_actions
        self.random_generator = np.random
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.h_v = np.zeros(self.e_v.shape, dtype=float)
        self.last_action = 0
        self.last_action_pro = np.zeros(num_actions, dtype=float)
        self.last_prediction = 0
        self.last_pho = 1
        self.saved_auxiliary = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u
        self.F = 0
        self.initial = 1
        self.M = 0
        self.behavior_policy = np.array([0.45, 0.1, 0.45])

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=self.behavior_policy)
        self.last_action_pro = pi
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)

        x = np.asarray(x, dtype=float)

        self.F = self.last_pho * self.gamma * self.F + self.initial
        self.M = (1 - self.lamda_u) * self.initial + self.lamda_u * self.F

        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=self.behavior_policy)

        """compute the importance sampling"""
        for j in range(self.num_actions):
            if action == j:
                pho = pi[j] / self.behavior_policy[j]

        """Update the critic"""
        prediction = np.dot(self.w_v, x)
        delta = reward - self.reward_bar + self.gamma * prediction - self.last_prediction
        self.w_v += self.alpha_v * pho * delta * x
        self.reward_bar += self.eta * delta

        """GTD"""
        self.e_v *= pho
        self.w_v += self.alpha_v * (delta * self.e_v - self.gamma * (1 - self.lamda_v) * x * np.dot(self.e_v, self.h_v))
        self.h_v += self.alpha_v * self.eta * (delta * self.e_v - self.saved_auxiliary)
        self.e_v *= self.lamda_v * self.gamma
        self.e_v += x
        # if replacing:
        #     self.e = np.clip(self.e, 0, 1)

        """Update the actor"""
        # self.w_u += self.alpha_u * delta * self.e_u
        self.w_u[:, action] += self.alpha_u * pho * self.M * delta * x
        for other in range(self.num_actions):
            self.w_u[:, other] -= self.alpha_u * pho * self.M * delta * x * pi[other]

        """"""
        self.last_pho = pho
        self.last_action = action
        self.last_action_pro = pi
        self.last_prediction = prediction
        self.saved_auxiliary = np.dot(self.h_v, x) * x
        return action, float(delta)


class GTD:

    def __init__(self, initial_x: np.ndarray):
        self.e = np.copy(initial_x)
        self.w = np.zeros(self.e.shape, dtype=float)
        self.h = np.zeros(self.e.shape, dtype=float)
        self.last_prediction = 0
        self.saved_auxiliary = 0

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return float(np.dot(self.w, x))

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               alpha: float,
               eta: float,
               lamda: float,
               rho: float=1,
               replacing: bool=False) -> float:
        delta = reward + gamma * self.predict(x) - self.last_prediction
        self.e *= rho
        self.w += alpha * (delta * self.e - gamma * (1 - lamda) * x * np.dot(self.e, self.h))
        self.h += alpha * eta * (delta * self.e - self.saved_auxiliary)
        self.e *= lamda * gamma
        self.e += x
        if replacing:
            self.e = np.clip(self.e, 0, 1)
        self.last_prediction = self.predict(x)
        self.saved_auxiliary = np.dot(self.h, x) * x
        return delta


class NaturalActorCritic:

    def __init__(self, n: int, num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 lamda_v: float,
                 lamda_u: float):

        assert (n > 0)
        assert (num_actions > 0)
        self.num_actions = num_actions
        self.random_generator = np.random
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_u = np.zeros((n, num_actions), dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_u = np.zeros((n, num_actions), dtype=float)
        self.last_action = 0
        self.last_prediction = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)
        x = np.asarray(x, dtype=float)
        prediction = np.dot(self.w_v, x)
        delta = reward - self.reward_bar + self.gamma * prediction - self.last_prediction
        self.w_v += self.alpha_v * delta * self.e_v
        self.w_u += self.alpha_u * delta * self.e_u
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        self.reward_bar += self.eta * delta
        self.e_v *= self.lamda_v * self.gamma
        self.e_v += x
        self.e_u *= self.lamda_u * self.gamma
        self.e_u[:, action] += x
        for other in range(self.num_actions):
            self.e_u[:, other] -= x * pi[other]
        self.last_action = action
        self.last_prediction = prediction
        return action, float(delta)


class ContinuousActorCritic:

    def __init__(self, n: int, random_generator=np.random):
        assert(n > 0)
        self.random_generator = random_generator
        self.reward_bar = 0
        self.e_v = np.zeros(n, dtype=float)
        self.e_mu = np.zeros(n, dtype=float)
        self.e_sigma = np.zeros(n, dtype=float)
        self.w_v = np.zeros(n, dtype=float)
        self.w_mu = np.zeros(n, dtype=float)
        self.w_sigma = np.zeros(n, dtype=float)
        self.last_prediction = 0

    def mu(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_mu, x)

    def sigma(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.exp(np.dot(self.w_sigma, x))

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray]) -> float:
        initial_x = np.asarray(initial_x, dtype=float)
        mu = self.mu(initial_x)
        sigma = self.sigma(initial_x)
        action = self.random_generator.normal(mu, sigma)
        self.e_v += initial_x
        self.e_mu += (action - mu) * initial_x
        self.e_sigma += ((action - mu) ** 2 - sigma ** 2) * initial_x
        return action

    def step(self,
             reward: float,
             gamma: float,
             x: Union[List[float], np.ndarray],
             alpha_v: float,
             alpha_mu: float,
             alpha_sigma: float,
             eta: float,
             lamda_v: float,
             lamda_mu: float,
             lamda_sigma: float) -> Tuple[float, float]:
        assert (0 <= gamma <= 1)
        assert (alpha_v > 0)
        assert (alpha_mu > 0)
        assert (alpha_sigma > 0)
        assert (0 <= lamda_v <= 1)
        assert (0 <= lamda_mu <= 1)
        assert (0 <= lamda_sigma <= 1)
        x = np.asarray(x, dtype=float)
        prediction = np.dot(self.w_v, x)
        delta = reward - self.reward_bar + gamma * prediction - self.last_prediction
        self.w_v += alpha_v * delta * self.e_v
        self.w_mu += alpha_mu * delta * self.e_mu
        self.w_sigma += alpha_sigma * delta * self.e_sigma
        mu = self.mu(x)
        sigma = self.sigma(x)
        action = self.random_generator.normal(mu, sigma)
        self.reward_bar += eta * delta
        self.e_v *= lamda_v * gamma
        self.e_v += x
        self.e_mu *= lamda_mu * gamma
        self.e_mu += (action - mu) / sigma ** 2 * x
        self.e_sigma *= lamda_sigma * gamma
        self.e_sigma += ((action - mu) ** 2 / sigma ** 2 - 1) * x
        self.last_prediction = prediction
        return action, float(delta)


#
# agent = 'AdvantageActorCritic'
# if agent == 'Reinforce':
#     LinearAC = Reinforce(MaxSize, env.action_space.n, 0.99, 0., 0., 0.0001, 0.3, 0.3)
# elif agent == 'Allactions':
#     LinearAC = Allactions(MaxSize, env.action_space.n, 0.99, 0., 0.001, 0.0001, 0.3, 0.3)
# elif agent == 'AdvantageActorCritic':
#     LinearAC = AdvantageActorCritic(MaxSize, env.action_space.n, 0.99, 0., 0.001, 0.0001, 0.3, 0.3)
# elif agent == 'DiscreteActorCritic':
#     LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, 0.99, 0., 0.01, 0.001, 0.3, 0.3)
# elif agent == 'OffDiscreteActorCritic':
#     LinearAC = OffDiscreteActorCritic(MaxSize, env.action_space.n, 0.99, 0., 0.001, 0.0001, 0.3, 0.3)
# else:
#     print('Please give the right agent!')
#
#
# espisode_reward = []
# espisode_reward_reinforce = []
# espisode_reward_AC = []
# load = False
#
# if load:
#     # with open('Ep_reward_allaction.bin', 'rb') as f:
#     #     try:
#     #         espisode_reward = pickle.load(f)
#     #     except:
#     #         print('There is no file!')
#     with open('Ep_reward_reinforce.bin', 'rb') as f:
#         try:
#             espisode_reward_reinforce = pickle.load(f)
#         except:
#             print('There is no file!')
#     with open('Ep_reward_AC.bin', 'rb') as f:
#         try:
#             espisode_reward_AC = pickle.load(f)
#         except:
#             print('There is no file!')
#     # plt.figure()
#     # espisode_step = np.linspace(0, MAX_EPISODE-1, num=MAX_EPISODE)
#     # plt.plot(espisode_step, espisode_reward_reinforce, espisode_reward_AC, label='Espisode_reward')
#     # plt.xlabel('episodes')
#     # plt.ylabel('reward of each episode')
#     # plt.legend()
#     # plt.show()
# else:
#
#     if agent == 'Allactions':
#         for i_espisode in range(MAX_EPISODE):
#
#             t = 0
#             track_r = []
#             observation = env.reset()
#             action = LinearAC.start(getValueFeature(observation))
#             while True:
#
#                 observation_, reward, done, info = env.step(action)
#                 action_ = LinearAC.choose_action(getValueFeature(observation_))
#                 track_r.append(reward)
#                 feature = []
#                 for i in range(env.action_space.n):
#                     feature.append(getQvalueFeature(observation, i))
#                 action, delta = LinearAC.step(reward, getValueFeature(observation), \
#                                               getQvalueFeature(observation, action), \
#                                               getQvalueFeature(observation_, action_),
#                                               feature)
#                 observation = observation_
#                 t += 1
#                 if done or t > MAX_EP_STEPS:
#
#                     ep_rs_sum = sum(track_r)
#                     if 'running_reward' not in globals():
#                         running_reward = ep_rs_sum
#                     else:
#                         running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#                     print("episode:", i_espisode,  "reward:", int(running_reward))
#                     espisode_reward.append(int(running_reward))
#                     break
#         with open('Ep_reward_allaction.bin', 'wb') as f:
#             pickle.dump(espisode_reward, f)
#     elif agent == 'AdvantageActorCritic':
#         for i_espisode in range(MAX_EPISODE):
#
#             t = 0
#             track_r = []
#             observation = env.reset()
#             action = LinearAC.start(getValueFeature(observation))
#             while True:
#
#                 observation_, reward, done, info = env.step(action)
#
#                 feature = []
#                 for i in range(env.action_space.n):
#                     feature.append(getQvalueFeature(observation, i))
#
#                 # action_ = LinearAC.choose_action(getValueFeature(observation_))
#                 # getQvalueFeature(observation_, action_),
#
#                 track_r.append(reward)
#
#                 action, delta = LinearAC.step(reward, getValueFeature(observation), \
#                                               getQvalueFeature(observation, action), \
#                                               feature)
#                 observation = observation_
#                 t += 1
#                 if done or t > MAX_EP_STEPS:
#
#                     ep_rs_sum = sum(track_r)
#                     if 'running_reward' not in globals():
#                         running_reward = ep_rs_sum
#                     else:
#                         running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#                     print("episode:", i_espisode,  "reward:", int(running_reward))
#                     espisode_reward.append(int(running_reward))
#                     break
#         with open('Ep_reward_AdvantageActorCritic.bin', 'wb') as f:
#             pickle.dump(espisode_reward, f)
#     elif agent == 'Reinforce':
#         for i_espisode in range(MAX_EPISODE):
#
#             t = 0
#             track_r = []
#             observation = env.reset()
#             action = LinearAC.start(getValueFeature(observation))
#             while True:
#
#                 observation_, reward, done, info = env.step(action)
#                 track_r.append(reward)
#                 LinearAC.store_trasition(getValueFeature(observation), action, reward)
#                 action = LinearAC.choose_action(getValueFeature(observation))
#                 observation = observation_
#                 t += 1
#                 if done or t > MAX_EP_STEPS:
#
#                     ep_rs_sum = sum(track_r)
#                     if 'running_reward' not in globals():
#                         running_reward = ep_rs_sum
#                     else:
#                         running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
#                     print("episode:", i_espisode, "reward:", int(running_reward))
#                     espisode_reward.append(int(running_reward))
#                     LinearAC.update()
#                     break
#         with open('Ep_reward_reinforce_without_baseline.bin', 'wb') as f:
#             pickle.dump(espisode_reward, f)
#     elif agent == 'OffDiscreteActorCritic':
#         for i_espisode in range(MAX_EPISODE):
#
#             t = 0
#             track_r = []
#             observation = env.reset()
#             action = LinearAC.start(getValueFeature(observation))
#             while True:
#
#                 observation_, reward, done, info = env.step(action)
#                 track_r.append(reward)
#                 action, delta = LinearAC.step(reward, getValueFeature(observation))
#                 observation = observation_
#                 t += 1
#                 if done or t > MAX_EP_STEPS:
#
#                     ep_rs_sum = sum(track_r)
#                     if 'running_reward' not in globals():
#                         running_reward = ep_rs_sum
#                     else:
#                         running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#                     print("episode:", i_espisode,  "reward:", int(running_reward))
#                     espisode_reward.append(int(running_reward))
#                     break
#         with open('Ep_reward_AC.bin', 'wb') as f:
#             pickle.dump(espisode_reward, f)
#     else:
#         for i_espisode in range(MAX_EPISODE):
#
#             t = 0
#             track_r = []
#             observation = env.reset()
#             action = LinearAC.start(getValueFeature(observation))
#             while True:
#
#                 observation_, reward, done, info = env.step(action)
#                 track_r.append(reward)
#                 action, delta = LinearAC.step(reward, getValueFeature(observation))
#                 observation = observation_
#                 t += 1
#                 if done or t > MAX_EP_STEPS:
#
#                     ep_rs_sum = sum(track_r)
#                     if 'running_reward' not in globals():
#                         running_reward = ep_rs_sum
#                     else:
#                         running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#                     print("episode:", i_espisode,  "reward:", int(running_reward))
#                     espisode_reward.append(int(running_reward))
#                     break
#         with open('Ep_reward_AC.bin', 'wb') as f:
#             pickle.dump(espisode_reward, f)