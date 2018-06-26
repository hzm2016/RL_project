# -*- coding: utf-8 -*-
"""
# @Time    : 23/06/18 12:41 PM
# @Author  : ZHIMIN HOU
# @FileName: LinearActorCritic.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

# -*- coding: ascii -*-

# MIT License
#
# Copyright (c) 2018 Dylan Robert Ashley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Tuple, Union

import numpy as np


__all__ = ['ContinuousActorCritic', 'DiscreteActorCritic']


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





