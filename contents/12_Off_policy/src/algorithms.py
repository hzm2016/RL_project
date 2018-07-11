# -*- coding: ascii -*-

import numpy as np
from typing import List, Tuple, Union


__all__ = ('GTD', 'DiscreteActorCritic', 'OffActorCritic', 'OffPAC', 'TIDBD', 'LGGTD', 'DLGGTD', 'LGTIDBD', 'DLGTIDBD', 'LGGTD2', 'DLGGTD2')


class GTD:

    def __init__(self,
                 n: int,
                 gamma: float,
                 eta: float,
                 alpha: float,
                 lamda: float,
                 ):
        self.e = np.zeros(n, dtype=float)
        self.w = np.zeros(self.e.shape, dtype=float)
        self.h = np.zeros(self.e.shape, dtype=float)
        self.gamma = gamma
        self.alpha = alpha
        self.eta = eta
        self.lamda = lamda
        self.last_prediction = 0
        self.saved_auxiliary = 0

    def predict(self,
                x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return float(np.dot(self.w, x))

    def start(self,
              initial_x: np.ndarray):
        self.e = np.copy(initial_x)

    def update(self,
               reward: float,
               x: np.ndarray,
               rho: float=1,
               replacing: bool=False) -> float:
        delta = reward + self.gamma * self.predict(x) - self.last_prediction
        self.e *= rho
        self.w += self.alpha * (delta * self.e - self.gamma * (1 - self.lamda) * x * np.dot(self.e, self.h))
        self.h += self.alpha * self.eta * (delta * self.e - self.saved_auxiliary)
        self.e *= self.lamda * self.gamma
        self.e += x
        if replacing:
            self.e = np.clip(self.e, 0, 1)
        self.last_prediction = self.predict(x)
        self.saved_auxiliary = np.dot(self.h, x) * x
        return delta


"""Emphatic Actor Critic and Off-policy actor critic"""
class OffActorCritic:

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
        self.behavior_policy = np.zeros(num_actions, dtype=float)

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = rv - max(rv)
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self,
              initial_x: Union[List[float], np.ndarray],
              behavior_policy: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        self.behavior_policy = np.asarray(behavior_policy, dtype=float)
        action = self.random_generator.choice(self.num_actions, p=self.behavior_policy)
        self.last_action_pro = pi
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def choose_action(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray],
             behavior_policy: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)

        x = np.asarray(x, dtype=float)
        self.behavior_policy = np.array(behavior_policy)
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


class OffPAC:

    def __init__(self, n: int, num_actions: int,
                 gamma: float,
                 eta: float,
                 alpha_v: float,
                 alpha_u: float,
                 alpha_h: float,
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
        self.h = np.zeros(self.e_v.shape, dtype=float)
        self.last_action = 0
        self.last_prediction = 0
        self.saved_auxiliary = 0
        self.gamma = gamma
        self.eta = eta
        self.alpha_v = alpha_v
        self.alpha_u = alpha_u
        self.alpha_h = alpha_h
        self.lamda_v = lamda_v
        self.lamda_u = lamda_u

    def softmax(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rv = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            rv[action] = np.dot(x, (self.w_u[:, action]).flatten())
        rv = rv - max(rv)
        rv = np.exp(rv)
        return rv / sum(rv)

    def predict(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        return np.dot(self.w_v, x)

    def start(self, initial_x: Union[List[float], np.ndarray],
              behavior_policy: Union[List[float], np.ndarray]):
        initial_x = np.asarray(initial_x, dtype=float)
        pi = self.softmax(initial_x)
        action = self.random_generator.choice(self.num_actions, p=behavior_policy)
        self.e_v += initial_x
        self.e_u[:, action] += initial_x * (1 - sum(pi))
        self.last_action = action
        return action

    def choose_action(self, x: Union[List[float], np.ndarray]):
        x = np.asarray(x, dtype=float)
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=pi)
        return action

    def step(self,
             reward: float,
             x: Union[List[float], np.ndarray],
             behavior_policy: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        assert (0 <= self.gamma <= 1)
        assert (self.eta >= 0)
        assert (self.alpha_v > 0)
        assert (self.alpha_u > 0)
        assert (0 <= self.lamda_v <= 1)
        assert (0 <= self.lamda_u <= 1)
        x = np.asarray(x, dtype=float)
        prediction = np.dot(self.w_v, x)
        delta = reward - self.reward_bar + self.gamma * prediction - self.last_prediction
        self.w_v += self.alpha_v * (delta * self.e_v - self.gamma * (1 - self.lamda_v) * x * np.dot(self.e_v, self.h))
        self.h += self.alpha_h * (delta * self.e_v - self.saved_auxiliary)
        self.w_u += self.alpha_u * delta * self.e_u

        """choose action"""
        pi = self.softmax(x)
        action = self.random_generator.choice(self.num_actions, p=behavior_policy)

        """compute the importance sampling"""
        pho = pi[action] / np.array(behavior_policy[action])

        self.reward_bar += self.eta * delta
        self.e_v *= pho * self.lamda_v * self.gamma
        self.e_v += pho * x
        self.e_u *= pho * self.lamda_u * self.gamma
        self.e_u[:, action] += pho * x
        for other in range(self.num_actions):
            self.e_u[:, other] -= pho * x * pi[other]
        self.last_action = action
        self.last_prediction = prediction
        self.saved_auxiliary = np.dot(self.h, x) * x
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


class TIDBD:

    def __init__(self, initial_x: np.ndarray, initial_alpha):
        assert(len(initial_x.shape) == 1)
        self._last_x = np.copy(initial_x)
        self.e = np.copy(initial_x)
        self.w = np.zeros(self.e.shape, dtype=float)
        self.h = np.zeros(self.w.shape, dtype=float)
        self.beta = np.ones(self.e.shape, dtype=float) * np.log(initial_alpha)

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return float(np.dot(self.w, x))

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               theta: float,
               lamda: float,
               replacing: bool=False) -> float:
        delta = reward + gamma * self.predict(x) - self.predict(self._last_x)
        self.beta += theta * delta * self._last_x * self.h
        alpha = np.e ** self.beta
        self.w += alpha * delta * self.e
        history_decay = 1 - alpha * self._last_x * self.e
        history_decay[history_decay < 0] = 0
        self.h *= history_decay
        self.h += alpha * delta * self.e
        self.e *= lamda * gamma
        self.e += x
        if replacing:
            self.e = np.clip(self.e, 0, 1)
        np.copyto(self._last_x, x)
        return delta


class LGGTD:

    def __init__(self, initial_x: np.ndarray, max_return: float):
        assert (len(initial_x.shape) == 1)
        self.learner = GTD(initial_x)
        self.err_learner = GTD(initial_x)
        self.err_learner.w.fill(max_return)
        self.err_learner.last_prediction = self.err_learner.predict(initial_x)
        self.sq_learner = GTD(initial_x)

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return self.learner.predict(x)

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               alpha: float,
               eta: float,
               rho: float = 1) -> Tuple[float, float]:
        err_prediction = self.err_learner.predict(x)
        self.err_learner.update(reward, gamma, x, alpha, 0, 1, rho=rho)
        sq_reward = (rho * reward) ** 2 + 2 * (rho ** 2) * gamma * reward * err_prediction
        sq_gamma = (rho * gamma) ** 2
        self.sq_learner.update(sq_reward, sq_gamma, x, alpha, 0, 1)
        errsq = (err_prediction - self.predict(x)) ** 2
        varg = max(0.0, self.sq_learner.predict(x) - err_prediction ** 2)
        lamda = errsq / (varg + errsq)
        delta = self.learner.update(reward, gamma, x, alpha, eta, lamda, rho=rho)
        return delta, lamda


class DLGGTD:

    def __init__(self, initial_x: np.ndarray, max_return: float):
        assert (len(initial_x.shape) == 1)
        self._last_x = np.array(initial_x)
        self.learner = GTD(initial_x)
        self.err_learner = GTD(initial_x)
        self.err_learner.w.fill(max_return)
        self.err_learner.last_prediction = self.err_learner.predict(initial_x)
        self.var_learner = GTD(initial_x)

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return self.learner.predict(x)

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               alpha: float,
               eta: float,
               rho: float=1) -> Tuple[float, float]:
        err_prediction = self.err_learner.predict(x)
        delta_err = self.err_learner.update(reward, gamma, x, alpha, 0, 1, rho=rho)
        var_reward = (rho * delta_err - (rho - 1) * self.err_learner.predict(self._last_x)) ** 2
        var_gamma = (rho * gamma) ** 2
        self.var_learner.update(var_reward, var_gamma, x, alpha, 0, 1)
        errsq = (err_prediction - self.predict(x)) ** 2
        varg = max(0.0, self.var_learner.predict(x))
        lamda = errsq / (varg + errsq)
        delta = self.learner.update(reward, gamma, x, alpha, eta, lamda, rho=rho)
        np.copyto(self._last_x, x)
        return delta, lamda


class LGTIDBD:

    def __init__(self,
                 initial_x: np.ndarray,
                 max_return: float,
                 initial_alpha: float = 0.05):
        assert (len(initial_x.shape) == 1)
        self.learner = TIDBD(initial_x, initial_alpha)
        self.err_learner = TIDBD(initial_x, initial_alpha)
        self.err_learner.w.fill(max_return)
        self.err_learner.last_prediction = self.err_learner.predict(initial_x)
        self.sq_learner = TIDBD(initial_x, initial_alpha)

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return self.learner.predict(x)

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               theta: float = 0.02) -> Tuple[float, float]:
        err_prediction = self.err_learner.predict(x)
        self.err_learner.update(reward, gamma, x, theta, 1)
        sq_reward = reward ** 2 + 2 * gamma * reward * err_prediction
        sq_gamma = gamma ** 2
        self.sq_learner.update(sq_reward, sq_gamma, x, theta, 1)
        errsq = (err_prediction - self.predict(x)) ** 2
        varg = max(0.0, self.sq_learner.predict(x) - err_prediction ** 2)
        lamda = errsq / (varg + errsq)
        delta = self.learner.update(reward, gamma, x, theta, lamda)
        return delta, lamda


class DLGTIDBD:

    def __init__(self,
                 initial_x: np.ndarray,
                 max_return: float,
                 initial_alpha: float = 0.05):
        assert (len(initial_x.shape) == 1)
        self._last_x = np.array(initial_x)
        self.learner = TIDBD(initial_x, initial_alpha)
        self.err_learner = TIDBD(initial_x, initial_alpha)
        self.err_learner.w.fill(max_return)
        self.err_learner.last_prediction = self.err_learner.predict(initial_x)
        self.var_learner = TIDBD(initial_x, initial_alpha)

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return self.learner.predict(x)

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               theta: float = 0.02) -> Tuple[float, float]:
        err_prediction = self.err_learner.predict(x)
        delta_err = self.err_learner.update(reward, gamma, x, theta, 1)
        var_reward = delta_err ** 2
        var_gamma = gamma ** 2
        self.var_learner.update(var_reward, var_gamma, x, theta, 1)
        errsq = (err_prediction - self.predict(x)) ** 2
        varg = max(0.0, self.var_learner.predict(x))
        lamda = errsq / (varg + errsq)
        delta = self.learner.update(reward, gamma, x, theta, lamda)
        np.copyto(self._last_x, x)
        return delta, lamda


class LGGTD2:

    EPS = 1e-3

    def __init__(self, initial_x, max_return):
        assert (len(initial_x.shape) == 1)
        n = len(initial_x)
        self._last_x = np.copy(initial_x)
        self._GTD = GTD(initial_x)
        self.w_err = np.ones(n) * max_return
        self.w_sq = np.zeros(n)
        self.e_bar = np.zeros(n)
        self.z_bar = np.ones(n, dtype=float) * initial_x

    def predict(self, x):
        """Return the current prediction for a given set of features x."""
        return np.dot(self._GTD.w, x)

    def update(self, reward, gamma, x, alpha, eta, rho=1):
        lamda = LGGTD2.lambda_greedy(
            self._last_x, reward, gamma, x, rho,
            self.w_err, self.w_sq, self._GTD.w, self.e_bar, self.z_bar, alpha)
        delta = self._GTD.update(reward, gamma, x, alpha, eta, lamda, rho)
        np.copyto(self._last_x, x)
        return delta, lamda

    @staticmethod
    def lambda_greedy(x, next_reward, next_gamma, next_x, rho, w_err,
                      w_sq, w, e_bar, z_bar, alpha) -> float:
        # use GTD to update w_err
        next_g_bar = np.dot(next_x, w_err)
        delta_err = next_reward + next_gamma * next_g_bar - np.dot(x, w_err)
        e_bar *= rho
        w_err += alpha * delta_err * e_bar
        e_bar *= next_gamma
        e_bar += next_x

        # use VTD to update w_sq
        next_reward_bar = (rho * next_reward) ** 2 + 2 * (rho ** 2) * next_gamma * next_reward * next_g_bar
        next_gamma_bar = (rho * next_gamma) ** 2
        delta_bar = next_reward_bar + next_gamma_bar * np.dot(next_x, w_sq) - np.dot(x, w_sq)
        w_sq += alpha * delta_bar * z_bar
        z_bar *= next_gamma_bar
        z_bar += next_x

        # compute lambda estimate
        errsq = (next_g_bar - np.dot(next_x, w)) ** 2
        varg = max(0.0, float(np.dot(next_x, w_sq) - next_g_bar ** 2))
        return float(errsq / float(varg + errsq))


class DLGGTD2:

    EPS = 1e-3

    def __init__(self, initial_x, max_return):
        assert (len(initial_x.shape) == 1)
        n = len(initial_x)
        self._last_x = np.copy(initial_x)
        self._GTD = GTD(initial_x)
        self.w_err = np.ones(n) * max_return
        self.w_var = np.zeros(n)
        self.e_bar = np.ones(n, dtype=float) * initial_x
        self.z_bar = np.ones(n, dtype=float) * initial_x

    def predict(self, x):
        """Return the current prediction for a given set of features x."""
        return np.dot(self._GTD.w, x)

    def update(self, reward, gamma, x, alpha, eta, rho=1):
        lamda = DLGGTD2.lambda_greedy(
            self._last_x, reward, gamma, x, rho,
            self.w_err, self.w_var, self._GTD.w, self.e_bar, self.z_bar, alpha)
        delta = self._GTD.update(reward, gamma, x, alpha, eta, lamda, rho)
        np.copyto(self._last_x, x)
        return delta, lamda

    @staticmethod
    def lambda_greedy(x, next_reward, next_gamma, next_x, rho, w_err,
                      w_var, w, e_bar, z_bar, alpha) -> float:
        # use GTD to update w_err
        next_g_bar = np.dot(next_x, w_err)
        delta_err = next_reward + next_gamma * next_g_bar - np.dot(x, w_err)
        e_bar *= rho
        w_err += alpha * delta_err * e_bar
        e_bar *= next_gamma
        e_bar += next_x

        # use DVTD to update w_var
        next_reward_bar = (rho * delta_err - (rho - 1) * np.dot(x, w_err)) ** 2
        next_gamma_bar = (rho * next_gamma) ** 2
        delta_bar = next_reward_bar + next_gamma_bar * np.dot(next_x, w_var) - np.dot(x, w_var)
        w_var += alpha * delta_bar * z_bar
        z_bar *= next_gamma_bar
        z_bar += next_x

        # compute lambda estimate
        errsq = (next_g_bar - np.dot(next_x, w)) ** 2
        varg = max(0.0, float(np.dot(next_x, w_var)))
        return float(errsq / float(varg + errsq))
