# -*- coding: ascii -*-

import numpy as np

from typing import Tuple


__all__ = ('GTD', 'TIDBD', 'LGGTD', 'DLGGTD', 'LGTIDBD', 'DLGTIDBD', 'LGGTD2', 'DLGGTD2')


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
