# -*- coding: utf-8 -*-
"""
# @Time    : 29/06/18 12:25 PM
# @Author  : ZHIMIN HOU
# @FileName: run_gridworld.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

from Grid_world import Maze
import numpy as np
import math as m
from Tile_coding import *
from LinearActorCritic import GTD


def feature_representation(observation):
    weight = np.array([[0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001], [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]])
    bias = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
    return np.array(np.matrix(observation) * np.matrix(weight) + bias)[0]


def behavior_policy():
    action = np.random.choice([0, 1, 2, 3], 1, p=[0.15, 0.35, 0.35, 0.15])
    return action


def target_policy():
    action = np.random.choice([1, 2], 1, p=[0.5, 0.5])
    return action


def recongnizers_policy():
    action = behavior_policy()
    c = np.array([0., 1., 1, 0.])
    if action == 1:
        rec = 1 / np.sum(c * [0.15, 0.35, 0.35, 0.15])
    elif action == 2:
        rec = 1 / np.sum(c * [0.15, 0.35, 0.35, 0.15])
    else:
        rec = 0.
    return action, rec


def compute_true_return(gamma, state):
    R = 0.
    for i in range(16):
        R += m.pow(gamma, i) * (-1)
    R += m.pow(gamma, 16) * 10
    return R


def compute_mean_square_error(weight_name, state):
    weight = np.load(weight_name)
    true_value = compute_true_return(0.8, state)
    pre_value = np.dot(weight, feature_representation(state))
    pre_error = np.sqrt(np.power((pre_value - true_value), 2))
    return pre_error


def train(env):
    delta = []
    observation = env.reset()
    learner = GTD(feature_representation(observation))
    for episode in range(100):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            # action = RL.choose_action(str(observation))
            # action = behavior_policy()

            """Compute the recongnizer"""
            action, rho = recongnizers_policy()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            """Compute the importance sampling: target policy: pi->0.5"""
            # if action == 1:
            #     rho = 0.5/0.25
            # elif action == 2:
            #     rho = 0.5/0.25
            # else:
            #     rho = 0

            # RL learn from this transition
            # RL.learn(str(observation), action, reward, str(observation_))
            delta.append(learner.update(reward, 1.0, feature_representation(observation), 0.00005, 0.01, 0.8, rho))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    # np.save('delta', delta)
    return learner


def run():
    env = Maze()
    print(env.reset())
    GTD = train(env)

    value = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            value[i, j] = (GTD.predict(feature_representation(np.array([i, j]))))

    print(GTD.w)
    np.save('weight', np.array(GTD.w))
    print(value)
    np.save('value', np.array(value))


if __name__ == "__main__":

    run()