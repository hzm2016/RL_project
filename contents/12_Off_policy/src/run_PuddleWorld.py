# -*- coding: utf-8 -*-
"""
# @Time    : 29/06/18 12:23 PM
# @Author  : ZHIMIN HOU
# @FileName: run_PuddleWorld.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import numpy as np
np.random.seed(1)
import time
import gym
import gym_puddle
import gym.spaces
import pickle
from algorithms import *
from Tile_coding import *


"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 4001
MAX_EP_STEPS = 5000
runs = 1
alphas = [5e-5]
lams = [0.3]
eta = 0.0
gamma = 0.99
agents = ['Allactions']

"""Environments Informations :: Puddle world"""
env = gym.make('PuddleWorld-v0')
env.seed(1)
env = env.unwrapped

env_test = gym.make('PuddleWorld-v0')
env_test.seed(1)

print("Environments information:")
print(env.action_space.n)
print(env.observation_space.shape[0])
print(env.observation_space.high)
print(env.observation_space.low)

"""Tile coding"""
NumOfTilings = 10
MaxSize = 10000
HashTable = IHT(MaxSize)

"""position and velocity needs scaling to satisfy the tile software"""
PositionScale = NumOfTilings / (env.observation_space.high[0] - env.observation_space.low[0])
VelocityScale = NumOfTilings / (env.observation_space.high[1] - env.observation_space.low[1])

def getQvalueFeature(obv, action):
    activeTiles = tiles(HashTable, NumOfTilings, [PositionScale * obv[0], VelocityScale * obv[1]], [action])
    return activeTiles

def getValueFeature(obv):
    activeTiles = tiles(HashTable, NumOfTilings, [PositionScale * obv[0], VelocityScale * obv[1]])
    return activeTiles


"""
########################Policy Evaluation#########################
utilized target policy to generate a trajectory 
sampled 2000 states from one trajectory
and run 500 Monte Carlo rollouts to compute an estimate true value
"""
def evaluat_policy(algorithm, target_policy):

    trajectory = np.zeros((10000, env.observation_space.shape[0]))
    state = env.reset()
    for i in range(10000):
        trajectory[i] = state
        action = np.random.choice(env.action_space.n, p=target_policy)
        state_next, reward, done, info = env.step(action)
        state = state_next
    print('trajectory generate finished')

    sample_index = np.random.choice(10000, 2000)
    sample_state = trajectory[sample_index, :]
    sample_reward = []
    for j in range(2000):
        start = sample_state[j]
        prediction = algorithm.predict(getValueFeature(start))
        episode_reward = []
        for num in range(500):
            track_r = []
            while True:
                take_action = np.random.choice(env.action_space.n, p=target_policy)
                state_next, reward, done, info = env.step(take_action)
                track_r.append(reward)
                if done or i > MAX_EP_STEPS:
                    episode_reward.append(sum(track_r))
                    break
        error = abs(prediction - np.mean(episode_reward))/np.mean(episode_reward)
        sample_reward.append(error)
    print('sample_generate finished')
    return np.mean(sample_reward)


"""build learners"""
behavior_policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
target_policy = np.array([0., 0., 0.5, 0., 0.5])
espisode_reward = []
observation = env.reset()
learner = GTD(getValueFeature(observation))

for i_espisode in range(MAX_EPISODE):
    t = 0
    track_r = []
    observation = env.reset()
    while True:
        action = np.random.choice(env.action_space.n, p=behavior_policy)
        observation_, reward, done, info = env.step(action)
        track_r.append(reward)
        rho = target_policy[action]/behavior_policy[action]
        delta = learner.update(reward, gamma, getValueFeature(observation), alphas[0], eta, lams[0], rho=rho)
        observation = observation_
        t += 1
        if done or t > MAX_EP_STEPS:
            break
    print('num_espisode', i_espisode)
    if i_espisode > 0 and i_espisode % 50 == 0:
        error = evaluat_policy(learner, target_policy)
        print('num_espisode %d, cumulative_reward %f' % (i_espisode, error))


"""
########################Control#########################
utilized target policy to generate a trajectory 
sampled 2000 states from one trajectory
and run 500 Monte Carlo rollouts to compute an estimate true value
"""

# def control_performance(off_policy, behavior_policy):
#
#     t = 0
#     track_r = []
#     observation = env_test.reset()
#     action_test = off_policy.start(getValueFeature(observation), behavior_policy)
#     while True:
#
#         observation_, reward, done, info = env.step(action_test)
#         track_r.append(reward)
#         action_test = off_policy.choose_action(getValueFeature(observation))
#         observation = observation_
#         t += 1
#         if done or t > MAX_EP_STEPS:
#             return sum(track_r)

# off_policy = OffActorCritic(MaxSize, env.action_space.n, \
#                             gamma, eta, alphas[0]*10, alphas[0], lams[0], lams[0])
#
# behavior_policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
#
# for i_espisode in range(MAX_EPISODE):
#
#     t = 0
#     track_r = []
#     observation = env.reset()
#     action = off_policy.start(getValueFeature(observation), behavior_policy)
#     while True:
#
#         observation_, reward, done, info = env.step(action)
#         track_r.append(reward)
#         optimal_action, delta = off_policy.step(reward, getValueFeature(observation), behavior_policy)
#         observation = observation_
#         action = np.random.choice(env.action_space.n, p=behavior_policy)
#         t += 1
#
#         if done or t > MAX_EP_STEPS:
#             break
#
#     if i_espisode % 100 == 0:
#         cum_reward = test(off_policy)
#         print('num_espisode %d, cumulative_reward %f' % (i_espisode, cum_reward))

# LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, 0.99, 0., 1e-4, 1e-5, 0.3, 0.3)
# espisode_reward = []
# observation = env.reset()
# action = LinearAC.start(getValueFeature(observation))
#
# for i_espisode in range(MAX_EPISODE):
#
#     t = 0
#     track_r = []
#     while True:
#
#         observation_, reward, done, info = env.step(action)
#         track_r.append(reward)
#         action, delta = LinearAC.step(reward, getValueFeature(observation))
#         observation = observation_
#         t += 1
#         if done or t > MAX_EP_STEPS:
#
#             observation = env.reset()
#             ep_rs_sum = sum(track_r)
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             print("episode:", i_espisode,  "reward:", int(running_reward))
#             espisode_reward.append(int(running_reward))
#             break

# LinearAC = OffPAC(MaxSize, env.action_space.n, 0.99, 0., 1e-4, 1e-5, 0.2, 0.2)
# behavior_policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
# espisode_reward = []
# observation = env.reset()
# action = LinearAC.start(getValueFeature(observation), behavior_policy)
#
# for i_espisode in range(MAX_EPISODE):
#
#     t = 0
#     track_r = []
#
#     while True:
#         observation_, reward, done, info = env.step(action)
#         track_r.append(reward)
#         action, delta = LinearAC.step(reward, getValueFeature(observation), behavior_policy)
#         observation = observation_
#         t += 1
#         if done or t > MAX_EP_STEPS:
#             env.reset()
#             break
#
#     if i_espisode % 100 == 0:
#         average_reward = []
#         for j in range(100):
#             cum_reward = test(LinearAC, behavior_policy)
#             average_reward.append(cum_reward)
#         print('num_espisode %d, cumulative_reward %f' % (i_espisode, np.mean(average_reward)))