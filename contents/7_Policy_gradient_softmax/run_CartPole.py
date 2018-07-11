# -*- coding: utf-8 -*-
"""
# @Time    : 06/07/18 9:22 AM
# @Author  : ZHIMIN HOU
# @FileName: run_LimitCartPole.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import gym
from Tile_coding import *
from LinearActorCritic import *
import numpy as np
import pickle

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 4001
MAX_EP_STEPS = 10000   # maximum time step in one episode
runs = 1
alphas = [5e-5, 1e-5]
lams = [0.0, 0.3]
eta = 0.0
gamma = 0.99
agents = ['Allactions', 'AdvantageActorCritic']

"""Environments Informations"""
env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
# env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped
# print("Environments information:")
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

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


# def play(LinearAC, agent):
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
#                 track_r.append(reward)
#                 feature = []
#                 for i in range(env.action_space.n):
#                     feature.append(getQvalueFeature(observation, i))
#                 action, delta = LinearAC.step(reward, getValueFeature(observation), \
#                                               getQvalueFeature(observation, action), \
#                                               feature)
#                 observation = observation_
#                 t += 1
#                 if done or t > MAX_EP_STEPS:
#                     return t, int(sum(track_r))
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
#                     return t, int(sum(track_r))
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
#                     return t, int(sum(track_r))
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
#                     return t, int(sum(track_r))
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
#                     return t, int(sum(track_r))
def play(LinearAC, agent):

    if agent == 'Allactions':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            feature = []
            for i in range(env.action_space.n):
                feature.append(getQvalueFeature(observation, i))
            action, delta = LinearAC.step(reward, getValueFeature(observation), \
                                          getQvalueFeature(observation, action), \
                                          feature)
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:
                return t, int(sum(track_r))
    elif agent == 'AdvantageActorCritic':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)

            feature = []
            for i in range(env.action_space.n):
                feature.append(getQvalueFeature(observation, i))

            # action_ = LinearAC.choose_action(getValueFeature(observation_))
            # getQvalueFeature(observation_, action_),

            track_r.append(reward)

            action, delta = LinearAC.step(reward, getValueFeature(observation), \
                                          getQvalueFeature(observation, action), \
                                          feature)
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:
                return t, int(sum(track_r))
    elif agent == 'Reinforce':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            LinearAC.store_trasition(getValueFeature(observation), action, reward)
            action = LinearAC.choose_action(getValueFeature(observation))
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:

                return t, int(sum(track_r))
    elif agent == 'OffDiscreteActorCritic':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            action, delta = LinearAC.step(reward, getValueFeature(observation))
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:

                return t, int(sum(track_r))
    else:
        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            action, delta = LinearAC.step(reward, getValueFeature(observation))
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:

                return t, int(sum(track_r))


if __name__ == '__main__':

    """Run all the parameters"""
    steps = np.zeros((len(lams), len(alphas), runs, MAX_EPISODE))
    rewards = np.zeros((len(lams), len(alphas), runs, MAX_EPISODE))
    for lamInd, lam in enumerate(lams):
        for alphaInd, alpha in enumerate(alphas):
            for run in range(runs):
                for agentInd, agent in enumerate(agents):
                    if agent == 'Reinforce':
                        LinearAC = Reinforce(MaxSize, env.action_space.n, gamma, eta, alpha*10, alpha, lam, lam)
                    elif agent == 'Allactions':
                        LinearAC = Allactions(MaxSize, env.action_space.n, gamma, eta, alpha*10, alpha, lam, lam)
                    elif agent == 'AdvantageActorCritic':
                        LinearAC = AdvantageActorCritic(MaxSize, env.action_space.n, gamma, eta, alpha*10, alpha, lam, lam)
                    elif agent == 'DiscreteActorCritic':
                        LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, gamma, eta, alpha*10, alpha, lam, lam)
                    else:
                        print('Please give the right agent!')
                    for ep in range(MAX_EPISODE):
                        step, reward = play(LinearAC, agent)
                        if 'running_reward' not in globals():
                            running_reward = reward
                        else:
                            running_reward = running_reward * 0.99 + reward * 0.01
                        steps[lamInd, alphaInd, run, ep] = step
                        rewards[lamInd, alphaInd, run, ep] = running_reward
                        print('lambda %f, alpha %f, run %d, episode %d, steps %d, rewards%d' %
                              (lam, alpha, run, ep, step, running_reward))
    with open('steps_all_agents.bin', 'wb') as f:
        pickle.dump(steps, f)
    with open('rewards_all_agents.bin', 'wb') as s:
        pickle.dump(rewards, s)