#!/usr/bin/env python -O
# -*- coding: utf-8 -*-
"""
# @Time    : 09/07/18 8:45 PM
# @Author  : ZHIMIN HOU
# @FileName: run_MountainCar.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import gym
import gym.spaces
from Tile_coding import *
from algorithms import *
import numpy as np
import argparse
import pickle

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 4001
MAX_EP_STEPS = 5000

"""Environments Informations"""
env = gym.make('MountainCar-v0')
env_test = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
# env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

"""Tile coding"""
NumOfTilings = 10
MaxSize = 100000
HashTable = IHT(MaxSize)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='../control_data')
    parser.add_argument('--alpha', type=float, default=np.array([0.05]))
    parser.add_argument('--alpha_h', type=float, default=np.array([0.0001]))
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--lambda', type=float, default=np.array([0.]))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.75)
    parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--num_runs', type=int, dest='num_runs', default=10)
    parser.add_argument('--num_states', type=int, dest='num_states', default=5)
    parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
    parser.add_argument('--num_episodes', type=int, dest='num_episodes', default=400)
    parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=1000)
    parser.add_argument('--num_change', type=int, dest='num_change', default=1000)
    parser.add_argument('--all_algorithms', type=str, dest='all_algorithms', default=['OffPAC'])
    parser.add_argument('--behavior_policy', type=float, dest='behavior_policy', default=np.array([0.3, 0.3, 0.4]))
    parser.add_argument('--test_name', default='control_test')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args


"""position and velocity needs scaling to satisfy the tile software"""
PositionScale = NumOfTilings / (env.observation_space.high[0] - env.observation_space.low[0])
VelocityScale = NumOfTilings / (env.observation_space.high[1] - env.observation_space.low[1])

def getQvalueFeature(obv, action):
    activeTiles = tiles(HashTable, NumOfTilings, [PositionScale * obv[0], VelocityScale * obv[1]], [action])
    return activeTiles

def getValueFeature(obv):
    activeTiles = tiles(HashTable, NumOfTilings, [PositionScale * obv[0], VelocityScale * obv[1]])
    return activeTiles


"""evaluate the control performance"""
def evalate(off_policy, behavior_policy):
    episode_reward = []
    for i in range(50):
        t = 0
        track_r = []
        observation = env.reset()
        action_test = off_policy.start(getValueFeature(observation), behavior_policy)
        while True:

            observation_, reward, done, info = env.step(action_test)
            track_r.append(reward)
            action_test = off_policy.choose_action(getValueFeature(observation))
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:
                episode_reward.append(sum(track_r))
                break
    return np.mean(episode_reward)


def play(LinearAC, agent):
    observation = env.reset()
    action = LinearAC.start(getValueFeature(observation), args['behavior_policy'])
    if agent == 'OffActorCritic':
        t = 0
        track_r = []
        env.reset()
        while True:
            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            action, delta = LinearAC.step(reward, getValueFeature(observation), args['behavior_policy'])
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:
                break
    elif agent == 'OffPAC':
        t = 0
        track_r = []
        env.reset()
        while True:
            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            action, delta = LinearAC.step(reward, getValueFeature(observation), args['behavior_policy'])
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:
                break
    else:
        print("Please give the agent you wanna use!!!")


if __name__ == '__main__':
    args = parse_args()

    """Run all the parameters"""
    rewards = np.zeros((len(args['lambda']), len(args['alpha']), args['num_runs'], int(args['num_episodes']/10)))
    for lamInd, lam in enumerate(args['lambda']):
        for alphaInd, alpha in enumerate(args['alpha']):
            for run in range(args['num_runs']):
                for agentInd, agent in enumerate(args['all_algorithms']):
                    if agent == 'OffActorCritic':
                        LinearAC = OffActorCritic(MaxSize, env.action_space.n, args['gamma'], args['eta'], alpha*10, alpha, lam, lam)
                    elif agent == 'OffPAC':
                        LinearAC = OffPAC(MaxSize, env.action_space.n, args['gamma'], args['eta'], alpha, 1.0, 0.0001, lam, lam)
                    else:
                        print('Please give the right agent!')
                    for ep in range(args['num_episodes']):
                        reward = play(LinearAC, agent)
                        if ep > 0 and ep % 10 == 0:
                            cum_reward = evalate(LinearAC, args['behavior_policy'])
                            if 'running_reward' not in globals():
                                running_reward = cum_reward
                            else:
                                running_reward = running_reward * 0.99 + cum_reward * 0.01
                            rewards[lamInd, alphaInd, run, int(ep/10)] = running_reward
                            print('lambda %f, alpha %f, run %d, episode %d, rewards%d' % (lam, alpha, run, ep, running_reward))
    with open('{}/rmse_{}.npy'.format(args['directory'], args['test_name']), 'wb') as outfile:
        np.save(outfile, rewards)