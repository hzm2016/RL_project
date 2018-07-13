# -*- coding: utf-8 -*-
"""
# @Time    : 10/07/18 5:42 PM
# @Author  : ZHIMIN HOU
# @FileName: run_Evaluation.py
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
import argparse

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 4001
MAX_EP_STEPS = 5000

"""Environments Informations :: Puddle world"""
env = gym.make('PuddleWorld-v0')
env.seed(1)
env = env.unwrapped

"""Tile coding"""
NumOfTilings = 10
MaxSize = 100000
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


"""Parameters"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default='../control_data')
    parser.add_argument('alpha', type=float, default=1e-4)  # np.array([1e-4, 5e-4, 1e-3, 1e-2, 0.5, 1.])
    parser.add_argument('lambda', type=float, default=0.0)  # np.array([0., 0.2, 0.4, 0.6, 0.8, 0.99])
    parser.add_argument('--alpha_h', type=float, default=np.array([0.0001]))
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.75)
    parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--num_runs', type=int, dest='num_runs', default=10)
    parser.add_argument('--num_states', type=int, dest='num_states', default=5)
    parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
    parser.add_argument('--num_episodes', type=int, dest='num_episodes', default=4000)
    parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=1000)
    parser.add_argument('--num_change', type=int, dest='num_change', default=1000)
    parser.add_argument('--all_algorithms', type=str, dest='all_algorithms', default=['OffPAC'])
    parser.add_argument('--behavior_policy', type=float, dest='behavior_policy',
                        default=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    parser.add_argument('--target_policy', type=float, dest='target_policy',
                        default=np.array([0., 0., 0.5, 0., 0.5]))
    parser.add_argument('--test_name', default='puddle_evaluation')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args


def evaluat_policy(learner, target_policy):
    trajectory = np.zeros((10000, env.observation_space.shape[0]))
    state = env.reset()
    for i in range(10000):
        trajectory[i] = state
        action = np.random.choice(env.action_space.n, p=target_policy)
        state_next, reward, done, info = env.step(action)
        state = state_next
    # print('trajectory generate finished')

    sample_index = np.random.choice(10000, 2000)
    sample_state = trajectory[sample_index, :]
    sample_reward = []
    for j in range(2000):
        start = sample_state[j]
        prediction = learner.predict(getValueFeature(start))
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
        error = abs(prediction - np.mean(episode_reward)) / np.mean(episode_reward)
        sample_reward.append(error)
    # print('sample_generate finished')
    return np.mean(sample_reward)


def play_evaluation(learner, behavior_policy):
    t = 0
    observation = env.reset()
    learner.start(getValueFeature(observation))
    action = np.random.choice(env.action_space.n, p=behavior_policy)
    while True:

        observation_, reward, done, info = env.step(action)
        rho = args['target_policy'][action] / args['behavior_policy'][action]

        delta = learner.update(reward, getValueFeature(observation), rho=rho)
        action = np.random.choice(env.action_space.n, p=behavior_policy)
        observation = observation_
        t += 1
        if done or t > MAX_EP_STEPS:
            break


"""
########################OffPolicy Evaluation#########################
utilized target policy to generate a trajectory 
sampled 2000 states from one trajectory
and run 500 Monte Carlo rollouts to compute an estimate true value
"""
if __name__ == '__main__':
    args = parse_args()
    """Run all the parameters"""
    rewards = np.zeros((args['num_runs'], int(args['num_episodes']/50)))
    for run in range(args['num_runs']):
        for agentInd, agent in enumerate(args['all_algorithms']):
            learner = GTD(MaxSize, args['gamma'], args['eta'], args['alpha'], args['lambda'])
            for ep in range(args['num_episodes']):
                play_evaluation(learner, args['behavior_policy'])
                if ep > 0 and ep % 50 == 0:
                    cum_reward = evaluat_policy(learner, args['target_policy'])
                    rewards[run, int(ep / 50)] = cum_reward
                    print('lambda %f, alpha %f, run %d, episode %d, rewards%d' % (run, ep, cum_reward))

    with open('{}/rmse_{}.npy'.format(args['dir'], args['test_name']), 'wb') as outfile:
        np.save(outfile, rewards)