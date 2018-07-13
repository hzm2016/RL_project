# -*- coding: utf-8 -*-
"""
# @Time    : 29/06/18 12:23 PM
# @Author  : ZHIMIN HOU
# @FileName: run_Control.py
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

# print("Environments information:")
# print(env.action_space.n)
# print(env.observation_space.shape[0])
# print(env.observation_space.high)
# print(env.observation_space.low)

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
    parser.add_argument('--dir', default='../control_data')
    parser.add_argument('alpha', type=float)  # default=np.array([1e-4, 5e-4, 1e-3, 1e-2, 0.5, 1.])
    parser.add_argument('--alpha_h', type=float, default=np.array([0.0001]))
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('lambda', type=float)  # default=np.array([0., 0.2, 0.4, 0.6, 0.8, 0.99])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.75)
    parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--num_runs', type=int, dest='num_runs', default=30)
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
    parser.add_argument('--test_name', default='puddle_control')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args


def control_performance(off_policy, behavior_policy):

    average_reward = []
    for j in range(100):
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
                average_reward.append(sum(track_r))
                break
    return np.mean(average_reward)


def play_control(learner, behavior_policy):

    t = 0
    observation = env.reset()
    action = learner.start(getValueFeature(observation), behavior_policy)
    while True:
        observation_, reward, done, info = env.step(action)
        action, delta = learner.step(reward, getValueFeature(observation), behavior_policy)
        observation = observation_
        t += 1
        if done or t > MAX_EP_STEPS:
            break


"""
################################offPolicyControl#########################
utilized target policy to generate a trajectory 
sampled 2000 states from one trajectory
and run 500 Monte Carlo rollouts to compute an estimate true value
"""
if __name__ == '__main__':

    args = parse_args()
    """Run all the parameters"""
    print('#############################offPolicyControl::Env_puddle#########################')
    rewards = np.zeros((len(args['all_algorithms']), args['num_runs'], int(args['num_episodes']/50)))
    for agentInd, agent in enumerate(args['all_algorithms']):
        for run in range(args['num_runs']):
            if agent == 'OffPAC':
                learner = OffPAC(MaxSize, env.action_space.n, args['gamma'], args['eta'], \
                                 args['alpha']*10, args['alpha'], args['alpha_h'], args['lambda'], args['lambda'])
            else:
                print('Please give the right agent!!')
            for ep in range(args['num_episodes']):
                play_control(learner, args['behavior_policy'])
                if ep > 0 and ep % 50 == 0:
                    cum_reward = control_performance(learner, args['behavior_policy'])
                    rewards[agentInd, run, int(ep/50)] = cum_reward
                    print('agent %s, run %d, episode %d, rewards%d' % (agent, run, ep, cum_reward))
    with open('{}/rmse_{}_alpha_{}_lambda_{}.npy'.format(args['directory'], args['test_name'], args['alpha'], args['lambda']), 'wb') as outfile:
        np.save(outfile, rewards)


# off_policy = OffActorCritic(MaxSize, env.action_space.n, \
#                             gamma, eta, alphas[0]*10, alphas[0], lams[0], lams[0])
# behavior_policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
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
#
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