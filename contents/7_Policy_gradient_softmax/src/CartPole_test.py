# -*- coding: utf-8 -*-
"""
# @Time    : 14/07/18 9:29 AM
# @Author  : ZHIMIN HOU
# @FileName: CartPole_test.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import gym
from Tile_coding import *
from LinearActorCritic import *
import numpy as np
import math
import pickle
import argparse

MAX_EP_STEPS = 10000   # maximum time step in one episode
"""Environments Informations"""
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

print("Environments information:")
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[0] = [-2.4, 2.4]
STATE_BOUNDS[1] = [-2., 2.]  # Cart velocity bound
STATE_BOUNDS[2] = [-np.pi/15.0, np.pi/15.0]
STATE_BOUNDS[3] = [-np.pi, np.pi]  # Pole velocity bound
NUM_BUCKETS = (1, 1, 6, 3)

# state_range = np.array([[-2.4, 2.4],  # Cart location bound
#                         [-6., 6.],  # Cart velocity bound
#                         [-np.pi * 12./180., np.pi * 12./180.],  # Pole angle bounds
#                         [-6., 6.]])  # Pole velocity bound


def state_bound(state):
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            state[i] = STATE_BOUNDS[i][0]
        elif state[i] >= STATE_BOUNDS[i][1]:
            state[i] = STATE_BOUNDS[i][1]
        else:
            state[i] = state[i]
    return state


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


# POLYNOMIAL_BASES == 0 :: FOURIER_BASES == 1
type = 1
class FourierBases:
    # @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, type):
        self.order = order
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        if type == 0:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))
        elif type == 1:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    # get the value of @state
    def get_feature(self, state):
        # map the state space into [0, 1]
        self.feature = np.zeros((len(state), self.order + 1))
        for i in range(len(state)):
            state[i] = float((state[i] - STATE_BOUNDS[i][0]) / (STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]))
            # get the feature vector
            self.feature[i, :] = np.asarray([func(state[i]) for func in self.bases])
        feature = np.reshape(self.feature, len(state) * (self.order + 1))
        return feature

"""================================Tile coding================================="""

NumOfTilings = 1
MaxSize = 1024
HashTable = IHT(MaxSize)

"""position and velocity needs scaling to satisfy the tile software"""
scale = np.zeros(4, dtype=float)
for i in range(len(STATE_BOUNDS)):
    scale[i] = NumOfTilings/(STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0])
#
def getQvalueFeature(obv, action):
    obv = state_bound(obv)
    activeTiles = tiles(HashTable, NumOfTilings, [scale[0] * obv[0], scale[1] * obv[1], scale[2] * obv[2], scale[3] * obv[3]], [action])
    return activeTiles

def getValueFeature(obv):
    obv = state_bound(obv)
    activeTiles = tiles(HashTable, NumOfTilings, [scale[0] * obv[0], scale[1] * obv[1], scale[2] * obv[2], scale[3] * obv[3]])
    return activeTiles

"""===============================FourierBases================================"""

# rb = FourierBases(20, 1)
# def getValueFeature(obv):
#     obv = state_bound(obv)
#     feature = rb.get_feature(obv)
#     return feature


"""Parameters"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='../logs')
    parser.add_argument('--alpha', type=float, default=np.array([0.00001]))  # np.array([5e-5, 1e-5, 1e-4, 1e-2, 0.5, 1.])
    parser.add_argument('--alpha_h', type=float, default=np.array([0.0001]))
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--lambda', type=float, default=np.array([0.99]))  # np.array([0., 0.2, 0.4])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.75)
    parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--num_runs', type=int, dest='num_runs', default=1)
    parser.add_argument('--num_states', type=int, dest='num_states', default=5)
    parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
    parser.add_argument('--num_episodes', type=int, dest='num_episodes', default=1000)
    parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=1000)
    parser.add_argument('--num_change', type=int, dest='num_change', default=1000)
    parser.add_argument('--all_algorithms', type=str, dest='all_algorithms', default=['DiscreteActorCritic'])
    parser.add_argument('--behavior_policy', type=float, dest='behavior_policy', default=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    parser.add_argument('--target_policy', type=float, dest='target_policy',
                        default=np.array([0., 0., 0.5, 0., 0.5]))
    parser.add_argument('--test_name', default='cartpole_control_sarsa_and_expected')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args

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

    args = parse_args()

    """Run all the parameters"""
    steps = np.zeros(
        (len(args['all_algorithms']), len(args['lambda']), len(args['alpha']), args['num_runs'], args['num_episodes']))
    rewards = np.zeros(
        (len(args['all_algorithms']), len(args['lambda']), len(args['alpha']), args['num_runs'], args['num_episodes']))
    for agentInd, agent in enumerate(args['all_algorithms']):
        for lamInd, lam in enumerate(args['lambda']):
            for alphaInd, alpha in enumerate(args['alpha']):
                for run in range(args['num_runs']):
                    if agent == 'Reinforce':
                        LinearAC = Reinforce(MaxSize, env.action_space.n, args['gamma'], args['eta'], alpha * 10, alpha,
                                             lam, lam)
                    elif agent == 'Allactions':
                        LinearAC = Allactions(MaxSize, env.action_space.n, args['gamma'], args['eta'], alpha * 10,
                                              alpha, lam, lam)
                    elif agent == 'AdvantageActorCritic':
                        LinearAC = AdvantageActorCritic(MaxSize, env.action_space.n, args['gamma'], args['eta'],
                                                        alpha * 10, alpha, lam, lam, False)
                    elif agent == 'A2CExpected':
                        LinearAC = AdvantageActorCritic(MaxSize, env.action_space.n, args['gamma'], args['eta'],
                                                        alpha * 10, alpha, lam, lam, True)
                    elif agent == 'DiscreteActorCritic':
                        LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, args['gamma'], args['eta'],
                                                       alpha * 10, alpha, lam, lam)
                    else:
                        print('Please give the right agent!')
                    print("++++++++++++++++++++++%s++++++++++++++++++++" % agent)
                    for ep in range(args['num_episodes']):

                        step, reward = play(LinearAC, agent)
                        if ep == 0:
                            running_reward = reward
                        else:
                            running_reward = running_reward * 0.99 + reward * 0.01
                        steps[agentInd, lamInd, alphaInd, run, ep] = step
                        rewards[agentInd, lamInd, alphaInd, run, ep] = running_reward
                        print('agent %s, lambda %f, alpha %f, run %d, episode %d, steps %d, rewards%d' %
                              (agent, lam, alpha, run, ep, step, running_reward))
    with open('{}/reward_{}.npy'.format(args['directory'], args['test_name']), 'wb') as outfile:
        np.save(outfile, rewards)



