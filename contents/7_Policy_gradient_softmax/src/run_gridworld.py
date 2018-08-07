# -*- coding: utf-8 -*-
"""
# @Time    : 29/06/18 12:25 PM
# @Author  : ZHIMIN HOU
# @FileName: run_gridworld.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import numpy as np
np.random.seed(1)
import gym
import gym_puddle
import gym.spaces
import pickle
from LinearActorCritic import *
from Tile_coding import *
import argparse

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 4001
MAX_EP_STEPS = 5000

"""Environments Informations :: Puddle world"""
env = gym.make('PuddleWorld-v0')
env = env.unwrapped

"""Tile coding"""
NumOfTilings = 10
MaxSize = 1000000
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
    parser.add_argument('--directory', default='./logs')
    parser.add_argument('alpha', type=float, default=np.array([5e-5]))  # np.array([5e-5, 5e-4, 1e-3, 1e-2, 0.5, 1.])
    parser.add_argument('lambda', type=float, default=np.array([0.]))  # np.array([0., 0.2, 0.4, 0.6, 0.8, 0.99]))
    parser.add_argument('num_runs', type=int, default=5)
    parser.add_argument('--all_algorithms', type=str, dest='all_algorithms',
                        default=['DiscreteActorCritic', 'Allactions', 'AdvantageActorCritic'])
    parser.add_argument('--alpha_h', type=float, default=np.array([0.0001]))
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--num_states', type=int, dest='num_states', default=5)
    parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
    parser.add_argument('--num_episodes', type=int, dest='num_episodes', default=1000)
    parser.add_argument('--behavior_policy', type=float, dest='behavior_policy', default=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    parser.add_argument('--target_policy', type=float, dest='target_policy', default=np.array([0., 0., 0.5, 0., 0.5]))
    parser.add_argument('--test_name', default='Gridworld_on_policy')
    args = vars(parser.parse_args())
    return args


def play(LinearAC, agent):

    if agent == 'Allactions':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.choose_action(getValueFeature(observation))
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
        action = LinearAC.choose_action(getValueFeature(observation))
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
    elif agent == 'Reinforce':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.choose_action(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            track_r.append(reward)
            LinearAC.store_trasition(getValueFeature(observation), action, reward)
            action = LinearAC.choose_action(getValueFeature(observation))
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:
                return t, int(sum(track_r))
    elif agent == 'DiscreteActorCritic':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.choose_action(getValueFeature(observation))
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
        action = LinearAC.choose_action(getValueFeature(observation))
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
    steps = np.zeros((len(args['all_algorithms']), args['num_runs'], args['num_episodes']))
    rewards = np.zeros((len(args['all_algorithms']), args['num_runs'], args['num_episodes']))
    for agentInd, agent in enumerate(args['all_algorithms']):
        for run in range(args['num_runs']):
            env.seed(run)
            if agent == 'Reinforce':
                LinearAC = Reinforce(MaxSize, env.action_space.n, args['gamma'], args['eta'], args['alpha']*10, args['alpha'], args['lambda'], args['lambda'],
                                     random_generator=np.random.RandomState(run))
            elif agent == 'Allactions':
                LinearAC = Allactions(MaxSize, env.action_space.n,  args['gamma'], args['eta'], args['alpha']*10, args['alpha'], args['lambda'], args['lambda'],
                                      random_generator=np.random.RandomState(run))
            elif agent == 'AdvantageActorCritic':
                LinearAC = AdvantageActorCritic(MaxSize, env.action_space.n,  args['gamma'], args['eta'], args['alpha']*10, args['alpha'], args['lambda'], args['lambda'], True,
                                                random_generator=np.random.RandomState(run))
            elif agent == 'DiscreteActorCritic':
                LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n,  args['gamma'], args['eta'], args['alpha']*10, args['alpha'], args['lambda'], args['lambda'],
                                               random_generator=np.random.RandomState(run))
            else:
                print('Please give the right agent!')
            print("++++++++++++++++++++++%s++++++++++++++++++++" % agent)
            observation = env.reset()
            action = LinearAC.start(getValueFeature(observation))
            for ep in range(args['num_episodes']):
                step, reward = play(LinearAC, agent)
                if ep == 0:
                    running_reward = reward
                else:
                    running_reward = running_reward * 0.9 + reward * 0.10
                steps[agentInd, run, ep] = step
                rewards[agentInd, run, ep] = running_reward
                print('agent %s, run %d, episode %d, steps %d, rewards%d' %
                      (agent, run, ep, step, running_reward))

    with open('{}/reward_{}_alpha_{}_lambda_{}.npy'.format(args['directory'], args['test_name'], args['alpha'], args['lambda']), 'wb') as outfile:
        np.save(outfile, rewards)