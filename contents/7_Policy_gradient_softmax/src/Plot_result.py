# -*- coding: utf-8 -*-
"""
# @Time    : 03/07/18 9:34 PM
# @Author  : ZHIMIN HOU
# @FileName: Plot_result.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse


"""Parameters"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='../logs')
    parser.add_argument('--alpha', type=float, default=1e-5)  # np.array([5e-5, 1e-5, 1e-4, 1e-2, 0.5, 1.])
    parser.add_argument('--lambda', type=float, default=0.1)  # np.array([0., 0.2, 0.4])
    parser.add_argument('--figure_name', type=str, default='13')
    parser.add_argument('--all_algorithms', type=str, dest='all_algorithms',
                        default=['DiscreteActorCritic', 'Allactions', 'AdvantageActorCritic'])
    parser.add_argument('--alpha_h', type=float, default=np.array([0.0001]))
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.75)
    parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--num_runs', type=int, dest='num_runs', default=1)
    parser.add_argument('--num_states', type=int, dest='num_states', default=5)
    parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
    parser.add_argument('--num_episodes', type=int, dest='num_episodes', default=3000)
    parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=1000)
    parser.add_argument('--num_change', type=int, dest='num_change', default=1000)
    parser.add_argument('--behavior_policy', type=float, dest='behavior_policy', default=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    parser.add_argument('--target_policy', type=float, dest='target_policy',
                        default=np.array([0., 0., 0.5, 0., 0.5]))
    parser.add_argument('--test_name', default='MountainCar_on_policy')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args


args = parse_args()
with open('{}/reward_{}_alpha_{}_lambda_{}.npy'.format(args['directory'], args['test_name'], args['alpha'], args['lambda']), 'rb') as outfile:
    rewards = np.load(outfile)

mean_rewards = np.mean(rewards, axis=1)
std_rewards = np.std(rewards, axis=1)

figureIndex = 0
plt.figure(figureIndex, figsize=(9, 6))
figureIndex += 1
for optionInd, name in enumerate(args['all_algorithms']):
    plt.fill_between(np.arange(len(mean_rewards[0, :])), mean_rewards[optionInd, :] - std_rewards[optionInd, :], mean_rewards[optionInd, :] + std_rewards[optionInd, :], alpha=0.3)
    plt.plot(np.arange(len(mean_rewards[0, :])), mean_rewards[optionInd, :], label=name, linewidth=2.5)
plt.title('alpha_{}_lambda_{}.png'.format(args['alpha'], args['lambda']), fontsize=18)
plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Cumulative Rewards', fontsize=16)
plt.legend(fontsize=16)
plt.savefig("../logs/{}.png".format(args['figure_name']))
plt.show()