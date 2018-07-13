# -*- coding: utf-8 -*-
"""
# @Time    : 29/06/18 12:02 PM
# @Author  : ZHIMIN HOU
# @FileName: Polt_single.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse

all_state = np.array([5])
all_frequency = np.array([1000])
all_alpha = np.array([0.005, 0.0005, 1e-5])
all_decay = np.array([1.0, 0.95, 0.90])
all_numchange = np.array([500, 1000, 5000])
all_name = ['unknown', 'normal', 'fade_frequency', 'sliding_window', 'Tree_backup']

## ================================================1================================================

# parser = argparse.ArgumentParser()
# parser.add_argument('--directory', default='../data')
# parser.add_argument('--alpha', type=float, default=0.0005)
# parser.add_argument('--eta', type=float, default=0.05)
# parser.add_argument('--lambda', type=float, default=0.99)
# parser.add_argument('--ISW', type=int, default=0)
# parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
# parser.add_argument('--left_probability2', type=float, dest='left_probability2', default=0.75)
# parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=20)
# parser.add_argument('--num_states', type=int, dest='num_states', default=10)
# parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
# parser.add_argument('--num_steps', type=int, dest='num_steps', default=50000)
# parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=3000)
# parser.add_argument('--test_name', default='test')
# args = vars(parser.parse_args())
# if 'num_steps' not in args:
#     args['num_steps'] = args['num_states'] * 100
#
#
# """all_rmse = np.ones((len(all_agent), len(all_state), len(all_frequency), args['num_seeds'], args['num_steps'])) * np.inf"""
#
# with open('{}/rmse_{}.npy'.format(args['directory'], args['test_name']), 'rb') as outfile:
#     all_rmse = np.load(outfile)
#
# # average all the seeds
# # all_rmse = np.mean(all_rmse, axis=3)
#
# figureIndex = 0
# step = np.linspace(0, len(all_rmse[0, 0, 0, 0, :])-1, num=len(all_rmse[0, 0, 0, 0, :]))
# for num_frequency, frequency in enumerate(all_frequency):
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for option in range(3):
#         plt.plot(step, all_rmse[option, 0, num_frequency, 0, :], label='behaviory policy = %s' % (all_name[option]))
#     plt.title('frequency update number = %s' % (str(frequency)))
#     plt.xlabel('Steps')
#     plt.ylabel('Episode_error')
#     plt.legend()
# plt.show()

## ================================================2================================================

# parser = argparse.ArgumentParser()
# parser.add_argument('--directory', default='../data')
# parser.add_argument('--alpha', type=float, default=0.0005)
# parser.add_argument('--eta', type=float, default=0.05)
# parser.add_argument('--lambda', type=float, default=0.99)
# parser.add_argument('--ISW', type=int, default=0)
# parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
# parser.add_argument('--left_probability2', type=float, dest='left_probability2', default=0.75)
# parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=1)
# parser.add_argument('--num_states', type=int, dest='num_states', default=10)
# parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
# parser.add_argument('--num_steps', type=int, dest='num_steps', default=50000)
# parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=3000)
# parser.add_argument('--test_name', default='test_different_states_seed')
# args = vars(parser.parse_args())
#
# if 'num_steps' not in args:
#     args['num_steps'] = args['num_states'] * 100
#
# with open('{}/rmse_{}.npy'.format(args['directory'], args['test_name']), 'rb') as outfile:
#     all_rmse = np.load(outfile)
#
# # # average all the seeds
# all_rmse = np.mean(all_rmse, axis=3)
#
# figureIndex = 0
# step = np.linspace(0, len(all_rmse[0, 0, 0, :])-1, num=len(all_rmse[0, 0, 0, :]))
# for num_state, state in enumerate(all_state):
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for option in range(3):
#         plt.plot(step, all_rmse[option, num_state, 2, :], label='behaviory policy = %s' % (all_name[option]))
#     plt.title('number of state= %s' % (str(state)))
#     plt.xlabel('Episodes')
#     plt.ylabel('True_error')
#     plt.legend()
# plt.show()

# without average over the seeds
# figureIndex = 0
# step = np.linspace(0, len(all_rmse[0, 0, 0, 0, :])-1, num=len(all_rmse[0, 0, 0, 0, :]))
# for num_state, state in enumerate(all_state):
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for option in range(3):
#         plt.plot(step, all_rmse[option, num_state, 2, 1, :], label='behaviory policy = %s' % (all_name[option]))
#     plt.title('number of state= %s' % (str(state)))
#     plt.xlabel('Episodes')
#     plt.ylabel('True_error')
#     plt.legend()
# plt.show()

## ================================================3================================================

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='../data')
parser.add_argument('--alpha', type=float, default=0.0005)
parser.add_argument('--eta', type=float, default=0.05)
parser.add_argument('--lambda', type=float, default=0.99)
parser.add_argument('--decay', type=float, default=0.90)
parser.add_argument('--ISW', type=int, default=0)
parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.95)
parser.add_argument('--left_probability2', type=float, dest='left_probability2', default=0.75)
parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=2)
parser.add_argument('--num_runs', type=int, dest='num_runs', default=100)
parser.add_argument('--num_states', type=int, dest='num_states', default=5)
parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
parser.add_argument('--num_steps', type=int, dest='num_steps', default=50000)
parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=1000)
parser.add_argument('--num_change', type=int, dest='num_change', default=1000)  # every 5000 steps behavior policy add
parser.add_argument('--test_name', default='five_agents_tree_backup')
args = vars(parser.parse_args())
if 'num_steps' not in args:
    args['num_steps'] = args['num_states'] * 100

with open('{}/rmse_{}.npy'.format(args['directory'], args['test_name']), 'rb') as outfile:
    all_rmse = np.load(outfile)

# average all the seeds
mean_rmse = np.mean(all_rmse, axis=4)
std_rmse = np.std(all_rmse, axis=4)

"""
all_rmse = np.ones((len(all_name), len(all_alpha), len(all_decay), \
len(all_numchange), args['num_seeds'], args['num_steps'])) * np.inf"""

figureIndex = 0
plt.figure(figureIndex)
plt.subplot(131)
figureIndex += 1
step = np.linspace(0, len(mean_rmse[0, 0, 0, 0, :])-1, num=len(mean_rmse[0, 0, 0, 0, :]))
# for decayInd, decay in enumerate(all_decay):
plt.subplot(131)
for optionInd, name in enumerate(all_name):
    plt.plot(step, mean_rmse[optionInd, 2, 0, 2, :], label=name)
    # plt.fill_between(step, mean_rmse[optionInd, 2, decayInd, 0, :] - std_rmse[optionInd, 2, decayInd, 0, :], mean_rmse[optionInd, 2, decayInd, 0, :] + std_rmse[optionInd, 2, decayInd, 0, :])
plt.title('decay= %s, alpha= %s, numchange= %s' % (str(all_decay[0]), str(all_alpha[2]), str(all_numchange[0])))
plt.xlabel('Episodes')
plt.ylabel('True_error')
plt.legend()
plt.show()

