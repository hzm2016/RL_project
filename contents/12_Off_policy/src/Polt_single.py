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

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='../data')
parser.add_argument('--alpha', type=float, default=0.0005)
parser.add_argument('--eta', type=float, default=0.05)
parser.add_argument('--lambda', type=float, default=0.99)
parser.add_argument('--ISW', type=int, default=0)
parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
parser.add_argument('--left_probability2', type=float, dest='left_probability2', default=0.75)
parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=20)
parser.add_argument('--num_states', type=int, dest='num_states', default=10)
parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
parser.add_argument('--num_steps', type=int, dest='num_steps', default=50000)
parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=3000)
parser.add_argument('--test_name', default='test')
args = vars(parser.parse_args())
if 'num_steps' not in args:
    args['num_steps'] = args['num_states'] * 100


all_state = np.array([5, 15, 25, 35, 45])
all_frequency = np.array([1000, 2500, 5000])
all_agent = np.array([0, 1, 2])
# all_rmse = np.ones((len(all_agent), len(all_state), len(all_frequency), args['num_seeds'], args['num_steps'])) * np.inf
# all_lambda = np.copy(all_rmse)

with open('rmse_{}.npy'.format(args['test_name']), 'r') as outfile:
    all_rmse = np.load(outfile)

step = np.linspace(0, len(all_rmse[0, 0, 0, :])-1, num=len(all_rmse[0, 0, 0, :]))
for num_state, state in enumerate(all_state):
    plt.figure()
    for option in range(len(all_agent)):
        plt.plot(step, all_rmse[option, num_state, 0, :], label='num_state = %s' (str(state)))
        plt.xlabel('Episodes')
        plt.ylabel('Episode_error')
        plt.legend()
        plt.show()


# for num in range(args['num_seeds']):
#     with open('{}/rmse_{}_{}.npy'.format(args['directory'], args['test_name'], '0'), 'rb') as outfile:
#         try:
#             espisode_error_0 = np.load(outfile)
#         except:
#             print('There is no file!')
#     with open('{}/rmse_{}_{}.npy'.format(args['directory'], args['test_name'], '1'), 'rb') as outfile:
#         try:
#             espisode_error_1 = np.load(outfile)
#         except:
#             print('There is no file!')
#     with open('{}/rmse_{}_{}.npy'.format(args['directory'], args['test_name'], '2'), 'rb') as outfile:
#         try:
#             espisode_error_2 = np.load(outfile)
#         except:
#             print('There is no file!')
#     plt.figure()
#     espisode_step = np.linspace(0, len(espisode_error_2[num]) - 1, num=len(espisode_error_2[num]))
#     plt.plot(espisode_step, espisode_error_0[num], label='unknown')
#     plt.plot(espisode_step, espisode_error_1[num], label='known')
#     # plt.plot(espisode_step, espisode_error_2[num], label='frequency')
#     plt.xlabel('Episodes')
#     plt.ylabel('Episode_error')
#     plt.legend()
#     plt.show()


# with open('../data/rmse_change_12.npy', 'rb') as f:
#     try:
#         espisode_error_0 = np.load(f)
#     except:
#         print('There is no file!')
#
# with open('../data/rmse_change_11.npy', 'rb') as f:
#     try:
#         espisode_error_1 = np.load(f)
#     except:
#         print('There is no file!')
#
# with open('../data/rmse_change_10.npy', 'rb') as f:
#     try:
#         espisode_error_2 = np.load(f)
#     except:
#         print('There is no file!')
#
# num = 1
# plt.figure()
# espisode_step = np.linspace(0, len(espisode_error_2[num]) - 1, num=len(espisode_error_2[num]))
# plt.plot(espisode_step, espisode_error_1[num], espisode_error_2[num], label='Espisode_1')
# plt.xlabel('episodes')
# plt.ylabel('reward of each episode')
# plt.legend()
# plt.show()
