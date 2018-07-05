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

"""Plot all the running results"""
with open('steps.bin', 'rb') as f:
    steps = pickle.load(f)
with open('rewards.bin', 'rb') as f:
    rewards =pickle.load(f)

# # average over episodes
# steps = np.mean(steps, axis=3)

# average over runs
# steps = np.mean(steps, axis=2)

# rewards = np.mean(rewards, axis=3)
# rewards = np.mean(rewards, axis=2)
print(steps[0, 0, 0, 0:10])

runs = 30
episodes = 3000
alphas = np.arange(1, 8) / 1000
lams = [0.99, 0.95, 0.5, 0]
eta = 0.0
gamma = 0.99
figureIndex = 0


plt.figure(figureIndex)
# print(rewards[1, 0, 0, :])
# step = np.linspace(0, len(steps[0, 0, 0, :])-1, num=len(steps[0, 0, 0, :]))

for lamInd, lam in enumerate(lams):
    plt.plot(alphas, rewards[lamInd, :], label='lambda = %s' % (str(lam)))
    print(rewards[lamInd, :])
plt.xlabel('alpha * # of tilings (8)')
plt.ylabel('averaged steps per episode')
# plt.ylim([180, 300])
plt.xlim([1, 100])
plt.legend()
plt.show()