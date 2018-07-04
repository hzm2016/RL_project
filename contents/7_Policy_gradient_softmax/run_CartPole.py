"""Linear Actor Critic"""
import gym
from Tile_coding import *
from LinearActorCritic import *
import numpy as np
import pickle

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 4001  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 10000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.005    # learning rate for actor
LR_C = 0.01     # learning rate for critic
EPSILON = 0
load = False

env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
# env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

print("Environments information:")
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

""""Tile coding"""
NumOfTilings = 10
MaxSize = 2048
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


# NPG = PolicyGradient(
#     n_actions=env.action_space.n,
#     n_features=MaxSize,
#     learning_rate=0.001,
#     reward_decay=0.99,
#     output_graph=False,
# )
# for i_episode in range(3000):
#
#     observation = env.reset()
#     t = 0
#
#     while True:
#         # if RENDER:
#         #     env.render()
#
#         action = NPG.choose_action(getValueFeature(observation))
#
#         observation_, reward, done, info = env.step(action)
#
#         NPG.store_transition(getValueFeature(observation), action, reward)
#
#         if done or t >= MAX_EP_STEPS:
#             ep_rs_sum = sum(NPG.ep_rs)
#
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
#             print("episode:", i_episode, "  reward:", int(running_reward))
#
#             vt = NPG.learn()
#
#             # if i_episode == 0:
#             #     plt.plot(vt)    # plot the episode vt
#             #     plt.xlabel('episode steps')
#             #     plt.ylabel('normalized state-action value')
#             #     plt.show()
#             break
#
#         t += 1
#
#         observation = observation_

# LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, 0.99, 0., 0.001, 0.0001, 0.3, 0.3)
#
# for i_espisode in range(3000):
#
#     t = 0.
#     track_r = []
#     observation = env.reset()
#     action = LinearAC.start(getValueFeature(observation))
#     while True:
#
#         # if RENDER:
#         #     env.render()
#
#         observation_, reward, done, info = env.step(action)
#
#         # if done:
#         #     reward = 10
#
#         track_r.append(reward)
#
#         action, delta = LinearAC.step(reward, getValueFeature(observation))
#
#         # print('delat', delta)
#
#         observation = observation_
#
#         t += 1
#
#         if done or t > MAX_EP_STEPS:
#
#             ep_rs_sum = sum(track_r)
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             # if running_reward > DISPLAY_REWARD_THRESHOLD:
#             #     RENDER = True     # rendering
#             print("episode:", i_espisode,  "reward:", int(running_reward))
#
#             break

def play(LinearAC, agent):

    if agent == 'Allactions':
        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            action_ = LinearAC.choose_action(getValueFeature(observation_))
            track_r.append(reward)
            feature = []
            for i in range(env.action_space.n):
                feature.append(getQvalueFeature(observation, i))
            action, delta = LinearAC.step(reward, getValueFeature(observation), \
                                          getQvalueFeature(observation, action), \
                                          getQvalueFeature(observation_, action_),
                                          feature)
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:

                ep_rs_sum = sum(track_r)
                running_reward = ep_rs_sum
                return t, int(running_reward)

    elif agent == 'AdvantageActorCritic':

        t = 0
        track_r = []
        observation = env.reset()
        action = LinearAC.start(getValueFeature(observation))
        while True:

            observation_, reward, done, info = env.step(action)
            action_ = LinearAC.choose_action(getValueFeature(observation_))
            track_r.append(reward)
            feature = []
            for i in range(env.action_space.n):
                feature.append(getQvalueFeature(observation, i))
            action, delta = LinearAC.step(reward, getValueFeature(observation), \
                                          getQvalueFeature(observation, action), \
                                          getQvalueFeature(observation_, action_),
                                          feature)
            observation = observation_
            t += 1
            if done or t > MAX_EP_STEPS:

                ep_rs_sum = sum(track_r)
                running_reward = ep_rs_sum
                return t, int(running_reward)
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

                ep_rs_sum = sum(track_r)
                running_reward = ep_rs_sum
                return t, int(running_reward)
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

                ep_rs_sum = sum(track_r)
                running_reward = ep_rs_sum
                return t, int(running_reward)


if __name__ == '__main__':

    runs = 30
    episodes = 3000
    alphas = np.arange(1, 8) / 1000
    lams = [0.99, 0.95, 0.5, 0]
    eta = 0.0
    gamma = 0.99
    agent = 'Reinforce'

    if load:
        with open('steps.bin', 'rb') as f:
            steps = pickle.load(f)
        with open('rewards.bin', 'rb') as s:
            rewards = pickle.load(s)
    else:

        steps = np.zeros((len(lams), len(alphas), runs, episodes))
        rewards = np.zeros((len(lams), len(alphas), runs, episodes))
        for lamInd, lam in enumerate(lams):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
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
                    for ep in range(episodes):
                        step, reward = play(LinearAC, agent)
                        steps[lamInd, alphaInd, run, ep] = step
                        if 'running_reward' not in globals():
                            running_reward = reward
                        else:
                            running_reward = running_reward * 0.99 + reward * 0.01
                        break
                        rewards[lamInd, alphaInd, run, ep] = running_reward
                        print('lambda %f, alpha %f, run %d, episode %d, steps %d' %
                              (lam, alpha, run, ep, step))
        with open('steps.bin', 'wb') as f:
            pickle.dump(steps, f)
        with open('rewards.bin', 'wb') as s:
            pickle.dump(rewards, s)

    # # average over episodes
    # steps = np.mean(steps, axis=3)
    #
    # # average over runs
    # steps = np.mean(steps, axis=2)

    # global figureIndex
    # plt.figure(figureIndex)
    # figureIndex += 1
    # for lamInd, lam in enumerate(lams):
    #     plt.plot(alphas, steps[lamInd, :], label='lambda = %s' % (str(lam)))
    # plt.xlabel('alpha * # of tilings (8)')
    # plt.ylabel('averaged steps per episode')
    # plt.ylim([180, 300])
    # plt.legend()


