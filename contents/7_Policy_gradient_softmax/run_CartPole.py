import gym
from RL_brain import *
from Tile_coding import *
from LinearActorCritic import DiscreteActorCritic
import numpy as np
import pickle

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 4000
DISPLAY_REWARD_THRESHOLD = 4001  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 5000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.005    # learning rate for actor
LR_C = 0.01     # learning rate for critic
EPSILON = 0
load = False

env = gym.make('MountainCar-v0')
env._max_episode_steps = 5000
# env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
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
#     n_features=env.observation_space.shape[0],
#     learning_rate=0.02,
#     reward_decay=0.99,
#     # output_graph=True,
# )
#
#
# for i_episode in range(3000):
#
#     observation = env.reset()
#
#     while True:
#         # if RENDER:
#         #     env.render()
#
#         action = NPG.choose_action(observation)
#
#         observation_, reward, done, info = env.step(action)
#
#         NPG.store_transition(observation, action, reward)
#
#         if done:
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
#         observation = observation_

# sess = tf.Session()
# type = 'AA'
#
# if type == 'AA':
#     actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
#     critic = ValueFunction(sess, n_features=N_F, n_actions=N_A, gamma=GAMMA, lr=LR_C)
# elif type == 'SA':
#     actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
#     critic = Critic(sess, n_features=N_F, lr=LR_C)
#
# sess.run(tf.global_variables_initializer())
#
# if OUTPUT_GRAPH:
#     summary_writer = tf.summary.FileWriter("logs/"+type, sess.graph)
#
# for i_episode in range(MAX_EPISODE):
#     s = env.reset()
#     t = 0
#     track_r = []
#     a, a_vector = actor.choose_action(s)
#     # print('a', a, 'aa', a_vector)
#
#     while True:
#
#         if RENDER:
#             env.render()
#
#         a, a_vector = actor.choose_action(s)
#         # print('a_vector', a_vector)
#
#         s_, r, done, info = env.step(a)
#
#         if done:
#             r = -20
#
#         track_r.append(r)
#
#         if type == 'AA':
#             aa_action_td_error = critic.learn(s, r, s_, int(done), a_vector)
#             # print('aa_q', aa_action_td_error[0])
#             a, b = actor.aa_learn(s, a, aa_action_td_error)
#             # print('prob', b)
#         else:
#             td_error = critic.learn(s, r, s_, GAMMA)  # gradient = grad[r + gamma * V(s_) - V(s)]
#             actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error
#
#         s = s_
#         t += 1
#
#         if done or t >= MAX_EP_STEPS:
#             ep_rs_sum = sum(track_r)
#
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
#
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
#             print("episode:", i_episode, "  reward:", int(running_reward))
#             record = summary_pb2.Summary.Value(tag='reward', simple_value=running_reward)
#             record_value = summary_pb2.Summary(value=[record])
#             summary_writer.add_summary(record_value, i_episode)
#             break


# LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, 0.99, 0., 0.001, 0.0001, 0.3, 0.3)


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


def play(LinearAC):

    t = 0
    track_r = []
    observation = env.reset()
    action = LinearAC.start(getValueFeature(observation))
    while True:

        # if RENDER:
        #     env.render()

        observation_, reward, done, info = env.step(action)

        # if done:
        #     reward = 10

        track_r.append(reward)

        action, delta = LinearAC.step(reward, getValueFeature(observation))

        # print('delat', delta)

        observation = observation_

        t += 1

        if done or t > MAX_EP_STEPS:

            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True     # rendering
            print("steps:", t, "  reward:", int(running_reward))

            break
    return t, int(running_reward)


if __name__ == '__main__':

    runs = 30
    episodes = 3000
    alphas = np.arange(1, 8) / 10000
    lams = [0.99, 0.95, 0.5, 0]
    eta = 0.0
    gamma = 0.9

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
                    LinearAC = DiscreteActorCritic(MaxSize, env.action_space.n, gamma, eta, alpha*10, alpha, lam, lam)
                    for ep in range(episodes):
                        step, reward = play(LinearAC)
                        steps[lamInd, alphaInd, run, ep] = step
                        rewards[lamInd, alphaInd, run, ep] = reward
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


