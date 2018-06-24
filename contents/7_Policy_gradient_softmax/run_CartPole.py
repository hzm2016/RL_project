import gym
from RL_brain import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import matplotlib.pyplot as plt
from Tile_coding import Tilecoder
from LinearActorCritic import DiscreteActorCritic

"""Superparameters"""
OUTPUT_GRAPH = True
MAX_EPISODE = 4000
DISPLAY_REWARD_THRESHOLD = 4001  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.00001    # learning rate for actor
LR_C = 0.0001     # learning rate for critic

env = gym.make('MountainCar-v0')
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
tile = Tilecoder(env, 10, 10)

LinearAC = DiscreteActorCritic(tile.numTiles, env.action_space.n)

for i_episode in range(3000):

    t = 0
    track_r = []
    observation = env.reset()

    action = LinearAC.start(tile.oneHotFeature(observation))

    while True:
        # if RENDER:
        #     env.render()

        observation_, reward, done, info = env.step(action)

        if done:
            reward = 10

        track_r.append(reward)

        action, delta = LinearAC.step(reward, 0.99, tile.oneHotFeature(observation), 0.1, 0.1/51, 0.1/51, 0.7, 0.7)

        observation = observation_

        if done or t > MAX_EP_STEPS:

            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            # if i_episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_



