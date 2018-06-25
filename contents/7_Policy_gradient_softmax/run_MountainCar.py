import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
from Tile_coding import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from LineartensorflowAC import *

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time
MAX_EPISODE = 3000
MAX_EP_STEPS = 5000   # maximum time step in one episode
OUTPUT_GRAPH = True

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

""""Tile coding"""
NumOfTilings = 10
MaxSize = 4096
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


sess = tf.Session()

DisAC = DiscreteActorCritic(sess, MaxSize, env.action_space.n, 0.99, 0.01, 0.001, 0.0001, 0.3, 0.3)

if OUTPUT_GRAPH:
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.global_variables_initializer())


for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:

        a = DisAC.choose_action(getValueFeature(s))

        s_, r, done, info = env.step(a)

        if done:
            r = -20

        track_r.append(r)

        DisAC.update(getValueFeature(s), r, getValueFeature(s_))

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            # record = summary_pb2.Summary.Value(tag='reward', simple_value=running_reward)
            # record_value = summary_pb2.Summary(value=[record])
            # summary_writer.add_summary(record_value, i_episode)
            break


# RL = PolicyGradient(
#     n_actions=env.action_space.n,
#     n_features=env.observation_space.shape[0],
#     learning_rate=0.02,
#     reward_decay=0.995,
#     # output_graph=True,
# )
#
#
# for i_episode in range(1000):
#
#     observation = env.reset()
#
#     while True:
#         if RENDER: env.render()
#
#         action = RL.choose_action(observation)
#
#         observation_, reward, done, info = env.step(action)     # reward = -1 in all cases
#
#         RL.store_transition(observation, action, reward)
#
#         if done:
#             # calculate running reward
#             ep_rs_sum = sum(RL.ep_rs)
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
#
#             print("episode:", i_episode, "  reward:", int(running_reward))
#
#             vt = RL.learn()  # train
#
#             if i_episode == 30:
#                 plt.plot(vt)  # plot the episode vt
#                 plt.xlabel('episode steps')
#                 plt.ylabel('normalized state-action value')
#                 plt.show()
#
#             break
#
#         observation = observation_
#
#
# LinearAC =
