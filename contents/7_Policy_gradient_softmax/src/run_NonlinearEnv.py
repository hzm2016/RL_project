import gym
from RL_brain import *
from Tile_coding import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 4001  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 10000   # maximum time step in one episode
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time
MAX_EPISODE = 3000
OUTPUT_GRAPH = True
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.005    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped
N_F = env.observation_space.shape[0]
N_A = env.action_space.n

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

""""Tile coding"""
NumOfTilings = 50
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

actor = Actor(sess, n_features=tile.numTiles, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=tile.numTiles, n_action=N_A, lr=LR_C)

if OUTPUT_GRAPH:
    summary_writer = tf.summary.FileWriter("nonlinear_results/", sess.graph)
sess.run(tf.global_variables_initializer())


if __name__ == '__main__':

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:

            a, _ = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            track_r.append(r)

            td_error = critic.learn(s, r, s_, GAMMA)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                print("episode:", i_episode, "  reward:", int(running_reward))
                record = summary_pb2.Summary.Value(tag='reward', simple_value=running_reward)
                record_value = summary_pb2.Summary(value=[record])
                summary_writer.add_summary(record_value, i_episode)
                break