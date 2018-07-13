#!/usr/bin/env python -O
# -*- coding: ascii -*-
import argparse
import numpy as np
import os
import signal
import sys
from domains import Ringworld


class GTD:

    def __init__(self, initial_x: np.ndarray):
        self.e = np.copy(initial_x)
        self.w = np.zeros(self.e.shape, dtype=float)
        self.h = np.zeros(self.e.shape, dtype=float)
        self.last_prediction = 0
        self.saved_auxiliary = 0

    def predict(self, x: np.ndarray) -> float:
        """Return the current prediction for a given set of features x."""
        return float(np.dot(self.w, x))

    def update(self,
               reward: float,
               gamma: float,
               x: np.ndarray,
               alpha: float,
               eta: float,
               lamda: float,
               rho: float=1,
               replacing: bool=False) -> float:
        delta = reward + gamma * self.predict(x) - self.last_prediction
        self.e *= rho
        self.w += alpha * (delta * self.e - gamma * (1 - lamda) * x * np.dot(self.e, self.h))
        self.h += alpha * eta * (delta * self.e - self.saved_auxiliary)
        self.e *= lamda * gamma
        self.e += x
        if replacing:
            self.e = np.clip(self.e, 0, 1)
        self.last_prediction = self.predict(x)
        self.saved_auxiliary = np.dot(self.h, x) * x
        return delta


def main(args):
    global siginfo_message

    all_state = np.array([5])
    all_frequency = np.array([1000])
    all_alpha = np.array([0.0005])  # [0.005, 0.0005, 1e-5]
    all_decay = np.array([1.0, 0.95, 0.90])  # [1.0, 0.95, 0.90]
    all_numchange = np.array([500])
    all_name = ['unknown', 'normal', 'fade_frequency', 'sliding_window', 'Retrace_lambda', 'Tree_backup']
    all_rmse = np.ones((len(all_name), len(all_alpha), len(all_decay), len(all_numchange), args['num_seeds'], args['num_steps'])) * np.inf
    for option, agent_name in enumerate(all_name):
        # for stateInd, state in enumerate(all_state):
        #     for frequencyInd, frequency in enumerate(all_frequency):
                for alphaInd, alpha in enumerate(all_alpha):
                    for decayInd, decay in enumerate(all_decay):
                        for changeInd, num_change in enumerate(all_numchange):
                            num_add = args['num_steps'] / num_change - 1
                            for seed in range(args['num_seeds']):

                                print("+++++++++++++++ %s +++++++++++++++" % agent_name)
                                """build domain"""
                                domain = Ringworld(
                                    args['num_states'],
                                    left_probability=args['left_probability'],
                                    random_generator=np.random.RandomState(seed))
                                last_s, action, reward, gamma, s = domain.next(args['left_probability'])
                                last_x = domain.state_to_features(last_s)
                                x = domain.state_to_features(s)
                                left_probability = args['left_probability']
                                fre_count = np.zeros((args['num_states'], args['num_actions']))

                                """build learners"""
                                learner = GTD(last_x)
                                for step in range(args['num_steps']):

                                    # # compute the frequency for every 1000 steps
                                    if step % args['num_frequency'] == 0:
                                        fre_behavior = np.zeros((args['num_states'], args['num_actions']))

                                    # compute the count
                                    fre_behavior[s, action] += 1

                                    # update the probability
                                    # for i in range(args['num_actions']):
                                    #     if i == action:
                                    #         fre_count[last_s, i] += 1
                                    #     else:
                                    #         fre_count[last_s, i] *= args['decay']

                                    # update the count
                                    for i in range(args['num_actions']):
                                        if i == action:
                                            fre_count[s, i] += 1
                                        else:
                                            fre_count[s, i] *= decay

                                    # update behavior policy
                                    # if step > 0 and step % args['num_change'] == 0:
                                    if step > 0 and step % num_change == 0:
                                        left_probability += (args['left_probability_end'] - args['left_probability'])/num_add

                                    # set message for siginfo
                                    siginfo_message = '[{0:3.2f}%] SEED: {1} of {2}, EPISODE: {3} of {4}, STEP: {5}'.format(
                                        100 * ((seed + step / args['num_steps']) / args['num_seeds']),
                                        seed + 1, args['num_seeds'], step + 1, args['num_steps'], step)

                                    """compute the importance sampling rho"""
                                    # compute in the same way
                                    if option == 0:
                                        # print('unknown')
                                        if action == domain.LEFT:
                                            rho = 0.05 / args['left_probability']
                                        else:
                                            rho = 0.95 / (1 - args['left_probability'])
                                    # compute with the changed probability
                                    elif option == 1:
                                        # print('right rho')
                                        if action == domain.LEFT:
                                            rho = 0.05 / left_probability
                                        else:
                                            rho = 0.95 / (1 - left_probability)
                                    # compute with the decay count
                                    elif option == 2:
                                        # print('decay')
                                        if action == domain.LEFT:
                                            rho = 0.05 / min(1, fre_count[s, action]/sum(fre_count[s, :]))
                                        else:
                                            rho = 0.95 / min(1, fre_count[s, action]/sum(fre_count[s, :]))
                                    # compute the count with a siding window
                                    elif option == 3:
                                        # print('count')
                                        if action == domain.LEFT:
                                            rho = 0.05 / min(1, fre_behavior[s, action]/sum(fre_behavior[s, :]))
                                            # print('0', fre_behavior[s, action]/sum(fre_behavior[s, :]))
                                        else:
                                            rho = 0.95 / min(1, fre_behavior[s, action]/sum(fre_behavior[s, :]))
                                            # print('1', fre_behavior[s, action]/sum(fre_behavior[s, :]))
                                    # compute the rho with Trace(lambda)
                                    elif option == 4:
                                        if action == domain.LEFT:
                                            rho = min(1, 0.05/left_probability)
                                        else:
                                            rho = min(1, 0.95/(1 - left_probability))
                                    # compute with tree backup
                                    else:
                                        # print('tree_backup')
                                        if action == domain.LEFT:
                                            rho = 0.05
                                        else:
                                            rho = 0.95

                                    learner.update(reward, gamma, x, alpha, args['eta'], args['lambda'], rho=rho)

                                    # record rmse and lambda :: compute return target policy
                                    all_rmse[option, alphaInd, decayInd, changeInd, seed, step] = domain.rmse(learner, left_probability=args['left_probability'])
                                    # all_rmse[option, seed, step] = domain.rmse(learner, left_probability=0.05)

                                    # move to next step :: with different behavior policy
                                    last_s, action, reward, gamma, s = domain.next(left_probability)

                                    last_x = domain.state_to_features(last_s)
                                    x = domain.state_to_features(s)

    with open('{}/rmse_{}.npy'.format(args['directory'], args['test_name']), 'wb') as outfile:
        np.save(outfile, all_rmse)


def multiple_hyperparameters(args):
    global siginfo_message

    all_state = np.array([5, 15, 25, 35, 45])
    all_frequency = np.array([1000, 2500, 5000])
    all_agent = np.array([0, 1, 2])
    all_rmse = np.ones((len(all_agent), len(all_state), len(all_frequency), args['num_seeds'], args['num_steps'])) * np.inf
    # all_lambda = np.copy(all_rmse)

    for option in range(len(all_agent)):
        for num_state, state in enumerate(all_state):
            for num_frequency, frequency in enumerate(all_frequency):
                for seed in range(args['num_seeds']):

                    """build domain"""
                    domain = Ringworld(
                        state,
                        left_probability=args['left_probability'],
                        random_generator=np.random.RandomState(seed))

                    last_s, action, reward, gamma, s = domain.next(args['left_probability'])
                    last_x = domain.state_to_features(last_s)
                    x = domain.state_to_features(s)

                    """build learners"""
                    learner = GTD(last_x)
                    for step in range(args['num_steps']):

                        # compute the frequency for every 1000 steps
                        if step % all_frequency[num_frequency] == 0:
                            fre_behavior = np.ones((state, args['num_actions']))

                        # compute the probability
                        fre_behavior[last_s, action] += 1

                        # set message for siginfo
                        siginfo_message = '[{0:3.2f}%] SEED: {1} of {2}, EPISODE: {3} of {4}, STEP: {5}'.format(
                            100 * ((seed + step / args['num_steps']) / args['num_seeds']),
                            seed + 1, args['num_seeds'], step + 1, args['num_steps'], step)

                        if option == 0:
                            # the change is unknown
                            if action == domain.LEFT:
                                rho = 0.05 / args['left_probability']
                            else:
                                rho = 0.95 / (1 - args['left_probability'])
                            # the change is known
                        elif option == 1:
                            if step < args['num_steps']//3:
                                if action == domain.LEFT:
                                    rho = 0.05 / args['left_probability']
                                else:
                                    rho = 0.95 / (1 - args['left_probability'])
                            else:
                                if action == domain.LEFT:
                                    rho = 0.05 / args['left_probability2']
                                else:
                                    rho = 0.95 / (1 - args['left_probability2'])
                        else:
                            if action == domain.LEFT:
                                rho = 0.05 / (fre_behavior[s, action]/sum(fre_behavior[s, :]))
                                # print('0', fre_behavior[s, action]/sum(fre_behavior[s, :]))
                            else:
                                rho = 0.95 / (fre_behavior[s, action]/sum(fre_behavior[s, :]))
                                # print('1', fre_behavior[s, action]/sum(fre_behavior[s, :]))

                        learner.update(reward, gamma, x, args['alpha'], args['eta'], args['lambda'], rho=rho)

                        # record rmse and lambda :: compute return target policy
                        all_rmse[option, num_state, num_frequency, seed, step] = domain.rmse(learner, left_probability=args['left_probability'])
                        # all_lambda[seed, step] = args['lambda']

                        # move to next step
                        if step < args['num_steps']//3:
                            last_s, action, reward, gamma, s = domain.next(args['left_probability'])
                        else:
                            last_s, action, reward, gamma, s = domain.next(args['left_probability2'])

                        last_x = domain.state_to_features(last_s)
                        x = domain.state_to_features(s)

                with open('{}/rmse_{}.npy'.format(args['directory'], args['test_name']), 'wb') as outfile:
                    np.save(outfile, all_rmse)
                # with open('{}/lambda_{}_{}.npy'.format(args['directory'], args['test_name'], str(option)), 'wb') as outfile:
                #     np.save(outfile, all_lambda)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='../data')
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--lambda', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--ISW', type=int, default=0)
    parser.add_argument('--left_probability', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--left_probability_end', type=float, dest='left_probability_end', default=0.95)
    parser.add_argument('--left_probability2', type=float, dest='left_probability2', default=0.75)
    parser.add_argument('--num_seeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--num_runs', type=int, dest='num_runs', default=1)
    parser.add_argument('--num_states', type=int, dest='num_states', default=5)
    parser.add_argument('--num_actions', type=int, dest='num_actions', default=2)
    parser.add_argument('--num_steps', type=int, dest='num_steps', default=50000)
    parser.add_argument('--num_frequency', type=int, dest='num_frequency', default=1000)
    parser.add_argument('--num_change', type=int, dest='num_change', default=1000)
    parser.add_argument('--test_name', default='Six_agents_trace_different_decay')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args


if __name__ == '__main__':
    # get command line arguments
    args = parse_args()

    # setup numpy
    np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

    # setup siginfo response system
    global siginfo_message
    siginfo_message = None
    if hasattr(signal, 'SIGINFO'):
        signal.signal(
            signal.SIGINFO,
            lambda signum, frame: sys.stderr.write('{}\n'.format(siginfo_message))
        )

    # multiple_hyperparameters(args)
    main(args)
