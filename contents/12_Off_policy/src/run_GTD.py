#!/usr/bin/env python -O
# -*- coding: ascii -*-
import argparse
import numpy as np
# import matplotlib.pyplot as plt
import os
import signal
import sys
from domains import Ringworld
from algorithms import GTD


def main(args):
    global siginfo_message

    all_rmse = np.ones((args['num_seeds'], args['num_steps'])) * np.inf
    all_lambda = np.copy(all_rmse)
    for option in range(3):
        for seed in range(args['num_seeds']):
            """build domain"""
            domain = Ringworld(
                args['num_states'],
                left_probability=args['left_probability'],
                random_generator=np.random.RandomState(seed))
            last_s, action, reward, gamma, s = domain.next(args['left_probability'])
            last_x = domain.state_to_features(last_s)
            x = domain.state_to_features(s)

            """build learners"""
            learner = GTD(last_x)
            for step in range(args['num_steps']):

                # compute the frequency for every 1000 steps
                if step % args['num_frequency'] == 0:
                    fre_behavior = np.ones((args['num_states'], args['num_actions']))

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
                        print('0', fre_behavior[s, action]/sum(fre_behavior[s, :]))
                    else:
                        rho = 0.95 / (fre_behavior[s, action]/sum(fre_behavior[s, :]))
                        print('1', fre_behavior[s, action]/sum(fre_behavior[s, :]))

                learner.update(reward, gamma, x, args['alpha'], args['eta'], args['lambda'], rho=rho)

                # record rmse and lambda :: compute return target policy
                all_rmse[seed, step] = domain.rmse(learner, left_probability=args['left_probability'])
                all_lambda[seed, step] = args['lambda']

                # move to next step
                if step < args['num_steps']//3:
                    last_s, action, reward, gamma, s = domain.next(args['left_probability'])
                else:
                    last_s, action, reward, gamma, s = domain.next(args['left_probability2'])

                last_x = domain.state_to_features(last_s)
                x = domain.state_to_features(s)

        with open('{}/rmse_{}_{}.npy'.format(args['directory'], args['test_name'], str(option)), 'wb') as outfile:
            np.save(outfile, all_rmse)
        with open('{}/lambda_{}_{}.npy'.format(args['directory'], args['test_name'], str(option)), 'wb') as outfile:
            np.save(outfile, all_lambda)


def multiple_hyperparameters(args):
    global siginfo_message

    all_state = np.array([5, 15, 25, 35, 45])
    all_frequency = np.array([1000, 2500, 5000])
    all_agent = np.array([0, 1, 2])
    all_rmse = np.ones((len(all_agent), len(all_state), len(all_frequency), args['num_seeds'], args['num_steps'])) * np.inf
    # all_lambda = np.copy(all_rmse)

    for option in range(len(all_agent)):
        for num_state in range(len(all_state)):
            for num_frequency in range(len(all_frequency)):
                for seed in range(args['num_seeds']):

                    """build domain"""
                    domain = Ringworld(
                        args['num_states'],
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
                            fre_behavior = np.ones((args['num_states'], args['num_actions']))

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

    # parse args and run
    # rmse_filename = '{}/rmse_change_12.npy'.format(args['directory'])
    # lambda_filename = '{}/lambda_change_12.npy'.format(args['directory'])
    # if not (os.path.exists(rmse_filename) and (os.path.exists(lambda_filename))):
    main(args)