#!/usr/bin/env python -O
# -*- coding: ascii -*-

import argparse
import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import numpy as np
import os
import scipy.stats as st

FONTSIZE = 14


def main(args):

    # load values
    with open(args['alpha_values'], 'r') as infile:
        alpha_values = np.array([line.strip() for line in infile])
    with open(args['eta_values'], 'r') as infile:
        eta_values = np.array([line.strip() for line in infile])
    with open(args['lambda_values'], 'r') as infile:
        lambda_values = np.array([line.strip() for line in infile])

    # load data
    GTD_rmse = np.ones((len(alpha_values), len(eta_values), len(lambda_values),
                        args['num_seeds'], args['num_steps'])) * np.inf
    GTD_lambda = np.copy(GTD_rmse)
    LGGTD_rmse = np.ones((len(alpha_values), len(eta_values),
                          args['num_seeds'], args['num_steps'])) * np.inf
    LGGTD_lambda = np.copy(LGGTD_rmse)
    DLGGTD_rmse = np.ones((len(alpha_values), len(eta_values),
                           args['num_seeds'], args['num_steps'])) * np.inf
    DLGGTD_lambda = np.copy(DLGGTD_rmse)
    for alpha_index, alpha_value in enumerate(alpha_values):
        for eta_index, eta_value in enumerate(eta_values):

            # load GTD data
            for lambda_index, lambda_value in enumerate(lambda_values):
                directory = '{}/{}/{}/GTD/{}/{}/{}'.format(
                    args['directory'], args['left_probability'], args['num_states'],
                    alpha_value, eta_value, lambda_value)
                all_rmse = None
                all_lambda = None
                try:
                    with open('{}/rmse.npy'.format(directory), 'rb') as infile:
                        all_rmse = np.load(infile)
                except FileNotFoundError:
                    pass
                try:
                    with open('{}/lambda.npy'.format(directory),
                              'rb') as infile:
                        all_lambda = np.load(infile)
                except FileNotFoundError:
                    pass
                if ((all_rmse is not None) and
                        (all_lambda is not None) and
                        (np.nan not in all_rmse) and
                        (np.nan not in all_lambda)):
                    np.copyto(GTD_rmse[alpha_index, eta_index, lambda_index, ...], all_rmse)
                    np.copyto(GTD_lambda[alpha_index, eta_index, lambda_index, ...], all_lambda)

            # load LGGTD data
            directory = '{}/{}/{}/LGGTD/{}/{}'.format(
                args['directory'], args['left_probability'], args['num_states'],
                alpha_value, eta_value)
            all_rmse = None
            all_lambda = None
            try:
                with open('{}/rmse.npy'.format(directory), 'rb') as infile:
                    all_rmse = np.load(infile)
            except FileNotFoundError:
                pass
            try:
                with open('{}/lambda.npy'.format(directory), 'rb') as infile:
                    all_lambda = np.load(infile)
            except FileNotFoundError:
                pass
            if ((all_rmse is not None) and
                    (all_lambda is not None) and
                    (np.nan not in all_rmse) and
                    (np.nan not in all_lambda)):
                np.copyto(LGGTD_rmse[alpha_index, eta_index, ...], all_rmse)
                np.copyto(LGGTD_lambda[alpha_index, eta_index, ...], all_lambda)

            # load DLGGTD data
            directory = '{}/{}/{}/DLGGTD/{}/{}'.format(
                args['directory'], args['left_probability'], args['num_states'],
                alpha_value, eta_value)
            all_rmse = None
            all_lambda = None
            try:
                with open('{}/rmse.npy'.format(directory), 'rb') as infile:
                    all_rmse = np.load(infile)
            except FileNotFoundError:
                pass
            try:
                with open('{}/lambda.npy'.format(directory),
                          'rb') as infile:
                    all_lambda = np.load(infile)
            except FileNotFoundError:
                pass
            if ((all_rmse is not None) and
                    (all_lambda is not None) and
                    (np.nan not in all_rmse) and
                    (np.nan not in all_lambda)):
                np.copyto(DLGGTD_rmse[alpha_index, eta_index, ...], all_rmse)
                np.copyto(DLGGTD_lambda[alpha_index, eta_index, ...], all_lambda)

    # derive stats from data
    GTD_mean_rmse = np.nanmean(GTD_rmse, axis=3)
    GTD_sem_rmse = st.sem(GTD_rmse, axis=3, nan_policy='omit')
    GTD_mean_lambda = np.nanmean(GTD_lambda, axis=3)
    GTD_sem_lambda = st.sem(GTD_lambda, axis=3, nan_policy='omit')
    LGGTD_mean_rmse = np.nanmean(LGGTD_rmse, axis=2)
    LGGTD_sem_rmse = st.sem(LGGTD_rmse, axis=2, nan_policy='omit')
    LGGTD_mean_lambda = np.nanmean(LGGTD_lambda, axis=2)
    LGGTD_sem_lambda = st.sem(LGGTD_lambda, axis=2, nan_policy='omit')
    DLGGTD_mean_rmse = np.nanmean(DLGGTD_rmse, axis=2)
    DLGGTD_sem_rmse = st.sem(DLGGTD_rmse, axis=2, nan_policy='omit')
    DLGGTD_mean_lambda = np.nanmean(DLGGTD_lambda, axis=2)
    DLGGTD_sem_lambda = st.sem(DLGGTD_lambda, axis=2, nan_policy='omit')

    # FROM HERE ON NEED GTD(0) AND GTD(1) #########################################################

    # filter to best hyperparamater combinations
    GTD_zero_best_auc = np.inf
    GTD_zero_best_alpha_index = GTD_zero_best_eta_index = np.nan
    GTD_one_best_auc = np.inf
    GTD_one_best_alpha_index = GTD_one_best_eta_index = np.nan
    LGGTD_best_auc = np.inf
    LGGTD_best_alpha_index = LGGTD_best_eta_index = np.nan
    DLGGTD_best_auc = np.inf
    DLGGTD_best_alpha_index = DLGGTD_best_eta_index = np.nan
    for alpha_index in range(len(alpha_values)):
        for eta_index in range(len(eta_values)):

            # for GTD(0)
            auc = np.sum(GTD_mean_rmse[alpha_index, eta_index, 0, :])
            if auc < GTD_zero_best_auc:
                GTD_zero_best_auc = auc
                GTD_zero_best_alpha_index = alpha_index
                GTD_zero_best_eta_index = eta_index

            # for GTD(1)
            auc = np.sum(GTD_mean_rmse[alpha_index, eta_index, -1, :])
            if auc < GTD_one_best_auc:
                GTD_one_best_auc = auc
                GTD_one_best_alpha_index = alpha_index
                GTD_one_best_eta_index = eta_index

            # for LGGTD
            auc = np.sum(LGGTD_mean_rmse[alpha_index, eta_index, :])
            if auc < LGGTD_best_auc:
                LGGTD_best_auc = auc
                LGGTD_best_alpha_index = alpha_index
                LGGTD_best_eta_index = eta_index

            # for DLGGTD
            auc = np.sum(DLGGTD_mean_rmse[alpha_index, eta_index, :])
            if auc < DLGGTD_best_auc:
                DLGGTD_best_auc = auc
                DLGGTD_best_alpha_index = alpha_index
                DLGGTD_best_eta_index = eta_index

    # setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), dpi=300)
    c_map = plt.get_cmap('viridis')

    # draw lines for GTD(0)
    ax1.errorbar(
        np.arange(args['num_steps']) + 1,
        GTD_mean_rmse[GTD_zero_best_alpha_index, GTD_zero_best_eta_index, 0, :],
        yerr=GTD_sem_rmse[GTD_zero_best_alpha_index, GTD_zero_best_eta_index, 0, :],
        color=c_map(float(0)),
        label='GTD(0)')
    ax2.errorbar(
        np.arange(args['num_steps']) + 1,
        GTD_mean_lambda[GTD_zero_best_alpha_index, GTD_zero_best_eta_index, 0, :],
        yerr=GTD_sem_lambda[GTD_zero_best_alpha_index, GTD_zero_best_eta_index, 0, :],
        color=c_map(float(0)),
        label='GTD(0)')

    # draw lines for GTD(1)
    ax1.errorbar(
        np.arange(args['num_steps']) + 1,
        GTD_mean_rmse[GTD_one_best_alpha_index, GTD_one_best_eta_index, -1, :],
        yerr=GTD_sem_rmse[GTD_one_best_alpha_index, GTD_one_best_eta_index, -1, :],
        color=c_map(float(1 / 3)),
        label='GTD(1)')
    ax2.errorbar(
        np.arange(args['num_steps']) + 1,
        GTD_mean_lambda[GTD_one_best_alpha_index, GTD_one_best_eta_index, -1, :],
        yerr=GTD_sem_lambda[GTD_one_best_alpha_index, GTD_one_best_eta_index, -1, :],
        color=c_map(float(1 / 3)),
        label='GTD(1)')

    # draw lines for LGGTD
    ax1.errorbar(
        np.arange(args['num_steps']) + 1,
        LGGTD_mean_rmse[LGGTD_best_alpha_index, LGGTD_best_eta_index, :],
        yerr=LGGTD_sem_rmse[LGGTD_best_alpha_index, LGGTD_best_eta_index, :],
        color=c_map(float(2 / 3)),
        label='LGGTD')
    ax2.errorbar(
        np.arange(args['num_steps']) + 1,
        LGGTD_mean_lambda[LGGTD_best_alpha_index, LGGTD_best_eta_index, :],
        yerr=LGGTD_sem_lambda[LGGTD_best_alpha_index, LGGTD_best_eta_index, :],
        color=c_map(float(2 / 3)),
        label='LGGTD')

    # draw lines for DLGGTD
    ax1.errorbar(
        np.arange(args['num_steps']) + 1,
        DLGGTD_mean_rmse[DLGGTD_best_alpha_index, DLGGTD_best_eta_index, :],
        yerr=DLGGTD_sem_rmse[DLGGTD_best_alpha_index, DLGGTD_best_eta_index, :],
        color=c_map(float(1)),
        label='DLGGTD')
    ax2.errorbar(
        np.arange(args['num_steps']) + 1,
        DLGGTD_mean_lambda[DLGGTD_best_alpha_index, DLGGTD_best_eta_index, :],
        yerr=DLGGTD_sem_lambda[DLGGTD_best_alpha_index, DLGGTD_best_eta_index, :],
        color=c_map(float(1)),
        label='DLGGTD')

    # tidy up plot
    ax1.set_xscale('log')
    ax1.set_ylabel('RMSE', fontsize=FONTSIZE, labelpad=10)
    ax1.set_yscale('log')
    ax2.set_xlabel('Timestep', fontsize=FONTSIZE, labelpad=10)
    ax2.set_ylabel('Lambda', fontsize=FONTSIZE, labelpad=10)
    ax2.set_yticks(np.array([0.2 * i for i in range(6)]))
    fig.suptitle('{}-State Ringworld with mu(s, left) = {}'.format(
        args['num_states'], args['left_probability']),
        fontsize=FONTSIZE)
    fig.subplots_adjust(hspace=0.25)
    ax1.legend(loc='best', frameon=False)

    # save figure
    fig.savefig(args['outfile'], bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, '')
    parser.add_argument('--directory', type=str)
    parser.add_argument('--num_seeds', type=int)
    parser.add_argument('--num_states', type=int)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--leftprob', type=float, dest='left_probability', default=0.05)
    parser.add_argument('--alpha-values', type=str, default='alpha_values.txt')
    parser.add_argument('--eta-values', type=str, default='eta_values.txt')
    parser.add_argument('--lambda-values', type=str, default='lambda_values.txt')
    return vars(parser.parse_args())


if __name__ == '__main__':
    # get command line arguments
    args = parse_args()

    # parse args and run
    if not os.path.exists(args['outfile']):
        main(args)
