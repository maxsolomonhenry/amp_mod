# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

# Modified by Max Henry, April 5, 2021 for use in his thesis; lord help him.

import numpy as np
import matplotlib.pylab as plt
import pickle
import os
from scipy.optimize import minimize, dual_annealing
import subprocess
import random


def kernel_optim_lbfgs_log(input_data,
                           target_data,
                           cost='correlation',
                           loss='exp_sum',
                           init_sig_mean=10.0,
                           init_sig_var=0.5,
                           dropout=0.0,
                           num_loops=10000,
                           method='L-BFGS-B',
                           log_foldername='./',
                           resume=None,
                           logging=False,
                           verbose=True,
                           allow_resume=True,
                           test_data=None):

    if (verbose):
        print(
            "* training gaussian kernels with cost '{}' and method '{}'".format(
                cost, method))

    num_model_features, num_stimuli = input_data.shape[0], input_data.shape[1]
    no_samples = num_stimuli * (num_stimuli - 1) / 2
    if resume != None:
        init_seed = resume['init_seed']
        init_sigmas = resume['sigmas']
        gradients = resume['gradients']
        correlations = resume['correlations']
        retrieved_loop = resume['retrieved_loop']
    else:
        init_sigmas = np.abs(
            init_sig_mean + init_sig_var * np.random.randn(num_model_features, 1))
        init_seed = init_sigmas
        gradients = np.zeros((num_model_features, 1))
        correlations = []  # = np.zeros((num_loops, ))
        retrieved_loop = 0

    num_model_features, num_stimuli = input_data.shape[0], input_data.shape[1]
    no_samples = num_stimuli * (num_stimuli - 1) / 2
    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    target_values = target_data[idx_triu]
    mean_target = np.mean(target_values)
    std_target = np.std(target_values)

    testing_correlations = []

    optimization_options = {'disp': None, 'maxls': 50, 'iprint': -1,
                            'gtol': 1e-36, 'eps': 1e-8, 'maxiter': num_loops,
                            'ftol': 1e-36}

    # Set bounds to 1, 1e15 for all model features.
    optimization_bounds = [
        (1.0 * i, 1e15 * i) for i in np.ones((num_model_features,))
    ]

    if logging:
        pickle.dump({
            'seed': init_seed,
            'cost': cost,
            'loss': loss,
            'method': method,
            'init_sig_mean': init_sig_mean,
            'init_sig_var': init_sig_var,
            'num_loops': num_loops,
            'log_foldername': log_foldername,
            'optimization_options': {'options': optimization_options, 'bounds': optimization_bounds}
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))

    def corr(x):
        kernel = np.zeros((num_stimuli, num_stimuli))

        # Make an index value for every entry in all the input data (all stims).
        idx = [i for i in range(len(input_data))]

        # Clip to the within the boundary values.
        x = np.clip(x, a_min=1.0, a_max=1e15)

        # If using dropout, randomly truncate index list.
        if dropout > 0.0:
            random.shuffle(idx)
            idx = idx[:int((1.0 - dropout) * len(idx))]

        for i in range(num_stimuli):
            for j in range(i + 1, num_stimuli):

                # Calculate the distance value, where x is sigma being trained.
                kernel[i, j] = -np.sum(
                    np.power(
                        np.divide(input_data[idx, i] - input_data[idx, j],
                                  (x[idx] + np.finfo(float).eps)), 2))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_values - mean_target))
        Jd = no_samples * std_target * std_kernel
        return Jn / Jd

    def grad_corr(sigmas):
        idx = [i for i in range(len(input_data))]
        if dropout > 0.0:
            random.shuffle(idx)
            idx = idx[:int((1.0 - dropout) * len(idx))]
        ndims = len(idx)
        kernel = np.zeros((num_stimuli, num_stimuli))
        dkernel = np.zeros((num_stimuli, num_stimuli, ndims))
        for i in range(num_stimuli):
            for j in range(i + 1, num_stimuli):
                kernel[i, j] = -np.sum(
                    np.power(
                        np.divide(input_data[idx, i] - input_data[idx, j],
                                  (sigmas[idx] + np.finfo(float).eps)), 2))
                dkernel[i, j, :] = 2 * np.power(
                    (input_data[idx, i] - input_data[idx, j]), 2) / (
                                           np.power(sigmas[idx], 3) + np.finfo(
                                           float).eps)
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_values - mean_target))
        Jd = no_samples * std_target * std_kernel

        for k in range(ndims):
            tmp = dkernel[:, :, k][idx_triu]
            dJn = np.sum(tmp * (target_values - mean_target))
            dJd = std_target / (std_kernel + np.finfo(float).eps) * np.sum(
                tmp * (kernel_v - mean_kernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (
                        np.power(Jd, 2) + np.finfo(float).eps)
        return gradients

    def print_corr(xk):
        kernel = np.zeros((num_stimuli, num_stimuli))
        for i in range(num_stimuli):
            for j in range(i + 1, num_stimuli):
                kernel[i, j] = np.exp(-np.sum(
                    np.power(
                        np.divide(input_data[:, i] - input_data[:, j],
                                  (xk + np.finfo(float).eps)), 2)))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_values - mean_target))
        Jd = no_samples * std_target * std_kernel

        if not os.path.isfile(os.path.join(log_foldername, 'tmp.pkl')):
            loop_cpt = 1
            pickle.dump({'loop': loop_cpt, 'correlation': [Jn / Jd]},
                        open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))
            correlations = [Jn / Jd]
            pickle.dump({
                'sigmas': xk,
                'kernel': kernel,
                'Jn': Jn,
                'Jd': Jd,
                'correlations': correlations,
            }, open(os.path.join(log_foldername,
                                 'optim_process_l={}.pkl'.format(loop_cpt)),
                    'wb'))
        else:
            last_loop = pickle.load(
                open(os.path.join(log_foldername, 'tmp.pkl'), 'rb'))
            loop_cpt = last_loop['loop'] + 1
            correlations = last_loop['correlation']
            correlations.append(Jn / Jd)

            if test_data != None:
                # testing data
                test_input = test_data[0]
                test_target = test_data[1]
                mean_target_test = np.mean(test_target)
                std_target_test = np.std(test_target)
                distances = np.zeros((input_data.shape[1], 1))
                for i in range(len(distances)):
                    distances[i, 0] = -np.sum(np.power(
                        np.divide(test_input - input_data[:, i],
                                  (xk + np.finfo(float).eps)), 2))
                mean_distances = np.mean(distances)
                stddev_distances = np.std(distances)
                Jn_ = np.sum(np.multiply(distances - mean_distances,
                                         test_target - mean_target_test))
                Jd_ = std_target_test * stddev_distances * (num_stimuli - 1)
                testing_correlations.append(Jn_ / Jd_)
            else:
                testing_correlations.append(0.0)

            monitoring_step = 25
            if (loop_cpt % monitoring_step == 0):
                print('  |_ loop={} J={:.6f} {:.6f}'.format(loop_cpt, Jn / Jd,
                                                            testing_correlations[
                                                                -1]))
                pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations
                }, open(os.path.join(log_foldername,
                                     'optim_process_l={}.pkl'.format(loop_cpt)),
                        'wb'))
                # plt.figure(figsize=(10,10))
                # plt.subplot(1,2,1)
                # plt.plot(xk)
                # plt.subplot(1,2,2)
                # plt.plot(correlations)
                # plt.plot(testing_correlations)
                # plt.savefig('log_sig_corr_lbfgs.pdf')
            pickle.dump(
                {'loop': loop_cpt, 'correlation': correlations, 'sigmas': xk},
                open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))

    res = minimize(corr, init_sigmas, args=(), method=method, jac=grad_corr,
                   callback=print_corr, options=optimization_options,
                   bounds=optimization_bounds)
    last_loop = pickle.load(open(os.path.join(log_foldername, 'tmp.pkl'), 'rb'))
    sigmas_ = last_loop['sigmas']
    correlations = last_loop['correlation']
    subprocess.run(["rm", os.path.join(log_foldername, 'tmp.pkl')])
    return correlations, sigmas_