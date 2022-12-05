import argparse
import os
import sys
from itertools import product

import model.multi_cause_influence as causal
import model.network_model as nm
import model.pmf as pmf
import numpy as np
from citation.process_dataset import (
    make_multi_covariate_simulation,
    process_dataset_multi_covariate,
)
from scipy.stats import poisson
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score


def calculate_ppc_dpf(heldout_idxs, obs_y, theta, beta):
    r"""Compute the predictive probability of the heldout items.
    Should be applied separately for network confounder and
    item confounder models -- the likelihood of each are
    different. This func is for item substitute -- the dPF.

    If a model reproduces the data well, then the probability
    of observing the actual heldout items should be similar
    to the probability of observing the replicated items.

    Under dPF, given user factors theta and item factors beta,
    shape (N,T,K) and (M,T,K) resp., the probability of
    observing a publication by user i on item m at t
    is Poisson(\sum_k exp(theta_{ik}^t) * exp(beta_{mk}^t)),
    so can calc rates given theta and beta, then calc logpmf.

    :param heldout_idxs: indices of heldout items
    :type heldout_idxs: tuple(np.ndarray, np.ndarray, np.ndarray)
    :param obs_y: actual observed user-item counts, shape (N,M,T)
    :type obs_y: np.ndarray
    :param theta: user factors inferred by dPF, shape (N,T,K)
    :type theta: np.ndarray
    :param beta: item factors inferred by dPF, shape (M,T,K)
    :type beta: np.ndarray
    :return: logll_heldout, logll_replicated
    :rtype: tuple(float, float)
    """
    expbeta, exptheta = np.exp(beta), np.exp(theta)
    rates = np.einsum("itk,mtk->imt", exptheta, expbeta)
    rates = rates[heldout_idxs]
    replicated = poisson.rvs(rates)
    heldout = obs_y[heldout_idxs]
    logll_heldout = poisson.logpmf(heldout, rates).sum()
    logll_replicated = poisson.logpmf(replicated, rates).sum()
    return logll_heldout, logll_replicated


def calculate_ppc_dsbmm(
    heldout_idxs,
    obs_a,
    node_probs: np.ndarray,
    block_probs: np.ndarray,
    deg_corr=True,
    directed=False,
):
    """Compute the predictive probability of the heldout edges.
    This func is for user-item substitute -- the DSBMM.

    See thesis for prob of edges under DSBMM.

    :param heldout_idxs: indices of heldout edges at each timestep
    :type heldout_idxs: list[tuple(np.ndarray, np.ndarray)]
    :param obs_a: actual observed adjacencies, length T list of (N,N)
                  sparse csr_arrays
    :type obs_a: list[sparse.csr_array]
    :param node_probs: node group marginal probs, shape (N,T,Q)
    :type node_probs: np.ndarray
    :param block_probs: block probs, shape (Q,Q,T)
    :type block_probs: np.ndarray
    :param deg_corr: use degree corrected version, defaults to True
    :type deg_corr: bool, optional
    :param directed: use directed version, defaults to False
    :type directed: bool, optional
    :return: logll_heldout, logll_replicated
    :rtype: tuple(float, float)
    """

    # replicated = pass
    heldout = [obs[idxs] for obs, idxs in zip(obs_a, heldout_idxs)]
    # logll_heldout = poisson.logpmf(heldout, rates).sum()
    # logll_replicated = poisson.logpmf(replicated, rates).sum()
    # return logll_heldout, logll_replicated


def evaluate_random_subset(items, array, z, w, metric="logll"):
    users = np.arange(array.shape[0])
    expected = z.dot(w.T)[users, items]
    truth = array[users, items]
    if metric == "auc":
        return roc_auc_score(truth, expected)
    else:
        return poisson.logpmf(truth, expected).sum()


def mask_items(N, M):
    items = np.arange(M)
    random_items = np.random.choice(items, size=N)
    return random_items


def main():
    num_exps = 20
    Ks = [3, 5, 8, 10]
    mixture_pr = 0.5
    noise = 10.0
    conf_strength = 50.0

    a_score = np.zeros((num_exps, len(Ks)))
    x_score = np.zeros((num_exps, len(Ks)))
    x_auc = np.zeros((num_exps, len(Ks)))
    for e_idx in range(num_exps):
        print("Working on experiment", e_idx)

        A, users, user_one_hots, item_one_hots, Beta = process_dataset_multi_covariate(
            datapath=datadir,
            sample_size=3000,
            num_items=1000,
            influence_shp=0.005,
            covar_2="random",
            covar_2_num_cats=5,
            use_fixed_graph=False,
        )
        Y, Y_past, Z, Gamma, Alpha, W = make_multi_covariate_simulation(
            A,
            user_one_hots,
            item_one_hots,
            Beta,
            noise=noise,
            confounding_strength=conf_strength,
            mixture_prob=mixture_pr,
        )

        N = Y_past.shape[0]
        M = Y_past.shape[1]
        masked_friends = mask_items(N, N)
        past_masked_items = mask_items(N, M)

        Y_past_train = Y_past.copy()
        A_train = A.copy()

        users = np.arange(N)
        A_train[users, masked_friends] = 0
        Y_past_train[users, past_masked_items] = 0

        for k_idx, K in enumerate(Ks):
            network_model = nm.NetworkPoissonMF(n_components=K, verbose=False)
            network_model.fit(A_train)
            Z_hat = network_model.Et
            pmf_model = pmf.PoissonMF(n_components=K, verbose=False)
            pmf_model.fit(Y_past_train)
            W_hat = pmf_model.Eb.T
            Theta_hat = pmf_model.Et

            replicates = 100
            A_predictive_score = 0.0
            YP_pred_score = 0.0
            for _ in range(replicates):
                A_logll_heldout, A_logll_replicated = calculate_ppc(
                    masked_friends, A, Z_hat, Z_hat
                )
                Y_logll_heldout, Y_logll_replicated = calculate_ppc(
                    past_masked_items, Y_past, Theta_hat, W_hat
                )
                if A_logll_replicated > A_logll_heldout:
                    A_predictive_score += 1.0
                if Y_logll_replicated > Y_logll_heldout:
                    YP_pred_score += 1.0

            a_score[e_idx][k_idx] = A_predictive_score / replicates
            x_score[e_idx][k_idx] = YP_pred_score / replicates
            x_auc[e_idx][k_idx] = evaluate_random_subset(
                past_masked_items, Y_past, Theta_hat, W_hat, metric="logll"
            )

    print("A ppc scores across choices of num components:", a_score.mean(axis=0))
    print("X ppc scores across choices of num components:", x_score.mean(axis=0))
    print("X auc across choices of num components:", x_auc.mean(axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", action="store", default="../dat/pokec/regional_subset"
    )
    args = parser.parse_args()
    datadir = args.data_dir

    main()
