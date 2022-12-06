import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import dsbmm_bp.data_processor as dsbmm_data_proc
import numpy as np

# from scipy import sparse
from scipy.stats import bernoulli, poisson
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pif_dsbmm_dpf.citation import utils
from pif_dsbmm_dpf.citation.process_dataset import CitationSimulator


def calculate_ppc_dpf(heldout_idxs, obs_y, theta, beta):
    r"""Compute the predictive probability of the heldout topics.
    Should be applied separately for network confounder and
    topic confounder models -- the likelihood of each are
    different. This func is for topic substitute -- the dPF.

    If a model reproduces the data well, then the probability
    of observing the actual heldout topics should be similar
    to the probability of observing the replicated topics.

    Under dPF, given au factors theta and topic factors beta,
    shape (N,T,K) and (M,T,K) resp., the probability of
    observing a publication by au i on topic m at t
    is Poisson(\sum_k exp(theta_{ik}^t) * exp(beta_{mk}^t)),
    so can calc rates given theta and beta, then calc logpmf.

    :param heldout_idxs: indices of heldout topics
    :type heldout_idxs: list[tuple(np.ndarray, np.ndarray)]
    :param obs_y: actual observed au-topic counts, length T list
                  of sparse csr_array shape (N,M)
    :type obs_y: list[sparse.csr_array]
    :param theta: au factors inferred by dPF, shape (N,T,K)
    :type theta: np.ndarray
    :param beta: topic factors inferred by dPF, shape (M,T,K)
    :type beta: np.ndarray
    :return: logll_heldout, logll_replicated
    :rtype: tuple(float, float)
    """
    expbeta, exptheta = np.exp(beta), np.exp(theta)
    # comb_idxs = (
    #     np.concatenate([x[0] for x in heldout_idxs]),
    #     np.concatenate([x[1] for x in heldout_idxs]),
    #     np.concatenate([t * np.ones(len(x[0])) for t, x in enumerate(heldout_idxs)]),
    # )
    subtheta = [
        exptheta[h_idx[0], t * np.ones(len(h_idx[0])), :]
        for t, h_idx in enumerate(heldout_idxs)
    ]
    subbeta = [
        expbeta[h_idx[1], t * np.ones(len(h_idx[1])), :]
        for t, h_idx in enumerate(heldout_idxs)
    ]
    rates = [(st * sb).sum(axis=-1) for st, sb in zip(subtheta, subbeta)]
    replicated = [poisson.rvs(rate) for rate in rates]
    heldout = [obs[h_idx] for h_idx, obs in zip(heldout_idxs, obs_y)]
    logll_heldout = np.sum(
        [poisson.logpmf(ho, rate).sum() for ho, rate in zip(heldout, rates)]
    )
    logll_replicated = np.sum(
        [poisson.logpmf(rep, rate).sum() for rep, rate in zip(replicated, rates)]
    )
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
    This func is for au-topic substitute -- the DSBMM.

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

    heldout = [obs[idxs] for obs, idxs in zip(obs_a, heldout_idxs)]
    i_idxs = [idxs[0] for idxs in heldout_idxs]
    j_idxs = [idxs[1] for idxs in heldout_idxs]
    i_probs = [node_probs[idxs, t, :] for t, idxs in enumerate(i_idxs)]
    j_probs = [node_probs[idxs, t, :] for t, idxs in enumerate(j_idxs)]
    if deg_corr:
        degs = [
            {"i": np.zeros((len(i_idx), 2)), "j": np.zeros((len(j_idx), 2))}
            for i_idx, j_idx in zip(i_idxs, j_idxs)
        ]
        for t, (obs, i_idx, j_idx) in enumerate(zip(obs_a, i_idxs, j_idxs)):
            # final degs dim is in, out
            degs[t]["i"][:, 0] = obs[:, i_idx].sum(axis=0).squeeze()
            degs[t]["i"][:, 1] = obs[i_idx, :].sum(axis=1).squeeze()
            degs[t]["j"][:, 0] = obs[:, j_idx].sum(axis=0).squeeze()
            degs[t]["j"][:, 1] = obs[j_idx, :].sum(axis=1).squeeze()

        if not directed:
            # undirected so in and out are the same
            # = d_i*d_j * p_qr
            tot_degs = [{k: v.sum(axis=1) for k, v in deg.items()} for deg in degs]
            i_degs = [deg["i"] for deg in tot_degs]
            j_degs = [deg["j"] for deg in tot_degs]
        else:
            # dir
            # = d_i^{in}*d_j^{out} * p_qr
            i_degs = [deg["i"][:, 1] for deg in degs]  # out deg
            j_degs = [deg["j"][:, 0] for deg in degs]  # in deg
        e_rates = [
            np.einsum("eq,qr,er->e", i_deg * i_prob, block_probs, j_deg * j_prob)
            for i_deg, i_prob, j_deg, j_prob in zip(i_degs, i_probs, j_degs, j_probs)
        ]
    else:
        # shouldn't matter if directed or not if ndc, as either block probs
        # will be symmetric (undir) or not (dir) and otherwise form
        # is same
        e_probs = [
            np.einsum("eq,qr,er->e", i_prob, block_probs, j_prob)
            for i_prob, j_prob in zip(i_probs, j_probs)
        ]
    if deg_corr:
        replicated = [poisson.rvs(er) for er in e_rates]
        logll_heldout = np.sum(
            [
                poisson.logpmf(ho.astype(int), er).sum()
                for ho, er in zip(heldout, e_rates)
            ]
        )
        logll_replicated = np.sum(
            [poisson.logpmf(rep, er).sum() for rep, er in zip(replicated, e_rates)]
        )
    else:
        replicated = [bernoulli.rvs(ep) for ep in e_probs]
        logll_heldout = np.sum(
            [
                bernoulli.logpmf((ho > 0).astype(int), ep).sum()
                for ho, ep in zip(heldout, e_probs)
            ]
        )
        logll_replicated = np.sum(
            [bernoulli.logpmf(rep, ep).sum() for rep, ep in zip(replicated, e_probs)]
        )
    return logll_heldout, logll_replicated


def evaluate_random_subset_dpf(heldout_idxs, obs_y, theta, beta, metric="logll"):
    """Only calc AUC for dPF, and on same heldout au-topic pairs as
    before so might as well just pass expected values in, but leave this
    for generality

    :param heldout_idxs: indices of heldout topics
    :type heldout_idxs: list[tuple(np.ndarray, np.ndarray)]
    :param obs_y: actual observed au-topic counts, length T list
                  sparse csr_array shape (N,M)
    :type obs_y: list[sparse.csr_array]
    :param theta: au factors inferred by dPF, shape (N,T,K)
    :type theta: np.ndarray
    :param beta: topic factors inferred by dPF, shape (M,T,K)
    :type beta: np.ndarray
    :return: score
    :rtype: float
    """
    expbeta, exptheta = np.exp(beta), np.exp(theta)
    subtheta = [
        exptheta[h_idx[0], t * np.ones(len(h_idx[0])), :]
        for t, h_idx in enumerate(heldout_idxs)
    ]
    subbeta = [
        expbeta[h_idx[1], t * np.ones(len(h_idx[1])), :]
        for t, h_idx in enumerate(heldout_idxs)
    ]
    expected = [(st * sb).sum(axis=-1).squeeze() for st, sb in zip(subtheta, subbeta)]
    truth = [obs[h_idx].toarray().squeeze() for h_idx, obs in zip(heldout_idxs, obs_y)]
    expected = np.concatenate(expected)
    truth = np.concatenate(truth)

    if metric == "auc":
        return roc_auc_score(truth, expected)
    else:
        return poisson.logpmf(truth, expected).sum()


def mask_topics(samp_size, n_cats):
    """Choose a random subset of topics to mask for each au,
    or a random subset of aus to mask (to mask edges)

    :param samp_size: return shape of mask
    :type samp_size: Union[int, tuple(int, int)]
    :param n_cats: number of topics/aus from which to subsample
    :type n_cats: int
    :return: random_topics
    :rtype: np.ndarray
    """
    cats = np.arange(n_cats)
    random_topics = np.random.choice(cats, size=samp_size)
    return random_topics


def main():
    num_exps = 20
    Qs = [4, 8, 16, 32]
    Ks = [3, 5, 8, 10]
    noise = 10.0
    conf_strength = 50.0
    window_len = 3

    a_score = np.zeros((num_exps, len(Ks)))
    x_score = np.zeros((num_exps, len(Ks)))
    x_auc = np.zeros((num_exps, len(Ks)))
    # for ct in tqdm(confounding_type, desc="Confounding type", position=0):
    #     for (noise, confounding) in tqdm(
    #         confounding_configs, desc="Confounding configs", position=1, leave=False
    #     ):
    for exp_idx in range(num_exps):
        print("Working on experiment", exp_idx)
        sim_model_path = datadir / f"sim_model_{exp_idx}.pkl"
        try:
            with open(sim_model_path, "rb") as f:
                simulation_model: CitationSimulator = pickle.load(f)
            tqdm.write(f"Loading prev sim of model w same seed and configs...")
        except FileNotFoundError:
            tqdm.write(
                f"Previous sim w given seed and configs not found, creating new sim..."
            )
            simulation_model = CitationSimulator(
                datapath=datadir,
                subnetwork_size=3000,
                sub_testsize=300,
                num_topics=1000,
                influence_shp=0.005,
                covar_2="random",
                covar_2_num_cats=5,
                seed=exp_idx,
                save_path=sim_model_path,
            )
            simulation_model.process_dataset()

        A = simulation_model.A
        print(f"Adj. size and mean: {A[0].shape}, {[f'{A_t.mean():.3g}' for A_t in A]}")
        print(f"T: {len(A)+1}")

        tqdm.write(
            f"""Working on confounding setting with prob: both
            and cov. 1/cov. 2 confounding strength:
            {(noise, conf_strength)}
            """
        )
        sys.stdout.flush()

        Y = simulation_model.make_multi_covariate_simulation(
            noise=noise, confounding_strength=conf_strength, confounding_to_use="both"
        )
        tqdm.write("Semi-synthetic data generated")

        N = Y[0].shape[0]
        M = Y[0].shape[1]
        T = len(Y)
        masked_friends = [mask_topics(N, N) for _ in range(T - 1)]
        past_masked_topics = [mask_topics(N, M) for _ in range(T - 1)]
        aus = np.arange(N)
        masked_friends = [(aus, mf) for mf in masked_friends]
        past_masked_topics = [(aus, pmt) for pmt in past_masked_topics]

        Y_past_train = [Y_t.copy() for Y_t in Y[:-1]]
        A_train = [A_t.copy() for A_t in A]

        for t in range(T - 1):
            A_train[t][masked_friends[t]] = 0
            Y_past_train[t][past_masked_topics[t]] = 0

        for k_idx, (Q, K) in enumerate(zip(Qs, Ks)):
            # change to train versions!
            dpf_repo_dir = main_code_dir / "DynamicPoissonFactorization"
            dpf_datadir = datadir / "dpf_data"  # see reqs in dpf repo
            dpf_datadir.mkdir(exist_ok=True)

            dpf_subdir = utils.gen_dpf_data(
                dpf_datadir,
                simulation_model.aus,
                sim_id=sim_model_path.stem,
                sim_tpcs=Y,
                window_len=window_len,
            )

            dpf_results_dir = datadir / "dpf_results"
            dpf_results_dir.mkdir(exist_ok=True)
            os.chdir(str(dpf_results_dir))
            dpf_settings = {
                "-n": N,
                "-m": M,
                "-dir": str(dpf_subdir),
                "-rfreq": 10,  # check ll every 10 iterations
                "-vprior": 10,  # prior on variance for transitions
                "-num_threads": 64,  # number of threads to use
                "-tpl": window_len,  # gap between time periods
                # -- assume passing time in years, so this
                # is window length in years
                "-max-iterations": 1000,  # max EM iterations
                # -- NB patience only
                # 3 checks w/o increase in ll
                # so unlikely to reach
                "-k": K,  # number of factors to fit
                # "-seed": int(seed),  # don't set random seed
            }
            # now make sure all strings so can pass to subprocess
            dpf_settings = {k: str(v) for k, v in dpf_settings.items()}

            # and load up DSBMM data
            try:
                dsbmm_data
            except NameError:
                dsbmm_datadir = datadir / "dsbmm_data"
                dsbmm_datadir.mkdir(exist_ok=True)
                # try:
                #     tqdm.write("Loading DSBMM data for given config")
                #     with open(
                #         dsbmm_datadir / f"{sim_model_path.stem}_dsbmm_data.pkl", "rb"
                #     ) as f:
                #         dsbmm_data = pickle.load(f)
                # except FileNotFoundError:
                tqdm.write("DSBMM data for given config not found, generating...")
                try:
                    all_dsbmm_data
                except NameError:
                    all_dsbmm_data = dsbmm_data_proc.load_data(
                        dsbmm_datadir,
                        edge_weight_choice="count",
                    )
                dsbmm_data = utils.subset_dsbmm_data(
                    all_dsbmm_data,
                    simulation_model.aus,
                    T,
                    sim_tpcs=Y,
                    meta_choices=["tpc_"],
                    remove_final=True,
                    save_path=dsbmm_datadir / f"{sim_model_path.stem}_dsbmm_data.pkl",
                )

            replicates = 20
            A_predictive_score = 0.0
            YP_pred_score = 0.0
            for _ in range(replicates):
                # now dataset gen, allow randomness again over replicates
                t = 1000 * time.time()  # current time in milliseconds
                np.random.seed(int(t) % 2**32)
                datetime_str = time.strftime("%d-%m_%H-%M", time.gmtime(time.time()))
                dsbmm_res_str = f"{sim_model_path.stem}_{datetime_str}"
                tqdm.write("Running DSBMM")
                Z_hat_joint, _, block_probs = utils.run_dsbmm(
                    dsbmm_data,
                    dsbmm_datadir,
                    Q,
                    ignore_meta=False,
                    datetime_str=dsbmm_res_str,
                    deg_corr=True,
                    directed=True,
                    ret_block_probs=True,
                )

                tqdm.write("Running dPF")
                W_hat, Theta_hat = utils.run_dpf(
                    dpf_repo_dir,
                    dpf_results_dir,
                    dpf_settings,
                    idx_map_dir=dpf_subdir,
                    true_N=N,
                    true_M=M,
                )

                A_logll_heldout, A_logll_replicated = calculate_ppc_dsbmm(
                    masked_friends,
                    A,
                    Z_hat_joint,
                    block_probs,
                )
                Y_logll_heldout, Y_logll_replicated = calculate_ppc_dpf(
                    past_masked_topics, Y[:-1], Theta_hat, W_hat
                )
                if A_logll_replicated > A_logll_heldout:
                    A_predictive_score += 1.0
                if Y_logll_replicated > Y_logll_heldout:
                    YP_pred_score += 1.0

            a_score[exp_idx][k_idx] = A_predictive_score / replicates
            x_score[exp_idx][k_idx] = YP_pred_score / replicates
            x_auc[exp_idx][k_idx] = evaluate_random_subset_dpf(
                past_masked_topics, Y[:-1], Theta_hat, W_hat, metric="logll"
            )

    print("A ppc scores across choices of num components:", a_score.mean(axis=0))
    print("X ppc scores across choices of num components:", x_score.mean(axis=0))
    print("X auc across choices of num components:", x_auc.mean(axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="/scratch/fitzgeraldj/data/caus_inf_data/"
    )
    args = parser.parse_args()
    datadir = Path(args.data_dir)
    main_code_dir = Path("~/Documents/main_project/post_confirmation/code").expanduser()
    main()
