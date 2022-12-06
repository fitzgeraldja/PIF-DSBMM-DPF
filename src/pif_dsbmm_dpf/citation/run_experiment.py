"""
This script runs the default experiments. In addition to what is noted in other
scripts, extras to note are:
- Influence is post-processed to be one if an author doesn't publish in a period
- Script will allow choice of identifier in DSBMM-type data that corresponds to 'region',
  but will overwrite previous data if exists -- should make separate directory
  for runs with different choices
- Require DSBMM-type data in subdir, dsbmm_data - see reqs in that repo
- Remove final time period for DSBMM data, so substitutes are for up to T-1
  -- final time period must currently be inferred using point estimate
     of corresponding value
- DSBMM here uses SIMULATED topics as topic metadata -- this means that
  including extra real meta should be roughly fine, in that such metadata
 should actually be more or less indep of simulated topics. However, (i)
 this won't necessarily be true when apply on real topics, (eg if include
 subjectarea codings as separate metadata) and (ii) that including such
 additional metadata may actually render the substitutes less effective in
 specifically capturing the sim/real topics, and rendering them cond. ind.,
 even if it provides a better model for the full dataset overall -- instead
 here by default we ONLY use the simulated topics, but others may be added
 using the meta_choice flag
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import dsbmm_bp.data_processor as dsbmm_data_proc
import numpy as np
import utils
from absl import app, flags
from process_dataset import CitationSimulator

# from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import NMF
from tqdm import tqdm

# local modules
from pif_dsbmm_dpf.model import joint_factor_model as joint
from pif_dsbmm_dpf.model import multi_cause_influence as causal
from pif_dsbmm_dpf.model import network_model as nm
from pif_dsbmm_dpf.model import pmf as pmf
from pif_dsbmm_dpf.model import spf as spf


def post_process_influence(X, Beta):
    for t, X_t in enumerate(X):
        total_X = X_t.sum(axis=1)
        no_X = total_X == 0
        Beta[no_X, t] = 1.0
    return Beta


def get_set_overlap(Beta_p, Beta, k=50):
    scores = np.zeros(Beta.shape[1])
    tmp_bp = Beta_p.copy()
    # set missing values to 0, rather than 1
    tmp_bp[tmp_bp == 1.0] = 0.0
    tmp_b = Beta.copy()
    tmp_b[tmp_b == 1.0] = 0.0
    for t, (beta_pt, beta_t) in enumerate(zip(Beta_p.T, Beta.T)):
        top = np.argsort(beta_t)[-k:]
        top_p = np.argsort(beta_pt)[-k:]

        scores[t] = (
            np.intersect1d(top, top_p).shape[0] / np.union1d(top, top_p).shape[0]
        )
    return scores


def main(argv):
    datadir = Path(FLAGS.data_dir)

    outdir = Path(FLAGS.out_dir)
    outdir.mkdir(exist_ok=True)
    model = FLAGS.model
    variant = FLAGS.variant

    # if not os.path.exists(outdir):
    # 	os.makedirs(outdir)

    confounding_type = FLAGS.confounding_type
    configs = FLAGS.configs
    Q = FLAGS.num_components
    K = FLAGS.num_exog_components
    seed = FLAGS.seed
    influence_shp = FLAGS.influence_strength

    use_old_subs = FLAGS.use_old_subs

    region_col_id = FLAGS.region_col_id
    meta_choices = FLAGS.meta_choice
    if meta_choices == "topics_only":
        meta_choices = ["tpc_"]
    elif meta_choices == "all":
        meta_choices = None
    else:
        meta_choices = meta_choices.split(",")
    edge_weight_choice = FLAGS.edge_weight_choice
    if edge_weight_choice == "none":
        edge_weight_choice = None

    confounding_type = confounding_type.split(",")
    confounding_configs = [
        (int(c.split(",")[0]), int(c.split(",")[1])) for c in configs.split(":")
    ]
    datetime_str = time.strftime("%d-%m_%H-%M", time.gmtime(time.time()))
    base_sim_str = f"cnfdr_type{confounding_type}_cnfdr_cfg{confounding_configs}"
    if seed is not None:
        sim_model_str = f"seed{seed}_{base_sim_str}"
    else:
        sim_model_str = f"{datetime_str}_{base_sim_str}"

    window_len = 3  # set window length for dPF

    # specify main code directory
    main_code_dir = Path("~/Documents/main_project/post_confirmation/code").expanduser()
    # NB if using dPF this should be the location which contains the dPF
    # repo (i.e. DynamicPoissonFactorization dir with code contained)

    print("Confounding configs:", confounding_configs)
    print("Model:", model)

    sim_model_path = datadir / f"{sim_model_str}.pkl"

    write = outdir / (model + "." + variant + "_model_fitted_params")
    write.mkdir(exist_ok=True)

    if seed is not None:
        try:
            tqdm.write(f"Loading prev sim of model w same seed and configs...")
            with open(sim_model_path, "rb") as f:
                simulation_model: CitationSimulator = pickle.load(f)
        except FileNotFoundError:
            tqdm.write(
                f"Previous sim w given seed and configs not found, creating new sim..."
            )
            simulation_model = CitationSimulator(
                datapath=datadir,
                subnetwork_size=3000,
                sub_testsize=300,
                num_topics=1000,
                influence_shp=influence_shp,
                covar_2="random",
                covar_2_num_cats=5,
                seed=seed,
                save_path=sim_model_path,
            )
            try:
                simulation_model.process_dataset()
            except FileNotFoundError:
                try:
                    dsbmm_datadir = datadir / "dsbmm_data"
                    dsbmm_data = dsbmm_data_proc.load_data(dsbmm_datadir)
                    dsbmm_data_proc.save_to_pif_form(
                        dsbmm_data["A"],
                        dsbmm_data["X"],
                        datadir,
                        dsbmm_data["meta_names"],
                        region_col_id=region_col_id,
                        age_col_id="career_age",
                    )
                    del dsbmm_data
                    simulation_model.process_dataset()
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "Data in suitable form for either PIF directly, or DSBMM, not found in specified directory."
                    )

    A = simulation_model.A
    print(f"Adj. size and mean: {A[0].shape}, {[f'{A_t.mean():.3g}' for A_t in A]}")
    print(f"T: {len(A)+1}")

    for ct in tqdm(confounding_type, desc="Confounding type", position=0):
        for (noise, confounding) in tqdm(
            confounding_configs, desc="Confounding configs", position=1, leave=False
        ):
            tqdm.write(
                f"""Working on confounding setting with prob: {ct}
                and cov. 1/cov. 2 confounding strength:
                {(noise, confounding)}
                """
            )
            sys.stdout.flush()

            Y = simulation_model.make_multi_covariate_simulation(
                noise=noise, confounding_strength=confounding, confounding_to_use=ct
            )
            tqdm.write("Semi-synthetic data generated")

            Beta = simulation_model.beta
            Z = simulation_model.au_embed_1
            Gamma = simulation_model.topic_embed_1
            Alpha = simulation_model.au_embed_2
            W = simulation_model.topic_embed_2

            Beta = post_process_influence(Y[:-1], Beta)

            N = Y[0].shape[0]
            M = Y[0].shape[1]
            T = len(Y)

            if model == "unadjusted":
                m = causal.CausalInfluenceModel(
                    n_components=Q,
                    n_exog_components=K,
                    verbose=True,
                    model_mode="influence_only",
                )

            # elif model == "network_pref_only":
            #     m = causal.CausalInfluenceModel(
            #         n_components=Q,
            #         n_exog_components=K,
            #         verbose=True,
            #         model_mode="network_preferences",
            #     )

            # elif model == "topic_only":
            #     m = causal.CausalInfluenceModel(
            #         n_components=Q,
            #         n_exog_components=K,
            #         verbose=True,
            #         model_mode="topic",
            #     )

            # elif model == "pif":
            #     m = causal.CausalInfluenceModel(
            #         n_components=Q + K,
            #         n_exog_components=K,
            #         verbose=True,
            #         model_mode="full",
            #     )

            # elif model == "spf":
            #     m = spf.SocialPoissonFactorization(n_components=Q + K, verbose=True)

            elif model == "dsbmm_dpf":
                m = causal.CausalInfluenceModel(
                    n_components=Q + K,
                    n_exog_components=K,
                    verbose=True,
                    model_mode="full",
                )

            elif model == "no_unobs":
                num_regions = Z.shape[-1]
                num_covar_comps = W.shape[-1]
                m = causal.CausalInfluenceModel(
                    n_components=num_regions,
                    n_exog_components=num_covar_comps,
                    verbose=True,
                    model_mode="full",
                )

            elif model == "topic_only_oracle":
                num_regions = Z.shape[-1]
                num_covar_comps = W.shape[-1]
                m = causal.CausalInfluenceModel(
                    n_components=num_regions,
                    n_exog_components=num_covar_comps,
                    verbose=True,
                    model_mode="topic",
                )

            elif model == "network_only_oracle":
                num_regions = Z.shape[-1]
                num_covar_comps = W.shape[-1]
                m = causal.CausalInfluenceModel(
                    n_components=num_regions,
                    n_exog_components=num_covar_comps,
                    verbose=True,
                    model_mode="network_preferences",
                )

            # if model == "spf":
            #     m.fit(Y[1:], A, Y[:-1])
            if model == "no_unobs":
                m.fit(Y[1:], A, Z, W, Y[:-1])
            elif model == "topic_only_oracle":
                m.fit(Y[1:], A, Z, W, Y[:-1])
            elif model == "network_only_oracle":
                m.fit(Y[1:], A, Z, W, Y[:-1])
            # elif model == "network_pref_only":
            #     network_model = nm.NetworkPoissonMF(n_components=Q)
            #     network_model.fit(A)
            #     Z_hat = network_model.Et
            #     W_hat = np.zeros((M, K))
            #     m.fit(Y[1:], A, Z_hat, W_hat, Y[:-1])
            # elif model == "topic_only":
            #     pmf_model = pmf.PoissonMF(n_components=K)
            #     pmf_model.fit(Y[:-1])
            #     W_hat = pmf_model.Eb.T
            #     Z_hat = np.zeros((N, Q))
            #     m.fit(Y[1:], A, Z_hat, W_hat, Y[:-1])
            elif model == "unadjusted":
                Z_hat = np.zeros((N, Q))
                W_hat = np.zeros((M, K))
                m.fit(Y[1:], A, Z_hat, W_hat, Y[:-1])
            # elif model == "pif":
            # if variant == "z-theta-joint":
            #     joint_model = joint.JointPoissonMF(n_components=Q)
            #     joint_model.fit(Y[:-1], A)
            #     Z_hat_joint = joint_model.Et
            #     # W_hat = joint_model.Eb.T

            #     pmf_model = pmf.PoissonMF(n_components=K)
            #     pmf_model.fit(Y[:-1])
            #     W_hat = pmf_model.Eb.T

            # elif variant == "theta-only":
            #     pmf_model = pmf.PoissonMF(n_components=K)
            #     pmf_model.fit(Y[:-1])
            #     W_hat = pmf_model.Eb.T
            #     Theta_hat = pmf_model.Et

            # elif variant == "z-theta-concat":
            #     network_model = nm.NetworkPoissonMF(n_components=Q)
            #     network_model.fit(A)
            #     Z_hat = network_model.Et

            #     pmf_model = pmf.PoissonMF(n_components=K)
            #     pmf_model.fit(Y[:-1])
            #     W_hat = pmf_model.Eb.T
            #     Theta_hat = pmf_model.Et
            # else:
            #     network_model = nm.NetworkPoissonMF(n_components=Q)
            #     network_model.fit(A)
            #     Z_hat = network_model.Et

            #     pmf_model = pmf.PoissonMF(n_components=K)
            #     pmf_model.fit(Y[:-1])
            #     W_hat = pmf_model.Eb.T
            #     Theta_hat = pmf_model.Et

            # Rho_hat = np.zeros((N, T - 1, Q + K))
            # if variant == "z-only":
            #         Rho_hat[:, :, :Q] = Z_hat
            #     elif variant == "theta-only":
            #         Rho_hat[:, :, :K] = Theta_hat
            #     elif variant == "z-theta-concat":
            #         Rho_hat[:, :, :Q] = Z_hat
            #         Rho_hat[:, :, Q:] = Theta_hat
            #     else:
            #         Rho_hat[:, :, :Q] = Z_hat_joint

            # m.fit(Y[1:], A, Rho_hat, W_hat, Y[:-1])

            elif model == "dsbmm_dpf":
                # setup for dpf as actually a c++ program
                # -- need to change cwd to scratch/ before running dPF
                # as saves in working directory
                # pass location of dpf code
                if "-ndc" in variant:
                    variant.replace("-ndc", "")
                    deg_corr = False
                else:
                    deg_corr = True
                if "-undir" in variant:
                    variant.replace("-undir", "")
                    directed = False
                else:
                    directed = True

                dpf_repo_dir = main_code_dir / "DynamicPoissonFactorization"
                dpf_datadir = datadir / "dpf_data"  # see reqs in dpf repo
                dpf_datadir.mkdir(exist_ok=True)

                dpf_subdir = utils.gen_dpf_data(
                    dpf_datadir,
                    simulation_model.aus,
                    sim_id=sim_model_str,
                    datetime_str=datetime_str,
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
                    "-seed": int(seed),  # set random seed
                }
                # now make sure all strings so can pass to subprocess
                dpf_settings = {k: str(v) for k, v in dpf_settings.items()}

                # and load up DSBMM data
                try:
                    dsbmm_data
                except NameError:
                    dsbmm_datadir = datadir / "dsbmm_data"
                    dsbmm_datadir.mkdir(exist_ok=True)
                    try:
                        tqdm.write("Loading DSBMM data for given config")
                        with open(
                            dsbmm_datadir / f"{sim_model_str}_dsbmm_data.pkl", "rb"
                        ) as f:
                            dsbmm_data = pickle.load(f)
                    except FileNotFoundError:
                        tqdm.write(
                            "DSBMM data for given config not found, generating..."
                        )
                        dsbmm_data = dsbmm_data_proc.load_data(
                            dsbmm_datadir,
                            edge_weight_choice=edge_weight_choice,
                        )
                        dsbmm_data = utils.subset_dsbmm_data(
                            dsbmm_data,
                            simulation_model.aus,
                            T,
                            sim_tpcs=Y,
                            meta_choices=meta_choices,
                            remove_final=True,
                            save_path=dsbmm_datadir / f"{sim_model_str}_dsbmm_data.pkl",
                        )
                dsbmm_res_str = f"{sim_model_str}_{'dc' if deg_corr else 'ndc'}_{'dir' if directed else 'undir'}_{'meta' if variant=='z-theta-joint' else 'nometa'}"
                dpf_res_name = f"{sim_model_str}_{variant}.pkl"
                if variant == "z-theta-joint":
                    # 'z-theta-joint' is DSBMM and dPF combo
                    try:
                        with open(
                            dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb"
                        ) as f:
                            Z_hat_joint, Z_trans = pickle.load(f)
                        tqdm.write("Loaded DSBMM results for given config")
                    except FileNotFoundError:
                        # only run if not already done
                        tqdm.write("Running DSBMM")
                        h_l = 2
                        tqdm.write(
                            f"using settings h_Q={np.round(np.exp(np.log(Q) / h_l)).astype(int)}, N={dsbmm_data['A'][0].shape[0]}, T-1={len(dsbmm_data['A'])}"
                        )
                        Z_hat_joint, Z_trans = utils.run_dsbmm(
                            dsbmm_data,
                            dsbmm_datadir,
                            Q,
                            ignore_meta=False,
                            datetime_str=dsbmm_res_str,
                            deg_corr=deg_corr,
                            directed=directed,
                        )
                        with open(
                            dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb"
                        ) as f:
                            pickle.dump((Z_hat_joint, Z_trans), f)
                    # now run dPF - likewise have set seed so should
                    # be identical between runs
                    try:
                        with open(dpf_results_dir / dpf_res_name, "rb") as f:
                            W_hat, Theta_hat = pickle.load(f)
                        tqdm.write("Loaded dPF results for given config")
                    except FileNotFoundError:
                        tqdm.write("Running dPF")
                        W_hat, Theta_hat = utils.run_dpf(
                            dpf_repo_dir,
                            dpf_results_dir,
                            dpf_settings,
                            idx_map_dir=dpf_subdir,
                            true_N=N,
                            true_M=M,
                        )
                        with open(dpf_results_dir / dpf_res_name, "wb") as f:
                            pickle.dump((W_hat, Theta_hat), f)

                elif variant == "theta-only":
                    # 'theta-only' is just dPF
                    try:
                        with open(dpf_results_dir / dpf_res_name, "rb") as f:
                            W_hat, Theta_hat = pickle.load(f)
                        tqdm.write("Loaded dPF results for given config")
                    except FileNotFoundError:
                        tqdm.write("Running dPF")
                        W_hat, Theta_hat = utils.run_dpf(
                            dpf_repo_dir,
                            dpf_results_dir,
                            dpf_settings,
                            idx_map_dir=dpf_subdir,
                            true_N=N,
                            true_M=M,
                        )
                        with open(dpf_results_dir / dpf_res_name, "wb") as f:
                            pickle.dump((W_hat, Theta_hat), f)

                elif variant == "z-theta-concat":
                    #  'z-theta-concat' is DSBM (no meta) and dPF combo
                    try:
                        with open(
                            dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb"
                        ) as f:
                            Z_hat_joint, Z_trans = pickle.load(f)
                        tqdm.write("Loaded DSBM results for given config")
                    except FileNotFoundError:
                        tqdm.write("Running DSBM (no meta)")
                        Z_hat, Z_trans = utils.run_dsbmm(
                            dsbmm_data,
                            dsbmm_datadir,
                            Q,
                            ignore_meta=True,
                            datetime_str=dsbmm_res_str,
                            deg_corr=deg_corr,
                            directed=directed,
                        )
                        with open(
                            dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb"
                        ) as f:
                            pickle.dump((Z_hat_joint, Z_trans), f)
                    try:
                        with open(dpf_results_dir / dpf_res_name, "rb") as f:
                            W_hat, Theta_hat = pickle.load(f)
                        tqdm.write("Loaded dPF results for given config")
                    except FileNotFoundError:
                        tqdm.write("Running dPF")
                        W_hat, Theta_hat = utils.run_dpf(
                            dpf_repo_dir,
                            dpf_results_dir,
                            dpf_settings,
                            idx_map_dir=dpf_subdir,
                            true_N=N,
                            true_M=M,
                        )
                        with open(dpf_results_dir / dpf_res_name, "wb") as f:
                            pickle.dump((W_hat, Theta_hat), f)
                else:
                    # 'z-only' is just DSBM (no meta)
                    try:
                        with open(
                            dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb"
                        ) as f:
                            Z_hat_joint, Z_trans = pickle.load(f)
                        tqdm.write("Loaded DSBM results for given config")
                    except FileNotFoundError:
                        tqdm.write("Running DSBM (no meta)")
                        Z_hat, Z_trans = utils.run_dsbmm(
                            dsbmm_data,
                            dsbmm_datadir,
                            Q,
                            ignore_meta=True,
                            datetime_str=dsbmm_res_str,
                            deg_corr=deg_corr,
                            directed=directed,
                        )
                        with open(
                            dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb"
                        ) as f:
                            pickle.dump((Z_hat_joint, Z_trans), f)
                    try:
                        with open(dpf_results_dir / dpf_res_name, "rb") as f:
                            W_hat, Theta_hat = pickle.load(f)
                        tqdm.write("Loaded dPF results for given config")
                    except FileNotFoundError:
                        tqdm.write("Running dPF")
                        W_hat, Theta_hat = utils.run_dpf(
                            dpf_repo_dir,
                            dpf_results_dir,
                            dpf_settings,
                            idx_map_dir=dpf_subdir,
                            true_N=N,
                            true_M=M,
                        )
                        with open(dpf_results_dir / dpf_res_name, "wb") as f:
                            pickle.dump((W_hat, Theta_hat), f)

                Rho_hat = np.zeros((N, T - 1, Q + K))
                if variant == "z-only":
                    Rho_hat[:, :, :Q] = Z_hat
                elif variant == "theta-only":
                    Rho_hat[:, :, :K] = Theta_hat
                elif variant == "z-theta-concat":
                    Rho_hat[:, :, :Q] = Z_hat
                    Rho_hat[:, :, Q:] = Theta_hat
                else:
                    Rho_hat[:, :, :Q] = Z_hat_joint

                m.fit(
                    Y[1:],
                    A,
                    Rho_hat,
                    W_hat,
                    Y[:-1],
                    Z_trans=Z_trans,
                    use_old_subs=use_old_subs,
                )

            Beta_p = m.E_beta
            # now that calculation is done, when ranking should not count
            # authors with no citations / no papers -- for both beta,
            # these would give beta = 1.0, but if many of them then will
            # artificially inflate the score -- scores below handle this
            scores = get_set_overlap(Beta_p, Beta)
            loss = utils.mse(Beta, Beta_p, for_beta=True)

            # now that beta and
            tqdm.write(f"Mean inferred infl: {Beta_p[Beta_p!=1].mean():.3g}")

            tqdm.write(f"Overlaps: {np.round(scores,3)}, \nMSE: {np.round(loss,3)}")
            tqdm.write(f"{'*' * 60}")
            sys.stdout.flush()
            outfile = write / ("conf=" + str((noise, confounding)) + ";conf_type=" + ct)
            np.savez_compressed(outfile, fitted=Beta_p, true=Beta)


if __name__ == "__main__":
    # TODO:
    # -- get running
    # -- write shell script to run w other vars
    # -- as for theirs, let run 10 times for each, with
    #    different random seeds
    # -- see if can get sensitivity analysis also going
    # -- sort PPCs -- very long!

    # -- for real data resort dPF full data: split first year of
    #    final period off as val, rest as test
    #    for held-out time period elsewhere, then move
    #    to dpf_datadir, + split some val data off
    #    this also (maybe first year or sth)
    # -- also will need to slightly modify, as for fully held-out data should
    #    allow substitutes to be constructed for final period -- the held-out
    #    dPF data should have the test as held-out, val as final year / some
    #    small amount of this BEFORE that, and DSBMM should not remove any info
    # -- consider updating impl of [pif, spf, network_pref_only, topic_only]
    #    models - would be good to compare to methods that treat each timestep
    #    separately, and should genuinely be fairly small modifications needed
    # -- handle final period properly - this might be used when observing completely held-out data,
    #    for which we don't have any confounder substitutes. Somewhat dubious using substitutes fit on the
    #    full data anyway, but otherwise would have to fit separate confounders on every subset here...
    #    Feel that observing performance on fully held-out data is reasonable compromise.
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "model",
        "dsbmm_dpf",
        """
        method to use selected from one of
        [
            dsbmm_dpf, unadjusted, topic_only_oracle,
            network_only_oracle, no_unobs (gold standard)
        ]
        default is dsbmm_dpf
        """,
    )
    flags.DEFINE_string(
        "data_dir",
        "/scratch/fitzgeraldj/data/caus_inf_data",
        "path to author profiles and network files (edgelist)",
    )
    # this should contain the full dataset in the right form for CitationSimulator
    # -- links in citation_links.npz, arr named "edge_list"
    # -- author profiles in au_profs.pkl
    # where files are as described in process_dataset.py
    # Should also be subdir, datadir / "dsbmm_data" -- will be created otherwise
    # -- if already have full dataset in DSBMM form, can upload here then
    #    rest should run, and will produce proper data for CitationSimulator
    #    on first run
    # Then final subdir, datadir / "dpf_data", which may be empty
    # -- if empty (/does not exist, in which case will be created),
    #    will be populated with data for DPF on first run, saved
    #    in a further subdir inside with a name according to seed
    #    / datetime, and results saved in datadir / dpf_results
    flags.DEFINE_string(
        "out_dir",
        "/scratch/fitzgeraldj/data/caus_inf_data/results",
        "directory to write output files to",
    )
    flags.DEFINE_string(
        "variant",
        "z-theta-joint",
        """
        variant for fitting per-author substitutes, chosen from one of
        [
            z-theta-joint (joint model),
            z-only (community model only),
            z-theta-concat (MF and community model outputs concatenated),
            theta-only (MF only)
        ]
        default is z-theta-joint
        """,
    )
    flags.DEFINE_string(
        "confounding_type",
        "both",
        """
        comma-separated list of types of confounding to simulate
        in outcome, chosen from [homophily, exog, both]
        default is both
        """,
    )
    flags.DEFINE_string(
        "configs",
        "50,50",
        """
        list of confounding strength configurations to use
        in simulation; must be in format
        "[confounding strength 1],[noise strength 1]:[confounding strength 2],[noise strength 2], ..."
        default is "50,50" (i.e. 50 confounding, 50 noise)
        """,
    )
    flags.DEFINE_integer(
        "num_components",
        16,
        """
        number of components to use to fit factor model for
        per-author substitutes, default 16
        """,
    )
    flags.DEFINE_integer(
        "num_exog_components",
        10,
        """
        number of components to use to fit factor model for
        per-topic substitutes, default 10
        """,
    )
    flags.DEFINE_integer(
        "seed", 42, "random seed passed to simulator in each experiment, default 42"
    )
    flags.DEFINE_float(
        "influence_strength",
        0.005,
        "Shape parameter that controls the average influence in network, default 0.005",
    )

    flags.DEFINE_string(
        "region_col_id",
        "main_adm1_1hot",
        """
        Identifier for metadata in dsbmm-type data corresponding to 'region'
        -- in paper, corresponds to main_adm1_1hot or main_ctry_1hot of author
        """,
    )

    flags.DEFINE_string(
        "meta_choice",
        "topics_only",
        """
        Choice of metadata to use in dsbmm-type data,
        either 'topics_only', 'all' or a comma-separated
        list of metadata columns to use -- in this case,
        matching is used to select columns, where each
        item passed is assume to identify the start of
        a column name (e.g. we use "tpc_" for topics)
        -- so more metadata columns than choices
        passed may be used, e.g. both weighted and
        unweighted topic columns
        """,
    )
    flags.DEFINE_string(
        "edge_weight_choice",
        "count",
        """
        Specify edge weight name to use, e.g. 'count' or 'weighted'
        for edge weights in network. Default is 'count', pass
        'none' to use unweighted (binary) network""",
    )
    flags.DEFINE_bool(
        "use_old_subs",
        True,
        "Use old substitutes (i.e. those from previous time period) in DSBMM",
    )  # NB this will require being passed as either
    # --flag (meaning true), or --noflag (meaning false)
    # if passing explicitly for flag
    app.run(main)
