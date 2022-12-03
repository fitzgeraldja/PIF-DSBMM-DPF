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
"""

import argparse
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import dsbmm_bp.data_processor as dsbmm_data_proc
import numpy as np
from absl import app, flags

# from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import NMF
from tqdm import tqdm

from citation.process_dataset import CitationSimulator

from ..model import joint_factor_model as joint
from ..model import multi_cause_influence as causal
from ..model import network_model as nm
from ..model import pmf as pmf
from ..model import spf as spf

# local modules
from . import utils


def post_process_influence(X, Beta):
    for t, X_t in enumerate(X):
        total_X = X_t.sum(axis=1)
        no_X = total_X == 0
        Beta[no_X, t] = 1.0
    return Beta


def get_set_overlap(Beta_p, Beta, k=50):
    scores = np.zeros(Beta.shape[1])
    for t, (beta_pt, beta_t) in enumerate(zip(Beta_p.T, Beta.T)):
        top = np.argsort(beta_t)[-k:]
        top_p = np.argsort(beta_pt)[-k:]

        scores[t] = (
            np.intersect1d(top, top_p).shape[0] / np.union1d(top, top_p).shape[0]
        )
    return scores


def main(argv):
    datadir = Path(FLAGS.data_dir)
    outdir = FLAGS.out_dir
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

    region_col_id = FLAGS.region_col_id
    meta_choices = FLAGS.meta_choice
    if meta_choices == "topics_only":
        meta_choices = ["tpc_"]
    elif meta_choices == "all":
        meta_choices = None
    else:
        meta_choices = meta_choices.split(",")

    datetime_str = time.strftime("%d-%m_%H-%M", time.gmtime(time.time()))
    dsbmm_label_str = f"seed{seed}_{datetime_str}" if seed is not None else datetime_str

    confounding_type = confounding_type.split(",")
    confounding_configs = [
        (int(c.split(",")[0]), int(c.split(",")[1])) for c in configs.split(":")
    ]

    print("Confounding configs:", confounding_configs)
    print("Model:", model)

    write = os.path.join(outdir, model + "." + variant + "_model_fitted_params")
    os.makedirs(write, exist_ok=True)

    simulation_model = CitationSimulator(
        datapath=datadir,
        subnetwork_size=3000,
        sub_testsize=300,
        num_topics=1000,
        influence_shp=influence_shp,
        covar_2="random",
        covar_2_num_cats=5,
        seed=seed,
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
    print("Adj. size and mean:", A[0].shape, [A_t.mean() for A_t in A])
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
                num_regions = Z.shape[1]
                num_covar_comps = W.shape[1]
                m = causal.CausalInfluenceModel(
                    n_components=num_regions,
                    n_exog_components=num_covar_comps,
                    verbose=True,
                    model_mode="full",
                )

            elif model == "topic_only_oracle":
                num_regions = Z.shape[1]
                num_covar_comps = W.shape[1]
                m = causal.CausalInfluenceModel(
                    n_components=num_regions,
                    n_exog_components=num_covar_comps,
                    verbose=True,
                    model_mode="topic",
                )

            elif model == "network_only_oracle":
                num_regions = Z.shape[1]
                num_covar_comps = W.shape[1]
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

            # Rho_hat = np.zeros((N, T, Q + K))
            # if variant == "z-only":
            #     Rho_hat[:, :Q] = Z_hat
            # elif variant == "theta-only":
            #     Rho_hat[:, :K] = Theta_hat
            # elif variant == "z-theta-concat":
            #     Rho_hat = np.column_stack((Z_hat, Theta_hat))
            # else:
            #     Rho_hat[:, :Q] = Z_hat_joint

            # m.fit(Y[1:], A, Rho_hat, W_hat, Y[:-1])

            elif model == "dsbmm_dpf":
                # setup for dpf as actually a c++ program
                # -- need to change cwd to scratch/ before running dPF
                # as saves in working directory
                # pass location of dpf code
                main_code_dir = Path(
                    "~/Documents/main_project/post_confirmation/code"
                ).expanduser()
                dpf_repo_dir = main_code_dir / "DynamicPoissonFactorization"
                dpf_datadir = datadir / "dpf_data"  # see reqs in dpf repo

                try:
                    dpf_train
                except NameError:
                    (
                        dpf_train,
                        dpf_val,
                        dpf_test,
                        dpf_au_idx_map,
                        dpf_tpc_idx_map,
                    ) = utils.get_dpf_data(
                        dpf_datadir,
                        simulation_model.aus,
                        seed=seed,
                        datetime_str=datetime_str,
                    )
                # TODO:
                # -- check that dpf subset size matches CI, and
                # that idx is same
                # -- make sure that DSBMM uses SIMULATED topics
                #    as meta -- this also means that including
                #    extra info should be roughly fine, in that
                #    real meta should actually be more or less
                #    indep of simulated topics. However, (i) this
                #    still won't be true when apply on real topics,
                #    and (ii) that including real meta may actually
                #    render the substitutes less effective in
                #    specifically capturing the sim/real topics
                #    -- probably better to ONLY use the topics,
                #    or at least pass a flag to choose
                # -- make sure attach region au metadata to DSBMM
                #    nets
                # -- check DSBMM hier trans works
                # -- gen au_profs, edgelist for CI if not done already
                # -- resort dPF full data: split first year of
                #    final period off as val, rest as test
                # for held-out time period elsewhere, then move
                # to dpf_datadir, + split some val data off
                # this also (maybe first year or sth), so only
                # train gets regenerated here, along w subsetting
                # val maybe?

                dpf_results_dir = datadir / "dpf_results"
                dpf_results_dir.mkdir(exist_ok=True)
                subprocess.run(["cd", str(dpf_results_dir)])
                dpf_settings = {
                    "-n": N,
                    "-m": M,
                    "-dir": str(dpf_datadir),
                    "-rfreq": 10,  # check ll every 10 iterations
                    "-vprior": 10,  # prior on variance for transitions
                    "-num_threads": 64,  # number of threads to use
                    "-tpl": 3,  # gap between time periods
                    # -- assume passing time in years, so this
                    # is window length in years
                    "-max-iterations": 1000,  # max EM iterations
                    # -- NB patience only
                    # 3 checks w/o increase in ll
                    # so unlikely to reach
                    "-k": K,  # number of factors to fit
                    "-seed": seed,  # set random seed
                }

                # and load up DSBMM data
                try:
                    dsbmm_data
                except NameError:
                    dsbmm_datadir = datadir / "dsbmm_data"
                    dsbmm_data = dsbmm_data_proc.load_data(
                        dsbmm_datadir,
                        edge_weight_choice="count",
                        sim_tpcs=Y,
                    )
                    dsbmm_data = utils.subset_dsbmm_data(
                        dsbmm_data,
                        simulation_model.aus,
                        T,
                        meta_choices=meta_choices,
                        remove_final=True,
                    )

                if variant == "z-theta-joint":
                    # 'z-theta-joint' is DSBMM and dPF combo
                    # TODO:
                    # -- make sure pass correct metadata -- in particular
                    #    final time period topics or not -- should hold out
                    #    at least one time period if possible for eval
                    # -- will remove very final period, so the substitutes
                    #    are of dim T-1
                    # -- check this works w how coded rest, + give
                    #    option of either using Z directly or
                    #    premultiplying w trans
                    # -- choose Q=16 so can use hierarchical w 2 layers, 8 groups per
                    # -- make new dir in scratch/ w DSBMM data in dsbmm_data/ subdir
                    # -- get running
                    Z_hat_joint, Z_trans = utils.run_dsbmm(
                        dsbmm_data,
                        dsbmm_datadir,
                        Q,
                        ignore_meta=False,
                        datetime_str=dsbmm_label_str,
                    )

                    W_hat, Theta_hat = utils.run_dpf(
                        dpf_repo_dir, dpf_results_dir, dpf_settings, n_components=K
                    )

                elif variant == "theta-only":
                    # 'theta-only' is just dPF
                    W_hat, Theta_hat = utils.run_dpf(
                        dpf_repo_dir, dpf_results_dir, dpf_settings, n_components=K
                    )

                elif variant == "z-theta-concat":
                    #  'z-theta-concat' is DSBM (no meta) and dPF combo
                    Z_hat, Z_trans = utils.run_dsbmm(
                        dsbmm_data,
                        dsbmm_datadir,
                        Q,
                        ignore_meta=True,
                        datetime_str=dsbmm_label_str,
                    )

                    W_hat, Theta_hat = utils.run_dpf(
                        dpf_repo_dir, dpf_results_dir, dpf_settings, n_components=K
                    )
                else:
                    # 'z-only' is just DSBM (no meta)
                    Z_hat, Z_trans = utils.run_dsbmm(
                        dsbmm_data,
                        dsbmm_datadir,
                        Q,
                        ignore_meta=True,
                        datetime_str=dsbmm_label_str,
                    )

                    W_hat, Theta_hat = utils.run_dpf(
                        dpf_repo_dir, dpf_results_dir, dpf_settings, n_components=K
                    )

                Rho_hat = np.zeros((N, T, Q + K))
                if variant == "z-only":
                    Rho_hat[:, :Q] = Z_hat
                elif variant == "theta-only":
                    Rho_hat[:, :K] = Theta_hat
                elif variant == "z-theta-concat":
                    Rho_hat = np.column_stack((Z_hat, Theta_hat))
                else:
                    Rho_hat[:, :Q] = Z_hat_joint

                m.fit(Y[1:], A, Rho_hat, W_hat, Y[:-1], Z_trans)

            Beta_p = m.E_beta
            scores = get_set_overlap(Beta_p, Beta)
            loss = utils.mse(Beta, Beta_p)

            tqdm.write(f"Mean inferred infl: {Beta_p.mean():.3g}")

            tqdm.write(
                f"Overlaps: {np.round(scores,3)}, \nMSE: {loss:np.round(loss,3)}"
            )
            tqdm.write("*" * 60)
            sys.stdout.flush()
            outfile = os.path.join(
                write, "conf=" + str((noise, confounding)) + ";conf_type=" + ct
            )
            np.savez_compressed(outfile, fitted=Beta_p, true=Beta)


if __name__ == "__main__":
    # TODO:
    # -- new variant options, possibly incl DC / NDC, and maybe weighted edges
    # -- update default datadir + outdir to scratch
    # -- get running
    # -- see if can get sensitivity analysis also going
    # -- consider updating impl of [pif, spf, network_pref_only, topic_only]
    # models - would be good to compare to methods that treat each timestep
    # separately, and should genuinely be fairly small modifications needed
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
        "../dat/citation/regional_subset",
        "path to author profiles and network files (edgelist)",
    )
    flags.DEFINE_string("out_dir", "../out/", "directory to write output files to")
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
        10,
        """
        number of components to use to fit factor model for
        per-author substitutes, default 10
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
        "region",
        "identifier for metadata in dsbmm-type data corresponding to 'region'",
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

    app.run(main)
