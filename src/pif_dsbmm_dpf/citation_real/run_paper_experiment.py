"""
This script runs the experiments for the real data.
In addition to what is noted in other scripts, extras to note are:
- Influence is post-processed to be one if an author doesn't publish in a period
- Script allows choice of
- Require DSBMM-type data in subdir, dsbmm_data - see reqs in that repo
- Remove final time period for DSBMM data, so substitutes are for up to T-1
  -- final time period must currently be inferred using point estimate
     of corresponding value
- DSBMM here uses real topics as topic metadata -- as such, metadata that
 is not indep from this (eg if include subjectarea codings as separate metadata)
 may actually render the substitutes less effective in specifically capturing
 the sim/real topics, and rendering them cond. ind., even if it provides a better
 model for the full dataset overall -- instead here by default we ONLY use the
 topics, but others may be added using the meta_choice flag
"""

import os
import pickle
import sys
import time
from pathlib import Path

import dsbmm_bp.data_processor as dsbmm_data_proc
import numpy as np
from absl import app, flags
from tqdm import tqdm

# local modules
from pif_dsbmm_dpf.citation import utils
from pif_dsbmm_dpf.citation_real.process_real import CitationProcessor

# from pif_dsbmm_dpf.model import joint_factor_model as joint
from pif_dsbmm_dpf.model import multi_cause_influence as causal

# from pif_dsbmm_dpf.model import network_model as nm
# from pif_dsbmm_dpf.model import pmf as pmf
# from pif_dsbmm_dpf.model import spf as spf


def post_process_influence(X, Beta):
    for t, X_t in enumerate(X):
        total_X = X_t.sum(axis=1)
        no_X = total_X == 0
        Beta[no_X, t] = 1.0
    return Beta


def main(argv):
    # TODO:
    # -- just set both facs to 20 for now
    # -- init all hat params to zeros
    # -- remove all 'true' data and oracle models
    # -- make write unique for each model, but now don't
    #    need to worry about other configs
    # -- save all params now not just beta
    # -- set up for allowing transductive inference,
    #    i.e. hold out part of last time period, infer suitable
    #    subs INCL using rest of last time period, then use
    #    label prop to infer labels for held out part and estim
    #    performance
    # -- get running
    # -- say in paper that possible reason for poor performance
    #    of DSBMM is that it is picking up actual topics, rather
    #    than simulated -- i.e. the simulated topics don't accurately
    #    capture the real topics, so the substitutes are not as useful,
    #    because of greater correlation between net and influence + topics
    #    than in PIF case
    # -- possibly try and simulate influence according to e.g. degree
    #    and see how this changes results

    datadir = Path(FLAGS.data_dir)

    outdir = Path(FLAGS.out_dir)
    outdir.mkdir(exist_ok=True)
    model = FLAGS.model
    variant = FLAGS.variant

    # if not os.path.exists(outdir):
    # 	os.makedirs(outdir)

    Q = FLAGS.num_components
    K = FLAGS.num_exog_components
    seed = FLAGS.seed

    use_old_subs = True
    try_pres_subs = True

    meta_choices = FLAGS.meta_choice
    if meta_choices == "topics_only":
        meta_choices = ["tpc_"]
    elif meta_choices == "all":
        meta_choices = None
    else:
        meta_choices = meta_choices.split(",")
    edge_weight_choice = FLAGS.edge_weight_choice
    # extra_str = "old_subs" if use_old_subs else "upd_subs"
    # using trans at least from hier model seems to make worse
    # but can improve by using pres subs
    extra_str = "_pres_subs" if try_pres_subs else "_old_subs"
    extra_str += f"_ewc{edge_weight_choice}"
    if edge_weight_choice == "none":
        edge_weight_choice = None

    datetime_str = time.strftime("%d-%m_%H-%M", time.gmtime(time.time()))
    if seed is not None:
        data_model_str = f"real_seed{seed}"
    else:
        data_model_str = f"real_{datetime_str}"

    window_len = 3  # set window length for dPF

    # specify main code directory
    main_code_dir = Path("~/Documents/main_project/post_confirmation/code").expanduser()
    # NB if using dPF this should be the location which contains the dPF
    # repo (i.e. DynamicPoissonFactorization dir with code contained)

    print("Model:", model)
    print("Variant:", variant)
    print("Subs:", "pres" if try_pres_subs else "old_subs")

    data_model_path = datadir / f"{data_model_str}.pkl"

    write = outdir / (model + "." + variant + extra_str + "_model_fitted_params")
    write.mkdir(exist_ok=True)

    if seed is not None:
        try:
            with open(data_model_path, "rb") as f:
                data_model: CitationProcessor = pickle.load(f)
            Y, Y_heldout, full_A_end = (
                data_model.Y,
                data_model.Y_heldout,
                data_model.full_A_end,
            )
            tqdm.write(f"Loading prev data subset of model w same seed and configs...")
        except (FileNotFoundError, AttributeError):
            tqdm.write(
                f"Previous data subset w given seed and configs not found, creating new data subset..."
            )
            data_model = CitationProcessor(
                datapath=datadir,
                subnetwork_size=8000,
                sub_testsize=300,
                num_topics=1000,
                test_prop=0.2,
                save_path=data_model_path,
            )
            try:
                Y, Y_heldout, full_A_end = data_model.process_dataset()
            except FileNotFoundError:
                try:
                    dsbmm_datadir = datadir / "dsbmm_data"
                    dsbmm_data = dsbmm_data_proc.load_data(dsbmm_datadir)
                    meta_tpc_col = [
                        mn for mn in dsbmm_data["meta_names"] if mn.startswith("tpc_")
                    ][0]
                    dsbmm_data_proc.save_to_pif_form(
                        dsbmm_data["A"],
                        dsbmm_data["X"],
                        datadir,
                        dsbmm_data["meta_names"],
                        tpc_col_id=meta_tpc_col,
                        synth=False,
                    )
                    del dsbmm_data
                    Y, Y_heldout, full_A_end = data_model.process_dataset()
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "Data in suitable form for either PIF directly, or DSBMM, not found in specified directory."
                    )

    A = data_model.A
    print(f"Adj. size and mean: {A[0].shape}, {[f'{A_t.mean():.3g}' for A_t in A]}")
    print(f"T: {len(A)}")

    outfile = write / "all_params.npz"
    try:
        tmp = np.load(outfile)
        assert not np.isnan(tmp["Beta_hat"]).any()
        tqdm.write("Skipping this config as already done.")
        return
    except (FileNotFoundError, AssertionError):
        tqdm.write("Starting procedure...")

    sys.stdout.flush()

    N = Y[0].shape[0]
    M = Y[0].shape[1]
    T = len(Y)

    Beta_hat = np.zeros((N, T - 1))
    Z_hat = np.zeros((N, T - 1, Q))
    Gamma_hat = np.zeros((M, T - 1, Q))
    Alpha_hat = np.zeros((N, T - 1, K))
    W_hat = np.zeros((M, T - 1, K))

    if model == "unadjusted":
        m = causal.CausalInfluenceModel(
            n_components=Q,
            n_exog_components=K,
            verbose=True,
            model_mode="influence_only",
        )

    elif model == "network_pref_only":
        m = causal.CausalInfluenceModel(
            n_components=Q,
            n_exog_components=K,
            verbose=True,
            model_mode="network_preferences",
        )

    elif model == "topic_only":
        m = causal.CausalInfluenceModel(
            n_components=Q,
            n_exog_components=K,
            verbose=True,
            model_mode="topic",
        )

    elif model == "dsbmm_dpf":
        m = causal.CausalInfluenceModel(
            n_components=Q + K,
            n_exog_components=K,
            verbose=True,
            model_mode="full",
        )

    if model == "unadjusted":
        m.fit(Y[1:], A, Z_hat, W_hat, Y[:-1])

    else:
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
            data_model.aus,
            sim_id=data_model_str,
            datetime_str=datetime_str,
            sim_tpcs=Y,
            window_len=window_len,
            split_test=False,
        )

        dpf_results_dir = datadir / "dpf_results"
        dpf_results_dir.mkdir(exist_ok=True)
        os.chdir(str(dpf_results_dir))
        dpf_settings = {
            "-n": N,
            "-m": M,
            "-dir": str(dpf_subdir),
            "-rfreq": 5,  # check ll every 5 iterations
            "-vprior": 10,  # prior on variance for transitions
            "-num_threads": 16,  # number of threads to use
            "-tpl": window_len,  # gap between time periods
            # -- assume passing time in years, so this
            # is window length in years
            "-max-iterations": 10,  # max EM iterations
            # - 10 usually enough
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
                    dsbmm_datadir / f"{data_model_str}_dsbmm_data.pkl", "rb"
                ) as f:
                    dsbmm_data = pickle.load(f)
            except FileNotFoundError:
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
                    data_model.aus,
                    T,
                    sim_tpcs=Y,
                    meta_choices=meta_choices,
                    remove_final=False,
                    save_path=dsbmm_datadir / f"{data_model_str}_dsbmm_data.pkl",
                )
        dsbmm_res_str = f"{data_model_str}_{'dc' if deg_corr else 'ndc'}_{'dir' if directed else 'undir'}_{'meta' if variant=='z-theta-joint' else 'nometa'}"
        dpf_res_name = f"{data_model_str}_{variant}.pkl"

        if model == "network_pref_only":
            # only run DSBM (no meta)
            try:
                with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb") as f:
                    Z_hat, Z_trans, block_probs = pickle.load(f)
                Z_hat, Z_trans, block_probs = utils.clean_dsbmm_res(
                    Q, Z_hat_joint, Z_trans, block_probs=block_probs
                )
                Z_hat, Z_trans = utils.verify_dsbmm_results(Q, Z_hat, Z_trans)
                tqdm.write("Loaded DSBMM results for given config")
            except (FileNotFoundError, AssertionError):
                # only run if not already done
                tqdm.write("Running DSBMM")
                h_l = 2
                tqdm.write(
                    f"using settings h_Q={np.round(np.exp(np.log(Q) / h_l)).astype(int)}, N={dsbmm_data['A'][0].shape[0]}, T-1={len(dsbmm_data['A'])}"
                )
                Z_hat, Z_trans, block_probs = utils.run_dsbmm(
                    dsbmm_data,
                    dsbmm_datadir,
                    Q,
                    ignore_meta=True,
                    datetime_str=dsbmm_res_str,
                    deg_corr=deg_corr,
                    directed=directed,
                    ret_block_probs=True,
                )
                Z_hat, Z_trans, block_probs = utils.clean_dsbmm_res(
                    Q, Z_hat, Z_trans, block_probs=block_probs
                )
                with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb") as f:
                    pickle.dump((Z_hat, Z_trans, block_probs), f)
            if Z_hat.shape[1] == T:
                if try_pres_subs:
                    # use present subs inferred where possible
                    Z_hat = Z_hat[:, 1:, :]
                    # W_hat[:,:-1,:] = W_hat[:,1:,:]
            else:
                if try_pres_subs:
                    # change to present subs inferred where possible
                    Z_hat[:, :-1, :] = Z_hat[:, 1:, :]
                    # W_hat[:,:-1,:] = W_hat[:,1:,:]

            m.fit(
                Y[1:],
                A,
                Z_hat,
                W_hat,
                Y[:-1],
                Z_trans=Z_trans,
                use_old_subs=use_old_subs,
            )
        elif model == "topic_only":
            # only run dPF
            try:
                with open(dpf_results_dir / dpf_res_name, "rb") as f:
                    W_hat, Theta_hat = pickle.load(f)
                assert W_hat.shape[-1] == K
                tqdm.write("Loaded dPF results for given config")
            except (FileNotFoundError, AssertionError):
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
            Z_trans = np.ones((Q, Q)) / Q
            if W_hat.shape[1] == T:
                if try_pres_subs:
                    # change to present subs inferred where possible
                    # Z_hat[:,:-1,:] = Z_hat[:,1:,:]
                    W_hat = W_hat[:, 1:, :]
            else:
                if try_pres_subs:
                    # change to present subs inferred where possible
                    # Z_hat[:,:-1,:] = Z_hat[:,1:,:]
                    W_hat[:, :-1, :] = W_hat[:, 1:, :]
            m.fit(
                Y[1:],
                A,
                Z_hat,
                W_hat,
                Y[:-1],
                Z_trans=Z_trans,
                use_old_subs=use_old_subs,
            )

        elif model == "dsbmm_dpf":
            if variant == "z-theta-joint":
                # 'z-theta-joint' is DSBMM and dPF combo
                try:
                    with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb") as f:
                        Z_hat_joint, Z_trans, block_probs = pickle.load(f)
                    Z_hat_joint, Z_trans, block_probs = utils.clean_dsbmm_res(
                        Q, Z_hat_joint, Z_trans, block_probs=block_probs
                    )
                    Z_hat_joint, Z_trans = utils.verify_dsbmm_results(
                        Q, Z_hat_joint, Z_trans
                    )
                    tqdm.write("Loaded DSBMM results for given config")
                except (FileNotFoundError, AssertionError):
                    # only run if not already done
                    tqdm.write("Running DSBMM")
                    h_l = 2
                    tqdm.write(
                        f"using settings h_Q={np.round(np.exp(np.log(Q) / h_l)).astype(int)}, N={dsbmm_data['A'][0].shape[0]}, T-1={len(dsbmm_data['A'])}"
                    )
                    Z_hat_joint, Z_trans, block_probs = utils.run_dsbmm(
                        dsbmm_data,
                        dsbmm_datadir,
                        Q,
                        ignore_meta=False,
                        datetime_str=dsbmm_res_str,
                        deg_corr=deg_corr,
                        directed=directed,
                        ret_block_probs=True,
                    )
                    with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb") as f:
                        pickle.dump((Z_hat_joint, Z_trans, block_probs), f)
                # now run dPF - likewise have set seed so should
                # be identical between runs
                try:
                    with open(dpf_results_dir / dpf_res_name, "rb") as f:
                        W_hat, Theta_hat = pickle.load(f)
                    assert W_hat.shape[-1] == K
                    tqdm.write("Loaded dPF results for given config")
                except (FileNotFoundError, AssertionError):
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
                # 'theta-only' is just dPF, but aiming
                # to try per-author subs. as well
                try:
                    with open(dpf_results_dir / dpf_res_name, "rb") as f:
                        W_hat, Theta_hat = pickle.load(f)
                    assert W_hat.shape[-1] == K
                    tqdm.write("Loaded dPF results for given config")
                except (FileNotFoundError, AssertionError):
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
                    with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb") as f:
                        Z_hat, Z_trans, block_probs = pickle.load(f)
                    Z_hat, Z_trans, block_probs = utils.clean_dsbmm_res(
                        Q, Z_hat, Z_trans, block_probs=block_probs
                    )
                    Z_hat, Z_trans = utils.verify_dsbmm_results(Q, Z_hat, Z_trans)
                    tqdm.write("Loaded DSBM results for given config")
                except (FileNotFoundError, AssertionError):
                    tqdm.write("Running DSBM (no meta)")
                    Z_hat, Z_trans, block_probs = utils.run_dsbmm(
                        dsbmm_data,
                        dsbmm_datadir,
                        Q,
                        ignore_meta=True,
                        datetime_str=dsbmm_res_str,
                        deg_corr=deg_corr,
                        directed=directed,
                        ret_block_probs=True,
                    )
                    Z_hat, Z_trans, block_probs = utils.clean_dsbmm_res(
                        Q, Z_hat, Z_trans, block_probs=block_probs
                    )
                    with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb") as f:
                        pickle.dump((Z_hat, Z_trans, block_probs), f)
                try:
                    with open(dpf_results_dir / dpf_res_name, "rb") as f:
                        W_hat, Theta_hat = pickle.load(f)
                    assert W_hat.shape[-1] == K
                    tqdm.write("Loaded dPF results for given config")
                except (FileNotFoundError, AssertionError):
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
                    with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "rb") as f:
                        Z_hat, Z_trans, block_probs = pickle.load(f)
                    Z_hat, Z_trans, block_probs = utils.clean_dsbmm_res(
                        Q, Z_hat, Z_trans, block_probs=block_probs
                    )
                    Z_hat, Z_trans = utils.verify_dsbmm_results(Q, Z_hat, Z_trans)

                    tqdm.write("Loaded DSBM results for given config")
                except (FileNotFoundError, AssertionError):
                    tqdm.write("Running DSBM (no meta)")
                    Z_hat, Z_trans, block_probs = utils.run_dsbmm(
                        dsbmm_data,
                        dsbmm_datadir,
                        Q,
                        ignore_meta=True,
                        datetime_str=dsbmm_res_str,
                        deg_corr=deg_corr,
                        directed=directed,
                        ret_block_probs=True,
                    )
                    Z_hat, Z_trans, block_probs = utils.clean_dsbmm_res(
                        Q, Z_hat, Z_trans, block_probs=block_probs
                    )
                    with open(dsbmm_datadir / f"{dsbmm_res_str}_subs.pkl", "wb") as f:
                        pickle.dump((Z_hat, Z_trans, block_probs), f)
                try:
                    with open(dpf_results_dir / dpf_res_name, "rb") as f:
                        W_hat, Theta_hat = pickle.load(f)
                    assert W_hat.shape[-1] == K
                    tqdm.write("Loaded dPF results for given config")
                except (FileNotFoundError, AssertionError):
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
            if Z_hat.shape[1] == T:
                if try_pres_subs:
                    Z_hat = Z_hat[:, 1:, :]
                else:
                    Z_hat = Z_hat[:, :-1, :]
            if W_hat.shape[1] == T:
                if try_pres_subs:
                    W_hat = W_hat[:, 1:, :]
                else:
                    W_hat = W_hat[:, :-1, :]

            if variant == "z-only":
                Rho_hat[:, :, :Q] = Z_hat
            elif variant == "theta-only":
                Rho_hat[:, :, :K] = Theta_hat
            elif variant == "z-theta-concat":
                Rho_hat[:, :, :Q] = Z_hat
                Rho_hat[:, :, Q:] = Theta_hat
            else:
                # z-theta-joint
                if Z_hat_joint.shape[1] == T:
                    if try_pres_subs:
                        Z_hat_joint = Z_hat_joint[:, 1:, :]
                    else:
                        Z_hat_joint = Z_hat_joint[:, :-1, :]
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

    Beta_hat = m.E_beta
    # now that calculation is done, when ranking should not count
    # authors with no citations / no papers -- for both beta,
    # these would give beta = 1.0, but if many of them then will
    # artificially inflate the score -- scores below handle this
    # scores = get_set_overlap(Beta_p, Beta)
    # loss = utils.mse(Beta, Beta_p, for_beta=True)

    tqdm.write(f"Mean inferred infl: {Beta_hat[Beta_hat!=1].mean():.3g}")

    # tqdm.write(f"Overlaps: {np.round(scores,3)}, \nMSE: {np.round(loss,3)}")
    tqdm.write(f"{'*' * 60}")
    sys.stdout.flush()
    if (model == "dsbmm_dpf") or (model == "network_pref_only"):
        Gamma_hat = m.E_gamma

    if model == "dsbmm_dpf" or model == "topic_only":
        Alpha_hat = m.E_alpha

    # if model == 'spf':
    # 	Z_hat = m.E_alpha

    if model == "dsbmm_dpf":
        Z_hat = Z_hat_joint

    np.savez_compressed(
        outfile,
        Z_hat=Z_hat,
        W_hat=W_hat,
        Alpha_hat=Alpha_hat,
        Gamma_hat=Gamma_hat,
        Beta_hat=Beta_hat,
    )
    tqdm.write(f"Saved results to {outfile}")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "model",
        "dsbmm_dpf",
        """
        method to use selected from one of
        [
            dsbmm_dpf, unadjusted, topic_only, network_pref_only
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
    # -- author pubs in au_pubs.pkl
    # where files are as described in process_real.py
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
        "/scratch/fitzgeraldj/data/caus_inf_data/real_results",
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

    flags.DEFINE_integer(
        "num_components",
        25,
        """
        number of components to use to fit factor model for
        per-author substitutes, default 16
        """,
    )
    flags.DEFINE_integer(
        "num_exog_components",
        20,
        """
        number of components to use to fit factor model for
        per-topic substitutes, default 10
        """,
    )
    flags.DEFINE_integer(
        "seed", 42, "random seed passed to simulator in each experiment, default 42"
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
        "none",
        """
        Specify edge weight name to use, e.g. 'count' or 'weighted'
        for edge weights in network. Default is 'none' (binary),
        pass 'count' to use weighted network""",
    )
    flags.DEFINE_bool(
        "use_old_subs",
        True,
        "Use old substitutes (i.e. those from previous time period) in DSBMM",
    )  # NB this will require being passed as either
    # --flag (meaning true), or --noflag (meaning false)
    # if passing explicitly for flag
    flags.DEFINE_bool(
        "try_pres_subs",
        True,
        """
        Try using subs from DSBMM for present time period if available, else use old subs.
        NB can be used in conjunction with use_old_subs, in which case would apply transition
        to the pres subs if available, else the old subs, then use the result as the subs
        """,
    )  # NB this will require being passed as either

    app.run(main)
