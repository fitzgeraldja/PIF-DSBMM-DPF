import pickle
import subprocess
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from typing_extensions import TypedDict

tqdm.pandas()

import dsbmm_bp.apply as dsbmm_apply


def sample_simple_markov(N: int, T: int, Q: int, eta: float, onehot=False):
    """Generate N discrete Markov chains over Q categories,
    length T, where

    p(x_i^t = q | x_i^{t-1} = q') = eta + (1-eta)/Q if q = q'
                                  = (1-eta)/Q otherwise

    :param N: Number of Markov chains to generate
    :type N: int
    :param Q: Number of discrete states
    :type Q: int
    :param eta: probability of assigning same state as previous state,
                else uniformly random sample from all states
    :type eta: float
    :param onehot: return one-hot (N,T,Q) array, else (N,T) array
                    of integers {0,...,Q-1}
    """
    Z = np.zeros((N, T), dtype=np.int32)
    Z[:, 0] = np.random.randint(0, Q, size=N)
    for t in range(1, T):
        rands = np.random.rand(N)
        stay_idxs = rands < eta
        Z[stay_idxs, t] = Z[stay_idxs, t - 1]
        Z[~stay_idxs, t] = np.random.randint(0, Q, size=N)
    if onehot:
        Z_onehot = np_to_onehot(Z)
        return Z_onehot
    else:
        return Z


def np_to_onehot(arr):
    """Take numpy array of integers corresponding to category
    idxs and convert to one-hot representation

    :param arr: int array of category indexes, of whatever shape
    :type arr: np.ndarray
    :return: one_hot representation of arr,
             shape (*arr.shape, arr.max()+1)
    :rtype: np.ndarray
    """
    tmp = arr.copy()
    tmp = tmp.astype(np.int32)
    Q = tmp.max() + 1
    out = np.zeros((*tmp.shape, Q), dtype=np.int32)
    np.put_along_axis(out, tmp[..., np.newaxis], 1, axis=-1)
    return out


def get_dpf_res_dir(dpf_settings):
    ret_dir = "-".join(
        [
            "n" + str(dpf_settings["-n"]),
            "m" + str(dpf_settings["-m"]),
            "k" + str(dpf_settings["-k"]),
            # assume not passing label
            # if (label != "")
            #     sa << "-" << label;
            # assume datfname cdn not met
            # else if (datfname.length() > 3) {
            #     string q = datfname.substr(0,2);
            #     if (isalpha(q[0]))
            #     sa << "-" << q;
            # }
            # assume only performing batch inference
            # if (batch)
            #     sa << "-batch";
            "batch",
            # assume only fixing to binary data
            # if (binary_data)
            #     sa << "-bin";
            "bin",
            # assume only using default prior
            # if (normal_priors)
            #     sa << "-normpriors";
            "normpriors",
            # assume not using dynamic item repr
            # if (dynamic_item_representations)
            #     sa << "-dynitemrep";
            # assume using default dui repr
            # if (dynamic_user_and_item_representations)
            #     sa << "-dui";
            "dui",
            "nthreads" + str(dpf_settings["-nthreads"]),
            # assume not fixing item params
            # if (fixed_item_param)
            #     sa << "-fip";
            # assume not using pf_init
            # if (pf_init)
            #     sa << "-pf_init";
            # assume not using static pf_init
            # if (pf_init_static)
            #     sa << "-pf_init_static";
            # assume not using normreps
            # if (normalized_representations)
            #     sa << "-normrep";
            "vprior" + str(dpf_settings["-vprior"]),
            "seed" + str(dpf_settings["-seed"]),
            "tpl" + str(dpf_settings["-tpl"]),
            "correction",
        ]
    )
    return ret_dir


def run_dpf(dpf_repo_dir, dpf_results_dir, dpf_settings):
    subprocess.run(
        [
            str(dpf_repo_dir / "src/dynnormprec"),
            *chain.from_iterable(dpf_settings.items()),
        ]
    )
    # now need to collect results from file
    run_res_dir = dpf_results_dir / get_dpf_res_dir(dpf_settings)
    # dPF saves item (topic) factors as beta,
    # and user (author) factors as theta
    # -- idea of using theta would be as a
    # additional author preference feature,
    # which also influences their pub topics,
    # but as we are using DSBMM with the
    # network to generate such features
    # this may not be necessary / could
    # deteriorate performance
    beta_files = sorted(
        run_res_dir.glob("beta_*[0-9].tsv"),
        key=lambda x: int(str(x.stem).split("_")[-1]),
    )
    W_hat = np.stack(
        list(
            map(
                lambda fname: pd.read_csv(
                    fname,
                    header=None,
                    names=["idx0", "idx1", *list(range(dpf_settings["-k"]))],
                    usecols=lambda name: "idx" not in str(name),
                    sep="\t",
                ).values,
                beta_files,
            )
        ),
        axis=1,
    )  # in shape (M, T, K), as dPF
    glob_beta_file = run_res_dir / "beta.tsv"
    glob_beta = pd.read_csv(
        glob_beta_file,
        header=None,
        names=["idx0", "idx1", *list(range(dpf_settings["-k"]))],
        usecols=lambda name: "idx" not in str(name),
        sep="\t",
    ).values
    W_hat += glob_beta[:, np.newaxis, :]
    # likewise now need to collect theta
    theta_files = sorted(
        run_res_dir.glob("theta_*[0-9].tsv"),
        key=lambda x: int(str(x.stem).split("_")[-1]),
    )
    Theta_hat = np.stack(
        list(
            map(
                lambda fname: pd.read_csv(
                    fname,
                    header=None,
                    names=["idx0", "idx1", *list(range(dpf_settings["-k"]))],
                    usecols=lambda name: "idx" not in str(name),
                    sep="\t",
                ).values,
                beta_files,
            )
        ),
        axis=1,
    )  # in shape (N, T, K)
    glob_theta_file = run_res_dir / "theta.tsv"
    glob_theta = pd.read_csv(
        glob_theta_file,
        header=None,
        names=["idx0", "idx1", *list(range(dpf_settings["-k"]))],
        usecols=lambda name: "idx" not in str(name),
        sep="\t",
    ).values
    Theta_hat += glob_theta[:, np.newaxis, :]

    return W_hat, Theta_hat


class Dsbmm_datatype(TypedDict):
    meta_names: list[str]
    meta_types: list[str]
    A: list[sparse.csr_array]
    X: list[np.ndarray]
    Q: int


def subset_dsbmm_data(
    data: Dsbmm_datatype,
    subset_idxs: np.ndarray,
    T: int,
    meta_choices: Optional[list[str]] = None,
    remove_final=True,
):
    # N = data["A"][0].shape[0]
    # bin_subset = np.arange(N)
    # bin_subset = np.isin(bin_subset, subset_aus)
    data["A"] = [A_t[np.ix_(subset_idxs, subset_idxs)] for A_t in data["A"]]
    if meta_choices is not None:
        chosen_meta = [
            mn
            for mn in data["meta_names"]
            if mn.startswith(mc)  # type:ignore
            for mc in meta_choices
        ]
        data["X"] = [
            X_s for mn, X_s in zip(data["meta_names"], data["X"]) if mn in chosen_meta
        ]
    data["X"] = [X_s[subset_idxs, ...] for X_s in data["X"]]
    if remove_final:
        # remove knowledge of final timestep
        if len(data["A"]) == T:
            data["A"] = data["A"][:-1]
        if data["X"][0].shape[1] == T:
            data["X"] = [X_s[:, :-1, :] for X_s in data["X"]]
    return data


def get_dpf_data(
    dpf_datadir: Path,
    subset_idxs: np.ndarray,
    seed: Optional[int] = None,
    datetime_str: Optional[str] = None,
):
    subdir_str = f"seed{seed}" if seed is not None else f"init:{datetime_str}"
    subdir = dpf_datadir / subdir_str
    end_names = ["train.tsv", "validation.tsv", "test.tsv"]
    sub_fnames = list(map(lambda x: subdir / x, end_names))
    try:
        dpf_train, dpf_val, dpf_test = map(
            lambda fname: pd.read_csv(
                fname,
                sep="\t",
                header=None,
                names=["auid_idx", "tpc_idx", "count", "windowed_year"],
            ),
            sub_fnames,
        )
    except FileNotFoundError:
        tqdm.write(f"No preexisting dpf subset data found for {subdir_str}")
        tqdm.write("Generating...")
        subdir.mkdir(exist_ok=True)
        dpf_train, dpf_val, dpf_test = gen_subset_dpf(
            dpf_datadir,
            subset_idxs,
            subdir,
        )
        tqdm.write("Done.")
    tqdm.write(
        f"train, val, test contain {len(dpf_train)}, {len(dpf_val)}, {len(dpf_test)} records resp."
    )
    tot_records = len(dpf_train) + len(dpf_val) + len(dpf_test)
    tqdm.write(
        f"""which corresponds to
        {len(dpf_train)/tot_records:.2g},
        {len(dpf_val)/tot_records:.2g},
        {len(dpf_test)/tot_records:.2g}
        split"""
    )
    with open(subdir / "au_n_tpc_maps.pkl", "rb") as f:
        dpf_au_idx_map, dpf_tpc_idx_map = pickle.load(f)

    return dpf_train, dpf_val, dpf_test, dpf_au_idx_map, dpf_tpc_idx_map


def gen_subset_dpf(
    dpf_datadir: Path,
    subset_idxs: np.ndarray,
    out_dir: Path,
    min_train_N: int = 1000,
    min_val_N: int = 100,
    min_test_N: int = 300,
):
    """Assume dpf_datadir contains all available data in correct format,
    so all that is required is subsetting and saving to new dir

    :param dpf_datadir: path to dir containing all data, in correct format
    :type dpf_datadir: Path
    :param subset_idxs: subset of idx column sampled
    :type subset_idxs: np.ndarray
    :param out_dir: dir to save subset data to
    :type out_dir: Path
    :param min_train_N: minimum number of author records in train set, defaults to 1000
    :type min_train_N: int, optional
    :param min_val_N: minimum number of author records in val set, defaults to 100
    :type min_val_N: int, optional
    :param min_test_N: minimum number of author records in test set, defaults to 300
    :type min_test_N: int, optional
    """
    # dPF expects data tsvs in form
    # auid_idx, topic_idx, n_pubs, time_idx
    # in three separate files
    # -- train.tsv, validation.tsv, test.tsv

    end_names = ["train.tsv", "validation.tsv", "test.tsv"]
    in_fnames = list(map(lambda x: dpf_datadir / x, end_names))
    out_fnames = list(map(lambda x: out_dir / x, end_names))
    all_train, all_val, all_test = map(
        lambda fname: pd.read_csv(
            fname,
            sep="\t",
            header=None,
            names=["auid_idx", "tpc_idx", "count", "windowed_year"],
        ),
        in_fnames,
    )
    dpf_train, dpf_val, dpf_test = map(
        lambda x: x[np.isin(x.auid_idx.values, subset_idxs)],
        [
            all_train,
            all_val,
            all_test,
        ],
    )
    # require that val,test aus + tpcs were present in train set
    train_aus = np.unique(dpf_train.auid_idx.values)
    train_tpcs = np.unique(dpf_train.tpc_idx.values)
    dpf_val = dpf_val[
        np.isin(dpf_val.auid_idx.values, train_aus)
        & np.isin(dpf_val.tpc_idx.values, train_tpcs)
    ]
    dpf_test = dpf_test[
        np.isin(dpf_test.auid_idx.values, train_aus)
        & np.isin(dpf_test.tpc_idx.values, train_tpcs)
    ]
    if len(dpf_train) < min_train_N:
        tqdm.write(
            f"Warning, train set likely too small: only {len(dpf_train)} records"
        )
    if len(dpf_val) < min_val_N:
        tqdm.write(f"Warning, val set likely too small: only {len(dpf_val)} records")
    if len(dpf_test) < min_test_N:
        tqdm.write(f"Warning, test set likely too small: only {len(dpf_test)} records")

    # now need to reindex aus, tpcs for dPF to work
    au_idx_map = dict(zip(train_aus, np.arange(len(train_aus))))
    tpc_idx_map = dict(zip(train_tpcs, np.arange(len(train_tpcs))))

    dpf_train, dpf_val, dpf_test = map(
        lambda df: df.replace({"auid_idx": au_idx_map, "tpc_idx": tpc_idx_map}),
        [dpf_train, dpf_val, dpf_test],
    )
    # save reidx maps in case this doesn't happen elsewhere
    with open(out_dir / "au_n_tpc_maps.pkl", "wb") as f:
        pickle.dump([au_idx_map, tpc_idx_map], f)
    # finally write to new files
    for fname, df in zip(out_fnames, [dpf_train, dpf_val, dpf_test]):
        df.to_csv(fname, sep="\t", header=False, index=False)

    return dpf_train, dpf_val, dpf_test


def run_dsbmm(
    dsbmm_data: Dsbmm_datatype,
    dsbmm_datadir: Path,
    Q: int,
    ignore_meta=False,
):
    dsbmm_data["Q"] = Q
    Tm1 = len(dsbmm_data["A"])
    N = dsbmm_data["A"][0].shape[0]
    h_l = 2  # 2 layers

    # # TODO: consider further sorting dsbmm meta here
    # meta_names = dsbmm_data["meta_names"]

    dsbmm_settings: dict[str, Union[None, bool, int, float, str]] = dict(
        ret_best_only=True,
        h_l=h_l,
        max_trials=None,
        n_runs=1,
    )
    pred_Z = dsbmm_apply.init_pred_Z(N, Tm1, **dsbmm_settings)
    # args.h_l, default=None, max. no. layers in hier
    # args.h_Q, default=8, max. no. groups at hier layer,
    # = 4 if h_Q > N_l / 4
    # args.h_min_N, default=20, min. nodes for split
    dsbmm_settings.update(
        dict(
            num_groups=None,
            h_Q=np.round(np.exp(np.log(Q) / h_l)).astype(int),
            h_min_N=10,
            min_Q=None,
            max_Q=None,
        )
    )
    trial_Q_settings = {k: v for k, v in dsbmm_settings.items() if k != "ret_best_only"}
    trial_Qs = dsbmm_apply.init_trial_Qs(**trial_Q_settings)
    if (
        not (
            dsbmm_settings["min_Q"] is not None
            or dsbmm_settings["max_Q"] is not None
            or dsbmm_settings["max_trials"] is not None
        )
        and dsbmm_settings["h_l"] is not None
    ):
        dsbmm_settings["ret_best_only"] = dsbmm_settings["n_runs"] == 1
        dsbmm_settings["n_runs"] = 1

    hierarchy_layers, RESULTS_DIR = dsbmm_apply.prepare_for_run(
        dsbmm_data, dsbmm_datadir, trial_Qs, h_l=h_l
    )
    dsbmm_settings.update(
        dict(
            verbose=True,
            link_choice="count",
            tuning_param=1.0,
            learning_rate=0.2,
            patience=5,
            deg_corr=True,
            directed=True,
            max_iter=100,
            max_msg_iter=30,
            ignore_meta=ignore_meta,
            alpha_use_first=False,
            partial_informative_dsbmm_init=True,
            planted_p=0.7,
            auto_tune=True,
        )
    )
    pred_Z, node_probs, pi = dsbmm_apply.run_hier_model(
        "pif_dsbmm",
        dsbmm_data,
        N,
        Tm1,
        pred_Z,
        trial_Qs,
        hierarchy_layers,
        RESULTS_DIR,
        ret_Z=True,
        ret_probs=True,
        ret_trans=True,
        save_to_file=True,
        **dsbmm_settings,
    )
    # TODO: consider trying with one-hot factors
    # from pred_Z also
    # Z_hat(_joint) = node_probs  # in shape (N,T,Q)
    if node_probs.shape[-1] != Q:
        if node_probs.shape[-1] < Q:
            dim_diff = Q - node_probs.shape[-1]
            node_probs = np.pad(
                node_probs, ((0, 0), (0, 0), (0, dim_diff)), mode="constant"
            )
        else:
            main_qs = np.argsort(np.sum(node_probs, axis=(0, 1)))[-Q:]
            node_probs = node_probs[..., main_qs]
            pi = pi[np.ix_(main_qs, main_qs)]
    return node_probs, pi


def mse(true, pred):
    """Return MSE at each timestep between true and pred,
    where time is final dim and shapes otherwise match

    :param true: true values
    :type true: np.ndarray
    :param pred: pred values
    :type pred: np.ndarray
    """
    return np.power(true - pred, 2).mean(axis=tuple(list(range(len(true.shape) - 1))))
