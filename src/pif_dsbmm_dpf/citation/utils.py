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
from dsbmm_bp.data_processor import clean_meta


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
        Z[~stay_idxs, t] = np.random.randint(0, Q, size=(~stay_idxs).sum())
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
            "nthreads" + str(dpf_settings["-num_threads"]),
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


def run_dpf(
    dpf_repo_dir,
    dpf_results_dir,
    dpf_settings,
    idx_map_dir: Path = None,
    true_N: int = None,
    true_M: int = None,
):
    subprocess.run(
        [
            str((dpf_repo_dir / "src") / "dynnormprec"),
            *chain.from_iterable(dpf_settings.items()),
        ]
    )
    # now need to collect results from file
    run_res_dir = dpf_results_dir / get_dpf_res_dir(dpf_settings)
    try:
        assert run_res_dir.exists()
    except AssertionError:
        raise RuntimeError("dPF run failed")
    # dPF saves item (topic) factors as beta,
    # and user (author) factors as theta
    # -- idea of using theta would be as a
    # additional author preference feature,
    # which also influences their pub topics,
    # but as we are using DSBMM with the
    # network to generate such features
    # this may not be necessary / could
    # deteriorate performance
    K = int(dpf_settings["-k"])
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
                    names=["idx0", "idx1", *list(range(K))],
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
        names=["idx0", "idx1", *list(range(K))],
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
                    names=["idx0", "idx1", *list(range(K))],
                    usecols=lambda name: "idx" not in str(name),
                    sep="\t",
                ).values,
                theta_files,
            )
        ),
        axis=1,
    )  # in shape (N, T, K)
    glob_theta_file = run_res_dir / "theta.tsv"
    glob_theta = pd.read_csv(
        glob_theta_file,
        header=None,
        names=["idx0", "idx1", *list(range(K))],
        usecols=lambda name: "idx" not in str(name),
        sep="\t",
    ).values
    Theta_hat += glob_theta[:, np.newaxis, :]

    if idx_map_dir is not None:
        with open(idx_map_dir / "au_n_tpc_maps.pkl", "rb") as f:
            dpf_au_idx_map, dpf_tpc_idx_map = pickle.load(f)
        # now need to map back to original indices
        try:
            assert (true_N is not None) and (true_M is not None)
        except AssertionError:
            raise ValueError(
                "true_N and true_M must be provided if idx_map_dir is provided"
            )
        _, T, K = W_hat.shape
        re_W = np.zeros((true_M, T, K))
        re_Theta = np.zeros((true_N, T, K))
        true_n_idxs = np.array(list(dpf_au_idx_map.keys()))
        new_n_idxs = np.array(list(dpf_au_idx_map.values()))
        true_m_idxs = np.array(list(dpf_tpc_idx_map.keys()))
        new_m_idxs = np.array(list(dpf_tpc_idx_map.values()))
        re_W[true_m_idxs] = W_hat[new_m_idxs]
        re_Theta[true_n_idxs] = Theta_hat[new_n_idxs]
        W_hat = re_W
        Theta_hat = re_Theta

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
    sim_tpcs: Optional[list[sparse.csr_array]] = None,
    meta_choices: Optional[list[str]] = None,
    remove_final=True,
    save_path: Optional[Path] = None,
):
    # N = data["A"][0].shape[0]
    # bin_subset = np.arange(N)
    # bin_subset = np.isin(bin_subset, subset_aus)
    data["A"] = [A_t[np.ix_(subset_idxs, subset_idxs)] for A_t in data["A"]]
    if meta_choices is not None:
        chosen_meta = list(
            {
                mn
                for mc in meta_choices
                for mn in data["meta_names"]
                if mn.startswith(mc)
            }
        )
        data["X"] = [
            X_s for mn, X_s in zip(data["meta_names"], data["X"]) if mn in chosen_meta
        ]
        data["meta_types"] = [
            mt
            for mt, mn in zip(data["meta_types"], data["meta_names"])
            if mn in chosen_meta
        ]
    data["X"] = [X_s[subset_idxs, ...] for X_s in data["X"]]

    if sim_tpcs is not None:
        # NB ordering should already have been done by subsetting, so
        # should be able to just assign directly
        tpc_idxs = [idx for idx, cm in enumerate(chosen_meta) if cm.startswith("tpc")]
        # currently need to make dense for DSBMM
        sim_tpcs = np.stack([X_t.toarray() for X_t in sim_tpcs], axis=1)
        data["X"][tpc_idxs[0]] = sim_tpcs
        if len(tpc_idxs) > 0:
            # remove any other topic features
            for tpc_idx in tpc_idxs[1:]:
                data["X"].pop(tpc_idx)
                data["meta_types"].pop(tpc_idx)
    if remove_final:
        # remove knowledge of final timestep
        if len(data["A"]) == T:
            data["A"] = data["A"][:-1]
        if data["X"][0].shape[1] == T:
            data["X"] = [X_s[:, :-1, :] for X_s in data["X"]]

    # now clean subset of meta according to specified dists
    tmp_meta_dims = [X_s.shape[-1] for X_s in data["X"]]
    max_cats = 30
    tqdm.write("Cleaning metadata of subset")
    if "indep bernoulli" in data["meta_types"]:
        tqdm.write(
            f"fixing that any author has at most {max_cats} categories for IB metadata"
        )
    data["X"] = clean_meta(
        data["meta_names"],
        data["meta_types"],
        tmp_meta_dims,
        data["X"],
        max_cats=max_cats,
    )
    if save_path is not None:
        tqdm.write(f"Saving subsetted DSBMM data to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    return data


def gen_dpf_data(
    dpf_datadir: Path,
    subset_idxs: np.ndarray,
    sim_id: Optional[str] = None,
    datetime_str: Optional[str] = None,
    sim_tpcs: Optional[sparse.csr_array] = None,
    window_len: int = 3,
):
    subdir_str = sim_id if sim_id is not None else f"init:{datetime_str}"
    subdir = dpf_datadir / subdir_str
    subdir.mkdir(exist_ok=True)
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
        if sim_tpcs is None:
            dpf_train, dpf_val, dpf_test = gen_subset_dpf(
                dpf_datadir, subset_idxs, subdir, window_len=window_len
            )
        else:
            dpf_train, dpf_val, dpf_test = convert_to_dpf_format(
                sim_tpcs, subdir, window_len=window_len
            )
        tqdm.write("Done.")
    tqdm.write(
        f"In dPF data, train, val, test contain {len(dpf_train)}, {len(dpf_val)}, {len(dpf_test)} records resp."
    )
    tot_records = len(dpf_train) + len(dpf_val) + len(dpf_test)
    tqdm.write(
        f"""which corresponds to
        {len(dpf_train)/tot_records:.2g},
        {len(dpf_val)/tot_records:.2g},
        {len(dpf_test)/tot_records:.2g}
        split"""
    )
    return subdir


def gen_subset_dpf(
    dpf_datadir: Path,
    subset_idxs: np.ndarray,
    out_dir: Path,
    min_train_N: int = 1000,
    min_val_N: int = 100,
    min_test_N: int = 300,
    window_len: int = 3,
):
    """Assume dpf_datadir contains all available REAL data in correct format,
    so all that is required is subsetting and saving to new dir

    dPF expects data tsvs in form
    auid_idx, topic_idx, n_pubs, time_idx
    in three separate files
    -- train.tsv, validation.tsv, test.tsv

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
    :param window_len: length of window to use for windowed year, defaults to 3
    :type window_len: int, optional
    """

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
    try:
        for df in [all_train, all_val, all_test]:
            assert np.all(np.diff(df.windowed_year.values) == window_len)
    except AssertionError:
        raise ValueError(
            f"Window length in data does not match specified value, {window_len}"
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
    dpf_train = dpf_train.astype(int)
    train_aus = np.unique(dpf_train.auid_idx.values)
    train_tpcs = np.unique(dpf_train.tpc_idx.values)
    dpf_val = dpf_val[
        np.isin(dpf_val.auid_idx.values, train_aus)
        & np.isin(dpf_val.tpc_idx.values, train_tpcs)
    ].astype(int)
    dpf_test = dpf_test[
        np.isin(dpf_test.auid_idx.values, train_aus)
        & np.isin(dpf_test.tpc_idx.values, train_tpcs)
    ].astype(int)
    if len(dpf_train) < min_train_N:
        tqdm.write(
            f"Warning, train set likely too small: only {len(dpf_train)} records"
        )
    if len(dpf_val) < min_val_N:
        tqdm.write(f"Warning, val set likely too small: only {len(dpf_val)} records")
    if len(dpf_test) < min_test_N:
        tqdm.write(f"Warning, test set likely too small: only {len(dpf_test)} records")

    # now need to reindex aus, tpcs for dPF to work
    au_idx_map = dict(zip(train_aus, np.arange(len(train_aus), dtype=int)))
    tpc_idx_map = dict(zip(train_tpcs, np.arange(len(train_tpcs), dtype=int)))

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


def convert_to_dpf_format(
    sim_tpcs: list[sparse.csr_array],
    out_dir: Path,
    min_train_N: int = 1000,
    min_val_N: int = 100,
    min_test_N: int = 300,
    window_len: int = 3,
    val_frac: float = 0.1,
    binarise: bool = True,
):
    """Converts a list of sparse author-topic counts, length T, each shape
    (N,M) to dPF format -- note that by default we assume a window length
    so must multiply time index by this to get in correct form

    dPF expects data tsvs in form
    auid_idx, topic_idx, n_pubs, time_idx
    in three separate files
    -- train.tsv, validation.tsv, test.tsv

    We will assume that all but final time period are train,
    then randomly split the final time period into val and test
    -- does cause some minor data leakage, in that we should overall
    choose model hyperparams that better suit test data than ideal,
    as some val data may be for authors in test set and/or for authors
    closely linked to authors in test set, but this semi-transductive
    'problem' would occur anyway unless greater care taken.

    When applying on real data, we'll manually split into train, val, test,
    then ensure that the validation data is from first year / three in final
    period, and test is remainder -- this is time series approach and should
    be reasonable.

    :param sim_tpcs: simulated topics
    :type sim_tpcs: list[sparse.csr_array]
    :param out_dir: path to dir to save output in
    :type out_dir: Path
    :param window_len: time window length, defaults to 3
    :type window_len: int, optional
    :param val_frac: fraction of final period data to use for validation,
                     remainder (1-test_frac) used for test, defaults to 0.1
    :type val_frac: float, optional
    :param binarise: binarise data, as suggested for dPF, defaults to True
    :type binarise: bool, optional
    :return: dpf_train, dpf_val, dpf_test
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    end_names = ["train.tsv", "validation.tsv", "test.tsv"]
    out_fnames = list(map(lambda x: out_dir / x, end_names))
    dpf_train = pd.DataFrame(
        np.concatenate(
            [
                np.stack(
                    [
                        *tpcs_t.nonzero(),
                        tpcs_t.data,
                        window_len * t * np.ones(tpcs_t.nnz),
                    ],
                    axis=1,
                )
                for t, tpcs_t in enumerate(sim_tpcs[:-1])
            ],
            axis=0,
        ),
        columns=["auid_idx", "tpc_idx", "count", "windowed_year"],
    )
    # now split final time period into val and test
    val_idxs = np.random.rand(len(sim_tpcs[-1].data)) < val_frac
    fin_data = pd.DataFrame(
        np.stack(
            [
                *sim_tpcs[-1].nonzero(),
                sim_tpcs[-1].data,
                window_len * (len(sim_tpcs) - 1) * np.ones(sim_tpcs[-1].nnz),
            ],
            axis=1,
        ),
        columns=["auid_idx", "tpc_idx", "count", "windowed_year"],
    )
    dpf_val = fin_data[val_idxs]
    dpf_test = fin_data[~val_idxs]

    # gen reidx maps again here just in case -- in theory likely to be
    # irrelevant here as data is synthetic, and should already have
    # ensured that authors in final period are present previously, but
    # keep for consistency
    # require that val,test aus + tpcs were present in train set
    dpf_train = dpf_train.astype(int)
    train_aus = np.unique(dpf_train.auid_idx.values)
    train_tpcs = np.unique(dpf_train.tpc_idx.values)
    dpf_val = dpf_val[
        np.isin(dpf_val.auid_idx.values, train_aus)
        & np.isin(dpf_val.tpc_idx.values, train_tpcs)
    ].astype(int)
    dpf_test = dpf_test[
        np.isin(dpf_test.auid_idx.values, train_aus)
        & np.isin(dpf_test.tpc_idx.values, train_tpcs)
    ].astype(int)
    if len(dpf_train) < min_train_N:
        tqdm.write(
            f"Warning, train set likely too small: only {len(dpf_train)} records"
        )
    if len(dpf_val) < min_val_N:
        tqdm.write(f"Warning, val set likely too small: only {len(dpf_val)} records")
    if len(dpf_test) < min_test_N:
        tqdm.write(f"Warning, test set likely too small: only {len(dpf_test)} records")

    # now need to reindex aus, tpcs for dPF to work
    au_idx_map = dict(zip(train_aus, np.arange(len(train_aus), dtype=int)))
    tpc_idx_map = dict(zip(train_tpcs, np.arange(len(train_tpcs), dtype=int)))

    dpf_train, dpf_val, dpf_test = map(
        lambda df: df.replace({"auid_idx": au_idx_map, "tpc_idx": tpc_idx_map}),
        [dpf_train, dpf_val, dpf_test],
    )
    with open(out_dir / "au_n_tpc_maps.pkl", "wb") as f:
        pickle.dump([au_idx_map, tpc_idx_map], f)
    if binarise:
        dpf_train, dpf_val, dpf_test = map(
            lambda df: df.assign(count=1),
            [dpf_train, dpf_val, dpf_test],
        )

    # finally write to new files
    for fname, df in zip(out_fnames, [dpf_train, dpf_val, dpf_test]):
        df.to_csv(fname, sep="\t", header=False, index=False)

    return dpf_train, dpf_val, dpf_test


def run_dsbmm(
    dsbmm_data: Dsbmm_datatype,
    dsbmm_datadir: Path,
    Q: int,
    ignore_meta=False,
    datetime_str=None,
    deg_corr=True,
    directed=True,
    ret_block_probs=False,
    use_1hot_Z=False,
):
    dsbmm_data["Q"] = Q
    Tm1 = len(dsbmm_data["A"])
    N = dsbmm_data["A"][0].shape[0]
    h_l = 2  # 2 layers

    # meta_names = dsbmm_data["meta_names"]
    # : dict[str, Union[None, bool, int, float, str]]
    dsbmm_settings = dict(
        ret_best_only=True,
        h_l=h_l,
        max_trials=None,
        n_runs=1,
        num_groups=None,
        h_Q=np.round(np.exp(np.log(Q) / h_l)).astype(int),
        h_min_N=10,
        min_Q=None,
        max_Q=None,
    )

    pred_Z, trial_Qs = dsbmm_apply.prep_Z_and_Qs(N, Tm1, **dsbmm_settings)
    # args.h_l, default=None, max. no. layers in hier
    # args.h_Q, default=8, max. no. groups at hier layer,
    # = 4 if h_Q > N_l / 4
    # args.h_min_N, default=20, min. nodes for split

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
    rmv_keys = ["num_groups", "max_trials", "min_Q", "max_Q"]
    dsbmm_settings = {k: v for k, v in dsbmm_settings.items() if k not in rmv_keys}
    dsbmm_settings.update(
        dict(
            verbose=False,
            link_choice="count",
            tuning_param=1.0,
            learning_rate=0.2,
            patience=5,
            deg_corr=deg_corr,
            directed=directed,
            max_iter=100,
            max_msg_iter=30,
            ignore_meta=ignore_meta,
            alpha_use_first=False,
            partial_informative_dsbmm_init=True,
            planted_p=0.7,
            auto_tune=True,
            datetime_str=datetime_str,
        )
    )
    res = dsbmm_apply.run_hier_model(
        "pif_dsbmm",
        dsbmm_data,
        N,
        Tm1,
        pred_Z,
        trial_Qs,
        hierarchy_layers,
        RESULTS_DIR,
        ret_Z=use_1hot_Z,
        ret_probs=not use_1hot_Z,
        ret_trans=True,
        ret_block_probs=ret_block_probs,
        save_to_file=True,
        **dsbmm_settings,
    )
    res = list(res)
    if use_1hot_Z:
        pred_Z = res.pop(0)
    else:
        node_probs = res.pop(0)
    # if ret_trans:
    pi = res.pop(0)
    if ret_block_probs:
        block_probs = res.pop(0)

    # TODO: consider trying with one-hot factors
    # from pred_Z also
    # Z_hat(_joint) = node_probs  # in shape (N,T,Q)
    if type(node_probs) == list:
        # only support single run (or ret_best_only) for now
        assert len(node_probs) == 1
        node_probs = node_probs[0]
        pi = pi[0]
    elif type(node_probs) == np.ndarray:
        if len(node_probs.shape) == 4:
            assert len(node_probs) == 1
            node_probs = node_probs[0]
            pi = pi[0]
    if node_probs.shape[-1] != Q:
        if node_probs.shape[-1] < Q:
            dim_diff = Q - node_probs.shape[-1]
            node_probs = np.pad(
                node_probs, ((0, 0), (0, 0), (0, dim_diff)), mode="constant"
            )
        else:
            group_probs = np.nansum(node_probs, axis=(0, 1))
            if np.any(group_probs == 0):
                # empty group idx, which should be removed
                # but also won't be counted for pi, so
                # can just drop
                node_probs = node_probs[:, :, group_probs > 0]
            if node_probs.shape[-1] > Q:
                # actually fit extra groups, so need to remove
                tqdm.write(f"Fit {node_probs.shape[-1]} groups, but only {Q} requested")
                tqdm.write("Removing extra groups")
                main_qs = np.argsort(group_probs)[-Q:]
                node_probs = node_probs[..., main_qs]
                try:
                    pi = pi[np.ix_(main_qs, main_qs)]
                except:
                    print(pi.shape, main_qs.shape, main_qs)
                    print(group_probs)
                    raise ValueError("Problem w pi shape")
    out_res = []
    if use_1hot_Z:
        out_res.append(pred_Z)
    else:
        out_res.append(node_probs)
    out_res.append(pi)
    if ret_block_probs:
        out_res.append(block_probs)
    return out_res


def mse(true, pred, for_beta=False):
    """Return MSE at each timestep between true and pred,
    where time is final dim and shapes otherwise match.

    When used for beta, also account for fact that many
    nodes may be missing, where both should take value 1.0
    but shouldn't be counted in MSE.

    :param true: true values
    :type true: np.ndarray
    :param pred: pred values
    :type pred: np.ndarray
    """
    if not for_beta:
        return np.power(true - pred, 2).mean(
            axis=tuple(list(range(len(true.shape) - 1)))
        )
    else:
        # know shape is (N,T-1)
        Tm1 = true.shape[-1]
        n_true_missing = np.sum(true == 1.0, axis=0)
        n_pred_missing = np.sum(pred == 1.0, axis=0)
        tqdm.write(f"Seems to be {n_true_missing} missing true values")
        tqdm.write(f"and {n_pred_missing} possibly missing pred values")
        mses = np.array(
            [
                np.power(
                    true[:, t] - pred[:, t], 2, where=true[:, t] != pred[:, t]
                ).mean()
                for t in range(Tm1)
            ]
        )


def safe_sparse_toarray(sparse_mat):
    """Convert potentially sparse array to dense

    :param sparse_mat: possible sparse array
    :type sparse_mat: Union[np.ndarray,sparse.csr_array]
    """
    try:
        return sparse_mat.toarray()
    except AttributeError:
        return sparse_mat
