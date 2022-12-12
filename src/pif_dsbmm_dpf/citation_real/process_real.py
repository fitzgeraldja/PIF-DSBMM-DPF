import os
import pickle
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.stats import poisson
from tqdm import tqdm


def label_propagation(
    new_au_idxs, old_au_idxs, full_A, old_node_probs, trans, block_probs, deg_corr=False
):
    # really want to fix params, then use message passing to update,
    # but this is tricky for hierarchical model
    # we'll basically do this, but manually here rather than via
    # DSBMM directly -- will also neglect topics themselves, i.e.
    # p(z_i^T|a_i^T,z_{-i,t}) =
    #               \sum_{z_{-i,T},z_{i,T-1} [
    #                           p(a_i|z_i^T,z_{-i})
    #                           * p(z_i^T|z_i^{T-1})p(z_{-i,T})
    #                           ]
    #                           /p(a_i|z_{-i})
    # know how to norm so don't need to calc p(a_i|z_{-i})
    # so can just calc p(a_i|z_i^T,z_{-i}) and p(z_i^T|z_i^{T-1})
    # so instead now want to sort new au idxs by the number of old
    # au idxs they are connected to, then iterate through them in
    # that order, and use old node probs and block probs to update
    # the new node probs
    # first make new node probs matrix for new and old nodes,
    # over prev and current timestep
    alpha = old_node_probs.mean(axis=(0, 1))
    Q = old_node_probs.shape[-1]
    full_node_probs = np.zeros((full_A.shape[0], 2, Q))
    full_node_probs[old_au_idxs[0], 0, :] = old_node_probs[:, -2, :]
    full_node_probs[old_au_idxs[1], 1, :] = old_node_probs[:, -1, :]
    sub_A_out = full_A[new_au_idxs, :][:, old_au_idxs[1]]
    sub_A_in = full_A[old_au_idxs[1], :][:, new_au_idxs]
    new_d_out = np.sum(sub_A_out, axis=1)
    new_d_in = np.sum(sub_A_in, axis=0)
    new_d_tot = new_d_out + new_d_in
    sorted_new = new_au_idxs[np.argsort(new_d_tot)[::-1]]
    full_d_out = np.sum(full_A, axis=1)
    full_d_in = np.sum(full_A, axis=0)
    full_new_d_out = full_d_out[sorted_new]
    full_new_d_in = full_d_in[sorted_new]
    for new_au_idx, d_in, d_out in zip(
        tqdm(sorted_new, desc="Label prop."), full_new_d_in, full_new_d_out
    ):
        prev_contrib = trans.T @ full_node_probs[new_au_idx, 0, :]
        if np.all(prev_contrib == 0.0):
            prev_contrib = alpha
        prev_contrib = np.log(
            prev_contrib, out=np.zeros_like(prev_contrib), where=prev_contrib != 0
        )
        out_nbrs = full_A[new_au_idx, :].flatnonzero()  # new node sends to
        in_nbrs = full_A[:, new_au_idx].flatnonzero()  # send to new node
        recip = np.intersect1d(out_nbrs, in_nbrs)  # reciprocated
        out_nbrs = np.setdiff1d(out_nbrs, recip)  # now only receive from new node
        in_nbrs = np.setdiff1d(in_nbrs, recip)  # now only send to new node
        out_nbr_d_out = full_d_out[out_nbrs]
        out_nbr_d_in = full_d_in[out_nbrs]
        in_nbr_d_in = full_d_in[in_nbrs]
        in_nbr_d_out = full_d_out[in_nbrs]
        recip_d_in = full_d_in[recip]
        recip_d_out = full_d_out[recip]
        if deg_corr:
            tmp_lam_outout = d_out * (
                block_probs.T[np.newaxis, ...] * out_nbr_d_in[:, np.newaxis, np.newaxis]
            )
            tmp_lam_outrecip = d_in * (
                block_probs[np.newaxis, ...] * out_nbr_d_out[:, np.newaxis, np.newaxis]
            )
            tmp_lam_inin = d_in * (
                block_probs[np.newaxis, ...] * in_nbr_d_out[:, np.newaxis, np.newaxis]
            )
            tmp_lam_inrecip = d_out * (
                block_probs.T[np.newaxis, ...] * in_nbr_d_in[:, np.newaxis, np.newaxis]
            )
            tmp_lam_recipin = d_out * (
                block_probs.T[np.newaxis, ...] * recip_d_in[:, np.newaxis, np.newaxis]
            )
            tmp_lam_recipout = d_in * (
                block_probs[np.newaxis, ...] * recip_d_out[:, np.newaxis, np.newaxis]
            )

            out_nbr_contrib = np.einsum(
                "eqr,q->er",
                np.exp(
                    poisson.logpmf(1, tmp_lam_outout)
                    + poisson.logpmf(0, tmp_lam_outrecip)
                ),
                full_node_probs[out_nbrs, 1, :],
            )
            in_nbr_contrib = np.einsum(
                "eqr,q->er",
                np.exp(
                    poisson.logpmf(1, tmp_lam_inin) + poisson.logpmf(0, tmp_lam_inrecip)
                ),
                full_node_probs[in_nbrs, 1, :],
            )
            recip_contrib = np.einsum(
                "eqr,q->er",
                np.exp(
                    poisson.logpmf(1, tmp_lam_recipin)
                    + poisson.logpmf(1, tmp_lam_recipout)
                ),
                full_node_probs[recip, 1, :],
            )
        else:
            tmp_lam_outout = block_probs.T
            tmp_lam_outrecip = block_probs
            tmp_lam_inin = block_probs
            tmp_lam_inrecip = block_probs.T
            tmp_lam_recipin = block_probs.T
            tmp_lam_recipout = block_probs

            out_nbr_contrib = np.einsum(
                "eqr,q->er",
                np.tile(tmp_lam_outout, (len(out_nbrs), 1, 1))
                * np.tile(1 - tmp_lam_outrecip, (len(out_nbrs), 1, 1)),
                full_node_probs[out_nbrs, 1, :],
            )
            in_nbr_contrib = np.einsum(
                "eqr,q->er",
                np.tile(tmp_lam_inin, (len(in_nbrs), 1, 1))
                * np.tile(1 - tmp_lam_inrecip, (len(in_nbrs), 1, 1)),
                full_node_probs[in_nbrs, 1, :],
            )
            recip_contrib = np.einsum(
                "eqr,q->er",
                np.tile(tmp_lam_recipin, (len(recip), 1, 1))
                * np.tile(tmp_lam_recipout, (len(recip), 1, 1)),
                full_node_probs[recip, 1, :],
            )
        tmp_out = np.log(
            out_nbr_contrib,
            out=np.zeros_like(out_nbr_contrib),
            where=out_nbr_contrib != 0,
        )
        tmp_out = tmp_out.sum(axis=0)
        tmp_in = np.log(
            in_nbr_contrib, out=np.zeros_like(in_nbr_contrib), where=in_nbr_contrib != 0
        )
        tmp_in = tmp_in.sum(axis=0)
        tmp_recip = np.log(
            recip_contrib, out=np.zeros_like(recip_contrib), where=recip_contrib != 0
        )
        tmp_recip = tmp_recip.sum(axis=0)

        log_out = tmp_out + tmp_in + tmp_recip + prev_contrib
        tmp_marg = np.exp(log_out - log_out.max())
        tmp_marg /= tmp_marg.sum()
        full_node_probs[new_au_idx, 1, :] = tmp_marg

    return full_node_probs


class CitationProcessor:
    def __init__(
        self,
        datapath="/scratch/fitzgeraldj/data/caus_inf_data",
        subnetwork_size=8000,
        sub_testsize=300,
        num_topics=1000,
        test_prop=0.2,
        save_path: Optional[Path] = None,
        **kwargs,
    ):
        self.datapath = Path(datapath)
        self.subnetwork_size = subnetwork_size
        self.sub_testsize = sub_testsize
        self.num_topics = num_topics
        self.test_prop = test_prop
        self.save_path = save_path

        self.parse_args(**kwargs)

    def parse_args(self, **kwargs):
        self.random_seed = int(kwargs.get("seed", 42))
        np.random.seed(self.random_seed)

    def snowball_sample(self):
        """Snowball sample over all timestep adjacencies
        -- at each iteration, starting from a random root author,
        follow citations (in either direction) until reach desired
        subsample size.
        -- require that sampled authors are present before final
        time period, so they are within train set at least once,
        and that warn if this fails to provide minimum specified
        number of authors at final period (test set), where an author
        is considered present there if we have a profile for them
        (i.e. real region data at least is available)

        :return: sampled_aus, subset of aus to use in experiment
        :rtype: np.ndarray
        """
        sampled_aus = set()
        explored_aus = set()
        test_aus = set()
        aus = np.arange(self.N)
        np.random.shuffle(aus)
        u_iter = 0

        with tqdm(total=self.subnetwork_size, desc="Snowball samp. size:") as pbar:
            while len(sampled_aus) < self.subnetwork_size:
                au = aus[u_iter]
                explored_aus.add(au)
                # get all connected aus to this au in any timeslice
                # NB column indices for row i are stored in
                # indices[indptr[i]:indptr[i+1]] in csr
                A_T = [A_t.T for A_t in self.A]
                conn_aus = reduce(
                    lambda x, y: np.union1d(x, y),
                    [
                        np.union1d(
                            A_t.indices[A_t.indptr[u_iter] : A_t.indptr[u_iter + 1]],
                            A_T_t.indices[
                                A_T_t.indptr[u_iter] : A_T_t.indptr[u_iter + 1]
                            ],
                        )
                        for A_t, A_T_t in zip(self.A, A_T)
                    ],
                )
                new_aus = set(conn_aus) - sampled_aus
                pbar.update(len(new_aus))
                sampled_aus.add(au)
                sampled_aus |= new_aus
                # add any sampled aus present in final time period to
                # testset
                test_aus |= set(self.uids[self.df_ts[-1]]) & sampled_aus
                unexplored_aus = new_aus - explored_aus
                if len(unexplored_aus) > 0:
                    # follow edge to one of these new aus
                    u_iter = np.random.choice(list(unexplored_aus))
                else:
                    # no new nodes found, start at new root
                    u_iter += 1
        if len(test_aus) < self.sub_testsize:
            tqdm.write(f"Warning: only {len(test_aus)} test authors sampled")
        return np.array(list(sampled_aus))

    def load_edgelist(self):
        arr = np.load(os.path.join(self.datapath, "citation_links.npz"))
        self.edgelist = arr["edge_list"]

    def load_au_pubs(self):
        """Load author pubs -- pandas df with columns
            'auid_idx' (author id idx, int
                        -- must match w corresponding idx
                        used in edgelist file),
            'windowed_year' (int for starting year of timeslice
                             for author info contained in each row),
            'tpc_idx' (tpc of pub in given timestep by that author),
        with duplicate (auid,windowed_year) combinations if more than
        one pub that year.
        """
        with open(os.path.join(self.datapath, "au_pubs.pkl"), "rb") as f:
            self.au_pubs = pickle.load(f)

        self.df_ts = np.unique(self.au_pubs.windowed_year.values).astype(int)
        self.uids = {
            t: self.au_pubs.auid_idx[self.au_pubs.windowed_year == t].unique()
            for t in self.df_ts
        }

    def make_adj_matrix(self):
        """Construct adjacency matrix at each timestep from provided edgelist.
        Given causal model specified, assume pass one fewer timestep worth of
        edge information than have author pubs for.
        """
        time_inds = self.edgelist[:, 2]
        self.timesteps = np.unique(time_inds)
        if not np.all(self.df_ts[:-1] == self.timesteps):
            try:
                self.T = len(self.timesteps) + 1
                assert len(self.df_ts) == self.T
                tqdm.write(
                    """Timesteps in author pubs and edgelist do not match,
                    but there are the correct number of each."""
                )
                tqdm.write("Assuming can match on order...")
            except AssertionError:
                if len(self.timesteps) == len(self.df_ts):
                    self.T = len(self.timesteps)
                    tqdm.write(
                        "Passed same number of adj mat and pub timesteps: assume want to match"
                    )
                else:
                    raise ValueError(
                        "Timesteps in edgelist and author pubs do not match."
                    )
        row_inds = [
            self.edgelist[time_inds == t, 0].astype(int) for t in self.timesteps
        ]
        col_inds = [
            self.edgelist[time_inds == t, 1].astype(int) for t in self.timesteps
        ]
        # self.N = len(set(self.edgelist[:, 0]) | set(self.edgelist[:, 1]))
        self.N = self.edgelist[:, :2].max().astype(int) + 1
        # tqdm.write(self.edgelist[:, :2].min(), self.edgelist[:, :2].max())
        try:
            assert self.N > self.subnetwork_size
            tqdm.write(f"Found {self.N} authors in edgelist, over {self.T} timesteps.")
        except AssertionError:
            raise ValueError(
                f"Network size is smaller than desired subnetwork size: {self.N} < {self.subnetwork_size}"
            )
        # fin_aus = np.array(list(set(row_inds[-1]) | set(col_inds[-1])),dtype=int)
        # self.test_aus = np.random.choice(fin_aus, int(self.test_prop*len(fin_aus)), replace=False)
        # # remove test authors from final timestep
        # row_inds[-1] = row_inds[-1][~np.isin(row_inds[-1], self.test_aus)]
        # col_inds[-1] = col_inds[-1][~np.isin(col_inds[-1], self.test_aus)]
        data = [np.ones(row_inds[t].shape[0]) for t in range(self.T)]
        # row_inds.shape, col_inds.shape, data.shape
        self.A = [
            csr_array((data[t], (row_inds[t], col_inds[t])), shape=(self.N, self.N))
            for t in range(self.T)
        ]
        # self.A = A.toarray()

    def process_dataset(self):
        # load real data
        self.load_edgelist()
        tqdm.write("Loaded edgelist")
        self.load_au_pubs()
        tqdm.write("Loaded author publications")
        self.make_adj_matrix()
        # snowball subsample roughly specified number of authors
        self.aus = self.snowball_sample()
        self.aus = np.sort(self.aus)
        # tqdm.write("Constructed adjacency matrix and subsampled")

        # restrict adjacencies to these aus
        self.A = [A_t[self.aus, :] for A_t in self.A]
        self.A = [A_t[:, self.aus] for A_t in self.A]

        # now restrict author pub records to these aus
        self.au_pubs = self.au_pubs[self.au_pubs.auid_idx.isin(self.aus)]
        # only take top num_topics topics
        tpc_counts = self.au_pubs.tpc_idx.value_counts()
        top_tpcs = tpc_counts.index[: self.num_topics]
        self.au_pubs = self.au_pubs[self.au_pubs.tpc_idx.isin(top_tpcs)].reset_index(
            drop=True
        )
        # and now can finally construct Y
        au_loc = {au: i for i, au in enumerate(self.aus)}
        unq_tpcs = np.unique(self.au_pubs.tpc_idx.values)
        tpc_loc = {tpc: i for i, tpc in enumerate(unq_tpcs)}
        self.au_pubs.replace({"auid_idx": au_loc}, inplace=True)
        self.au_pubs.replace({"tpc_idx": tpc_loc}, inplace=True)
        self.Y = []
        for wyear in self.df_ts:
            sub_df = self.au_pubs.loc[self.au_pubs.windowed_year == wyear]
            Y_t = csr_array(
                (
                    np.ones(sub_df.shape[0], dtype=int),
                    (sub_df.auid_idx.values, sub_df.tpc_idx.values),
                ),
                shape=(len(self.aus), self.num_topics),
            )
            self.Y.append(Y_t)
        # now construct held out testset from final timestep
        # for transductive eval
        pres_aus = np.flatnonzero(self.Y[-1].sum(axis=1))
        self.test_aus = np.random.choice(
            pres_aus, int(self.test_prop * len(pres_aus)), replace=False
        )
        self.Y_heldout = self.Y[-1][self.test_aus, :].copy()
        self.Y[-1][self.test_aus, :] = 0
        self.Y[-1].eliminate_zeros()
        self.full_A_end = self.A[-1].copy()
        self.A[-1][self.test_aus, :] = 0
        self.A[-1][:, self.test_aus] = 0
        self.A[-1].eliminate_zeros()
        if self.save_path is not None:
            with open(self.save_path, "wb") as f:
                pickle.dump(self, f)
        return self.Y, self.Y_heldout, self.full_A_end
