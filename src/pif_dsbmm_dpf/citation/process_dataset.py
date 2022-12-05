r"""
This script generates semi-synthetic data from real citation and author
profile data.

Data reqs:

datapath assumed to be path to a folder containing

edgelist file
- npz format, file named 'citation_links.npz', and array named 'edge_list'
- first col idx of author sending citation,
- second col idx of author receiving citation,
- third col timestep at which this happens

au_profs file
- pickled pandas df, named 'au_profs.pkl'
- columns are 'auid_idx', 'windowed_year', 'region', 'career_age'
- no duplicate (auid,windowed_year) combinations

There should be one more timestep in the au_profs file than the edgelist file,
and up to the final timestep they should either match, or it is assumed that
they can be paired according to their rank.

Important default choices in semi-synthetic data generation:
- Perform snowball sampling on aggregated adjacency matrix to subsample authors
  to more manageable size (by default = 3000)
- Choose to make random topic embeddings w.r.t. covariate 1
  (author region by default) stable over time, so each region
  has stable topics in which it 'specialises' - sample randomly at first timestep
  then use these values throughout
- Other random covariate by default follows simple Markov process on given
  number of categories, where at time t an author/topic is either assigned to
  their previous category with probability eta (default = 0.8) else randomly
  chooses out of all categories
- Assume influence at time t realistically should depend to some degree on
  influence at previous timestep -- in real data this will be observed via
  the topic themselves, but as we are generating these here we enforce this
  artificially by imposing that
    p(infl at t | infl at t-1) = infl at t-1 + Norm(0,s.d.)
  where by default s.d. = 0.05 * mean influence (i.e. fluctuations ~5%),
  then thresholding to enforce that influence is always positive
- If an author is never cited in a time period, their influence is set to one

- Added in extra flags to control some extra features:
  --rand_cv_mode \in ["non-markov","markov","stable"], default "markov"
  --rand_cv_eta = float \in [0,1], default 0.8
  --influence_mode \in ["gp","non-gp"], default "gp"
  --influence_gpvar = float \in [0,1], default 0.05 (av. percentage of mean infl)
- Lowered default num topics to be simulated to 1000 from 3000, as
  closer to real data
- make_multi_covariate_simulation now only return Y rather than Y,Y_past,
  and this is a list length T of csr_arrays
"""


import os
import pickle
import sys
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.special import expit
from scipy.stats import bernoulli, gamma, poisson
from tqdm import tqdm
from utils import sample_simple_markov


class CitationSimulator:
    # -- Allow either
    # (i) Independent random covariates at each timestep, or
    # (ii) Sample the random covariates according to some Markov chain
    # NB even in case (i), the region/country indicators will still have
    # some temporal correlation, so will choose to only sample the
    # corresponding random covariates for topics once, then use these
    # throughout -- the actual confounders are only noisy observations
    # of these anyway so shouldn't be a major issue.
    def __init__(
        self,
        datapath="/scratch/fitzgeraldj/data/caus_inf_data",
        subnetwork_size=3000,
        sub_testsize=300,
        influence_shp=0.005,
        num_topics=1000,
        covar_1="region_categorical",
        covar_2="random",
        covar_2_num_cats=5,
        save_path: Optional[Path] = None,
        **kwargs,
    ):
        self.datapath = Path(datapath)
        self.subnetwork_size = subnetwork_size
        self.sub_testsize = sub_testsize
        self.influence_shp = influence_shp
        self.num_topics = num_topics
        self.covar_1 = covar_1
        self.covar_2 = covar_2
        self.covar_2_num_cats = covar_2_num_cats
        self.save_path = save_path

        self.parse_args(**kwargs)

    def parse_args(self, **kwargs):
        self.random_seed = int(kwargs.get("seed", 42))
        self.do_sensitivity = bool(kwargs.get("do_sensitivity", False))
        self.sensitivity_parameter = float(kwargs.get("sensitivity_parameter", 1.0))
        self.error_rate = float(kwargs.get("error_rate", 0.3))
        self.rand_cv_mode = kwargs.get("rand_cv_mode", "markov")
        self.rand_cv_eta = float(kwargs.get("rand_cv_eta", 0.8))
        self.influence_mode = kwargs.get("influence_mode", "gp")
        # assume s.d. of change in influence between timesteps is
        # ~5% of mean influence
        self.influence_gpvar = float(kwargs.get("influence_gpvar", 0.05))
        self.influence_gpvar = self.influence_shp * 10 * self.influence_gpvar
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

    def load_au_profs(self):
        """Load author profiles -- pandas df with columns
            'auid_idx' (author id idx, int
                        -- must match w corresponding idx
                        used in edgelist file),
            'windowed_year' (int for starting year of timeslice
                             for author info contained in each row),
            'region' (region to which author is affiliated most often
                      at that timestep),
            'career_age' (duration for which author has been publishing)
        with no duplicate (auid,windowed_year) combinations.
        """
        with open(os.path.join(self.datapath, "au_profs.pkl"), "rb") as f:
            self.au_profs = pickle.load(f)
        # enforce that (auid_idx,windowed_year) should be unique
        print(len(self.au_profs))
        self.au_profs.drop_duplicates(
            subset=["auid_idx", "windowed_year"], inplace=True
        )
        print(len(self.au_profs))
        self.au_profs["region_categorical"] = pd.Categorical(self.au_profs.region).codes
        code = {
            r: i for (i, r) in enumerate(np.unique(self.au_profs["region_categorical"]))
        }
        self.au_profs["region_categorical"] = self.au_profs["region_categorical"].apply(
            lambda x: code[x]
        )
        age_bins = np.arange(0, 100, step=5)
        self.au_profs["career_age_binned"] = np.digitize(
            self.au_profs.career_age.values, age_bins
        )
        self.au_profs["career_age_binned"] -= 1
        self.df_ts = np.unique(self.au_profs.windowed_year.values).astype(int)
        self.uids = {
            t: self.au_profs.auid_idx[self.au_profs.windowed_year == t]
            for t in self.df_ts
        }

    def make_adj_matrix(self):
        """Construct adjacency matrix at each timestep from provided edgelist.
        Given causal model specified, assume pass one fewer timestep worth of
        edge information than have author profiles for.
        """
        time_inds = self.edgelist[:, 2]
        self.timesteps = np.unique(time_inds)
        self.T = len(self.timesteps) + 1
        if not np.all(self.df_ts[:-1] == self.timesteps):
            try:
                assert len(self.df_ts) == self.T
                tqdm.write(
                    """Timesteps in author profiles and edgelist do not match,
                    but there are the correct number of each."""
                )
                tqdm.write("Assuming can match on order...")
            except AssertionError:
                raise ValueError(
                    "Timesteps in edgelist and author profiles do not match."
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
        data = [np.ones(row_inds[t].shape[0]) for t in range(self.T - 1)]
        # row_inds.shape, col_inds.shape, data.shape
        self.A = [
            csr_array((data[t], (row_inds[t], col_inds[t])), shape=(self.N, self.N))
            for t in range(self.T - 1)
        ]
        # self.A = A.toarray()

    def get_one_hot_covariate_encoding(self, covar):
        """Return one-hot encoding of specified covariate for each timestep,
        in shape (N,T,num_cats)

        :param covar: chosen covariate -- name of column in au_profs df,
                      which must be a categorical index - this is automatically
                      generated for region and career_age, suitably named
                      region_categorical and career_age_binned resp.
        :type covar: str
        :return: one_hot_encoding, shape (N,T,num_cats)
        :rtype: np.ndarray
        """
        tqdm.write(f"Getting one-hot encoding for {covar}...")
        tot_sub = self.au_profs.loc[self.au_profs.auid_idx.isin(self.aus), :]
        categories = np.unique(tot_sub[covar])
        num_cats = len(categories)
        one_hot_encoding = np.zeros((self.aus.shape[0], self.T, num_cats))
        tot_sub1hot = pd.get_dummies(tot_sub[covar])
        self.covar1_codedict = {i: c for i, c in enumerate(tot_sub1hot.columns)}
        for t, year in enumerate(self.df_ts):
            pres_u = np.isin(self.aus, self.uids[year].values)
            try:
                one_hot_encoding[pres_u, t, :] = tot_sub1hot.loc[
                    tot_sub.windowed_year == year, :
                ].values
            except ValueError:
                print(t)
                print(pres_u.sum())
                print((tot_sub.windowed_year == year).sum())
                raise ValueError("Something went wrong with one-hot encoding.")
            # u_idx = np.arange(self.aus.shape[0])
            # # not guaranteed that every author that gets cited has a profile
            # # available at every timestep, so must take subset
            # # also might have missing data for some columns
            # pres_u = pres_u[~np.isnan(data)]
            # data = data[~np.isnan(data)]
            # one_hot_encoding[pres_u, t, data] = 1
        return one_hot_encoding

    def sample_random_covariate(
        self, num_categories: int, num_samples: int, mode="non-markov", eta=0.8
    ):
        """Generate random covariate values for each author at each timestep.
        If mode == 'non-markov', each timestep is uniformly random sampled over
        the number of categories specified.

        If mode == 'markov', the covariate values at the first timestep is
        uniformly randomly sampled, then subsequent timesteps are sampled
        sequentially, where the probability of an author being assigned to
        the same category as the previous timestep is eta, else the category
        is uniformly randomly sampled, i.e.

            p(covariate_t = covariate_t-1) = eta + (1-eta)/num_categories,
            p(covariate_t = cv != covariate_t-1) = (1-eta)/num_categories.

        Finally if mode == 'stable', the covariate values at the first
        timestep are uniformly sampled, then these values are used for all
        subsequent times.

        :param num_categories: Number of categories to sample
        :type num_categories: int
        :param num_samples: Number of samples at each timestep
        :type num_samples: int
        :param mode: Mode to generate random series of covariates.
                     Options are "non-markov", "markov", and "stable",
                     as described above - defaults to "non-markov"
        :type mode: str, optional
        :param eta: probability of assigning same category as prev timestep
                    if mode=='markov', defaults to 0.7
        :type eta: float, optional
        :return: one-hot encoding of sampled covariates,
                 shape (num_samples, T, num_categories)
        :rtype: np.ndarray
        """
        one_hot_encoding = np.zeros((num_samples, self.T, num_categories))
        if mode == "non-markov":
            for t in range(self.T):
                one_hot_encoding[
                    np.arange(num_samples),
                    t,
                    np.random.randint(0, num_categories, size=num_samples),
                ] = 1
        elif mode == "markov":
            one_hot_encoding = sample_simple_markov(
                num_samples, self.T, num_categories, eta, onehot=True
            )
        elif mode == "stable":
            one_hot_encoding[
                np.arange(num_samples),
                :,
                np.random.randint(0, num_categories, size=num_samples),
            ] = 1
        else:
            raise NotImplementedError(
                "Only non-markov, markov, and stable modes for random cv gen implemented"
            )
        return one_hot_encoding

    def make_embeddings(
        self,
        au_encoding,
        topic_encoding,
        num_cats,
        noise=10.0,
        confounding_strength=10.0,
        gamma_mean=0.1,
        gamma_scale=0.1,
    ):
        M, T, _ = topic_encoding.shape
        N = au_encoding.shape[0]
        gamma_shp = gamma_mean / gamma_scale
        embedding_shp = (num_cats * gamma_shp * noise) / (noise + (num_cats - 1))
        loadings_shp = (num_cats * gamma_shp * confounding_strength) / (
            confounding_strength + (num_cats - 1)
        )

        embedding = au_encoding * gamma.rvs(
            embedding_shp, scale=gamma_scale, size=(N, T, num_cats)
        )
        embedding += (1 - au_encoding) * gamma.rvs(
            embedding_shp / noise, scale=gamma_scale, size=(N, T, num_cats)
        )

        loadings = topic_encoding * gamma.rvs(
            loadings_shp, scale=gamma_scale, size=(M, T, num_cats)
        )
        loadings += (1 - topic_encoding) * gamma.rvs(
            loadings_shp / confounding_strength,
            scale=gamma_scale,
            size=(M, T, num_cats),
        )
        return embedding, loadings

    def make_simulated_influence(self) -> np.ndarray:
        N = self.A[0].shape[0]
        # by default influence_shp == 0.005, so expected influence
        # (at least in first timestep)
        # = shp * scale = 0.005 * 10 = 0.05
        if self.influence_shp > 0:
            if self.influence_mode == "non-gp":
                influence = gamma.rvs(
                    self.influence_shp, scale=10.0, size=(N, self.T - 1)
                )
            elif self.influence_mode == "gp":
                influence = np.zeros((N, self.T - 1))
                influence[:, 0] = gamma.rvs(self.influence_shp, scale=10.0, size=(N,))
                for t in range(1, self.T - 1):
                    tmp = influence[:, t - 1] + np.random.normal(
                        0, scale=self.influence_gpvar, size=(N,)
                    )
                    tmp[tmp < 0] = 0
                    influence[:, t] = tmp
            else:
                raise NotImplementedError(
                    "influence_mode must be 'non-gp' or 'gp' if influence_shp > 0"
                )
        else:
            influence = np.zeros((N, self.T - 1))
        return influence  # type: ignore

    def process_dataset(self):
        # load real data
        self.load_edgelist()
        tqdm.write("Loaded edgelist")
        self.load_au_profs()
        tqdm.write("Loaded author profiles")
        self.make_adj_matrix()
        # snowball subsample roughly specified number of authors
        self.aus = self.snowball_sample()
        tqdm.write("Constructed adjacency matrix and subsampled")

        # restrict adjacencies to these aus
        self.A = [A_t[self.aus, :] for A_t in self.A]
        self.A = [A_t[:, self.aus] for A_t in self.A]

        self.au_one_hot_covar_1 = self.get_one_hot_covariate_encoding(self.covar_1)
        tqdm.write(f"Generated {self.covar_1} au covariate")
        if self.covar_2 == "career_age_binned":
            self.au_one_hot_covar_2 = self.get_one_hot_covariate_encoding(self.covar_2)
        else:
            self.au_one_hot_covar_2 = self.sample_random_covariate(
                self.covar_2_num_cats,
                self.A[0].shape[0],
                mode=self.rand_cv_mode,
                eta=self.rand_cv_eta,
            )
        tqdm.write(f"Generated {self.covar_2} au covariate")

        num_regions = self.au_one_hot_covar_1.shape[1]
        num_cats = self.covar_2_num_cats
        # choose to make random topic embeddings w.r.t. covariate 1
        # (author region by default) stable over time, so each region
        # has stable topics in which it 'specialises'
        self.topic_one_hot_covar_1 = self.sample_random_covariate(
            num_regions,
            self.num_topics,
            mode="stable",
        )
        tqdm.write(f"Generated {self.covar_1} tpc covariate")
        # use same mode for topic rand cv as au rand cv
        self.topic_one_hot_covar_2 = self.sample_random_covariate(
            num_cats, self.num_topics, mode=self.rand_cv_mode, eta=self.rand_cv_eta
        )
        tqdm.write(f"Generated {self.covar_2} tpc covariate")
        tqdm.write("Simulating influence...")
        self.beta: np.ndarray = self.make_simulated_influence()
        no_cit_aus = np.stack([A_t.sum(axis=0) == 0 for A_t in self.A])
        self.beta[no_cit_aus] = 1.0

    def make_multi_covariate_simulation(
        self,
        noise=10.0,
        confounding_strength=10.0,
        gamma_mean=0.1,
        gamma_scale=0.1,
        confounding_to_use="both",
    ):
        covar_1_num_cats = self.au_one_hot_covar_1.shape[1]
        covar_2_num_cats = self.covar_2_num_cats

        tqdm.write("Simulating embeddings...")
        self.au_embed_1, self.topic_embed_1 = self.make_embeddings(
            self.au_one_hot_covar_1,
            self.topic_one_hot_covar_1,
            covar_1_num_cats,
            noise=noise,
            confounding_strength=confounding_strength,
            gamma_mean=gamma_mean,
            gamma_scale=gamma_scale,
        )
        tqdm.write(
            f"Done for {self.covar_1}, now simulating {self.covar_2} embeddings..."
        )
        self.au_embed_2, self.topic_embed_2 = self.make_embeddings(
            self.au_one_hot_covar_2,
            self.topic_one_hot_covar_2,
            covar_2_num_cats,
            noise=noise,
            confounding_strength=confounding_strength,
            gamma_mean=gamma_mean,
            gamma_scale=gamma_scale,
        )
        tqdm.write("Done")
        tqdm.write("Finally simulating publication topics...")
        Y = self.make_mixture_preferences_outcomes(
            confounding_to_use=confounding_to_use
        )
        if self.save_path is not None:
            tqdm.write(f"Finished, saving to {self.save_path}...")
            with open(self.save_path, "wb") as f:
                pickle.dump(self, f)
        tqdm.write("Done")
        return Y

    def make_mixture_preferences_outcomes(self, confounding_to_use):
        # author preferences for topics should should be dot product
        # of corresponding embeddings
        homophily_pref = np.einsum("itk,mtk->imt", self.au_embed_1, self.topic_embed_1)
        random_pref = np.einsum("itk,mtk->imt", self.au_embed_2, self.topic_embed_2)

        if confounding_to_use == "homophily":
            base_rate = homophily_pref
        elif confounding_to_use == "both":
            base_rate = homophily_pref + random_pref
        else:
            base_rate = random_pref

        N = self.au_embed_1.shape[0]
        M = self.topic_embed_1.shape[0]
        pres_aus = set()  # aus who 'published' before t=T
        pres_tpcs = set()  # tpcs that were 'published' before t=T
        if self.do_sensitivity:
            bias = self.create_bias()
            # influence rate for i at t then just
            # \sum_j a_{ij}^{t-1} y_{jk}^{t-1}
            # for t=0 only use base rate, then generate rest sequentially
            y_tm1 = csr_array(
                poisson.rvs(base_rate[..., 0] + bias[..., 0]), shape=(N, M)
            )
            pres_aus |= set(np.flatnonzero(y_tm1.sum(axis=1)))
            pres_tpcs |= set(np.flatnonzero(y_tm1.sum(axis=0)))
            y = [y_tm1]
            for t in range(self.T - 1):
                influence_rate = (self.beta[:, t] * self.A[t]) @ y_tm1
                y_tm1 = csr_array(
                    poisson.rvs(base_rate + influence_rate + bias[..., t + 1]),
                    shape=(N, M),
                )
                if t == self.T - 2:
                    # unlikely but possible that simulate an au / tpc in
                    # final period that was never simulated in earlier periods
                    # as we're using this to test + val, we remove these
                    y_tm1[np.ix_(list(pres_aus), list(pres_tpcs))] = 0
                    y_tm1 = csr_array(y_tm1, shape=(N, M))
                y.append(y_tm1)

        else:
            y_tm1 = csr_array(poisson.rvs(base_rate[..., 0]), shape=(N, M))
            y = [y_tm1]
            for t in range(self.T - 1):
                influence_rate = (self.beta[:, t] * self.A[t]) @ y_tm1
                y_tm1 = csr_array(
                    poisson.rvs(base_rate + influence_rate),
                    shape=(N, M),
                )
                y.append(y_tm1)

        return y

    def create_bias(self, gamma_mean=0.1, gamma_scale=0.1):
        N = self.A[0].shape[0]
        M = self.num_topics

        bias = np.zeros((N, M, self.T))
        for tm1, A_t in enumerate(self.A):
            t = tm1 + 1
            # (au1, au2) = np.nonzero(A_t)
            mask: np.ndarray = bernoulli.rvs(self.error_rate, size=(len(A_t.data),))  # type: ignore
            bias_mean = gamma_mean / self.sensitivity_parameter
            n_biased = mask.sum()
            bias_vals = gamma.rvs(
                (bias_mean / gamma_scale), scale=gamma_scale, size=(n_biased,)
            )
            random_topics = bernoulli.rvs(
                self.error_rate, size=(n_biased, self.num_topics)
            )
            # in csr format:
            # column indices for row i are stored in
            # indices[indptr[i]:indptr[i+1]]
            # and their corresponding values are stored in
            # data[indptr[i]:indptr[i+1]]
            i_idxs = np.repeat(np.arange(N), np.diff(A_t.indptr))
            j_idxs = A_t.indices
            # now can get idxs of biased aus by edge
            i_idxs = i_idxs[mask == 1]
            j_idxs = j_idxs[mask == 1]
            # now can get idxs of biased topics
            tpc_idxs = np.nonzero(random_topics)[1]
            # and must repeat the corresponding indices the right no. times
            tpc_bias_cnts = random_topics.sum(axis=1)
            i_idxs = np.repeat(i_idxs, tpc_bias_cnts)
            j_idxs = np.repeat(j_idxs, tpc_bias_cnts)
            # then finally
            bias[i_idxs, tpc_idxs, t] = bias_vals
            bias[j_idxs, tpc_idxs, t] = bias_vals
            # and don't need to loop over all edges like below anymore,
            # which would take v long for large dense graphs...
            # for edge_iter in range(au1.shape[0]):
            #     if mask[edge_iter]:
            #         i = au1[edge_iter]
            #         j = au2[edge_iter]
            #         bias_mean = gamma_mean / self.sensitivity_parameter
            #         bias_val = gamma.rvs((bias_mean / gamma_scale), scale=gamma_scale)
            #         random_topics = bernoulli.rvs(self.error_rate, size=self.num_topics)
            #         for k in np.nonzero(random_topics):
            #             bias[i, k] = bias_val
            #             bias[j, k] = bias_val

        return bias
