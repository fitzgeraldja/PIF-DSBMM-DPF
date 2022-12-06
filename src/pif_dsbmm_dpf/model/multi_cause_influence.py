import argparse
import sys
from functools import reduce

import numpy as np
from scipy import sparse, special
from scipy.stats import gamma, poisson, truncnorm
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

from pif_dsbmm_dpf.citation import utils


class CausalInfluenceModel:
    # TODO:
    # -- Allow inference where the A used for inference at each timestep is either
    # (i) the A at the previous timestep (logic being that publications take time, so inspiration comes at an earlier stage), and
    # (ii) the A at the current timestep (logic being that citations are a good proxy for influence already). For the latter,
    # presumably we would need to allow observation of the network to permit held-out influence estimation, but this would still be
    # legitimate to some degree if we didn't allow observation of the topics themselves.
    # -- [DONE] Allow both
    # (i) time-homogeneous params (but independent observations at each timestep), which thus use all data for each param
    # -- assume that can just use eqns below but sum over t as well as whatever else, and
    # (ii) time-varying params, that only use the present timestep -- means held-out data won't have available params, but can
    # just use the latest available instead
    # -- Check that DSBMM is approximately rendering citations and topics independent given the groups, as this is necessary
    # for the causal model to be valid.
    # -- Change all below from sparse to ndarray as v minimal savings anyway
    # - ahh incorrect -- any dots w the Ys or A will be sparse and also
    # more importantly will prevent calculating all the other
    # multipliers, so save a lot of time if these are very sparse
    # -- Also means can't use npnewaxis for Y_t or A_t, nor einsum
    def __init__(
        self,
        n_components=100,
        n_exog_components=100,
        max_iter=100,
        tol=0.0005,
        random_state=None,
        verbose=False,
        model_mode="full",
        use_current_A=False,
        time_homog=False,
        **kwargs,
    ):
        self.n_components = n_components
        self.n_exog_components = n_exog_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.mode = model_mode
        self.use_current_A = use_current_A
        self.time_homog = time_homog

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(0)

        self._parse_args()

    def _parse_args(self, **kwargs):
        self.learning_rate = float(kwargs.get("learning_rate", 0.1))
        self.batch_size = int(kwargs.get("batch_size", 100))

        self.inf_rate = float(kwargs.get("a", 0.1))
        self.inf_shp = float(kwargs.get("b", 0.1))

        self.topic_mean = float(kwargs.get("c", 0.01))
        self.topic_rate = float(kwargs.get("d", 10.0))
        self.topic_shp = self.topic_mean * self.topic_rate

        self.au_mean = float(kwargs.get("e", 0.01))
        self.au_rate = float(kwargs.get("f", 10.0))
        self.au_shp = self.au_mean * self.au_rate

    def _init_beta(self, N, T):
        """Initialise beta -- these are the multipliers for the topics at the previous timestep
        for authors that were cited, i.e. the influence scores we're interested in

        :param N: Number of authors
        :type N: int
        :param T: Number of timesteps
        :type T: int
        """
        if self.time_homog:
            self.beta_shape = self.inf_shp + truncnorm.rvs(0, 1, size=(N))
        else:
            self.beta_shape = self.inf_shp + truncnorm.rvs(0, 1, size=(N, T - 1))

    def _init_gamma(self, M, K, T):
        """Initialise gamma -- these are the per-topic attributes, i.e. the multipliers for the
        author-topic confounder zeta

        :param M: Number of topics
        :type M: int
        :param K: Number of factors for author-topic confounder
        :type K: int
        :param T: Number of timesteps
        :type T: int
        """
        if self.time_homog:
            self.gamma_shape = self.topic_shp + truncnorm.rvs(0, 1, size=(M, K))
        else:
            self.gamma_shape = self.topic_shp + truncnorm.rvs(0, 1, size=(M, T - 1, K))

    def _init_alpha(self, N, K, T):
        """Initialise alpha -- these are the per-author attributes, i.e. the multipliers for the topic confounder tau

        :param N: Number of authors
        :type N: int
        :param K: Number of factors for topic confounder
        :type K: int
        :param T: Number of timesteps
        :type T: int
        """
        if self.time_homog:
            self.alpha_shape = self.au_shp + truncnorm.rvs(0, 1, size=(N, K))
        else:
            self.alpha_shape = self.au_shp + truncnorm.rvs(0, 1, size=(N, T - 1, K))

    def _init_rates(self, A, Y_past, Z, W, M, N):
        """Initialise rates

        :param A: Network (A^{t-1} for us), list of T-1 sparse matrices, each shape (N,N), or (N,N,T-1) if array (not currently handled)
        :type A: Union[np.ndarray, List[sparse.csr_array]]
        :param Y_past: Previous timestep's observed number of publications in each topic for each author, shape (N,M,T-1) or list of T-1 sparse mats (N,M)
        :type Y_past: Union[np.ndarray, sparse.csr_array]
        :param Z: Substitute inferred for network + topic confounder at current timestep, shape (N,T-1,Q) (first index now t=1 not t=0)
        :type Z: np.ndarray
        :param W: Substitute inferred for topic confounder at current timestep, shape (N,T-1,K)
        :type W: np.ndarray
        :param M: Number of topics
        :type M: int
        :param N: Number of authors
        :type N: int
        """
        if self.time_homog:
            # want to init beta rate with prior + \sum_{t=1} \sum_{j,k} a_{ji}^{t-1} y_{ik}^{t-1}
            self.beta_rates = self.inf_rate + np.stack(
                [
                    A_tm1.sum(axis=0) * Ytm1.sum(axis=1)
                    for A_tm1, Ytm1 in zip(A, Y_past)
                ],
                axis=1,
            ).sum(axis=1)
        else:
            self.beta_rates = self.inf_rate + np.stack(
                [
                    A_tm1.sum(axis=0) * Ytm1.sum(axis=1)
                    for A_tm1, Ytm1 in zip(A, Y_past)
                ],
                axis=1,
            )

        if self.mode == "network_preferences":
            # want to init gamma rate w prior + \sum_{t=1} \sum_i zeta_i^t (topic-link confounder substitute)
            if self.time_homog:
                self.gamma_rates = self.topic_rate + np.expand_dims(
                    Z.sum(axis=(0, 1)), 0
                )
            else:
                self.gamma_rates = self.topic_rate + Z.sum(axis=0, keepdims=True)
        elif self.mode == "topic":
            # want to init alpha rate w prior + \sum_{t=1} \sum_k tau_k^t # (topic confounder substitute)
            if self.time_homog:
                self.alpha_rates = self.au_rate + np.expand_dims(W.sum(axis=(0, 1)), 0)
            else:
                self.alpha_rates = self.au_rate + W.sum(axis=0, keepdims=True)
        elif self.mode == "full":
            if self.time_homog:
                self.gamma_rates = self.topic_rate + np.expand_dims(
                    Z.sum(axis=(0, 1)), 0
                )
                self.alpha_rates = self.au_rate + np.expand_dims(W.sum(axis=(0, 1)), 0)
            else:
                self.gamma_rates = self.topic_rate + Z.sum(axis=0, keepdims=True)
                self.alpha_rates = self.au_rate + W.sum(axis=0, keepdims=True)

    def _init_expectations(self):
        self.E_log_beta, self.E_beta = self._compute_expectations(
            self.beta_shape, self.beta_rates
        )

        if self.mode == "network_preferences":
            self.E_log_gamma, self.E_gamma = self._compute_expectations(
                self.gamma_shape, self.gamma_rates
            )
        elif self.mode == "topic":
            self.E_log_alpha, self.E_alpha = self._compute_expectations(
                self.alpha_shape, self.alpha_rates
            )
        elif self.mode == "full":
            self.E_log_gamma, self.E_gamma = self._compute_expectations(
                self.gamma_shape, self.gamma_rates
            )
            self.E_log_alpha, self.E_alpha = self._compute_expectations(
                self.alpha_shape, self.alpha_rates
            )

    def _compute_expectations(self, shp, rte):
        return special.psi(shp) - np.log(rte), shp / rte

    def _compute_terms_and_normalizers(
        self, A, Y_past, Y, Z, W, Z_trans=None, use_old_subs=True
    ):
        """Compute terms necessary for SVI updates of params

        :param A: Network (A^{t-1} for us), list of T-1 sparse matrices, each shape (N,N), or (N,N,T-1) if array (not currently handled)
        :type A: Union[np.ndarray, List[sparse.csr_array]]
        :param Y_past: Previous timestep's observed number of publications in each topic for each author, shape (N,M,T-1) or list of T-1 sparse mats (N,M)
        :type Y_past: Union[np.ndarray, sparse.csr_array]
        :param Z: Substitute inferred for network + topic confounder at current timestep, shape (N,T-1,Q) (first index now t=1 not t=0)
        :type Z: np.ndarray
        :param W: Substitute inferred for topic confounder at current timestep, shape (N,T-1,K)
        :type W: np.ndarray
        :param Z_trans: Transition matrix for topic-link confounder, shape (Q,Q)
        :type Z_trans: np.ndarray, optional
        :param use_old_subs: Either use the old substitutes (default), or point
                             estimates of the new ones, using pi*Z^{t-1} for DSBMM and
                             \\mu_{v_k} + v_{k,t-1} for dPF
        :type use_old_subs: bool, optional
        """
        # NB all these terms will be time-varying even if fitting time-homogeneous params, as the network and topics themselves
        # are time-varying

        # now influence component of normaliser, i.e. for psi variational param, requires
        # E_log_beta (N,T-1) or (N,),
        # A ([(N,N) for _ in T - 1])
        # Y_past ([(N,M) for _ in T - 1])
        # and term itself is then just
        # \sum_j exp(E_log_beta)_j^{t-1} a_{ij}^{t-1} y_{jk}^{t-1}, as want beta term still (N,T-1)
        self.beta_term = np.exp(self.E_log_beta)

        if self.time_homog:
            # NB all of these variational params components will now be of size N,M,T, so very much want to keep sparse if possible
            # -- also makes clear reason for snowball sampling only, as otherwise NM gets very large even without enormous number of
            # items, so fills memory even if computations are relatively fast, as this isn't possible given dense params
            influence_component = np.stack(
                [
                    utils.safe_sparse_toarray((self.beta_term * A_t) @ Ytm1)
                    for A_t, Ytm1 in zip(A, Y_past)
                ],
                axis=-1,
            )
        else:
            influence_component = np.stack(
                [
                    utils.safe_sparse_toarray((beta_t * A_t) @ Ytm1)
                    for beta_t, A_t, Ytm1 in zip(self.beta_term.T, A, Y_past)
                ],
                axis=-1,
            )

        if not use_old_subs:
            try:
                assert Z_trans is not None
            except AssertionError:
                raise ValueError("Must pass Z_trans if want to use updated subs")
            Z = np.einsum("qr,ntq->ntr", Z_trans, Z)
            # NB no need to do update for W if dPF and only
            # using point estimates, as would only use
            # v_{k,t-1} = \hat{v}_{k,t-1} - \mu_{v_k}
            # then the update would return \hat{v}_{k,t-1}
            # anyway

        preference_component = 1e-10

        if self.mode == "network_preferences":
            # now network preference component of normaliser needs
            # E_log_gamma (M,T-1,Q) or (M,Q)
            # Z (N,T-1,Q)
            # and term itself is then just
            # \sum_q exp(E_log_gamma)_{mq}^t z_{iq}^t
            self.gamma_term = np.exp(self.E_log_gamma)
            # NB gamma, Z are dense in N,M,T so nothing to be done to keep sparse here
            if self.time_homog:
                preference_component = np.einsum("mq,itq->imt", self.gamma_term, Z)
            else:
                preference_component = np.einsum("mtq,itq->imt", self.gamma_term, Z)

        elif self.mode == "topic":
            # now topic preference component of normaliser needs
            # E_log_alpha (N,T-1,K) or (N,K)
            # W (M,T-1,K)
            # and term itself is then just
            # \sum_k exp(E_log_alpha)_{ik}^t w_{mk}^t
            self.alpha_term = np.exp(self.E_log_alpha)
            if self.time_homog:
                preference_component = np.einsum("iq,mtq->imt", self.alpha_term, W)
            else:
                preference_component = np.einsum("itk,mtk->imt", self.alpha_term, W)

        elif self.mode == "full":
            self.gamma_term = np.exp(self.E_log_gamma)
            self.alpha_term = np.exp(self.E_log_alpha)
            if self.time_homog:
                preference_component = np.einsum(
                    "mq,itq->imt", self.gamma_term, Z
                ) + np.einsum("iq,mtq->imt", self.alpha_term, W)
            else:
                preference_component = np.einsum(
                    "mtq,itq->imt", self.gamma_term, Z
                ) + np.einsum("itk,mtk->imt", self.alpha_term, W)

        self.normaliser = preference_component + influence_component

    ##for now, this will simply return log likelihood under the Poisson model for Y
    def _compute_elbo(self, Y, A, Y_past, Z, W, Z_trans=None, use_old_subs=True):
        """Compute the ELBO for all observed data
        -- just compute the \\mu_{ik}^t
        = \alpha_i^{t,\top}\tau_k^t
        + \\zeta_i^{t,\top}\\gamma_k^t
        + \\sum_j \beta_j^{t-1} a_{ij}^{t-1} y_{jk}^{t-1}
        then calc logpmf of y ~ Poisson(\\mu) accordingly

        :param Y: All observed data from t=1, T-1 sparse matrices of shape (N,M)
        :type Y: List[sparse.csr_array]
        :param A: All observed adjacencies up to T-1, list of T-1 sparse matrices of shape (N,N)
        :type A: List[sparse.csr_array]
        :param Y_past: Observed data up to T-1, list of T-1 sparse matrices of shape (N,M)
        :type Y_past: List[sparse.csr_array]
        :param Z: Substitute for topic-link confounder, shape (N,T-1,Q) -- NB this can't include most recent period,
        so will need to take expectation over trans for final period
        :type Z: np.ndarray
        :param W: Substitute for topic confounder, shape (M,T-1,Q) -- likewise can't include most recent period,
        so need to take expectation over trans for final period
        :type W: np.ndarray
        :param Z_trans: Transition matrix for topic-link confounder, shape (Q,Q)
        :type Z_trans: np.ndarray, optional
        :param use_old_subs: Either use the old substitutes (default), or point
                             estimates of the new ones, using pi*Z^{t-1} for DSBMM and
                             \\mu_{v_k} + v_{k,t-1} for dPF
        :type use_old_subs: bool, optional

        :return: ELBO
        :rtype: float
        """

        if self.time_homog:
            influence_rate = np.stack(
                [
                    utils.safe_sparse_toarray((self.E_beta[np.newaxis, :] * A_t) @ Ytm1)
                    for A_t, Ytm1 in zip(A, Y_past)
                ],
                axis=-1,
            )
        else:
            influence_rate = np.stack(
                [
                    utils.safe_sparse_toarray((beta_t[np.newaxis, :] * A_t) @ Ytm1)
                    for beta_t, A_t, Ytm1 in zip(self.E_beta.T, A, Y_past)
                ],
                axis=-1,
            )
        pref_rate = 1e-10
        # NB could have neg Z from missing nodes in DSBMM, in which case should
        # set to zero
        if np.sum(Z < 0) > 0:
            tqdm.write(f"found {np.sum(Z < 0)} Z < 0 -- setting to 0")
            Z[Z < 0] = 0.0

        # check for W also just in case
        if np.sum(W < 0) > 0:
            tqdm.write(f"found {np.sum(W < 0)} W < 0 -- setting to 0")
            W[W < 0] = 0.0

        if not use_old_subs:
            try:
                assert Z_trans is not None
            except AssertionError:
                raise ValueError("Must pass Z_trans if want to use updated subs")
            Z = np.einsum("qr,ntq->ntr", Z_trans, Z)
            # NB no need to do update for W if dPF and only
            # using point estimates, as would only use
            # v_{k,t-1} = \hat{v}_{k,t-1} - \mu_{v_k}
            # then the update would return \hat{v}_{k,t-1}
            # anyway

        if self.mode == "network_preferences":
            if self.time_homog:
                pref_rate = np.einsum("itq,mq->imt", Z, self.E_gamma)
            else:
                pref_rate = np.einsum("itq,mtq->imt", Z, self.E_gamma)

        elif self.mode == "topic":
            if self.time_homog:
                pref_rate = np.einsum("iq,mtq->imt", self.E_alpha, W)
            else:
                pref_rate = np.einsum("itq,mtq->imt", self.E_alpha, W)

        elif self.mode == "full":
            if self.time_homog:
                pref_rate = np.einsum("itq,mq->imt", Z, self.E_gamma) + np.einsum(
                    "iq,mtq->imt", self.E_alpha, W
                )
            else:
                pref_rate = np.einsum("itq,mtq->imt", Z, self.E_gamma) + np.einsum(
                    "itq,mtq->imt", self.E_alpha, W
                )

        rate = influence_rate + pref_rate
        # check rate is positive
        if np.sum(rate < 1e-10) > 0:
            tqdm.write(
                f"found {np.sum(rate < 1e-10)} rate params < 1e-10 -- setting to 1e-10"
            )
            rate[rate < 1e-10] = 1e-10
        for Y_t in Y:
            # make sure no neg / zero vals in Y_t
            if (Y_t.data < 0).any():
                tqdm.write(f"found {np.sum(Y_t.data < 0)} Y_t < 0 -- setting to 0")
                Y_t[Y_t < 0] = 0
            if (Y_t.data == 0).any():
                tqdm.write(f"found {np.sum(Y_t.data == 0)} Y_t == 0 -- will drop")
                Y_t[Y_t == 0] = 0
                Y_t.eliminate_zeros()

        # can't directly use sparse in logpmf -- instead only op
        # over non-zero entries
        return np.sum(
            [
                poisson.logpmf(Y_t.data, rate_t[Y_t.nonzero()]).sum()
                for Y_t, rate_t in zip(Y, rate.transpose(2, 0, 1))
            ]
        )

    def _update_gamma(self, Y, Z):
        norm_obs = [
            Y_t / norm_t for Y_t, norm_t in zip(Y, self.normaliser.transpose(2, 0, 1))
        ]
        # want (\sum_t) \sum_i gamma_term_{mtq} z_{itq} y_{im}^t
        if self.time_homog:
            expected_aux = self.gamma_term * reduce(
                lambda x, y: x + y,
                [
                    utils.safe_sparse_toarray(nrm_ob_t.T @ z_t)
                    for nrm_ob_t, z_t in zip(norm_obs, Z.transpose(1, 0, 2))
                ],
            )
        else:
            expected_aux = self.gamma_term * np.stack(
                [
                    utils.safe_sparse_toarray(nrm_ob_t.T @ z_t)
                    for nrm_ob_t, z_t in zip(
                        norm_obs,
                        Z.transpose(1, 0, 2),
                    )
                ],
                axis=1,
            )
        self.gamma_shape = self.topic_shp + expected_aux
        self.E_log_gamma, self.E_gamma = self._compute_expectations(
            self.gamma_shape, self.gamma_rates
        )

    def _update_alpha(self, Y, W):
        norm_obs = [
            Y_t / norm_t for Y_t, norm_t in zip(Y, self.normaliser.transpose(2, 0, 1))
        ]
        # want (\sum_t) \sum_m alpha_term_{itq} w_{mtq} y_{im}^t
        if self.time_homog:
            expected_aux = self.alpha_term * reduce(
                lambda x, y: x + y,
                [
                    utils.safe_sparse_toarray(nrm_ob_t @ w_t)
                    for nrm_ob_t, w_t in zip(norm_obs, W.transpose(1, 0, 2))
                ],
            )
        else:
            expected_aux = self.alpha_term * np.stack(
                [
                    utils.safe_sparse_toarray(nrm_ob_t @ w_t)
                    for nrm_ob_t, w_t in zip(
                        norm_obs,
                        W.transpose(1, 0, 2),
                    )
                ],
                axis=1,
            )
        self.alpha_shape = self.au_shp + expected_aux
        self.E_log_alpha, self.E_alpha = self._compute_expectations(
            self.alpha_shape, self.alpha_rates
        )

    def _update_beta(self, Y, Y_past, A):
        norm_obs = [
            Y_t / norm_t for Y_t, norm_t in zip(Y, self.normaliser.transpose(2, 0, 1))
        ]
        # want (\sum_t) beta_term_{i,t-1} \sum_{j,m} y_{jm}^t a_{ji,t-1} y_{im}^{t-1}
        if self.time_homog:
            expected_aux = self.beta_term * reduce(
                lambda x, y: x + y,
                [
                    utils.safe_sparse_toarray(((A_t.T @ nrm_ob_t) * ytm1).sum(axis=1))
                    for nrm_ob_t, A_t, ytm1 in zip(norm_obs, A, Y_past)
                ],
            )

        else:
            expected_aux = self.beta_term * np.stack(
                [
                    utils.safe_sparse_toarray(((A_t.T @ nrm_ob_t) * ytm1).sum(axis=1))
                    for nrm_ob_t, A_t, ytm1 in zip(norm_obs, A, Y_past)
                ],
                axis=1,
            )
        self.beta_shape = self.inf_shp + expected_aux
        self.E_log_beta, self.E_beta = self._compute_expectations(
            self.beta_shape, self.beta_rates
        )

    def fit(self, Y, A, Z, W, Y_past, Z_trans=None, use_old_subs=True):
        T = len(Y)
        N, M = Y[0].shape
        T += 1

        # K = Z.shape[1]
        Q = self.n_components
        K = self.n_exog_components
        try:
            assert Q == Z.shape[-1]
            assert K == W.shape[-1]
        except AssertionError:
            raise ValueError(
                """
                Dimensions of provided substitutes do not match one
                or more specified number of components
                """
            )

        self._init_beta(N, T)

        if self.mode == "network_preferences":
            self._init_gamma(M, Q, T)

        elif self.mode == "topic":
            self._init_alpha(N, K, T)

        elif self.mode == "full":
            self._init_gamma(M, Q, T)
            self._init_alpha(N, K, T)

        self._init_rates(A, Y_past, Z, W, M, N)
        self._init_expectations()

        old_bd = float("-inf")
        bd = self._compute_elbo(
            Y, A, Y_past, Z, W, Z_trans=Z_trans, use_old_subs=use_old_subs
        )

        for i in range(self.max_iter):
            if self.verbose:
                tqdm.write(f"ELBO: {bd:.4g}")
                sys.stdout.flush()

            old_bd = bd

            self._compute_terms_and_normalizers(
                A, Y_past, Y, Z, W, Z_trans=Z_trans, use_old_subs=use_old_subs
            )
            self._update_beta(Y, Y_past, A)
            if self.mode == "network_preferences":
                self._update_gamma(Y, Z)

            elif self.mode == "topic":
                self._update_alpha(Y, W)

            elif self.mode == "full":
                self._update_gamma(Y, Z)
                self._update_alpha(Y, W)

            bd = self._compute_elbo(
                Y, A, Y_past, Z, W, Z_trans=Z_trans, use_old_subs=use_old_subs
            )

            if (bd - old_bd) / abs(old_bd) < self.tol:
                print(old_bd, bd)
                break


def get_set_overlap(Beta_p, Beta, k=20):
    top = np.argsort(Beta)[-k:]
    top_p = np.argsort(Beta_p)[-k:]
    return np.intersect1d(top, top_p).shape[0] / k


if __name__ == "__main__":
    N = 1000
    Q = 10
    K = Q
    M = 1000
    T = 5

    # TODO: finish fully synthetic data gen (?)
    Z = gamma.rvs(0.5, scale=0.1, size=(N, Q))
    Z_trans = np.random.rand(Q, Q)
    Gamma = gamma.rvs(0.5, scale=0.1, size=(M, Q))
    Alpha = gamma.rvs(0.5, scale=0.1, size=(N, K))
    W = gamma.rvs(0.5, scale=0.1, size=(M, K))
    Beta = gamma.rvs(0.005, scale=10.0, size=N)

    # now need to simulate T-1 As
    A = poisson.rvs(Z.dot(Z.T))
    non_id = 1 - np.identity(N)
    A = A * non_id

    rate_topic = Alpha.dot(W.T)
    rate_pref = Z.dot(Gamma.T)
    # Y_past = poisson.rvs(rate_topic + rate_pref)
    Y_0 = poisson.rvs(rate_topic + rate_pref)  # sample initial time
    rate_inf = sparse.csr_array(Beta * A).dot(sparse.csr_array(Y_0))
    Y = poisson.rvs(rate_pref + rate_inf + rate_topic)  # sample remainder
    # Y_past = Y[...,:-1].copy()
    # Y = Y[...,1:]
    print("Sparsity of data matrices:", A.mean(), Y[..., :-1].mean(), Y[..., 1:].mean())

    pmf = CausalInfluenceModel(n_components=K, n_exog_components=K, verbose=True)
    pmf.fit(Y[..., 1:], A, Z, W, Y[..., :-1], Z_trans)

    print("Beta overlap:", get_set_overlap(pmf.E_beta, Beta))
    print("MSE Beta:", mse(Beta, pmf.E_beta))
    print("MSE Random Beta:", mse(Beta, gamma.rvs(0.1, scale=10.0, size=N)))
