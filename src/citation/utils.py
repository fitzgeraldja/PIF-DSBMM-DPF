import numpy as np


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
