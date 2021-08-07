import random
import numpy as np
from itertools import product


def _check_param(values):
    """Check the parameter boundaries passed in dict values.

    Returns
    -------
    list of checked parameters.
    """

    if isinstance(values, (list, tuple, np.ndarray)):
        return list(set(values))

    elif hasattr(values, 'rvs'):
        return values

    else:
        return [values]


def _clear_callbacks(callbacks, trial, fold):
    """Assign the correct saving path to callbacks (if needed).

    Returns
    -------
    list of callbacks.
    """

    file_paths = ['filepath', 'log_dir', 'filename', 'path', 'root']

    for c in callbacks:
        for f in file_paths:
            if hasattr(c, f):
                if fold is not None:
                    setattr(c, f, getattr(c, f).replace('{fold}', str(fold)))
                setattr(c, f, getattr(c, f).replace('{trial}', str(trial)))

    return callbacks


def _create_fold(X, ids):
    """Create folds from the data received.

    Returns
    -------
    Data fold.
    """

    if isinstance(X, list):
        return [x[ids] for x in X]

    elif isinstance(X, dict):
        return {k: v[ids] for k, v in X.items()}

    else:
        return X[ids]


def _check_data(X, is_target=False):
    """Data controls for cross validation."""

    if isinstance(X, list):
        data_len = []
        for x in X:
            if not isinstance(x, np.ndarray):
                raise ValueError(
                    "Received data in list format. Take care to cast each "
                    "value of the list to numpy array.")
            data_len.append(len(x))

        if len(set(data_len)) > 1:
            raise ValueError("Data must have the same cardinality. "
                             "Got {}".format(data_len))

    elif isinstance(X, dict):
        data_len = []
        for x in X.values():
            if not isinstance(x, np.ndarray):
                raise ValueError(
                    "Received data in dict format. Take care to cast each "
                    "value of the dict to numpy array.")
            data_len.append(len(x))

        if len(set(data_len)) > 1:
            raise ValueError("Data must have the same cardinality. "
                             "Got {}".format(data_len))

    elif isinstance(X, np.ndarray):
        x = X
        data_len = [len(x)]

    else:
        raise ValueError(
            "Data format not appropriate for Keras CV search. "
            "Supported types are list, dict or numpy array.")

    if not is_target:
        x = np.zeros(data_len[0])

    return x


def _is_multioutput(y):
    """Check if multioutput task."""

    if isinstance(y, list):
        return len(y) > 1

    elif isinstance(y, dict):
        return len(y) > 1

    else:
        return False


class ParameterSampler(object):
    # modified from scikit-learn ParameterSampler
    """Generator on parameters sampled from given distributions.
    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : integer, default None
        Number of parameter settings that are produced.

    random_state : int, default None
        Pass an int for reproducible output across multiple
        function calls.

    is_random: bool, default True
        If it's a random search.

    Returns
    -------
    param_combi : list of tuple
        list of sampled parameter combination
    """

    def __init__(self, param_distributions, n_iter=None,
                 random_state=None, is_random=False):

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions
        self.is_random = is_random

    def sample(self):

        param_distributions = self.param_distributions.copy()

        all_lists = all(not hasattr(p, "rvs")
                        for p in param_distributions.values())

        seed = (random.randint(1, 100) if self.random_state is None
                else self.random_state + 1)
        random.seed(seed)

        if all_lists:
            param_combi = list(product(*param_distributions.values()))

            if self.is_random:
                grid_size = len(param_combi)
                if grid_size < self.n_iter:
                    raise ValueError(
                        "The total space of parameters {} is smaller "
                        "than n_iter={}. Try with KerasGridSearch.".format(
                            grid_size, self.n_iter))
                param_combi = random.sample(param_combi, self.n_iter)

        else:

            if self.n_iter is None:
                raise ValueError(
                    "n_iter must be an integer >0 when parameter "
                    "distributions are provided. Get None.")

            param_combi = []
            k = self.n_iter
            for i in range(self.n_iter):
                dist = param_distributions.copy()
                params = []
                for j, v in enumerate(dist.values()):
                    if hasattr(v, "rvs"):
                        params.append(v.rvs(random_state=seed * (k + j)))
                    else:
                        params.append(v[random.randint(0, len(v) - 1)])
                    k += i + j
                param_combi.append(tuple(params))

        # reset seed
        np.random.mtrand._rand

        return param_combi