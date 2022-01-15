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
    elif 'scipy' in str(type(values)).lower():
        return values
    elif 'hyperopt' in str(type(values)).lower():
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
                             "Got {}.".format(data_len))

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
                             "Got {}.".format(data_len))

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
    """Generator on parameters sampled from given distributions.
    If all parameters are presented as a list, sampling without replacement is
    performed. If at least one parameter is given as a scipy distribution,
    sampling with replacement is used. If all parameters are given as hyperopt
    distributions Tree of Parzen Estimators searching from hyperopt is computed.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for random sampling (such as those from scipy.stats.distributions)
        or be hyperopt distributions for bayesian searching.
        If a list is given, it is sampled uniformly.

    n_iter : integer, default=None
        Number of parameter configurations that are produced.

    random_state : int, default=None
        Pass an int for reproducible output across multiple
        function calls.

    Returns
    -------
    param_combi : list of dicts or dict of hyperopt distributions
        Parameter combinations.

    searching_type : str
        The searching algorithm used.
    """

    def __init__(self, param_distributions, n_iter=None, random_state=None):

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def sample(self):
        """Generator parameter combinations from given distributions."""

        param_distributions = self.param_distributions.copy()

        is_grid = all(isinstance(p, list)
                      for p in param_distributions.values())
        is_random = all(isinstance(p, list) or 'scipy' in str(type(p)).lower()
                        for p in param_distributions.values())
        is_hyperopt = all('hyperopt' in str(type(p)).lower()
                          or (len(p) < 2 if isinstance(p, list) else False)
                          for p in param_distributions.values())

        if is_grid:
            param_combi = list(product(*param_distributions.values()))
            param_combi = [
                dict(zip(param_distributions.keys(), combi))
                for combi in param_combi
            ]
            return param_combi, 'grid'

        elif is_random:
            if self.n_iter is None:
                raise ValueError(
                    "n_iter must be an integer >0 when scipy parameter "
                    "distributions are provided. Get None."
                )

            seed = (random.randint(1, 100) if self.random_state is None
                    else self.random_state + 1)
            random.seed(seed)

            param_combi = []
            k = self.n_iter
            for i in range(self.n_iter):
                dist = param_distributions.copy()
                combi = []
                for j, v in enumerate(dist.values()):
                    if 'scipy' in str(type(v)).lower():
                        combi.append(v.rvs(random_state=seed * (k + j)))
                    else:
                        combi.append(v[random.randint(0, len(v) - 1)])
                    k += i + j
                param_combi.append(
                    dict(zip(param_distributions.keys(), combi))
                )
            np.random.mtrand._rand

            return param_combi, 'random'

        elif is_hyperopt:
            if self.n_iter is None:
                raise ValueError(
                    "n_iter must be an integer >0 when hyperopt "
                    "search spaces are provided. Get None."
                )
            param_distributions = {
                k: p[0] if isinstance(p, list) else p
                for k, p in param_distributions.items()
            }

            return param_distributions, 'hyperopt'

        else:
            raise ValueError(
                "Parameters not recognized. "
                "Pass lists, scipy distributions (also in conjunction "
                "with lists), or hyperopt search spaces."
            )