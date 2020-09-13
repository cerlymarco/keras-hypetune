import random
import numpy as np
from itertools import product


def _check_param(values):
    
    """
    Check the parameter boundaries passed in dict values.
    
    Returns
    -------
    list of checked parameters.
    """

    if isinstance(values, (list,tuple,np.ndarray)):
        return list(set(values))
    elif hasattr(values, 'rvs'):
        return values
    else:
        return [values]


def _safeformat_str(str, **kwargs):
    
    """
    Safe naming formatting for 'trial' and 'fold' token.
    
    Returns
    -------
    string filled correctly.
    """
    
    class SafeDict(dict):
        def __missing__(self, key):
            return '{' + key + '}'
    
    replacements = SafeDict(**kwargs)
    
    return str.format_map(replacements)


def _get_callback_paths(callbacks):
    
    """
    Extract the saving paths of Keras callbacks that allow the
    possibility to create external files.
    
    Returns
    -------
    list of extracted paths.
    """
    
    paths = []
    
    if isinstance(callbacks, list):
        for c in callbacks:
            if hasattr(c, 'filepath'):
                paths.append(c.filepath)
            elif hasattr(c, 'log_dir'):
                paths.append(c.log_dir)
            elif hasattr(c, 'filename'):
                paths.append(c.filename)
            elif hasattr(c, 'path'):
                paths.append(c.path)
            elif hasattr(c, 'root'):
                paths.append(c.root)
            else:
                paths.append(None)
    else:
        if hasattr(callbacks, 'filepath'):
            paths.append(callbacks.filepath)
        elif hasattr(callbacks, 'log_dir'):
            paths.append(callbacks.log_dir)
        elif hasattr(callbacks, 'filename'):
            paths.append(callbacks.filename)
        elif hasattr(callbacks, 'path'):
            paths.append(callbacks.path)
        elif hasattr(callbacks, 'root'):
            paths.append(callbacks.root)
        else:
            paths.append(None)
            
    return paths


def _clear_callbacks(callbacks, paths, trial, fold, start_score):
    
    """
    Assign the correct saving path to callbacks (if needed) and
    restore the starting score.
    
    Returns
    -------
    list of callbacks.
    """
    
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    
    for i,c in enumerate(callbacks):
        if hasattr(c, 'filepath'):
            c.filepath = _safeformat_str(paths[i], 
                                         trial=trial, fold=fold)
        elif hasattr(c, 'log_dir'):
            c.log_dir = _safeformat_str(paths[i], 
                                        trial=trial, fold=fold)
        elif hasattr(c, 'filename'):
            c.filename = _safeformat_str(paths[i], 
                                         trial=trial, fold=fold)
        elif hasattr(c, 'path'):
            c.path = _safeformat_str(paths[i], 
                                     trial=trial, fold=fold)
        elif hasattr(c, 'root'):
            c.root = _safeformat_str(paths[i], 
                                     trial=trial, fold=fold)
        if hasattr(c, 'best'):
            c.best = start_score

    return callbacks


def _create_fold(X, ids):
    
    """
    Create folds from the data received.
    
    Returns
    -------
    arrays/list or array/dict of arrays containing fold data.
    """
    
    if isinstance(X, list):
        return [x[ids] for x in X]
    
    elif isinstance(X, dict):
        return {k:v[ids] for k,v in X.items()}
    
    else:
        return X[ids]
    

def _check_data(X):
    
    """
    Data controls for cross validation.
    """
    
    if isinstance(X, list):
        for x in X:
            if not isinstance(x, np.ndarray):
                raise ValueError(
                    "Received data in list format. If you are dealing with "
                    "multi-input or multi-output model, take care to cast each "
                    "element of the list to numpy array. In case of single-input or "
                    "single-output, list are not supported: cast them to numpy array.")
    
    elif isinstance(X, dict):
        for x in X.values():
            if not isinstance(x, np.ndarray):
                raise ValueError(
                    "Received data in dict format. Take care to cast each "
                    "value of the dict to numpy array.")
                
    elif isinstance(X, np.ndarray):
        pass
    
    else:
        raise ValueError(
            "Data format not appropriate for Keras CV search. "
            "Supported types are list, dict or numpy array.")
        
        

class ParameterSampler(object):

    # modified from scikit-learn ParameterSampler
    """
    Generator on parameters sampled from given distributions.
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
    n_iter : integer
        Number of parameter settings that are produced.
    random_state : int, default None
        Pass an int for reproducible output across multiple
        function calls.
    
    Returns
    -------
    param_combi : list of tuple
        list of sampled parameter combination
    """

    def __init__(self, param_distributions, n_iter, random_state=None):
        
        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def __init__(self, param_distributions, n_iter, random_state=None):
        
        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def sample(self):
        
        self.param_distributions = self.param_distributions.copy()
        
        for p_k, p_v in self.param_distributions.items():
            self.param_distributions[p_k] = _check_param(p_v)
         
        all_lists = all(not hasattr(p, "rvs") 
                        for p in self.param_distributions.values())
            
        seed = (random.randint(1, 100) if self.random_state is None 
                else self.random_state+1)
        random.seed(seed)
        
        if all_lists:
            param_combi = list(product(*self.param_distributions.values()))
            grid_size = len(param_combi)

            if grid_size < self.n_iter:
                raise ValueError(
                    f"The total space of parameters {grid_size} is smaller "
                    f"than n_iter={self.n_iter}. Try with KerasGridSearch.")
            param_combi = random.sample(param_combi, self.n_iter)

        else:
            param_combi = []
            k = self.n_iter
            for i in range(self.n_iter):
                dist = self.param_distributions
                params = []
                for j,v in enumerate(dist.values()):
                    if hasattr(v, "rvs"):
                        params.append(v.rvs(random_state=seed*(k+j)))
                    else:
                        params.append(v[random.randint(0,len(v)-1)])
                    k += i+j
                param_combi.append(tuple(params))
        
        # reset seed
        np.random.mtrand._rand
                
        return param_combi