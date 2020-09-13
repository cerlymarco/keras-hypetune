import random
import numpy as np
from itertools import product

from .utils import (ParameterSampler, _check_param, _safeformat_str, _get_callback_paths,
                    _clear_callbacks, _create_fold, _check_data)



class KerasGridSearch(object):
    
    """
    Grid hyperparamater searching and optimization on a fixed validation set.
    
    Pass a Keras model (in Sequential or Functional format), and 
    a dictionary with the parameter boundaries for the experiment.

    For searching, takes in the same arguments available in Keras model.fit(...).
    All the input format supported by Keras model are accepted.
    
    
    Parameters
    ----------
    hypermodel : function
        A callable that takes parameters in dict format and returns a TF Model instance.
    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict keys present 
        in the hypermodel function.
    monitor : str, default val_loss
        Quantity to monitor in order to detect the best model.
    greater_is_better : bool, default False
        Whether the quantity to monitor is a score function, meaning high is good, 
        or a loss function (as default), meaning low is good.
    store_model : bool, default True
        If True the best model is stored inside the KerasGridSearch object.
    savepath : str, default None
        String or path-like, path to save the best model file. If None, no saving is applied.
    tuner_verbose : int, default 1
        0 or 1. Verbosity mode. 0 = silent, 1 = print trial logs with the connected score.
        
        
    Attributes
    ----------
    trials : list
        A list of dicts. The dicts are all the hyperparameter combinations tried and 
        derived from the param_grid 
    scores : list 
        The monitor quantities achived on the validation data by all the models tried.
    best_params : dict, default None
        The dict containing the best combination (in term of score) of hyperparameters.
    best_score : float, default None
        The best score achieved by all the possible combination created.
    best_model : TF Model, default None
        The best model (in term of score). Accessible only if store_model is set to True. 
        
    
    Notes
    ----------
    KerasGridSearch allows the usage of every callbacks available in Keras (also the 
    custom one). The callbacks, that provide the possibility to save any output as
    external files, support naming formatting options. This is true for ModelCheckpoint,
    CSVLogger, TensorBoard and RemoteMonitor. 'trial' is the custom token that can be used
    to personalize the name formatting. 
    
    For example: if filepath in ModelCheckpoint is model_{trial}.hdf5, then the model 
    checkpoints will be saved with the relative number of trial in the filename.
    This enables to save and differentiate each model created in the searching trials. 
    """
    
    def __init__(self,
                 hypermodel,
                 param_grid,
                 monitor='val_loss',
                 greater_is_better=False,
                 store_model=True,
                 savepath=None,
                 tuner_verbose=1):
        
        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose
        self.trials = []
        self.scores = []
        self.best_params = None
        self.best_score = None
        self.best_model = None
        
        
    def set_seed(self,
                 seed_fun,
                 **seedargs):
        
        """
        Pass a function to set the seed in every trial: optional.
        
        Parameters
        ---------- 
        seed_fun : callable, default None
            Function used to set the seed in each trial.
        seedargs : Additional arguments of seed_fun.
            
        Examples
        --------
        >>> def seed_setter(seed):
        >>>     tf.random.set_seed(seed)
        >>>     os.environ['PYTHONHASHSEED'] = str(seed)
        >>>     np.random.seed(seed)
        >>>     random.seed(seed)
        >>>
        >>> kgs = KerasGridSearch(...)
        >>> kgs.set_seed(seed_setter, seed=1234)
        >>> kgs.search(...)
        """
        
        if not callable(seed_fun):
            raise ValueError("seed_fun must be a callable function")
        
        self.seed_fun = seed_fun
        self.seedargs = seedargs
        
    
    def search(self, 
               x, 
               y = None, 
               validation_data = None, 
               validation_split = 0.0, 
               **fitargs):
        
        """
        Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation set provided.
        
        Parameters
        ----------       
        x : multi types
            Input data. All the input format supported by Keras model are accepted.
        y : multi types, default None
            Target data. All the target format supported by Keras model are accepted.
        validation_data : multi types, default None
            Data on which to evaluate the loss and any model metrics at the end of each epoch. 
            All the validation_data format supported by Keras model are accepted.
        validation_split : float, default 0.0
            Float between 0 and 1. Fraction of the training data to be used as validation data.
        **fitargs : Additional fitting arguments, the same accepted in Keras model.fit(...).
        """
        
        # retrive utility params from CV process (if applied)
        fold = self._fold if hasattr(self, '_fold') else ''
        callback_paths = (self._callback_paths if hasattr(self, '_callback_paths') 
                          else '')
        
        if validation_data is None and validation_split == 0.0:
            raise ValueError("Pass at least one of validation_data or validation_split")
            
        if not isinstance(self.param_grid, dict):
            raise ValueError("Pass param_grid in dict format")
        self.param_grid = self.param_grid.copy()
        
        tunable_fitargs = ['batch_size', 'epochs', 'steps_per_epoch', 'class_weight']
            
        if 'callbacks' in fitargs.keys() and fold == '':
            callback_paths = _get_callback_paths(fitargs['callbacks'])
            
        for p_k, p_v in self.param_grid.items():
            self.param_grid[p_k] = _check_param(p_v)
        
        start_score = -np.inf if self.greater_is_better else np.inf
        self.best_score = start_score 

        eval_epoch = np.argmax if self.greater_is_better else np.argmin
        eval_score = np.max if self.greater_is_better else np.min
        
        total_trials = np.prod([len(p) for p in self.param_grid.values()])
        verbose = fitargs['verbose'] if 'verbose' in fitargs.keys() else 0
        
        if self.tuner_verbose == 1:
            print(f"\n{total_trials} trials detected for {tuple(self.param_grid.keys())}")
                
        for trial,param in enumerate(product(*self.param_grid.values())):
            
            if hasattr(self, 'seed_fun'):
                self.seed_fun(**self.seedargs)
                
            if 'callbacks' in fitargs.keys():
                fitargs['callbacks'] = _clear_callbacks(fitargs['callbacks'], 
                                                        callback_paths,
                                                        trial+1, fold,
                                                        start_score)
            
            param = dict(zip(self.param_grid.keys(), param))
            model = self.hypermodel(param)
            
            fit_param = {k:v for k,v in param.items() if k in tunable_fitargs} 
            all_fitargs = dict(list(fitargs.items()) + list(fit_param.items()))
            
            if self.tuner_verbose == 1:
                print(f"\n***** ({trial+1}/{total_trials}) *****\nSearch({param})")
            else:
                verbose = 0
            all_fitargs['verbose'] = verbose
                        
            model.fit(x = x, 
                      y = y, 
                      validation_split = validation_split, 
                      validation_data = validation_data,
                      **all_fitargs)
                                    
            epoch = eval_epoch(model.history.history[self.monitor])
            param['epochs'] = epoch+1
            param['steps_per_epoch'] = model.history.params['steps']
            param['batch_size'] = (all_fitargs['batch_size'] if 'batch_size' 
                                   in all_fitargs.keys() else None)
            score = np.round(model.history.history[self.monitor][epoch],5)
            evaluate = eval_score([self.best_score, score])
                    
            if self.best_score != evaluate:

                self.best_params = param

                if self.store_model:
                    self.best_model = model

                if self.savepath is not None:
                    model.save(self.savepath.format(fold=fold))
            
            self.best_score = evaluate
            self.trials.append(param)
            self.scores.append(score)
            
            if self.tuner_verbose == 1:
                print(f"SCORE: {score} at epoch {epoch+1}")



class KerasRandomSearch(object):
    
    """
    Random hyperparamater searching and optimization on a fixed validation set.
    
    Pass a Keras model (in Sequential or Functional format), and 
    a dictionary with the parameter boundaries for the experiment.
    
    In contrast to grid-search, not all parameter values are tried out, 
    but rather a fixed number of parameter settings is sampled from 
    the specified distributions. The number of parameter settings that 
    are tried is given by n_iter.

    If all parameters are presented as a list, sampling without replacement 
    is performed. If at least one parameter is given as a distribution 
    (random variable from scipy.stats.distribution), sampling with replacement 
    is used. It is highly recommended to use continuous distributions 
    for continuous parameters.

    For searching, takes in the same arguments available in Keras model.fit(...).
    All the input format supported by Keras model are accepted.
    
    
    Parameters
    ----------
    hypermodel : function
        A callable that takes parameters in dict format and returns a TF Model instance.
    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict keys present 
        in the hypermodel function.
    n_iter : int
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.
    sampling_seed : int, default 0
        The seed used to sample from the hyperparameter distributions.
    monitor : str, default val_loss
        Quantity to monitor in order to detect the best model.
    greater_is_better : bool, default False
        Whether the quantity to monitor is a score function, meaning high is good, 
        or a loss function (as default), meaning low is good.
    store_model : bool, default True
        If True the best model is stored inside the KerasRandomSearch object.
    savepath : str, default None
        String or path-like, path to save the best model file. If None, no saving is applied.
    tuner_verbose : int, default 1
        0 or 1. Verbosity mode. 0 = silent, 1 = print trial logs with the connected score.
        
        
    Attributes
    ----------
    trials : list
        A list of dicts. The dicts are all the hyperparameter combinations tried and 
        derived from the param_grid 
    scores : list 
        The monitor quantities achived on the validation data by all the models tried.
    best_params : dict, default None
        The dict containing the best combination (in term of score) of hyperparameters.
    best_score : float, default None
        The best score achieved by all the possible combination created.
    best_model : TF Model, default None
        The best model (in term of score). Accessible only if store_model is set to True. 
        
    
    Notes
    ----------
    KerasRandomSearch allows the usage of every callbacks available in Keras (also the 
    custom one). The callbacks, that provide the possibility to save any output as
    external files, support naming formatting options. This is true for ModelCheckpoint,
    CSVLogger, TensorBoard and RemoteMonitor. 'trial' is the custom token that can be used
    to personalize the name formatting. 
    
    For example: if filepath in ModelCheckpoint is model_{trial}.hdf5, then the model 
    checkpoints will be saved with the relative number of trial in the filename.
    This enables to save and differentiate each model created in the searching trials. 
    """
    
    def __init__(self,
                 hypermodel,
                 param_grid,
                 n_iter,
                 sampling_seed = 0,
                 monitor ='val_loss',
                 greater_is_better = False,
                 store_model = True,
                 savepath = None,
                 tuner_verbose = 1):
        
        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed        
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose
        self.trials = []
        self.scores = []
        self.best_params = None
        self.best_score = None
        self.best_model = None
        
        
    def set_seed(self,
                 seed_fun,
                 **seedargs):
        
        """
        Pass a function to set the seed in every trial: optional.
        
        Parameters
        ---------- 
        seed_fun : callable, default None
            Function used to set the seed in each trial.
        seedargs : Additional arguments of seed_fun.
            
        Examples
        --------
        >>> def seed_setter(seed):
        >>>     tf.random.set_seed(seed)
        >>>     os.environ['PYTHONHASHSEED'] = str(seed)
        >>>     np.random.seed(seed)
        >>>     random.seed(seed)
        >>>
        >>> kgs = KerasRandomSearch(...)
        >>> kgs.set_seed(seed_setter, seed=1234)
        >>> kgs.search(...)
        """
        
        if not callable(seed_fun):
            raise ValueError("seed_fun must be a callable function")
        
        self.seed_fun = seed_fun
        self.seedargs = seedargs
    
    
    def search(self, 
               x, 
               y = None, 
               validation_data = None, 
               validation_split = 0.0, 
               **fitargs):
        
        """
        Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation set provided.
        
        Parameters
        ----------       
        x : multi types
            Input data. All the input format supported by Keras model are accepted.
        y : multi types, default None
            Target data. All the target format supported by Keras model are accepted.
        validation_data : multi types, default None
            Data on which to evaluate the loss and any model metrics at the end of each epoch. 
            All the validation_data format supported by Keras model are accepted.
        validation_split : float, default 0.0
            Float between 0 and 1. Fraction of the training data to be used as validation data.
        **fitargs : Additional fitting arguments, the same accepted in Keras model.fit(...).
        """
        
        # retrive utility params from CV process (if applied)
        fold = self._fold if hasattr(self, '_fold') else ''
        callback_paths = (self._callback_paths if hasattr(self, '_callback_paths') 
                          else '')
        
        if validation_data is None and validation_split == 0.0:
            raise ValueError("Pass at least one of validation_data or validation_split")
            
        if not isinstance(self.param_grid, dict):
            raise ValueError("Pass param_grid in dict format")
        self.param_grid = self.param_grid.copy()
        
        tunable_fitargs = ['batch_size', 'epochs', 'steps_per_epoch', 'class_weight']
            
        if 'callbacks' in fitargs.keys() and fold == '':
            callback_paths = _get_callback_paths(fitargs['callbacks'])
        
        start_score = -np.inf if self.greater_is_better else np.inf
        self.best_score = start_score 

        eval_epoch = np.argmax if self.greater_is_better else np.argmin
        eval_score = np.max if self.greater_is_better else np.min
        
        verbose = fitargs['verbose'] if 'verbose' in fitargs.keys() else 0
                
        rs = ParameterSampler(n_iter = self.n_iter, 
                              param_distributions = self.param_grid,
                              random_state = self.sampling_seed)
        sampled_params = rs.sample()
        
        if self.tuner_verbose == 1:
            print(f"\n{self.n_iter} trials detected for {tuple(self.param_grid.keys())}")
                
        for trial,param in enumerate(sampled_params):
            
            if hasattr(self, 'seed_fun'):
                self.seed_fun(**self.seedargs)
                
            if 'callbacks' in fitargs.keys():
                fitargs['callbacks'] = _clear_callbacks(fitargs['callbacks'], 
                                                        callback_paths,
                                                        trial+1, fold,
                                                        start_score)
            
            param = dict(zip(self.param_grid.keys(), param))
            model = self.hypermodel(param)
            
            fit_param = {k:v for k,v in param.items() if k in tunable_fitargs} 
            all_fitargs = dict(list(fitargs.items()) + list(fit_param.items()))
            
            if self.tuner_verbose == 1:
                print(f"\n***** ({trial+1}/{self.n_iter}) *****\nSearch({param})")
            else:
                verbose = 0
            all_fitargs['verbose'] = verbose
                        
            model.fit(x = x, 
                      y = y, 
                      validation_split = validation_split, 
                      validation_data = validation_data,
                      **all_fitargs)
                                    
            epoch = eval_epoch(model.history.history[self.monitor])
            param['epochs'] = epoch+1
            param['steps_per_epoch'] = model.history.params['steps']
            param['batch_size'] = (all_fitargs['batch_size'] if 'batch_size' 
                                   in all_fitargs.keys() else None)
            score = np.round(model.history.history[self.monitor][epoch],5)
            evaluate = eval_score([self.best_score, score])
                    
            if self.best_score != evaluate:

                self.best_params = param

                if self.store_model:
                    self.best_model = model

                if self.savepath is not None:
                    model.save(self.savepath.format(fold=fold))
            
            self.best_score = evaluate
            self.trials.append(param)
            self.scores.append(score)
            
            if self.tuner_verbose == 1:
                print(f"SCORE: {score} at epoch {epoch+1}")
                


class KerasGridSearchCV(object):
    
    """
    Grid hyperparamater searching and optimization with cross-validation.
    
    Pass a Keras model (in Sequential or Functional format), and 
    a dictionary with the parameter boundaries for the experiment.
    The cross-validation strategies are the same provided by the 
    scikit-learn cross-validation generator.
    
    For searching, takes in the same arguments available in Keras model.fit(...).
    Only input in array format are supported. In case of multi-input or
    multi-output is it possible to wrap arrays in list or dictionaries like in
    Keras.
    
    
    Parameters
    ----------
    hypermodel : function
        A callable that takes parameters in dict format and returns a TF Model instance.
    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict keys present 
        in the hypermodel function.
    cv : scikit-learn cross-validation generator
        An sklearn.model_selection splitter class. Used to determine how samples 
        are split up into groups for cross-validation.
    monitor : str, default val_loss
        Quantity to monitor in order to detect the best models.
    greater_is_better : bool, default False
        Whether the quantity to monitor is a score function, meaning high is good, 
        or a loss function (as default), meaning low is good.
    store_model : bool, default True
        If True the best models are stored inside the KerasGridSearchCV object. The best model
        of each fold is stored.
    savepath : str, default None
        String or path-like, path to save the best model files. If None, no saving is applied. 
        savepath can contain named formatting options ('fold' is a special useful key). 
        For example: if filepath is model_{fold}.h5, then the best model of each fold is saved 
        with the number of the relative fold in the name.
    tuner_verbose : int, default 1
        0 or 1. Verbosity mode. 0 = silent, 1 = print trial logs with the connected score.
        
        
    Attributes
    ----------
    folds_trials : dict
        A dicts of list. The lists contain all the hyperparameter combinations tried 
        in each fold and derived from the param_grid. 
    folds_scores : dict
        A dicts of list. The lists contain the monitor quantities achived on the 
        validation data by all the models tried in each fold.
    folds_best_params : dict
        The dict containing the best combination (in term of score) of hyperparameters 
        in each fold.
    folds_best_score : dict
        The best scores achieved by all the possible combination created in each fold.
    folds_best_model : dict
        The best models (in term of score) in each fold. Accessible only if store_model 
        is set to True. 
    best_params_score : float, default None
        The best average score in all the available folds.
    best_params : dict, default None
        The paramareter combination related to the best average score 
        in all the available folds.
    
    Notes
    ----------
    KerasGridSearchCV allows the usage of every callbacks available in Keras (also the 
    custom one). The callbacks, that provide the possibility to save any output as
    external files, support naming formatting options. This is true for ModelCheckpoint,
    CSVLogger, TensorBoard and RemoteMonitor. 'trial' and 'fold' are custom tokens that 
    can be used to personalize the name formatting. 
    
    For example: if filepath in ModelCheckpoint is model_{fold}_{trial}.hdf5, then 
    the model checkpoints will be saved with the relative number of trial, obtained at
    a certain fold, in the filename. This enables to save and differentiate each model 
    created in the searching trials. 
    """
    
    def __init__(self,
                 hypermodel,
                 param_grid,
                 cv,
                 monitor = 'val_loss',
                 greater_is_better = False,
                 store_model = True,
                 savepath = None,
                 tuner_verbose = 1):
        
        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.cv = cv
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose
        self.folds_trials = {}
        self.folds_scores = {}
        self.folds_best_params = {}
        self.folds_best_score = {}
        self.folds_best_models = {}
        self.best_params_score = None
        self.best_params = None
        
        
    def set_seed(self,
                 seed_fun,
                 **seedargs):
        
        """
        Pass a function to set the seed in every trial: optional.
        
        Parameters
        ---------- 
        seed_fun : callable, default None
            Function used to set the seed in each trial.
        seedargs : Additional arguments of seed_fun.
            
        Examples
        --------
        >>> def seed_setter(seed):
        >>>     tf.random.set_seed(seed)
        >>>     os.environ['PYTHONHASHSEED'] = str(seed)
        >>>     np.random.seed(seed)
        >>>     random.seed(seed)
        >>>
        >>> kgs = KerasGridSearchCV(...)
        >>> kgs.set_seed(seed_setter, seed=1234)
        >>> kgs.search(...)
        """

        if not callable(seed_fun):
            raise ValueError("seed_fun must be a callable function")
        
        self.seed_fun = seed_fun
        self.seedargs = seedargs
        
        
    def search(self, 
               x, 
               y, 
               sample_weight = None, 
               groups = None, 
               **fitargs): 
        
        """
        Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation folder created
        following the validation strategy.
        
        Parameters
        ----------       
        x : multi types
            Input data. Accepted types are arrays or list/dict in case of multi-input/output.
        y : multi types, default None
            Target data. Accepted types are arrays or list/dict in case of multi-input/output.
        sample_weight : multi types, default None
            Optional Numpy array of weights for the training samples, used for weighting 
            the loss function (during training only). Accepted types are arrays or 
            list/dict in case of multi-input/output
        groups : array-like, default None
            Group labels for the samples used while splitting the dataset into train/valid set.
        **fitargs : Additional fitting arguments, the same accepted in Keras model.fit(...).
            The validation set is automatically created accordingly the cv strategy.
        """
                
        if 'validation_split' in fitargs.keys() or 'validation_data' in fitargs.keys():
            raise ValueError("Validation is automatically created by the cv strategy")
        
        _check_data(x)
        _check_data(y)
        if sample_weight is not None: _check_data(sample_weight)
        
        for fold,(train_id,val_id) in enumerate(self.cv.split(x, y, groups)):
            
            if self.tuner_verbose == 1:
                print("\n{}\n{}  Fold {}  {}\n{}".format(
                    '#'*18, '#'*3, str(fold+1).zfill(3), '#'*3, '#'*18))
            
            if 'callbacks' in fitargs.keys() and fold == 0:
                callback_paths = _get_callback_paths(fitargs['callbacks'])
                            
            x_train = _create_fold(x, train_id)
            y_train = _create_fold(y, train_id)
            sample_weight_train = (_create_fold(sample_weight, train_id) if sample_weight 
                                   is not None else None)
            
            x_val = _create_fold(x, val_id)
            y_val = _create_fold(y, val_id)
            sample_weight_val = (_create_fold(sample_weight, val_id) if sample_weight 
                                 is not None else None)
        
            kgs_fold = KerasGridSearch(hypermodel = self.hypermodel,   
                                       param_grid = self.param_grid,
                                       monitor = self.monitor,
                                       greater_is_better = self.greater_is_better,
                                       store_model = self.store_model,
                                       savepath = self.savepath,
                                       tuner_verbose = self.tuner_verbose)
            
            kgs_fold._fold = fold+1
            if 'callbacks' in fitargs.keys():
                kgs_fold._callback_paths = callback_paths
            
            if hasattr(self, 'seed_fun'):
                kgs_fold.set_seed(self.seed_fun, **self.seedargs)

            kgs_fold.search(x = x_train, 
                            y = y_train, 
                            sample_weight = sample_weight_train,
                            validation_data = (x_val, y_val, sample_weight_val),
                            **fitargs)
                                    
            self.folds_trials[f"fold {fold+1}"] = kgs_fold.trials
            self.folds_scores[f"fold {fold+1}"] = kgs_fold.scores
            self.folds_best_params[f"fold {fold+1}"] = kgs_fold.best_params
            if self.store_model:
                self.folds_best_models[f"fold {fold+1}"] = kgs_fold.best_model
            self.folds_best_score[f"fold {fold+1}"] = kgs_fold.best_score
            
        eval_score = np.argmax if self.greater_is_better else np.argmin
        mean_score_params = np.mean(list(self.folds_scores.values()), axis=0).round(5)
        evaluate = eval_score(mean_score_params)
        
        self.best_params = [list(f)[evaluate] for f in self.folds_trials.values()]
        self.best_params_score = mean_score_params[evaluate]
        
        
        
class KerasRandomSearchCV(object):
    
    """
    Random hyperparamater searching and optimization with cross-validation.
    
    Pass a Keras model (in Sequential or Functional format), and 
    a dictionary with the parameter boundaries for the experiment.
    The cross-validation strategies are the same provided by the 
    scikit-learn cross-validation generator.
    
    In contrast to grid-search, not all parameter values are tried out, 
    but rather a fixed number of parameter settings is sampled from 
    the specified distributions. The number of parameter settings that 
    are tried is given by n_iter.

    If all parameters are presented as a list, sampling without replacement 
    is performed. If at least one parameter is given as a distribution 
    (random variable from scipy.stats.distribution), sampling with replacement 
    is used. It is highly recommended to use continuous distributions 
    for continuous parameters.
    
    For searching, takes in the same arguments available in Keras model.fit(...).
    Only input in array format are supported. In case of multi-input or
    multi-output is it possible to wrap arrays in list or dictionaries like in
    Keras.
    
    
    Parameters
    ----------
    hypermodel : function
        A callable that takes parameters in dict format and returns a TF Model instance.
    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict keys present 
        in the hypermodel function.
    cv : scikit-learn cross-validation generator
        An sklearn.model_selection splitter class. Used to determine how samples 
        are split up into groups for cross-validation.
    n_iter : int
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.
    sampling_seed : int, default 0
        The seed used to sample from the hyperparameter distributions.
    monitor : str, default val_loss
        Quantity to monitor in order to detect the best models.
    greater_is_better : bool, default False
        Whether the quantity to monitor is a score function, meaning high is good, 
        or a loss function (as default), meaning low is good.
    store_model : bool, default True
        If True the best models are stored inside the KerasRandomSearchCV object. The best model
        of each fold is stored.
    savepath : str, default None
        String or path-like, path to save the best model files. If None, no saving is applied. 
        savepath can contain named formatting options ('fold' is a special useful key). 
        For example: if filepath is model_{fold}.h5, then the best model of each fold is saved 
        with the number of the relative fold in the name.
    tuner_verbose : int, default 1
        0 or 1. Verbosity mode. 0 = silent, 1 = print trial logs with the connected score.
        
        
    Attributes
    ----------
    folds_trials : dict
        A dicts of list. The lists contain all the hyperparameter combinations tried 
        in each fold and derived from the param_grid. 
    folds_scores : dict
        A dicts of list. The lists contain the monitor quantities achived on the 
        validation data by all the models tried in each fold.
    folds_best_params : dict
        The dict containing the best combination (in term of score) of hyperparameters 
        in each fold.
    folds_best_score : dict
        The best scores achieved by all the possible combination created in each fold.
    folds_best_model : dict
        The best models (in term of score) in each fold. Accessible only if store_model 
        is set to True. 
    best_params_score : float, default None
        The best average score in all the available folds.
    best_params : dict, default None
        The paramareter combination related to the best average score 
        in all the available folds.
    
    Notes
    ----------
    KerasRandomSearchCV allows the usage of every callbacks available in keras (also the 
    custom one). The callbacks, that provide the possibility to save any output as
    external files, support naming formatting options. This is true for ModelCheckpoint,
    CSVLogger, TensorBoard and RemoteMonitor. 'trial' and 'fold' are custom tokens that 
    can be used to personalize the name formatting. 
    
    For example: if filepath in ModelCheckpoint is model_{fold}_{trial}.hdf5, then 
    the model checkpoints will be saved with the relative number of trial, obtained at
    a certain fold, in the filename. This enables to save and differentiate each model 
    created in the searching trials. 
    """
    
    def __init__(self,
                 hypermodel,
                 param_grid,
                 cv,
                 n_iter,
                 sampling_seed = 0,
                 monitor = 'val_loss',
                 greater_is_better = False,
                 store_model = True,
                 savepath = None,
                 tuner_verbose = 1):
        
        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.cv = cv
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose
        self.folds_trials = {}
        self.folds_scores = {}
        self.folds_best_params = {}
        self.folds_best_score = {}
        self.folds_best_models = {}
        self.best_params_score = None
        self.best_params = None
        
        
    def set_seed(self,
                 seed_fun,
                 **seedargs):
        
        """
        Pass a function to set the seed in every trial: optional.
        
        Parameters
        ---------- 
        seed_fun : callable, default None
            Function used to set the seed in each trial.
        seedargs : Additional arguments of seed_fun.
            
        Examples
        --------
        >>> def seed_setter(seed):
        >>>     tf.random.set_seed(seed)
        >>>     os.environ['PYTHONHASHSEED'] = str(seed)
        >>>     np.random.seed(seed)
        >>>     random.seed(seed)
        >>>
        >>> kgs = KerasRandomSearchCV(...)
        >>> kgs.set_seed(seed_setter, seed=1234)
        >>> kgs.search(...)
        """

        if not callable(seed_fun):
            raise ValueError("seed_fun must be a callable function")
        
        self.seed_fun = seed_fun
        self.seedargs = seedargs
        
        
    def search(self, 
               x, 
               y, 
               sample_weight = None, 
               groups = None, 
               **fitargs): 
        
        """
        Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation folder created
        following the validation strategy.
        
        Parameters
        ----------       
        x : multi types
            Input data. Accepted types are arrays or list/dict in case of multi-input/output.
        y : multi types, default None
            Target data. Accepted types are arrays or list/dict in case of multi-input/output.
        sample_weight : multi types, default None
            Optional Numpy array of weights for the training samples, used for weighting 
            the loss function (during training only). Accepted types are arrays or 
            list/dict in case of multi-input/output
        groups : array-like, default None
            Group labels for the samples used while splitting the dataset into train/valid set.
        **fitargs : Additional fitting arguments, the same accepted in Keras model.fit(...).
            The validation set is automatically created accordingly the cv strategy.
        """
                
        if 'validation_split' in fitargs.keys() or 'validation_data' in fitargs.keys():
            raise ValueError("Validation is automatically created by the cv strategy")
        
        _check_data(x)
        _check_data(y)
        if sample_weight is not None: _check_data(sample_weight)
        
        for fold,(train_id,val_id) in enumerate(self.cv.split(x, y, groups)):
            
            if self.tuner_verbose == 1:
                print("\n{}\n{}  Fold {}  {}\n{}".format(
                    '#'*18, '#'*3, str(fold+1).zfill(3), '#'*3, '#'*18))
            
            if 'callbacks' in fitargs.keys() and fold == 0:
                callback_paths = _get_callback_paths(fitargs['callbacks'])
                            
            x_train = _create_fold(x, train_id)
            y_train = _create_fold(y, train_id)
            sample_weight_train = (_create_fold(sample_weight, train_id) if sample_weight 
                                   is not None else None)
            
            x_val = _create_fold(x, val_id)
            y_val = _create_fold(y, val_id)
            sample_weight_val = (_create_fold(sample_weight, val_id) if sample_weight 
                                 is not None else None)
                        
            kgs_fold = KerasRandomSearch(hypermodel = self.hypermodel,   
                                         param_grid = self.param_grid,
                                         n_iter = self.n_iter,
                                         sampling_seed = self.sampling_seed,
                                         monitor = self.monitor,
                                         greater_is_better = self.greater_is_better,
                                         store_model = self.store_model,
                                         savepath = self.savepath,
                                         tuner_verbose = self.tuner_verbose)
            
            kgs_fold._fold = fold+1
            if 'callbacks' in fitargs.keys():
                kgs_fold._callback_paths = callback_paths
            
            if hasattr(self, 'seed_fun'):
                kgs_fold.set_seed(self.seed_fun, **self.seedargs)

            kgs_fold.search(x = x_train, 
                            y = y_train, 
                            sample_weight = sample_weight_train,
                            validation_data = (x_val, y_val, sample_weight_val),
                            **fitargs)
                                    
            self.folds_trials[f"fold {fold+1}"] = kgs_fold.trials
            self.folds_scores[f"fold {fold+1}"] = kgs_fold.scores
            self.folds_best_params[f"fold {fold+1}"] = kgs_fold.best_params
            if self.store_model:
                self.folds_best_models[f"fold {fold+1}"] = kgs_fold.best_model
            self.folds_best_score[f"fold {fold+1}"] = kgs_fold.best_score
            
        eval_score = np.argmax if self.greater_is_better else np.argmin
        mean_score_params = np.mean(list(self.folds_scores.values()), axis=0).round(5)
        evaluate = eval_score(mean_score_params)
        
        self.best_params = [list(f)[evaluate] for f in self.folds_trials.values()]
        self.best_params_score = mean_score_params[evaluate]