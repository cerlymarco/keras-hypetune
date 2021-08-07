import random
import inspect
import numpy as np
from copy import deepcopy

from .utils import (ParameterSampler, _check_param, _check_data,
                    _clear_callbacks, _create_fold, _is_multioutput)


class _KerasSearch:
    """Base class for KerasSearch meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 hypermodel,
                 param_grid,
                 n_iter=None,
                 sampling_seed=None,
                 monitor='val_loss',
                 greater_is_better=False,
                 store_model=True,
                 savepath=None,
                 tuner_verbose=1):

        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose

    def __repr__(self):
        return "<kerashypetune.{}>".format(self.__class__.__name__)

    def __str__(self):
        return "<kerashypetune.{}>".format(self.__class__.__name__)

    def _search(self,
                x, y=None,
                validation_data=None,
                validation_split=0.0,
                is_random=False,
                id_fold=None,
                **fitargs):
        """Private method to perform a search on a fixed validation set for
        the best parameters configuration."""

        self.trials = []
        self.scores = []

        if validation_data is None and validation_split == 0.0:
            raise ValueError(
                "Pass at least one of validation_data or validation_split.")

        if not isinstance(self.param_grid, dict):
            raise ValueError("Pass param_grid in dict format.")
        self._param_grid = self.param_grid.copy()

        tunable_fitargs = ['batch_size', 'epochs',
                           'steps_per_epoch', 'class_weight']

        for p_k, p_v in self._param_grid.items():
            self._param_grid[p_k] = _check_param(p_v)

        eval_epoch = np.argmax if self.greater_is_better else np.argmin
        eval_score = np.max if self.greater_is_better else np.min
        start_score = -np.inf if self.greater_is_better else np.inf
        self.best_score = start_score

        rs = ParameterSampler(n_iter=self.n_iter,
                              param_distributions=self._param_grid,
                              random_state=self.sampling_seed,
                              is_random=is_random)
        self._param_combi = rs.sample()

        if 'callbacks' in fitargs:
            if isinstance(fitargs['callbacks'], list):
                _callbacks = deepcopy(fitargs['callbacks'])
            else:
                _callbacks = deepcopy([fitargs['callbacks']])

        if self.tuner_verbose > 0:
            print("\n{} trials detected for {}".format(
                len(self._param_combi), tuple(self._param_grid.keys())))
            verbose = fitargs['verbose'] if 'verbose' in fitargs else 0
        else:
            verbose = 0
        fitargs['verbose'] = verbose

        for trial, param in enumerate(self._param_combi):

            if 'callbacks' in fitargs:
                fitargs['callbacks'] = _clear_callbacks(
                    deepcopy(_callbacks), trial + 1, id_fold)

            param = dict(zip(self._param_grid.keys(), param))
            model = self.hypermodel(param)

            fit_param = {k: v for k, v in param.items() if k in tunable_fitargs}
            all_fitargs = dict(list(fitargs.items()) + list(fit_param.items()))

            if self.tuner_verbose > 0:
                print("\n***** ({}/{}) *****\nSearch({})".format(
                    trial + 1, len(self._param_combi), param))

            model.fit(x=x,
                      y=y,
                      validation_split=validation_split,
                      validation_data=validation_data,
                      **all_fitargs)

            epoch = eval_epoch(model.history.history[self.monitor])
            param['epochs'] = epoch + 1
            param['steps_per_epoch'] = model.history.params['steps']
            param['batch_size'] = (all_fitargs['batch_size'] if 'batch_size'
                                   in all_fitargs else None)
            score = round(model.history.history[self.monitor][epoch], 5)
            evaluate = eval_score([self.best_score, score])

            if self.best_score != evaluate:

                self.best_params = param

                if self.store_model:
                    self.best_model = model

                if self.savepath is not None:
                    if id_fold is not None:
                        model.save(self.savepath.replace('{fold}', str(id_fold)))
                    else:
                        model.save(self.savepath)

            self.best_score = evaluate
            self.trials.append(param)
            self.scores.append(score)

            if self.tuner_verbose > 0:
                print("SCORE: {} at epoch {}".format(score, epoch + 1))

        return self


class KerasGridSearch(_KerasSearch):
    """Grid hyperparamater searching and optimization on a fixed
    validation set.

    Pass a Keras model (in Sequential or Functional format), and
    a dictionary with the parameter boundaries for the experiment.
    For searching, takes in the same arguments available in Keras
    model.fit(...). All the input format supported by Keras model
    are accepted.

    Parameters
    ----------
    hypermodel : callable
        A callable that takes parameters in dict format and returns a
        TF Model instance.

    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict
        keys present in the hypermodel function.

    monitor : str, default='val_loss'
        Quantity to monitor in order to detect the best model.

    greater_is_better : bool, default=False
        Whether the quantity to monitor is a score function, meaning high
        is good, or a loss function (as default), meaning low is good.

    store_model : bool, default=True
        If True the best model is stored in the KerasGridSearch object.

    savepath : str, default=None
        String or path-like, path to save the best model file.
        If None, no saving is applied.

    tuner_verbose : int, default=1
        Verbosity mode. <=0 silent all; >0 print trial logs with the
        connected score.

    Attributes
    ----------
    trials : list
        A list of dicts. The dicts are all the hyperparameter combinations
        tried and derived from the param_grid.

    scores : list
        The monitor quantities achived on the validation data by all the
        models tried.

    best_params : dict
        The dict containing the best combination (in term of score) of
        hyperparameters.

    best_score : float
        The best score achieved by all the possible combination created.

    best_model : TF Model
        The best model (in term of score). Accessible only if store_model
        is set to True.

    Notes
    ----------
    KerasGridSearch allows the usage of every callbacks available in Keras
    (also the custom one). The callbacks, that provide the possibility to
    save any output as external files, support naming formatting options.
    This is true for ModelCheckpoint, CSVLogger, TensorBoard and RemoteMonitor.
    'trial' is the custom token that can be used to personalize name formatting.

    For example: if filepath in ModelCheckpoint is model_{trial}.hdf5, then
    the model checkpoints will be saved with the relative number of trial in
    the filename. This enables to save and differentiate each model created in
    the searching trials.
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
        self.n_iter = None
        self.sampling_seed = None

    def search(self,
               x, y=None,
               validation_data=None,
               validation_split=0.0,
               **fitargs):
        """Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation set provided.

        Parameters
        ----------
        x : multi types
            Input data. All the input formats supported by Keras model are
            accepted.

        y : multi types, default=None
            Target data. All the target formats supported by Keras model are
            accepted.

        validation_data : multi types, default=None
            Data on which evaluate the loss and any model metrics at the end of
            each epoch. All the validation_data formats supported by Keras model
            are accepted.

        validation_split : float, default=0.0
            Float between 0 and 1. Fraction of the training data to be used as
            validation data.

        **fitargs : Additional fitting arguments, the same accepted in Keras
                    model.fit(...).

        Returns
        -------
        self : object
        """

        self._search(x=x, y=y,
                     validation_data=validation_data,
                     validation_split=validation_split,
                     is_random=False,
                     **fitargs)

        return self


class KerasRandomSearch(_KerasSearch):
    """Random hyperparamater searching and optimization on a fixed
    validation set.

    Pass a Keras model (in Sequential or Functional format), and
    a dictionary with the parameter boundaries for the experiment.
    For searching, takes in the same arguments available in Keras
    model.fit(...). All the input format supported by Keras model
    are accepted.

    In contrast to grid-search, not all parameter values are tried out,
    but rather a fixed number of parameter settings is sampled from
    the specified distributions. The number of parameter settings that
    are tried is given by n_iter.
    If all parameters are presented as a list/floats/integers, sampling
    without replacement is performed. If at least one parameter is given
    as a distribution (random variable from scipy.stats.distribution),
    sampling with replacement is used. It is highly recommended to use
    continuous distributions for continuous parameters.

    Parameters
    ----------
    hypermodel : callable
        A callable that takes parameters in dict format and returns a
        TF Model instance.

    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict
        keys present in the hypermodel function.

    n_iter : int
        Number of parameter settings that are sampled.
        n_iter trades off runtime vs quality of the solution.

    sampling_seed : int, default=0
        The seed used to sample from the hyperparameter distributions.

    monitor : str, default='val_loss'
        Quantity to monitor in order to detect the best model.

    greater_is_better : bool, default=False
        Whether the quantity to monitor is a score function, meaning high
        is good, or a loss function (as default), meaning low is good.

    store_model : bool, default=True
        If True the best model is stored in the KerasGridSearch object.

    savepath : str, default=None
        String or path-like, path to save the best model file.
        If None, no saving is applied.

    tuner_verbose : int, default=1
        Verbosity mode. <=0 silent all; >0 print trial logs with the
        connected score.

    Attributes
    ----------
    trials : list
        A list of dicts. The dicts are all the hyperparameter combinations
        tried and derived from the param_grid.

    scores : list
        The monitor quantities achived on the validation data by all the
        models tried.

    best_params : dict
        The dict containing the best combination (in term of score) of
        hyperparameters.

    best_score : float
        The best score achieved by all the possible combination created.

    best_model : TF Model
        The best model (in term of score). Accessible only if store_model
        is set to True.

    Notes
    ----------
    KerasGridSearch allows the usage of every callbacks available in Keras
    (also the custom one). The callbacks, that provide the possibility to
    save any output as external files, support naming formatting options.
    This is true for ModelCheckpoint, CSVLogger, TensorBoard and RemoteMonitor.
    'trial' is the custom token that can be used to personalize name formatting.

    For example: if filepath in ModelCheckpoint is model_{trial}.hdf5, then
    the model checkpoints will be saved with the relative number of trial in
    the filename. This enables to save and differentiate each model created in
    the searching trials.
    """

    def __init__(self,
                 hypermodel,
                 param_grid,
                 n_iter,
                 sampling_seed=0,
                 monitor='val_loss',
                 greater_is_better=False,
                 store_model=True,
                 savepath=None,
                 tuner_verbose=1):
      
        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.sampling_seed = sampling_seed
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose

    def search(self,
               x, y=None,
               validation_data=None,
               validation_split=0.0,
               **fitargs):
        """Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation set provided.

        Parameters
        ----------
        x : multi types
            Input data. All the input formats supported by Keras model are
            accepted.

        y : multi types, default=None
            Target data. All the target formats supported by Keras model are
            accepted.

        validation_data : multi types, default=None
            Data on which evaluate the loss and any model metrics at the end of
            each epoch. All the validation_data formats supported by Keras model
            are accepted.

        validation_split : float, default=0.0
            Float between 0 and 1. Fraction of the training data to be used as
            validation data.

        **fitargs : Additional fitting arguments, the same accepted in Keras
                    model.fit(...).

        Returns
        -------
        self : object
        """

        self._search(x=x, y=y,
                     validation_data=validation_data,
                     validation_split=validation_split,
                     is_random=True,
                     **fitargs)

        return self


class _KerasSearchCV:
    """Base class for KerasSearchCV meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 hypermodel,
                 param_grid,
                 cv,
                 n_iter=None,
                 sampling_seed=None,
                 monitor='val_loss',
                 greater_is_better=False,
                 store_model=True,
                 savepath=None,
                 tuner_verbose=1):

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

    def __repr__(self):
        return "<kerashypetune.{}>".format(self.__class__.__name__)

    def __str__(self):
        return "<kerashypetune.{}>".format(self.__class__.__name__)

    def _search(self,
                x, y=None,
                sample_weight=None,
                groups=None,
                is_random=False,
                **fitargs):
        """Private method to perform a CV search for the best parameters
        configuration."""

        self.folds_trials = {}
        self.folds_scores = {}
        self.folds_best_params = {}
        self.folds_best_score = {}
        if self.store_model:
            self.folds_best_models = {}

        if 'validation_split' in fitargs or 'validation_data' in fitargs:
            raise ValueError(
                "Validation is automatically created by the cv strategy.")

        if not hasattr(self.cv, 'split'):
            raise ValueError(
                "Expected cv as cross-validation object with split method to "
                "generate indices to split data into training and test set "
                "(like from sklearn.model_selection).")
        else:
            split_args = inspect.signature(self.cv.split).parameters
            split_args = {k: v.default for k, v in split_args.items()}
            split_need_y = split_args['y'] is not None

        _x = _check_data(x)

        if y is not None:
            _y = _check_data(y, is_target=True)
            if _is_multioutput(y) and split_need_y:
                raise ValueError(
                    "{} not supports multioutput.".format(self.cv))
        else:
            _y = None

        if sample_weight is not None:
            _ = _check_data(sample_weight)

        for fold, (train_id, val_id) in enumerate(self.cv.split(_x, _y, groups)):

            if self.tuner_verbose > 0:
                print("\n{}\n{}  Fold {}  {}\n{}".format(
                    '#' * 18, '#' * 3, str(fold + 1).zfill(3), '#' * 3, '#' * 18))

            x_train = _create_fold(x, train_id)
            y_train = None if y is None else _create_fold(y, train_id)
            sample_weight_train = (_create_fold(sample_weight, train_id)
                                   if sample_weight is not None else None)

            x_val = _create_fold(x, val_id)
            y_val = None if y is None else _create_fold(y, val_id)
            sample_weight_val = (_create_fold(sample_weight, val_id)
                                 if sample_weight is not None else None)

            ks_fold = _KerasSearch(hypermodel=self.hypermodel,
                                   param_grid=self.param_grid,
                                   n_iter=self.n_iter,
                                   sampling_seed=self.sampling_seed,
                                   monitor=self.monitor,
                                   greater_is_better=self.greater_is_better,
                                   store_model=self.store_model,
                                   savepath=self.savepath,
                                   tuner_verbose=self.tuner_verbose)

            ks_fold._search(x=x_train, y=y_train,
                            sample_weight=sample_weight_train,
                            validation_data=(x_val, y_val, sample_weight_val),
                            is_random=is_random,
                            id_fold=fold + 1,
                            **fitargs)

            fold_id = "fold {}".format(fold + 1)
            self.folds_trials[fold_id] = ks_fold.trials
            self.folds_scores[fold_id] = ks_fold.scores
            self.folds_best_params[fold_id] = ks_fold.best_params
            if self.store_model:
                self.folds_best_models[fold_id] = ks_fold.best_model
            self.folds_best_score[fold_id] = ks_fold.best_score

        eval_score = np.argmax if self.greater_is_better else np.argmin
        mean_score_params = np.mean(
            list(self.folds_scores.values()), axis=0).round(5)
        evaluate = eval_score(mean_score_params)

        self.best_params = [list(f)[evaluate] for f in self.folds_trials.values()]
        self.best_params_score = mean_score_params[evaluate]

        return self


class KerasGridSearchCV(_KerasSearchCV):
    """Grid hyperparamater searching and optimization on a fixed
    validation set.

    Pass a Keras model (in Sequential or Functional format), and
    a dictionary with the parameter boundaries for the experiment.
    For searching, takes in the same arguments available in Keras
    model.fit(...). The cross-validation strategies are the same
    provided by the scikit-learn cross-validation generator.
    Only input in array format are supported. In case of multi-input or
    multi-output is it possible to wrap arrays in list or dictionaries
    like in Keras.

    Parameters
    ----------
    hypermodel : callable
        A callable that takes parameters in dict format and returns a
        TF Model instance.

    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict
        keys present in the hypermodel function.

    cv : scikit-learn cross-validation generator
        An sklearn.model_selection splitter class. Used to determine
        how samples are split up into groups for cross-validation.

    monitor : str, default='val_loss'
        Quantity to monitor in order to detect the best model.

    greater_is_better : bool, default=False
        Whether the quantity to monitor is a score function, meaning high
        is good, or a loss function (as default), meaning low is good.

    store_model : bool, default=True
        If True the best model of each fold is stored inside the
        KerasGridSearchCV object.

    savepath : str, default=None
        String or path-like, path to save the best model file.
        If None, no saving is applied.

    tuner_verbose : int, default=1
        Verbosity mode. <=0 silent all; >0 print trial logs with the
        connected score.

    Attributes
    ----------
    folds_trials : dict
        A dicts of list. The lists contain all the hyperparameter combinations
        tried in each fold and derived from the param_grid.

    folds_scores : dict
        A dicts of list. The lists contain the monitor quantities achived on
        the validation data by all the models tried in each fold.

    folds_best_params : dict
        The dict containing the best combination (in term of score) of
        hyperparameters in each fold.

    folds_best_score : dict
        The best scores achieved by all the possible combination created in
        each fold.

    folds_best_model : dict
        The best models (in term of score) in each fold. Accessible only if
        store_model is set to True.

    best_params_score : float
        The best average score in all the available folds.

    best_params : dict
        The paramareter combination related to the best average score in all
        the available folds.


    Notes
    ----------
    KerasGridSearchCV allows the usage of every callbacks available in Keras
    (also the custom one). The callbacks, that provide the possibility to
    save any output as external files, support naming formatting options.
    This is true for ModelCheckpoint, CSVLogger, TensorBoard and RemoteMonitor.
    'trial' and 'fold' are custom tokens that can be used to personalize the
    name formatting.

    For example: if filepath in ModelCheckpoint is model_{fold}_{trial}.hdf5,
    then the model checkpoints will be saved with the relative number of trial,
    obtained at a certain fold, in the filename. This enables to save and
    differentiate each model created in the searching trials.
    """

    def __init__(self,
                 hypermodel,
                 param_grid,
                 cv,
                 monitor='val_loss',
                 greater_is_better=False,
                 store_model=True,
                 savepath=None,
                 tuner_verbose=1):
      
        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.cv = cv
        self.monitor = monitor
        self.greater_is_better = greater_is_better
        self.store_model = store_model
        self.savepath = savepath
        self.tuner_verbose = tuner_verbose
        self.n_iter = None
        self.sampling_seed = None

    def search(self,
               x, y=None,
               sample_weight=None,
               groups=None,
               **fitargs):
        """Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation folds
        created following the validation strategy.

        Parameters
        ----------
        x : multi types
            Input data. Accepted types are arrays or list/dict in case of
            multi-input/output.

        y : multi types, default=None
            Target data. Accepted types are arrays or list/dict in case of
            multi-input/output.

        sample_weight : multi types, default=None
            Optional Numpy array of weights for the training samples, used
            for weighting the loss function (during training only). Accepted
            types are arrays or list/dict in case of multi-input/output.

        groups : array-like, default=None
            Group labels for the samples used while splitting the dataset into
            train/valid set.

        **fitargs : Additional fitting arguments, the same accepted in Keras
                    model.fit(...).
            The validation set is automatically created accordingly to the
            cv strategy.

        Returns
        -------
        self : object
        """

        self._search(x=x, y=y,
                     sample_weight=sample_weight,
                     groups=groups,
                     is_random=False,
                     **fitargs)

        return self


class KerasRandomSearchCV(_KerasSearchCV):
    """Random hyperparamater searching and optimization on a fixed
    validation set.

    Pass a Keras model (in Sequential or Functional format), and
    a dictionary with the parameter boundaries for the experiment.
    For searching, takes in the same arguments available in Keras
    model.fit(...). The cross-validation strategies are the same
    provided by the scikit-learn cross-validation generator.
    Only input in array format are supported. In case of multi-input or
    multi-output is it possible to wrap arrays in list or dictionaries
    like in Keras.

    In contrast to grid-search, not all parameter values are tried out,
    but rather a fixed number of parameter settings is sampled from
    the specified distributions. The number of parameter settings that
    are tried is given by n_iter.
    If all parameters are presented as a list/floats/integers, sampling
    without replacement is performed. If at least one parameter is given
    as a distribution (random variable from scipy.stats.distribution),
    sampling with replacement is used. It is highly recommended to use
    continuous distributions for continuous parameters.

    Parameters
    ----------
    hypermodel : callable
        A callable that takes parameters in dict format and returns a
        TF Model instance.

    param_grid : dict
        Hyperparameters to try, 1-to-1 mapped with the parameters dict
        keys present in the hypermodel function.

    cv : scikit-learn cross-validation generator
        An sklearn.model_selection splitter class. Used to determine
        how samples are split up into groups for cross-validation.

    n_iter : int
        Number of parameter settings that are sampled.
        n_iter trades off runtime vs quality of the solution.

    sampling_seed : int, default=0
        The seed used to sample from the hyperparameter distributions.

    monitor : str, default='val_loss'
        Quantity to monitor in order to detect the best model.

    greater_is_better : bool, default=False
        Whether the quantity to monitor is a score function, meaning high
        is good, or a loss function (as default), meaning low is good.

    store_model : bool, default=True
        If True the best model of each fold is stored inside the
        KerasRandomSearchCV object.

    savepath : str, default=None
        String or path-like, path to save the best model file.
        If None, no saving is applied.

    tuner_verbose : int, default=1
        Verbosity mode. <=0 silent all; >0 print trial logs with the
        connected score.

    Attributes
    ----------
    folds_trials : dict
        A dicts of list. The lists contain all the hyperparameter combinations
        tried in each fold and derived from the param_grid.

    folds_scores : dict
        A dicts of list. The lists contain the monitor quantities achived on
        the validation data by all the models tried in each fold.

    folds_best_params : dict
        The dict containing the best combination (in term of score) of
        hyperparameters in each fold.

    folds_best_score : dict
        The best scores achieved by all the possible combination created in
        each fold.

    folds_best_model : dict
        The best models (in term of score) in each fold. Accessible only if
        store_model is set to True.

    best_params_score : float
        The best average score in all the available folds.

    best_params : dict
        The paramareter combination related to the best average score in all
        the available folds.

    Notes
    ----------
    KerasRandomSearchCV allows the usage of every callbacks available in Keras
    (also the custom one). The callbacks, that provide the possibility to
    save any output as external files, support naming formatting options.
    This is true for ModelCheckpoint, CSVLogger, TensorBoard and RemoteMonitor.
    'trial' and 'fold' are custom tokens that can be used to personalize the
    name formatting.

    For example: if filepath in ModelCheckpoint is model_{fold}_{trial}.hdf5,
    then the model checkpoints will be saved with the relative number of trial,
    obtained at a certain fold, in the filename. This enables to save and
    differentiate each model created in the searching trials.
    """

    def __init__(self,
                 hypermodel,
                 param_grid,
                 cv,
                 n_iter,
                 sampling_seed=0,
                 monitor='val_loss',
                 greater_is_better=False,
                 store_model=True,
                 savepath=None,
                 tuner_verbose=1):
      
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

    def search(self,
               x, y=None,
               sample_weight=None,
               groups=None,
               **fitargs):
        """Performs a search for best hyperparameter configurations creating
        all the possible trials and evaluating on the validation folds
        created following the validation strategy.

        Parameters
        ----------
        x : multi types
            Input data. Accepted types are arrays or list/dict in case of
            multi-input/output.

        y : multi types, default=None
            Target data. Accepted types are arrays or list/dict in case of
            multi-input/output.

        sample_weight : multi types, default=None
            Optional Numpy array of weights for the training samples, used
            for weighting the loss function (during training only). Accepted
            types are arrays or list/dict in case of multi-input/output.

        groups : array-like, default=None
            Group labels for the samples used while splitting the dataset into
            train/valid set.

        **fitargs : Additional fitting arguments, the same accepted in Keras
                    model.fit(...).
            The validation set is automatically created accordingly to the
            cv strategy.

        Returns
        -------
        self : object
        """

        self._search(x=x, y=y,
                     sample_weight=sample_weight,
                     groups=groups,
                     is_random=True,
                     **fitargs)

        return self
