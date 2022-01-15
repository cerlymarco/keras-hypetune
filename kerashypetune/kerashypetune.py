from ._classes import _KerasSearch, _KerasSearchCV


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

        self._search(tuning_type='grid',
                     x=x, y=y,
                     validation_data=validation_data,
                     validation_split=validation_split,
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
    KerasRandomSearch allows the usage of every callbacks available in Keras
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

        self._search(tuning_type='random',
                     x=x, y=y,
                     validation_data=validation_data,
                     validation_split=validation_split,
                     **fitargs)

        return self


class KerasBayesianSearch(_KerasSearch):
    """Bayesian hyperparamater searching and optimization on a fixed
    validation set.

    Pass a Keras model (in Sequential or Functional format), and
    a dictionary with the parameter boundaries for the experiment.
    For searching, takes in the same arguments available in Keras
    model.fit(...). All the input format supported by Keras model
    are accepted.

    In contrast to random-search, the parameter settings are not sampled
    randomly. The parameter values are chosen according to bayesian
    optimization algorithms based on gaussian processes and regression trees.
    The number of parameter settings that are tried is given by n_iter.
    Parameters must be given as hyperopt distributions.
    It is highly recommended to use continuous distributions for continuous
    parameters.

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
    KerasBayesianSearch allows the usage of every callbacks available in Keras
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
               trials=None,
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

        trials : hyperopt.Trials() object, default=None
            A hyperopt trials object, used to store intermediate results for all
            optimization runs. Effective (and required) only when hyperopt
            parameter searching is computed.

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

        self._search(tuning_type='hyperopt',
                     x=x, y=y,
                     trials=trials,
                     validation_data=validation_data,
                     validation_split=validation_split,
                     **fitargs)

        return self


class KerasGridSearchCV(_KerasSearchCV):
    """Grid hyperparamater searching and optimization with cross
    validation. Out-of-fold samples generated by CV are used automatically
    as validation_data in each model fitting.

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

        self._search(tuning_type='grid',
                     x=x, y=y,
                     sample_weight=sample_weight,
                     groups=groups,
                     **fitargs)

        return self


class KerasRandomSearchCV(_KerasSearchCV):
    """Random hyperparamater searching and optimization with cross
    validation. Out-of-fold samples generated by CV are used automatically
    as validation_data in each model fitting.

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

        self._search(tuning_type='random',
                     x=x, y=y,
                     sample_weight=sample_weight,
                     groups=groups,
                     **fitargs)

        return self


class KerasBayesianSearchCV(_KerasSearchCV):
    """Bayesian hyperparamater searching and optimization with cross
    validation. Out-of-fold samples generated by CV are used automatically
    as validation_data in each model fitting.

    Pass a Keras model (in Sequential or Functional format), and
    a dictionary with the parameter boundaries for the experiment.
    For searching, takes in the same arguments available in Keras
    model.fit(...). The cross-validation strategies are the same
    provided by the scikit-learn cross-validation generator.
    Only input in array format are supported. In case of multi-input or
    multi-output is it possible to wrap arrays in list or dictionaries
    like in Keras.

    In contrast to random-search, the parameter settings are not sampled
    randomly. The parameters values are chosen according to bayesian
    optimization algorithms based on gaussian processes and regression trees.
    The number of parameter settings that are tried is given by n_iter.
    Parameters must be given as hyperopt distributions.
    It is highly recommended to use continuous distributions for continuous
    parameters.

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
    KerasBayesianSearchCV allows the usage of every callbacks available in Keras
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
               trials=None,
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

        trials : hyperopt.Trials() object, default=None
            A hyperopt trials object, used to store intermediate results for all
            optimization runs. Effective (and required) only when hyperopt
            parameter searching is computed.

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

        self._search(tuning_type='hyperopt',
                     x=x, y=y,
                     trials=trials,
                     sample_weight=sample_weight,
                     groups=groups,
                     **fitargs)

        return self