import inspect
import numpy as np
from copy import deepcopy

from hyperopt import fmin, tpe

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

    def _fit(self,
             x, y,
             params,
             callbacks,
             validation_split,
             validation_data,
             id_fold, fitargs):
        """Private method to fit a Keras model."""

        if not callable(self.hypermodel):
            raise ValueError("hypermodel must be a callable function.")

        results = {}
        tunable_fitargs = ['batch_size', 'epochs',
                           'steps_per_epoch', 'class_weight']

        if callbacks is not None:
            fitargs['callbacks'] = _clear_callbacks(
                deepcopy(callbacks), self._trial_id, id_fold)

        model = self.hypermodel(params)
        fit_params = {k: v for k, v in params.items() if k in tunable_fitargs}
        all_fitargs = dict(list(fitargs.items()) + list(fit_params.items()))

        if self.tuner_verbose > 0:
            print(
                "\n***** ({}/{}) *****\nSearch({})".format(
                    self._trial_id,
                    self.n_iter if self._tuning_type is 'hyperopt' \
                        else len(self._param_combi),
                    params
                )
            )

        model.fit(x=x,
                  y=y,
                  validation_split=validation_split,
                  validation_data=validation_data,
                  **all_fitargs)

        epoch = self._eval_score(model.history.history[self.monitor])
        params['epochs'] = epoch + 1
        params['steps_per_epoch'] = model.history.params['steps']
        params['batch_size'] = (all_fitargs['batch_size'] if 'batch_size'
                                                             in all_fitargs else None)

        score = round(model.history.history[self.monitor][epoch], 5)

        if self.tuner_verbose > 0:
            print("SCORE: {} at epoch {}".format(score, epoch + 1))

        results = {
            'params': params, 'status': 'ok',
            'loss': score * self._score_sign,
            'model': model
        }
        self._trial_id += 1

        return results

    def _search(self,
                tuning_type,
                x, y=None,
                trials=None,
                validation_data=None,
                validation_split=0.0,
                id_fold=None,
                **fitargs):
        """Private method to perform a search on a fixed validation set for
        the best parameters configuration."""

        if validation_data is None and validation_split == 0.0:
            raise ValueError(
                "Pass at least one of validation_data or validation_split."
            )

        if not isinstance(self.param_grid, dict):
            raise ValueError("Pass param_grid in dict format.")
        self._param_grid = self.param_grid.copy()

        for p_k, p_v in self._param_grid.items():
            self._param_grid[p_k] = _check_param(p_v)

        self._eval_score = np.argmax if self.greater_is_better else np.argmin
        self._score_sign = -1 if self.greater_is_better else 1

        rs = ParameterSampler(n_iter=self.n_iter,
                              param_distributions=self._param_grid,
                              random_state=self.sampling_seed)
        self._param_combi, self._tuning_type = rs.sample()
        self._trial_id = 1

        tuning_class = {
            'grid': 'KerasGridSearch',
            'random': 'KerasRandomSearch',
            'hyperopt': 'KerasBayesianSearch'
        }
        if tuning_type != self._tuning_type:
            raise ValueError(
                "The chosen param_grid is incompatible with {}. "
                "Maybe you are looking for {}.".format(
                    tuning_class[tuning_type] + \
                    ('CV' if id_fold is not None else ''),
                    tuning_class[self._tuning_type] + \
                    ('CV' if id_fold is not None else '')
                )
            )

        if 'callbacks' in fitargs:
            if isinstance(fitargs['callbacks'], list):
                callbacks = deepcopy(fitargs['callbacks'])
            else:
                callbacks = deepcopy([fitargs['callbacks']])
        else:
            callbacks = None

        if self.tuner_verbose > 0:
            n_trials = self.n_iter if self._tuning_type is 'hyperopt' \
                else len(self._param_combi)
            print("\n{} trials detected for {}".format(
                n_trials, tuple(self._param_grid.keys())))
            verbose = fitargs['verbose'] if 'verbose' in fitargs else 0
        else:
            verbose = 0
        fitargs['verbose'] = verbose

        if self._tuning_type == 'hyperopt':
            if trials is None:
                raise ValueError(
                    "trials must be not None when using hyperopt."
                )

            search = fmin(
                fn=lambda p: self._fit(
                    params=p, x=x, y=y,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    id_fold=id_fold, fitargs=fitargs
                ),
                space=self._param_combi, algo=tpe.suggest,
                max_evals=self.n_iter, trials=trials,
                rstate=np.random.RandomState(self.sampling_seed),
                show_progressbar=False, verbose=0
            )
            all_results = trials.results

        else:
            all_results = [
                self._fit(
                    params=params, x=x, y=y,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    id_fold=id_fold, fitargs=fitargs
                )
                for params in self._param_combi
            ]

        self.trials, self.scores, models = [], [], []
        for res in all_results:
            self.trials.append(res['params'])
            self.scores.append(self._score_sign * res['loss'])
            models.append(res['model'])

        # get the best
        id_best = self._eval_score(self.scores)
        self.best_score = self.scores[id_best]
        self.best_params = self.trials[id_best]
        best_model = models[id_best]
        if self.store_model:
            self.best_model = best_model

        if self.savepath is not None:
            if id_fold is not None:
                best_model.save(self.savepath.replace('{fold}', str(id_fold)))
            else:
                best_model.save(self.savepath)

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

    def _search(self,
                tuning_type,
                x, y=None,
                trials=None,
                sample_weight=None,
                groups=None,
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

            _trials = trials if trials is None else deepcopy(trials)
            ks_fold._search(tuning_type=tuning_type,
                            x=x_train, y=y_train,
                            trials=_trials,
                            sample_weight=sample_weight_train,
                            validation_data=(x_val, y_val, sample_weight_val),
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