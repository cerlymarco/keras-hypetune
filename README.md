# keras-hypetune
A friendly python package for Keras Hyperparameters Tuning based only on NumPy.

## Overview

A very simple wrapper for fast Keras hyperparameters optimization. keras-hypetune lets you use the power of Keras without having to learn a new syntax. All you need it's just create a python dictionary where to put the parameter boundaries for the experiments and define your Keras model (in any format: Functional or Sequential) inside a callable function.

```python
def get_model(param):
        
    model = Sequential()
    model.add(Dense(param['unit_1'], activation=param['activ']))
    model.add(Dense(param['unit_2'], activation=param['activ']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=param['lr']), 
                  loss='mse', metrics=['mae'])
    
    return model
```

The optimization process is easily trackable using the callbacks provided by Keras. At the end of the searching, you can access all you need by querying the keras-hypetune searcher. The best solutions can be automatically saved in proper locations.

## Installation

```shell
pip install keras-hypetune
```

Tensorflow and Keras are not needed requirements. keras-hypetune is specifically for tf.keras with TensorFlow 2.0. The usage of GPU is normally available.

## Fixed Validation Set

This tuning modality operates the optimization on a fixed validation set. The parameter combinations are evaluated always on the same set of data. In this case, it's allowed the usage of any kind of input data format accepted by Keras.

### KerasGridSearch

All the passed parameter combinations are created and evaluated.

```python
param_grid = {
    'unit_1': [128,64], 
    'unit_2': [64,32],
    'lr': [1e-2,1e-3], 
    'activ': ['elu','relu'],
    'epochs': 100, 
    'batch_size': 512
}

kgs = KerasGridSearch(get_model, param_grid, monitor='val_loss', greater_is_better=False)
kgs.search(x_train, y_train, validation_data=(x_valid, y_valid))
```

### KerasRandomSearch

Only random parameter combinations are created and evaluated.

The number of parameter combinations that are tried is given by n_iter. If all parameters are presented as a list, sampling without replacement is performed. If at least one parameter is given as a distribution (from scipy.stats random variables), sampling with replacement is used.

```python
param_grid = {
    'unit_1': [128,64], 
    'unit_2': stats.randint(32, 128),
    'lr': stats.uniform(1e-4, 0.1), 
    'activ': ['elu','relu'],
    'epochs': 100, 
    'batch_size': 512
}

krs = KerasRandomSearch(get_model, param_grid, monitor='val_loss', greater_is_better=False, 
                        n_iter=15, sampling_seed=33)
krs.search(x_train, y_train, validation_data=(x_valid, y_valid))
```

## Cross Validation

This tuning modality operates the optimization using a cross-validation approach. The CV strategies available are the same provided by scikit-learn splitter classes. The parameter combinations are evaluated on the mean score of the folds. In this case, it's allowed the usage of only numpy array data. For tasks involving multi-input/output, the arrays can be wrapped into list or dict like in normal Keras.

### KerasGridSearchCV

All the passed parameter combinations are created and evaluated.

```python
param_grid = {
    'unit_1': [128,64], 
    'unit_2': [64,32],
    'lr': [1e-2,1e-3], 
    'activ': ['elu','relu'],
    'epochs': 100, 
    'batch_size': 512
}

cv = KFold(n_splits=3, random_state=33, shuffle=True)

kgs = KerasGridSearchCV(get_model, param_grid, cv=cv, monitor='val_loss', greater_is_better=False)
kgs.search(X, y)
```

### KerasRandomSearchCV

Only random parameter combinations are created and evaluated.

The number of parameter combinations that are tried is given by n_iter. If all parameters are presented as a list, sampling without replacement is performed. If at least one parameter is given as a distribution (from scipy.stats random variables), sampling with replacement is used.

```python
param_grid = {
    'unit_1': [128,64], 
    'unit_2': stats.randint(32, 128),
    'lr': stats.uniform(1e-4, 0.1), 
    'activ': ['elu','relu'],
    'epochs': 100, 
    'batch_size': 512
}

cv = KFold(n_splits=3, random_state=33, shuffle=True)

krs = KerasRandomSearchCV(get_model, param_grid, cv=cv, monitor='val_loss', greater_is_better=False,
                          n_iter=15, sampling_seed=33)
krs.search(X, y)
```
