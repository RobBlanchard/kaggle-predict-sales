import os
import pickle
import warnings
import random
import pickle

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bayes_opt import BayesianOptimization

import config as cfg

warnings.filterwarnings('ignore')

#Load Data
print("Loading Data")
df = pd.load_pickle("cleaned_sets/df_fs_done.pkl")

identificators = ["shop_id","item_id","date_block_num"]
predictors = [x for x in train.columns if x not in not_predictors]
label = "item_cnt_next_month"

X_train = pd.concat([train[predictors],valid[predictors]], axis=0)
y_train = pd.concat([train[label], valid[label]],  axis=0)
dtrain = xgb.DMatrix(X_train, label=y_train)

X_test = test[predictors+identificators]

def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
    
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)})
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')

params = xgb_bo.res['max']['max_params']
params['max_depth'] = int(params['max_depth'])
print(params)