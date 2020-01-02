import os
import pickle
import warnings
import random
import pickle
import time
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import config as cfg
import data_processing as dp


warnings.filterwarnings('ignore')
plt.style.use('seaborn-dark-palette')

df= pd.read_pickle("cleaned_sets/df_fs_done.pkl")

train = df.loc[(df["date_block_num"]>=20) & (df["date_block_num"]<=32)].sample(frac=0.05)
test = df.loc[(df["date_block_num"]>=33)]

identificators = ["shop_id","item_id","date_block_num"]
predictors = [x for x in train.columns if x not in identificators]
label = "item_cnt_next_month"

X_train = train[predictors]
y_train = train[label]

X_test = test[predictors+identificators]

dtrain = xgb.DMatrix(X_train, label=y_train)

def dict_to_iterlist(d):
    keys=d.keys()
    lists=d.values()
    return list(keys), list(itertools.product(*lists))

def fine_tune_xgb(initial_params, gridsearch_params, dtrain, early_stopping_rounds=10, cv_fold=5):
    min_rmse = float("Inf")
    best_params = None
    
    params=initial_params
    gs_param_names, combinations = dict_to_iterlist(gridsearch_params)
    nb_gs_params = len(gs_param_names)
    
    boosting_rounds=100
    
    for combi in combinations:
        start_time=time.time()
        print(", ".join([f"{gs_param_names[i]}={combi[i]}" for i in range(nb_gs_params)]))

        # Update our parameters
        for i in range(nb_gs_params):
            params[gs_param_names[i]] = combi[i]
            
        if "num_boost_round" in gs_param_names:
            boosting_rounds=combi[gs_param_names.index("num_boost_round")]
        
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=boosting_rounds,
            seed=42,
            nfold=cv_fold,
            metrics={'rmse'},
            early_stopping_rounds=early_stopping_rounds
        )
        # Update best RMSE
        
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        print("Time taken for this round {}".format(time.time()-start_time))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = combi
    
    print(best_params)
    print("".join(["Best params:", 
                   ", ".join([f"{gs_param_names[i]}={best_params[i]}" for i in range(nb_gs_params)]),
                  f", RMSE: {min_rmse}",]))
    
params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 2,
    'gamma':0,
    'eta':.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # Other parameters
    'objective':'reg:squarederror',
    'eval_metric':'rmse',
    "tree_method":'gpu_hist',
    "gpu_id":0
}

gridsearch_params = {
    "num_boost_round":[50,100,200,300,400,500]
        }

fine_tune_xgb(params, gridsearch_params, dtrain)