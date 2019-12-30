import os
import pickle
import warnings
import random
import click

import numpy as np
import pandas as pd
import xgboost as xgb

import config as cfg

warnings.filterwarnings('ignore')

IDENTIFICATORS = ["shop_id","item_id","date_block_num"]
PREDICTORS = []

def load_data():
    df= pd.read_pickle("cleaned_sets/df_fs_done.pkl")
    test = df.loc[(df["date_block_num"]>=33)]
    
    global PREDICTORS
    PREDICTORS = [x for x in test.columns if x not in IDENTIFICATORS]
    
    X_test = test[PREDICTORS+IDENTIFICATORS]
    
    return X_test

def load_model(path):
    xgbr = xgb.XGBRegressor()
    xgbr.load_model(path)
    return xgbr

def fine_tune_results(submission):
    submission.loc[submission["item_cnt_next_month_pred"]<0, "item_cnt_next_month_pred"]=0

def predict(X_test, model, path_to_output, fill_method, clip):

    y_pred = model.predict(X_test[PREDICTORS])
    X_test["item_cnt_next_month_pred"] = y_pred

    to_pred = pd.read_csv(cfg.FILENAMES['TEST_SALES'])

    submission = pd.merge(to_pred, X_test[["item_id","shop_id","item_cnt_next_month_pred"]],
                      how="left", on=["item_id","shop_id"])
    
    if fill_method=="zero":
        submission = submission.fillna(0)
    elif fill_method=="categoryxshop_mean":
        pass
    elif fill_method=="category_mean":
        pass
    elif fill_method=="shop_mean":
        pass
    
    if clip:
        submission.loc[submission["item_cnt_next_month_pred"]>20,"item_cnt_next_month_pred"]=20
    
    submission = fine_tune_results(submission)

    submission_formated = (submission[["ID", "item_cnt_next_month_pred"]]
                       .rename({"item_cnt_next_month_pred":"item_cnt_month"}, axis=1))

    submission_formated.to_csv(os.path.join("submissions", path_to_output), index=False)

    
@click.command()
@click.argument('model_path')
@click.argument('output_path')
@click.option("--fill_method", default="zero", help="Method to fill missing values in the predictions")
@click.option("--clip/--no-clip", default=False, is_flag=True, help="Boolean to clip predictions to [0,20]")
def main(model_path, output_path, fill_method, clip):
    
    print("Loading data...")
    X_test = load_data()
    print("Data loaded.\n")
    
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.\n")
    
    print("Predicting...")
    predict(X_test, model, output_path, fill_method, clip)
    print("Predictions done!\n")
    
    
if __name__=="__main__":
    main()
    
    
    