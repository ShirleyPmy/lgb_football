#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:10:44 2022

@author: pmy
"""

from CONSTANT import *
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from process_data import random_split_data

version = datetime.now().strftime("%Y%m%d%H%M%S")
print('Version: ', version)

def get_model_input(df_model, drop_columns):
    #有些特征并不计划参与训练， 比如steps_left和match_num
    df_model = df_model.drop(drop_columns, axis=1)
    return df_model

def modeling(df_model):
    drop_columns = ['match_num', 'steps_left']
    df_model = get_model_input(df_model, drop_columns)
    
    # 类别型变量
    categoricals = []
    
    train_X, valid_X, test_X = random_split_data(df_model)
    
    train_Y = train_X.pop('label')
    valid_Y = valid_X.pop('label')
    test_Y = test_X.pop('label')
    
    
    model = lightgbm_modeling(train_X, train_Y, valid_X, valid_Y, categoricals)
    
    return model

def lightgbm_modeling(train_X, train_Y, valid_X, valid_Y, categoricals, OPT_ROUNDS=600, weight=None):
    EARLY_STOP = 50
    OPT_ROUNDS = OPT_ROUNDS
    MAX_ROUNDS = 3000
    params = {
        'boosting': 'gbdt',
        'objective': 'multiclass',
        'metric' : ['multi_logloss'],
        'learning_rate': 0.001,
        'max_depth': -1,
        'min_child_samples': 20,
        'max_bin': 255,
        'subsample': 0.85,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.001,
        'subsample_for_bin': 200000,
        'min_split_gain': 0.001,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'num_leaves':63,
        'seed': SEED,
        'nthread': 16,
        'num_class': 3,
        'verbosity': -1
        #'is_unbalance': True,
    }
    print(f'Now Version {version}')
    print('Start train and validate...')
    print('feature number:', len(train_X.columns))
    feat_cols = list(train_X.columns)
    dtrain = lgb.Dataset(data=train_X, label=train_Y, feature_name=feat_cols,weight=weight)
    dvalid = lgb.Dataset(data=valid_X, label=valid_Y, feature_name=feat_cols)
    model = lgb.train(params,
                      dtrain,
                      categorical_feature=categoricals,
                      num_boost_round=MAX_ROUNDS,
                      early_stopping_rounds=EARLY_STOP,
                      verbose_eval=50,
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid']
                      )
    importances = pd.DataFrame({'features':model.feature_name(),
                            'importances':model.feature_importance()})
    importances.sort_values('importances',ascending=False,inplace=True)
    print(importances)
    importances.to_csv( (feat_imp_dir+'{}_imp.csv').format(version), index=False )
    return model
