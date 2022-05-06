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
        'metric': ['multi_logloss'],
        'learning_rate': 0.05,
        'max_depth': -1,
        'min_child_samples': 100,
        'max_bin': 255,
        'subsample': 0.85,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.01,
        'subsample_for_bin': 200000,
        'min_split_gain': 0.001,
        'reg_alpha': 0.005,
        'reg_lambda': 0.005,
        'num_leaves':511,
        'seed': SEED,
        'nthread': 16,
        'num_class': 3,
        'verbosity': -1,
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
def select(df_model):
    train_X, valid_X, test_X = random_split_data(df_model)
    df_select = train_X.loc[(train_X['ball_position_x'] < 0) & (train_X['label'] == 1)]
    print(df_select.shape)
    print('select specific dataframe:',df_select)

def google_to_pitch(x, y):
    x = (x+1) * 60
    y = (y+0.42) * 80/0.84
    return x, y

def positions(df, df_model):

    position_left = df['left_team'].values
    position_left = np.concatenate(position_left, axis=0)
    position_left = position_left.reshape(df.shape[0], 11, 2)

    for i in range(11):
        for j in range(2):
            df_model[f'left_{i + 1}_{j + 1}'] = position_left[:, i, j]

    position_right = df['right_team'].values
    position_right = np.concatenate(position_right, axis=0)
    position_right = position_right.reshape(df.shape[0], 11, 2)

    for i in range(11):
        for j in range(2):
            df_model[f'right_{i + 1}_{j + 1}'] = position_right[:, i, j]
    return df_model


def predict(data):
    #df = data.df.copy()
    df_model = data.df_model.copy()
    drop_columns = ['match_num', 'steps_left']
    df_model = get_model_input(df_model, drop_columns)
    # 类别型变量
    categoricals = []

    train_X, valid_X, test_X = random_split_data(df_model)
    #train_X = shuffle_data(train_X)
    #valid_X = shuffle_data(valid_X)
    #test_X = shuffle_data(test_X)

    train_Y = train_X.pop('label')
    valid_Y = valid_X.pop('label')
    test_Y = test_X.pop('label')
    #model = lightgbm_modeling(train_X, train_Y, valid_X, valid_Y, categoricals)
    model = lgb.Booster(model_file='../processed_data/model.txt')
    pred_Y = model.predict(test_X, num_iteration=model.best_iteration)
    results = np.argmax(pred_Y, axis=1)
    print("recall_score on testing data...")
    print(recall_score(test_Y, results, average=None))
    print("predicted value :", Counter(results))
    # print(results,type(results))
    # print(test_Y,type(test_Y))
    print("label:", Counter(test_Y))
    return pred_Y




def predict_and_draw(data):
    df_model = data.df_model.copy()
    df = data.df.copy()
    pred_Y = predict(data)
    df_pred = pd.DataFrame(pred_Y, columns=['pred_0', 'pred_1', 'pred_2'])
    df_position = positions(df,df_model)
    train_X, valid_X, test_X = random_split_data(df_position)
    df_pred.index = test_X.index
    df_new = pd.concat([test_X, df_pred], axis=1)
    df_select = df_new.loc[(df_new['ball_position_x'] > 0.5) & (abs(df_new['ball_position_y']) < 0.2) & (df_new['pred_1'] > 0.1) & (df_new['pred_1'] < 0.3)]
    print('select specific dataframe:', df_select)
    print('the shape of specific dataframe:', df_select.shape)

    for index in df_select.index:
        pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
        fig, ax = pitch.draw()  # 场地大小120*80
        for i in range(11):
            x, y = google_to_pitch(df_select['left' + '_' + str(i + 1) + '_' + '1'].loc[index],
                                   df_select['left' + '_' + str(i + 1) + '_' + '2'].loc[index])
            plt.scatter(x, y, color='blue', s=60)  # 绘制左队球员位置
        for i in range(11):
            x, y = google_to_pitch(df_select['right' + '_' + str(i + 1) + '_' + '1'].loc[index],
                                   df_select['right' + '_' + str(i + 1) + '_' + '2'].loc[index])
            plt.scatter(x, y, color='black', s=60)  # 绘制右队球员位置
        x, y = google_to_pitch(df_select['ball_position_x'].loc[index], df_select['ball_position_y'].loc[index])
        plt.scatter(x, y, color='red', marker='8', s=30)  # 绘制球位置
        second = 1.8 * (3000 - df_select['steps_left'])
        print(second)
        min = (second / 60).astype(int)
        second = (second - 60 * min).astype(int)
        time = str(min) + ':' + str(second)

        res = round(df_select['pred_1'].loc[index], 4)
        plt.savefig(f'{fig_dir}{index}_{res}.png')
        plt.show()
