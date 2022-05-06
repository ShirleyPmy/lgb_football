#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:32:13 2022

@author: pmy
"""

from process_data import *
from data import Data
from feature import feature_engineering
from model import *
from CONSTANT import *


if __name__ == '__main__':
    if False:
        # 第一次处理数据的时候才走这个流程，之后存成二进制处理好的文件，直接读取，可以加快速度
        df, df_model = process_original_data()
        data = Data(df,df_model)
        df = data.df
        df_model = data.df_model
        save_processed_data(df,'df') #存df
        save_processed_data(df_model,'df_model') #存df_model
    else:
        df = load_processed_data('df') #load df
        df_model = load_processed_data('df_model') #load df_model
        data = Data(df, df_model) #重新将df, df_model 打包，不想再修改feature.py
    
    #如果只是调整模型，那么特征也可以不更新，不需要重新生成
    feature_engineering(data)
    
    
    # 模型的预测等部分
    #model = modeling(data.df_model)
    #result = predict(data)
    predict_and_draw(data)
