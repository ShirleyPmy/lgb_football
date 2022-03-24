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


if __name__ == '__main__':
    if True:
        # 第一次处理数据的时候才走这个流程，之后存成二进制处理好的文件，直接读取，可以加快速度
        df, df_model = process_original_data()
        data = Data(df, df_model)
#        save_processed_data(data)
    else:
        data = load_processed_data()
    
    #如果只是调整模型，那么特征也可以不更新，不需要重新生成
    feature_engineering(data)
    
    
    # 模型的预测等部分
    model = modeling(data.df_model)
    