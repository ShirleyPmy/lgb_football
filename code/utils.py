#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:52:31 2022

@author: pmy
"""

import numpy as np
import pandas as pd
import pickle

def downcast(series):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()
        
        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif  max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series
        
    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)
        
        tmp = series[~pd.isna(series)]
        if(len(tmp)==0):
            return series.astype(np.float16)
        
        if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
            return series.astype(np.float16)
        elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
            return series.astype(np.float32)
       
        else:
            return series
            
    else:
        return series
    

def load_pickle(file_path):
    return pickle.load(open(file_path,'rb'))
    
    
def dump_pickle(obj,file_path):
    pickle.dump(obj,open(file_path,'wb'))