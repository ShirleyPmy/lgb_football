#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:59:50 2022

@author: pmy
"""
import os

SEED = 2021

data_dir = '../data/'

feat_imp_dir = '../feat_importances/'
if not os.path.exists(feat_imp_dir):
    os.makedirs(feat_imp_dir)
    

processed_data = '../processed_data/'
if not os.path.exists(processed_data):
    os.makedirs(processed_data)
