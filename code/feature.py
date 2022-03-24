#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:36:36 2022

@author: pmy
"""
import pandas as pd
import numpy as np
import time
import utils
import sys

pd.set_option('mode.chained_assignment', None)

def steps_left_bin(data):
    df_model = data.df_model.copy()
    
    df_model['steps_left_bin'] = np.ceil(df_model['steps_left']/60).astype(int)
    
    data.df_model = df_model
    
def ball_position(data):
    df = data.df.copy()
    df_model = data.df_model.copy()
    
    # 小数位数太多容易过拟合
    df_model['ball_x'] = data.df['ball'].apply(lambda x:round(x[0],4))
    df_model['ball_y'] = data.df['ball'].apply(lambda x:round(x[1],4))
    df_model['ball_z'] = data.df['ball'].apply(lambda x:round(x[2],4))
    
    data.df_model = df_model
    
def owned_player_position(data):
    df = data.df.copy()
    df_model = data.df_model.copy()
    
    # 小数位数太多容易过拟合
    df_model['owned_player_position_x'] = -10
    df_model['owned_player_position_y'] = -10
    
    def from_player_index_get_position(line):
        team = line['ball_owned_team']
        index = line['ball_owned_player']
        if(team==0):
            position = line['left_team']
        elif(team==1):
            position = line['right_team']
        else:
            return np.array([-10,10])
        
        return position[index]
    
    owned_player_position = data.df.apply(from_player_index_get_position,axis=1)
    
    df_model['owned_player_position_x'] = owned_player_position.apply(lambda x:round(x[0],4))
    df_model['owned_player_position_y'] = owned_player_position.apply(lambda x:round(x[1],4))
    
    data.df_model = df_model
    
def owned_ball_distance_horizontal(data):
    df_model = data.df_model.copy()
    
    #这里是偷懒的写法，因为直接使用df_model中的位置计算，而改位置只取了4位小数，再计算距离时不够精确
    df_model['owned_ball_distance_horizontal'] = ((df_model['ball_x'] - df_model['owned_player_position_x'])**2 + (df_model['ball_y'] - df_model['owned_player_position_y'])**2)**0.5
    
    data.df_model = df_model
    # 这里有一些思路， 比如我算了控球的队员离球的水平距离，那么其实也可以算一下离球最近的对方队员的距离，或许也对预测最终结果有帮助

def dis_owned_left(data):
    df = data.df.copy()
    #temp = df['left_team']
    df_model = data.df_model.copy()
    distance_0 = []
    distance_1 = []
    distance_2 = []
    for i in range(len(df_model['owned_player_position_x'])):
        temp_dis = []
        x = df_model['owned_player_position_x'][i]
        y = df_model['owned_player_position_y'][i]
        temp2 = df['left_team'][i]
        #dis = float('inf')
        for j in range(11):
            temp_dis.append(((x-temp2[j][0])**2+(y-temp2[j][1])**2)**0.5)
            #dis = min(dis, ((x-temp2[j][0])**2+(y-temp2[j][1])**2)**0.5)
        distance_0.append(sorted(temp_dis)[0])
        distance_1.append(sorted(temp_dis)[1])
        distance_2.append(sorted(temp_dis)[2])
    df_model['dis_owned_left_0'] = distance_0
    df_model['dis_owned_left_1'] = distance_1
    df_model['dis_owned_left_2'] = distance_2
    data.df_model = df_model

def dis_owned_right(data):
    df = data.df.copy()
    #temp = df['left_team']
    df_model = data.df_model.copy()
    distance_0 = []
    distance_1 = []
    distance_2 = []
    for i in range(len(df_model['owned_player_position_x'])):
        temp_dis = []
        x = df_model['owned_player_position_x'][i]
        y = df_model['owned_player_position_y'][i]
        temp2 = df['right_team'][i]
        #dis = float('inf')
        for j in range(11):
            temp_dis.append(((x-temp2[j][0])**2+(y-temp2[j][1])**2)**0.5)
            #dis = min(dis, ((x-temp2[j][0])**2+(y-temp2[j][1])**2)**0.5)
        distance_0.append(sorted(temp_dis)[0])
        distance_1.append(sorted(temp_dis)[1])
        distance_2.append(sorted(temp_dis)[2])
    df_model['dis_owned_right_0'] = distance_0
    df_model['dis_owned_right_1'] = distance_1
    df_model['dis_owned_right_2'] = distance_2
    data.df_model = df_model


def team_tired_factor_mean(data):
    df = data.df.copy()
    df_model = data.df_model.copy()
    
    df_model['left_team_tired_mean'] = df['left_team_tired_factor'].apply(lambda x:np.mean(x))
    df_model['right_team_tired_mean'] = df['right_team_tired_factor'].apply(lambda x:np.mean(x))
    
    data.df_model = df_model
    
    #比如这里可以做一些，控球球员的疲惫值，周围xx米内队友/对手的平均疲惫值，离你最近的3个队友/对手的平均疲惫值
    

def goal_position_mean(data):
    # 统计特征，统计上一场比赛的平均进球时（label=1），球的平均位置
    df = data.df.copy()
    df_model = data.df_model.copy()
    
    position = df_model.loc[df_model['label']==1,['ball_x','ball_y','ball_z','match_num']]
    position = position.groupby('match_num').mean()
    position.columns = ['last_match_goal_position_mean_x','last_match_goal_position_mean_y','last_match_goal_position_mean_z']
    
    position = position.reset_index()
    position['match_num'] += 1
    
    df_model = df_model.merge(position,how='left',on='match_num')
    
    data.df_model = df_model
    
    #类似的统计特征可以多搞搞，本场比赛的，其他比赛的，就是注意不要leak（标签泄漏）


#说一点特征思路，正常指定要计算一些，球，球门，球员的位置信息，距离，速度，前方是不是有队友接应等，可以自己试试，能做的很多，前方的敌人数量等
#但是比较tricky的思路的话，感觉可以把球场建模成一个矩阵，然后上面有球、人的位置，状态等，用比如cnn等方式抽取一些embedding表示，让模型自己学习上述特征




def feature_engineering(data):
    # 做特征就写一个函数，然后把函数名字加到这里， 后续做实验发现这个特征没什么用，那就可以在列表中删除它，但是函数还可以留着
    good_funcs = [ steps_left_bin, ball_position, owned_player_position, owned_ball_distance_horizontal,
                  team_tired_factor_mean, goal_position_mean, dis_owned_left, dis_owned_right]
    
    
    origin_columns = data.df_model.columns
    origin_index = data.df_model.shape[0]
    
    funcs = good_funcs
    
    print('feature_engineering')
    print('running features num: ', len(good_funcs))
    print('data shape: ', data.df_model.shape)
    
    for func in funcs:
        tmp_columns = data.df_model.columns
        t1 = time.time()
        func(data)
        
        #数据条数不能变化
        if(data.df_model.shape[0]!=origin_index):
            print('origin data num: ', origin_index, ', now data num: ', data.df_model.shape[0])
            print('数据错误，样本数变化')
            sys.exit()
        
        add_columns = [i for i in data.df_model.columns if i not in tmp_columns]
        
        # 减内存
        for col in add_columns:
            data.df_model[col] = utils.downcast(data.df_model[col])
            
        t2 = time.time()
        print(f'do feature {func.__name__}, add columns {add_columns}, use time {t2-t1:.4f} s')
        print(data.df_model)
    
    