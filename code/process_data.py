import pandas as pd
import numpy as np
import os
import re
import utils
import time

from CONSTANT import *

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


t1 = time.time()
def episode_process(episode):
    '''
    处理一个episode的函数
    episode:一场比赛的全部raw_data
    该函数不对step进行筛选全部保留
    只打标签
    return: episode 打好标签的raw_data
    '''
    episode = episode.tolist()
    for i in range(len(episode)):
        if 'ball' not in episode[i]:
            episode[i] = episode[i][0]
    while episode[0]['ball_owned_team'] == -1:
        episode.remove(episode[0])  # 对开局那几帧进行截断，现在还不够严谨，更好的是需要对每一次游戏终止后重新开始的几帧进行截断
    for i in range(len(episode)):  # 新建一个标注球权的属性，为了方便后面打标签
        if episode[i]['ball_owned_team'] != -1:
            episode[i]['ball_owned_team_new'] = episode[i]['ball_owned_team']  # 如果有人控球，球权不变
        else:
            episode[i]['ball_owned_team_new'] = episode[i - 1]['ball_owned_team_new']  # 无人控球，与上一时刻相同

    # 下面这一部分打标签，从后往前打
    if episode[-1]['score'] == episode[-2]['score']:  # 判断一下最后一帧有没有进球，虽然这种可能性微乎其微
        goal = 0  # 没有进球
    elif episode[-1]['score'][0] > episode[-2]['score'][0]:
        goal = 1  # 左边球队进球
    else:
        goal = -1  # 右边球队进球
    episode[-1]['value'] = goal  # 给最后一帧打标签
    for i in range(1, len(episode)):
        index = -i - 1  # 我现在要给episode[index]打标签
        # 判断和下一时刻比球权是否改变，变了直接0（说明这个球被截断了下一时刻），没变标签和下一时刻一样
        if episode[index]['ball_owned_team_new'] != episode[index + 1]['ball_owned_team_new']:
            episode[index]['value'] = 0
        else:
            episode[index]['value'] = episode[index + 1]['value']
        # 判断这一时刻有没有进球，有进球，重新赋值
        if i < len(episode) - 1:
            if episode[index]['score'][0] > episode[index - 1]['score'][0]:
                episode[index]['value'] = 1
            if episode[index]['score'][1] > episode[index - 1]['score'][1]:
                episode[index]['value'] = -1
    return episode
t2 = time.time()
print(f'labelling uses time {t2-t1:.4f} s')


def process_original_data():
    t1 = time.time()
    files = os.listdir(data_dir)
    files.sort()
    
    print(files)
    df = []
    
    cnt = 0
    
    for file_name in files:
        file_i = np.load(data_dir+file_name, allow_pickle=True)
        
        match_num = int(re.findall('([\d]+)',file_name)[0])
        
        file_i = episode_process(file_i)
        
        columns = sorted([i for i in list(file_i[0].keys())])
        data_list = []
        
        for line in file_i:
            line_list = [line[key] for key in columns]
            data_list.append(line_list)
        
        df_tmp = pd.DataFrame(data=data_list, columns=columns)
        
        df_tmp.drop('ball_owned_team_new',axis=1,inplace=True)
        
        df_tmp['match_num'] = match_num
        
        df.append(df_tmp)

        cnt += 1

        if cnt%100 == 0:
            print('has processed:', cnt)
    
    # match_num 和 steps_left做联合主键
    df = pd.concat(df, axis=0).reset_index(drop=True)
    
    # 降内存
    if type(df) == pd.DataFrame:
        for col in df.columns:
            df[col] = utils.downcast(df[col])
    
    df.loc[df['value']==-1,'value'] = 2
    
    df_model = df[['match_num','steps_left']]
    df_model['label'] = df['value']

    t2 = time.time()
    print(f'constructing dataframe {t2 - t1:.4f} s')

    #print("看看df:",df)
    #print("看看df_model:",df_model)
    return df, df_model

def random_split_data(df_model):
    """ 
    这里训练测试验证集合的划分，基本的几种
    1.随机划分，
    2.拿几场完整比赛做测试集，
    3.每场比赛抽取相同比例的数据作为测试集，
    4.每场比赛以session维度（一次完整的控球）抽取测试集合
    ....
    
    

    """
    t1 = time.time()
    data_num = df_model.shape[0]
    train_point = int(np.ceil(data_num*0.7))
    valid_point = int(np.ceil(data_num*0.9))
    
    
    df_model = df_model.sample(frac=1, random_state=SEED)
    
    train_df = df_model[:train_point]
    valid_df = df_model[train_point:valid_point]
    test_df = df_model[valid_point:]

    t2 = time.time()
    print(f'spliting data uses time {t2-t1:.4f} s')
    return train_df, valid_df, test_df

def save_processed_data(obj, s):
    block_len = len(obj)//block_num
    path = file_dir+'/'+ s
    if not os.path.exists( path ):
        os.makedirs( path )
    for block_id in range(block_num):
        save_dir = path+'/'+str(block_id)+'.pkl'
        l = block_id * block_len
        r = (block_id+1) * block_len
        utils.pickle.dump( obj.iloc[l:r], open(save_dir,'wb') )

def load_processed_data(s):
    path = file_dir+'/'+s
    if os.path.exists(path):
        datas = []
        for block_id in range(block_num):
            save_dir = path+'/'+str(block_id)+'.pkl'
            datas.append( utils.pickle.load( open(save_dir,'rb') ) )
        data = pd.concat( datas )
        return data
    else:
        return utils.pickle.load( open(file_dir,'rb') )
      
t3 = time.time()
print(f'processing data uses time {t3-t2: .4f} s')
