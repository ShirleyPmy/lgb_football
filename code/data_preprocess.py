import os
import time
from os.path import isfile
import numpy as np

class Preprocess:
    def __init__(self, arg):
        # 其他参数
        self.team = arg['ave_tune']
        # 预处理参数
        self.feature = arg['feature']

    def generate_data(self):
        '''
        主函数，把需要用到的episodes返回的raw_data做一下特征工程并保存
        '''
        if self.team == 'ave':
            print('正在处理ave模型的训练数据，特征：', self.feature)
            self.generate_average_data()
        else:
            print('正在处理tune_', self.team, '模型的训练数据，特征：', self.feature)
            self.generate_tuned_data()

    def episode_process(self, episode):
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

    def feature_engineer(self, data):  # 特征提取方式一：完全提取
        '''
        做特征工程的函数，可以处理一条数据，也可以处理多条数据
        '''
        if self.feature == 'full':
            if type(data) == dict:  # 如果只是提取一条数据的attribute和flag
                item = data
                flag = item['value']
                feature = []
                feature.extend(item['left_team'].flatten())  # 左队坐标
                feature.extend(item['left_team_direction'].flatten())  # 左队速度
                feature.extend(item['left_team_tired_factor'])  # 左队疲劳值
                feature.extend(item['right_team'].flatten())  # 右队坐标
                feature.extend(item['right_team_direction'].flatten())  # 右队速度
                feature.extend(item['right_team_tired_factor'])  # 右队疲劳值
                feature.extend(item['ball'])  # 球三维
                feature.extend(item['ball_direction'])  # 球速
                feature.extend(item['ball_rotation'])  # 球转速
                if item['ball_owned_team'] == 0:  # 控球球队one-hot
                    feature.extend([1, 0, 0])
                if item['ball_owned_team'] == -1:
                    feature.extend([0, 1, 0])
                if item['ball_owned_team'] == 1:
                    feature.extend([0, 0, 1])
                feature.extend(item['score'])  # 比分
                feature.extend([item['steps_left']])  # 剩余时间
                game_mode = [0, 0, 0, 0, 0, 0, 0]
                game_mode[item['game_mode']] = 1
                feature.extend(game_mode)
                return feature, flag
            attributes = []  # 如果data包含多条数据，放进列表里
            flags = []
            for item in data:
                flag = item['value']
                feature = []
                feature.extend(item['left_team'].flatten())  # 左队坐标
                feature.extend(item['left_team_direction'].flatten())  # 左队速度
                feature.extend(item['left_team_tired_factor'])  # 左队疲劳值
                feature.extend(item['right_team'].flatten())  # 右队坐标
                feature.extend(item['right_team_direction'].flatten())  # 右队速度
                feature.extend(item['right_team_tired_factor'])  # 右队疲劳值
                feature.extend(item['ball'])  # 球三维
                feature.extend(item['ball_direction'])  # 球速
                feature.extend(item['ball_rotation'])  # 球转速
                if item['ball_owned_team'] == 0:  # 控球球队one-hot
                    feature.extend([1, 0, 0])
                if item['ball_owned_team'] == -1:
                    feature.extend([0, 1, 0])
                if item['ball_owned_team'] == 1:
                    feature.extend([0, 0, 1])
                feature.extend(item['score'])  # 比分
                feature.extend([item['steps_left']])  # 剩余时间
                game_mode = [0, 0, 0, 0, 0, 0, 0]
                game_mode[item['game_mode']] = 1
                feature.extend(game_mode)
                attributes.append(feature)
                flags.append(flag)
            return attributes, flags

    def generate_tuned_data(self):
        '''
        生成训练tune模型需要的数据并保存
        '''
        basepath = os.path.abspath(os.path.dirname(__file__)) + '/games/tune/' + self.team  # 当前文件的绝对路径
        attribute_set = []
        flag_set = []
        i = 0
        while True:
            filepath = basepath + '/episode' + str(i) + '.npy'  # 该遍历哪一个episode的数据了
            if not isfile(filepath):  # 看看有没有全部遍历完
                break
            episode = np.load(filepath, allow_pickle=True)  # 读取该episode的数据
            episode_with_tag = self.episode_process(episode)  # 打标签
            attributes, flags = self.feature_engineer(episode_with_tag)  # 特征工程
            for attribute in attributes:
                attribute_set.append(attribute)
            for flag in flags:
                flag_set.append(flag)
            i += 1
        # os.makedirs(os.path.abspath(os.path.dirname(__file__)) + '/data_set/' + self.team + '/' + self.feature)
        np.save('data_set/' + self.team + '/' + self.feature +'/attributes.npy', attribute_set, allow_pickle=True)  # 保存数据
        np.save('data_set/' + self.team + '/' + self.feature +'/flags.npy', flag_set, allow_pickle=True)  # 保存数据
        print('训练数据已保存')

    def generate_average_data(self):
        '''
        生成训练ave模型所需要的数据并保存
        '''
        basepath = os.path.abspath(os.path.dirname(__file__)) + '/games/ave/'  # 当前文件的绝对路径
        attribute_set = []
        flag_set = []
        for root, dirs, files in os.walk(basepath):  # 遍历文件夹
            for dir in dirs:
                dir_path = os.path.join(basepath, dir)
                i = 0
                while True:
                    filepath = dir_path + '/episode' + str(i) + '.npy'  # 该遍历哪一个episode的数据了
                    if not isfile(filepath):  # 看看有没有全部遍历完
                        break
                    episode = np.load(filepath, allow_pickle=True)  # 读取该episode的数据
                    episode_with_tag = self.episode_process(episode)  # 打标签
                    attributes, flags = self.feature_engineer(episode_with_tag)  # 特征工程
                    for attribute in attributes:
                        attribute_set.append(attribute)
                    for flag in flags:
                        flag_set.append(flag)
                    i += 1
                    if i % 1000 == 0:
                        print(i)
                        time.sleep(20)
        os.makedirs(os.path.abspath(os.path.dirname(__file__)) + '/data_set/' + self.team + '/' + self.feature)
        np.save('data_set/ave/' + self.feature + '/attributes.npy', attribute_set, allow_pickle=True)  # 保存数据
        np.save('data_set/ave/' + self.feature + '/flags.npy', flag_set, allow_pickle=True)  # 保存数据
        print('训练数据已保存')

