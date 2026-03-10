# -*- coding: utf-8 -*-
# @Time    : 2025/6/12 20:41
# @Author  : sjx_alo！！
# @FileName: 显著性检验.py
# @Algorithm ：
# @Description:   对数据进行检验

import os
import pickle

import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_test
from scipy.io import loadmat
from scipy import signal
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from utils import Prod_oa_data, natural_sort_key, channelNameCheck, mean_remove_bad, mean_remove_bad1, \
    replace_outliers_with_mean
import h5py

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题?

fs = 1000
allmarkers = ['22', '24', '28', '26']
ch_names = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T7',
            'T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
channelReInd = [1,0,2,4,3,6,5,8,7,10,9,11,13,12,15,14,17,16,19,18,20,22,21,24,23,26,25,27,29,28]


People_list = pd.read_excel('../患者列表.xlsx')
name_list = People_list['姓名'].values
side_list = People_list['患侧'].values

folder_path = 'F:\\D613\\国康\\柴老师脑电数据\\论文\\XXXX-卒中患者运动脑机接口神经耦合分析\\ERD\\data\\'
data_list = os.listdir(folder_path)
path_list = np.array([os.path.join(folder_path, path) for path in data_list])

montages_list = mne.channels.get_builtin_montages()
montages = mne.channels.make_standard_montage(montages_list[7])

all_ERD= []
for name in name_list:
    # 选择这个人的数据进行处理
    sel_ind = [i for i in range(len(path_list)) if name in path_list[i]]
    rfile_list = sorted(path_list[sel_ind], key=natural_sort_key, reverse=False)

    tmp_ERD_list = []
    for i in range(3):
        ERD_data = np.load(rfile_list[i])
        tmp_ERD_list.append(ERD_data)
    all_ERD.append(tmp_ERD_list)

# 将数据分为左和右  分别提取出来并计算
left_ind = [1,3,6]
right_ind = [0,2,5,7] # 右侧患病
all_ERD_left = np.array(all_ERD)[left_ind]
all_ERD_right = np.array(all_ERD)[right_ind]

# 将左右进行合并  左右翻转后合并
all_ERD_left1 = np.array(all_ERD_left).reshape(len(left_ind),3, 4,-1)[:, :, :, channelReInd]
all_ERD_right1 = np.array(all_ERD_right).reshape(len(right_ind),3, 4,-1)

all_ERD_list = np.concatenate((all_ERD_left1, all_ERD_right1),axis=0)

# 比较三组之间是否有统计差异
data = np.array(all_ERD_list).reshape(-1, 12, 30)

# 计算 IQR
Q1 = np.percentile(data, 25, axis=0)
Q3 = np.percentile(data, 75, axis=0)
IQR = Q3 - Q1

# 找出异常值的位置
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (data < lower_bound) | (data > upper_bound)

# 替换异常值为同组数据的均值（不包括异常值本身）
data_cleaned = data.copy()
n, ch, t = data.shape

for i in range(ch):
    for j in range(t):
        # 取出当前通道、时间点的所有样本值
        values = data[:, i, j]
        is_outlier = outliers[:, i, j]

        # 只用非异常值计算均值
        if np.sum(~is_outlier) > 0:
            mean_value = np.mean(values[~is_outlier])
        else:
            # 如果全部是异常值，就用全体均值
            mean_value = np.mean(values)

        # 替换异常值为均值
        data_cleaned[is_outlier, i, j] = mean_value



data = data_cleaned

n_subjects, n_conds, n_channels = data.shape

# condition mapping
paradigm_labels = ['T1', 'T2', 'T3']
action_labels = ['A1', 'A2', 'A3', 'A4']
conditions = [(p, a) for p in paradigm_labels for a in action_labels]

# 展开成长格式
records = []
for subj in range(n_subjects):
    for cond_idx, (paradigm, action) in enumerate(conditions):
        for ch in range(n_channels):
            records.append({
                'subject': f'S{subj+1}',
                'time': paradigm,
                'action': action,
                'channel': f'Ch{ch+1}',
                'value': data[subj, cond_idx, ch]
            })

df_long = pd.DataFrame.from_records(records)

# 做三因素重复测量 ANOVA
aov = AnovaRM(data=df_long,
                depvar='value',
                subject='subject',
                within=['time', 'action', 'channel'])
res = aov.fit()
# 输出ANOVA分析结果
print(res)

results = []
task_idx = 0
new_data = data.reshape(-1, 3, 4, 30)

# 通道名
channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
                 'PO3', 'PO4', 'O1', 'Oz', 'O2']

# 创建info对象
info = mne.create_info(ch_names=channel_names, sfreq=1, ch_types='eeg')
info.set_montage('standard_1020')  # 设置标准10-20电极排列

# 生成通道邻接矩阵
adjacency, ch_names = find_ch_adjacency(info, ch_type='eeg')

all_cluster = {
    'action':[],
    '天数A':[],
    '天数B':[],
    '通道索引':[],
    '对应通道名':[],
    'p_val':[],
'effect_size':[],
'CI_low':[],
    'cohens_d':[],
'CI_high':[]
}

# sub:10  action:4  条件：3  channel:30
# 条件:fs-1,fs-2,fs-3

# [sub, channel] -> channel 在 具体action和（配对条件a  配对条件b）
for action in range(4):
    for p1, p2 in [(0,1), (0,2), (1,2)]:
        data1 = new_data[:, p1, action, :]
        data2 = new_data[:, p2, action, :]

        X = [data1, data2]  # 受试者 × 通道，两个范式组成列表

        # 运行 permutation cluster test
        T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
            X, n_permutations=1000, threshold=None,
            tail=0, adjacency=adjacency, out_type='indices', seed=42
        )

        # 查看结果
        for i_c, p_val in enumerate(cluster_p_values):
            if p_val < 0.05:
                print(f'显著 cluster #{i_c}，p = {p_val:.4f}')
                print('通道索引:', clusters[i_c][0])
                print('对应通道名:', [channel_names[idx] for idx in clusters[i_c][0]])

                ch_idx = clusters[i_c][0]

                cluster_data1 = data1[:, ch_idx].mean(axis=1)
                cluster_data2 = data2[:, ch_idx].mean(axis=1)

                diff = cluster_data2 - cluster_data1

                # Cohen d
                cohens_d = diff.mean() / diff.std(ddof=1)

                # Hedges g
                n = len(diff)
                hedges_g = cohens_d * (1 - (3 / (4 * n - 9)))

                # CI
                mean_diff = diff.mean()
                sem = stats.sem(diff)

                ci_low, ci_high = stats.t.interval(
                    0.95,
                    n - 1,
                    loc=mean_diff,
                    scale=sem
                )
                all_cluster['cohens_d'].append(cohens_d)
                all_cluster['effect_size'].append(hedges_g)
                all_cluster['CI_low'].append(ci_low)
                all_cluster['CI_high'].append(ci_high)


                all_cluster['action'].append(action)
                all_cluster['天数A'].append(p1)
                all_cluster['天数B'].append(p2)
                all_cluster['通道索引'].append(clusters[i_c][0])
                all_cluster['对应通道名'].append([channel_names[idx] for idx in clusters[i_c][0]])
                all_cluster['p_val'].append(p_val)

all_cluster_pd = pd.DataFrame(all_cluster)
all_cluster_pd.to_csv('./ERD显著性0310.csv', encoding='gbk', index=False)