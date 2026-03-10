# -*- coding: utf-8 -*-
# @Time    : 2025/6/7 20:16
# @Author  : sjx_alo！！
# @FileName: ERD.py
# @Algorithm ：
# @Description: 计算ERD指标

import os

import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import natural_sort_key, channelNameCheck
import h5py

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题?


allmarkers = ['22', '24', '28', '26']
ch_names = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T7',
            'T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']

People_list = pd.read_excel('../患者列表.xlsx')
name_list = People_list['姓名'].values
side_list = People_list['患侧'].values

folder_path = 'H:\\OA\\ProdData\\MI(长期)_sel\\'
data_list = os.listdir(folder_path)
path_list = np.array([os.path.join(folder_path, path) for path in data_list])

all_data_x = []
all_data_y = []
ERD_names = []
channel_list = []


for name in name_list:
    # 选择这个人的数据进行处理
    sel_ind = [i for i in range(len(path_list)) if name in path_list[i]]
    # 对选择到的文件名进行排序
    rfile_list = sorted(path_list[sel_ind], key=natural_sort_key, reverse=False)

    # 放置每个人的数据
    tmp_datax = []
    tmp_datay = []
    tmp_channel_list = []

    # 对数据循环进行处理
    for i in range(len(rfile_list)):
        data = h5py.File(rfile_list[i], 'r')['EEG']
        data_event = data['event']
        # 读取 HDF5 object reference 对应的数据
        event_type_list0 = [data[data_event['type'][()][i][0]][()] for i in range(len(data_event['type']))]
        event_time_list = [int(data[data_event['latency'][()][i][0]][()][0][0]) for i in
                           range(len(data_event['latency']))]

        channel_name_ref = [data[ref[0]] for ref in data['chanlocs']['labels']]
        channel_name = [''.join([chr(c[0]) for c in channel_name_ref[i]]) for i in range(len(channel_name_ref))]

        assert channel_name == ch_names

        # select_channel = [np.where(np.array(channel_name) == select_channel_name[i])[0] for i in
        #                   range(len(select_channel_name))]
        # # 把数值转换为字符串的格式
        event_type_list = [''.join([chr(c[0]) for c in row]) for row in event_type_list0[:-1]]
        #
        markers_list = np.concatenate(
            [np.where(np.array(event_type_list) == allmarkers[i])[0] for i in range(len(allmarkers))])

        eegdata = np.array(data['data']).T

        each_datax = []
        each_datay = []
        for marker_ind in range(len(allmarkers)):
            # 存放一类数据

            each_PSD_trial = []

            tmp_list = np.concatenate([np.where(np.array(event_type_list) == allmarkers[marker_ind])[0]])

            tmp_sel_list = [np.where(np.array(markers_list) == tmp_list[tmp_ind])[0][0] for tmp_ind in
                            range(len(tmp_list))]

            # if len(tmp_sel_list) < 4:
            #     break
            # 为了保证数据是整齐的  所以固定6s
            each_datax.append(eegdata[:, :, tmp_sel_list])
            each_datay.extend([marker_ind for _ in range(len(tmp_sel_list))])

        # each_datax = np.array(each_datax).squeeze()
        # each_datay = np.array(each_datay)
        if len(each_datax) < 4:
            continue
        tmp_channel_list.append(channel_name)
        tmp_datax.append(each_datax)
        tmp_datay.append(each_datay)
    channel_list.append(tmp_channel_list)
    all_data_x.append(tmp_datax)
    all_data_y.append(tmp_datay)

montages_list = mne.channels.get_builtin_montages()
freq_band = [13, 30]
mean_ERD = []

for i in range(len(channel_list)):
    channels = channel_list[i][0]
    tmp_data_x = all_data_x[i]
    for k in range(3):
        each_ERD = []
        data_x = tmp_data_x[k]
        for j in range(4):
            org_data = data_x[j]

            # 计算ERD
            tmp_rest = np.transpose(org_data[:, :2000, :], axes=[2,0,1])
            # tmp_dataC3_rest = np.expand_dims(tmp_data[:, 1, :2500], axis=1)
            tmp_task = np.transpose(org_data[:, 4000:6000, :], axes=[2,0,1])
            # tmp_dataC3_task = np.expand_dims(tmp_data[:, 1, 5000:], axis=1)

            info = mne.create_info(ch_names=channels, sfreq=1000, ch_types='eeg')  # 创建信号的信息
            # 计算任务功率
            events = np.array([[i, 0, 1] for i in range(len(tmp_task))])  # 虚拟事件，n个事件
            raw = mne.EpochsArray(tmp_task, info, events=events)
            # 使用 Morlet 小波计算时频图
            frequencies = np.logspace(np.log10(4), np.log10(40), 40)  # 频率范围6Hz到30Hz
            n_cycles = frequencies / 2  # 频率越高，周期数越多
            task_power = mne.time_frequency.tfr_morlet(raw, freqs=frequencies, n_cycles=n_cycles,
                                                       return_itc=False)

            info = mne.create_info(ch_names=channels, sfreq=1000, ch_types='eeg')  # 创建信号的信息
            events = np.array([[i, 0, 1] for i in range(len(tmp_rest))])  # 虚拟事件，n个事件
            raw = mne.EpochsArray(tmp_rest, info, events=events)
            # 使用 Morlet 小波计算时频图
            frequencies = np.logspace(np.log10(4), np.log10(40), 40)  # 频率范围6Hz到30Hz
            n_cycles = frequencies / 2  # 频率越高，周期数越多
            rest_power = mne.time_frequency.tfr_morlet(raw, freqs=frequencies, n_cycles=n_cycles,
                                                       return_itc=False)

            freq_idx = np.where((rest_power.freqs >= freq_band[0]) & (rest_power.freqs <= freq_band[1]))[0]

            # 分别计算C3 和 C4的ERD，  计算beta频带的功率

            cmc_list = np.mean((np.mean(task_power.data[:, freq_idx, :], axis=-1)
                                - np.mean(rest_power.data[:, freq_idx, :], axis=-1)) \
                               / np.mean(rest_power.data[:, freq_idx, :], axis=-1), axis=-1)

            montages = mne.channels.make_standard_montage(montages_list[0])
            rechannelindex, reName = channelNameCheck(montages, channels, cmc_list)

            reWeight = cmc_list[rechannelindex]
            if len(channels) == 30:
                mean_ERD.append(reWeight)
                mean_Name = reName
                # ind_list.append(i)
            each_ERD.append(reWeight)
            # reName = channelName[rechannelindex]
            info = mne.create_info(
                ch_names=list(reName),
                ch_types=['eeg'] * len(reName),  # 通道个数
                sfreq=500)  # 采样频率
            info.set_montage(montages)
            # fig, ax = plt.subplots(figsize=(12,8))
            im, cn = mne.viz.plot_topomap(reWeight,
                                          info,
                                          names=reName,
                                          # contours = 0, # 参数表示不绘制等值线
                                          size=10,
                                          show=False,
                                          vlim=(-1, 1),
                                          cmap='jet',
                                          )


            cbar = plt.colorbar(im)
            plt.savefig('../ERD/Fig/' + name_list[i]+'_'+str(k) +'_'+str(j)+ '.png',
                        bbox_inches='tight', dpi=900)
            plt.show()

        # 4个动作 每个动作30个数
        # np.save('ind_list.npy', np.unique(np.array(ind_list)))
        np.save('../ERD/data/'+name_list[i]+str(k)+'ERD.npy', np.array(each_ERD))
np.save('../ERD/Names.npy', np.array(name_list))