# -*- coding: utf-8 -*-
# @Time    : 2025/6/7 20:16
# @Author  : sjx_alo！！
# @FileName: CMC.py
# @Algorithm ：
# @Description:  计算CMC指标
import os
import pickle

import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy import signal


from utils import Prod_oa_data, natural_sort_key, channelNameCheck
import h5py
from mne_connectivity import spectral_connectivity_epochs

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题?

fs = 1000
allmarkers = ['22', '24', '28', '26']
ch_names = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T7',
            'T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']

People_list = pd.read_excel('../患者列表.xlsx')
name_list = People_list['姓名'].values
side_list = People_list['患侧'].values

folder_path = 'H:\\OA\\ProdData\\MI(长期)_sel\\'
data_list = os.listdir(folder_path)
path_list = np.array([os.path.join(folder_path, path) for path in data_list])

EMG_path = folder_path + 'EMG\\'
EMGfile_list = np.array([EMG_path + file for file in os.listdir(EMG_path) if file[-4:] == '.mat'])

ERD_names = []
all_data_x = []
all_data_emg = []
all_data_y = []
channel_list = []

for name in name_list:
    # 选择这个人的数据进行处理
    sel_ind = [i for i in range(len(path_list)) if name in path_list[i]]
    EMG_sel_ind = [i for i in range(len(EMGfile_list)) if name in EMGfile_list[i]]
    # 对选择到的文件名进行排序
    rfile_list = sorted(path_list[sel_ind], key=natural_sort_key, reverse=False)
    EMG_rfile_list = sorted(EMGfile_list[EMG_sel_ind], key=natural_sort_key, reverse=False)

    # 放置每个人的数据
    tmp_datax = []
    tmp_datay = []
    tmp_emg = []
    tmp_channel_list = []

    # 对数据循环进行处理
    for i in range(len(rfile_list)):
        data = h5py.File(rfile_list[i], 'r')['EEG']
        data_emg = loadmat(EMG_rfile_list[i])['result']
        data_event = data['event']
        # 读取 HDF5 object reference 对应的数据
        event_type_list0 = [data[data_event['type'][()][i][0]][()] for i in range(len(data_event['type']))]

        event_time_list = [int(data[data['epoch']['marker_latency'][()][i][0]][()][0][0]) for i in
                           range(len(data['epoch']['marker_latency']))]

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
        each_dataemg = []
        each_datay = []
        for marker_ind in range(len(allmarkers)):
            each_PSD_trial = []

            tmp_list = np.concatenate([np.where(np.array(event_type_list) == allmarkers[marker_ind])[0]])

            tmp_sel_list = [np.where(np.array(markers_list) == tmp_list[tmp_ind])[0][0] for tmp_ind in
                            range(len(tmp_list))]

            # 为了保证数据是整齐的  所以固定6s
            each_datax.append(eegdata[:, :, tmp_sel_list])
            tmp_data_emg = []
            for tmp_marker_ind in tmp_sel_list:
                tmp_data_emg.append(data_emg[:, event_time_list[tmp_marker_ind]:event_time_list[tmp_marker_ind] + 5000])
            each_dataemg.append(tmp_data_emg)
            each_datay.extend([marker_ind for _ in range(len(tmp_sel_list))])

        tmp_datax.append(each_datax)
        tmp_emg.append(each_dataemg)
        tmp_datay.append(each_datay)
        tmp_channel_list.append(channel_name)


    channel_list.append(tmp_channel_list)
    all_data_x.append(tmp_datax)
    all_data_emg.append(tmp_emg)
    all_data_y.append(tmp_datay)


# 计算CMC指标
montages_list = mne.channels.get_builtin_montages()
freq_band = [13, 30]
mean_ERD = []

for i in range(len(channel_list)):
    for k in range(3):
        all_CMC = []
        channels = channel_list[i][k]
        data_x = all_data_x[i][k]
        for j in range(4):
            each_CMC = []
            org_data = data_x[j]

            # 计算CMC
            tmp_task_eeg = np.transpose(org_data[:, 2000:, :], axes=[2,0,1])

            con = spectral_connectivity_epochs(
                data=tmp_task_eeg,  # shape: (n_epochs, n_channels, n_times)
                method='wpli',  # 加权相位滞后指数
                mode='multitaper',
                sfreq=1000,
                fmin=freq_band[0], fmax=30,
                faverage=True,  # 对频率区间内平均
                tmin=0.0, tmax=5.0,
                mt_adaptive=False, n_jobs=1
            )

            fig = plt.figure(figsize=(8, 6))
            plt.imshow(con.get_data("dense"), vmin=0, vmax=1)
            plt.show()


            all_CMC.append(con.get_data("dense"))

        with open('../wPLI/data'+'\\'+ name_list[i]+'_'+str(k) +'wPLI.pkl', 'wb') as f:
            pickle.dump({'wPLI': all_CMC,
                'frequencies': freq_band,
                'EEG_data': data_x,
                'label_data': all_data_y[i],
                'EEG_channelName': channels}, f)