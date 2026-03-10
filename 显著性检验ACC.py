# -*- coding: utf-8 -*-
# @Time    : 2025/6/17 14:44
# @Author  : sjx_alo！！
# @FileName: 显著性检验ACC.py
# @Algorithm ：
# @Description: 对ACC指标进行显著性检验  看是否存在显著差异

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, wilcoxon

acc_eeg = pd.read_csv('../ACC_res/ACC_EEGNet.csv').iloc[:, 1:].values
acc_fbcnet = pd.read_csv('./ACC_FBCNet.csv').iloc[:, 1:].values
acc_lmda = pd.read_csv('../ACC_res/ACC_LMDA.csv').iloc[:, 1:].values
acc_conformer = pd.read_csv('../ACC_res/ACC_Conformer.csv').iloc[:, 1:].values
a = np.array([0.142857, 0.000000, 0.600000, 0.000000, 0.285714, 0.666667, 0.142857, 0.571429])
b = np.array([0.571429, 0.478261, 0.333333, 0.600000, 0.428571, 0.166667, 0.500000, 0.428571])

diff = a - b
from scipy.stats import shapiro
stat, p = shapiro(diff)
print(f"Shapiro-Wilk p = {p:.5f}")

a = acc_fbcnet[:8,0]
b = acc_fbcnet[:8,2]
diff = a - b
stat, p = shapiro(diff)
print(f"Shapiro-Wilk 检验 p值: {p:.3f}")
p_list = []
stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox', correction=False)
print(f"Wilcoxon 统计量: {stat}, p值: {p:.5f}")
p_list.append(p)

stat, p = wilcoxon(acc_fbcnet[:8,0], acc_fbcnet[:8,2], alternative='two-sided')
print(f"T检验统计量: {stat:.3f}, p值: {p:.3f}")
p_list.append(p)


stat, p = wilcoxon(acc_lmda[:8,0], acc_lmda[:8,2], alternative='two-sided')
print(f"T检验统计量: {stat:.3f}, p值: {p:.3f}")
p_list.append(p)

stat, p = wilcoxon(acc_conformer[:,0], acc_conformer[:,2], alternative='two-sided')
print(f"T检验统计量: {stat:.3f}, p值: {p:.3f}")
p_list.append(p)

acc = []
acc1 = []
acc.extend(acc_eeg[:8, 0])
acc1.extend(acc_eeg[:8, 2])
acc.extend(acc_lmda[:8, 0])
acc1.extend(acc_lmda[:8, 2])
acc.extend(acc_fbcnet[:8, 0])
acc1.extend(acc_fbcnet[:8, 2])
acc.extend(acc_conformer[:8, 0])
acc1.extend(acc_conformer[:8, 2])

stat, p = ttest_ind(acc, acc1)
print(f"T检验统计量: {stat:.3f}, p值: {p:.3f}")
p_list.append(p)
