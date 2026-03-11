# -*- coding: utf-8 -*-
# @Time    : 2024/5/25 15:43
# @Author  : sjx_alo！！
# @FileName: EEGNet.py
# @Algorithm ：
# @Description: EEGNet模型



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class SegmentVarianceLayer(nn.Module):
    def __init__(self, segment_size, dim=-1, keepdim=False):
        super(SegmentVarianceLayer, self).__init__()
        self.segment_size = segment_size
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        # 确定输入张量的形状
        shape = x.shape

        # 将张量沿指定维度分割
        segments = torch.split(x, self.segment_size, dim=self.dim)

        # 计算每个分段的方差
        variances = [segment.var(dim=self.dim, keepdim=self.keepdim) for segment in segments]

        # 将结果拼接回原来的维度
        if self.keepdim:
            result = torch.cat(variances, dim=self.dim)
        else:
            result = torch.stack(variances, dim=self.dim)

        return result


class EegNet(nn.Module):
    def __init__(self,
                 chunk_size: int = 1125,
                 num_electrodes: int = 22,
                 in_depth = 1,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 4,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 activation = False
                 ):
        super(EegNet, self).__init__()

        self.F1 = F1
        self.F2 = F2
        self.in_depth = in_depth
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(in_depth, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))
        if activation == False:
            self.lin = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)
        else:
            self.lin = nn.Sequential(
                nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False),
                nn.Sigmoid()
            )

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_depth, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x.squeeze(-1)