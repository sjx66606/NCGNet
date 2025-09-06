# -*- coding: utf-8 -*-
# @Time    : 2025/6/17 21:48
# @Author  : sjx_alo！！
# @FileName: GModel.py
# @Algorithm ：
# @Description:


# @Description:  创建脑电和近红外的解码模型

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import stats
from torch.nn import Linear
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from model.EEGNet import EegNet
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import ChebConv, global_mean_pool
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
device = 'cuda'



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        # Define trainable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] Correlation coefficient of graph attention (unnormalized)

        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GatingNetwork1(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(64, num_experts)
        )
        self.temperature = 1.0

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / self.temperature, dim=-1)


class GatingNetwork0(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(64, num_experts)
        )
        self.temperature = 1.0

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / self.temperature, dim=-1)

class MyGCN(nn.Module):
    def __init__(self):
        super(MyGCN, self).__init__()

        self.eegNet = EegNet(chunk_size= 5000,
                 num_electrodes= 30,
                 in_depth = 1,
                 F1= 8,
                 F2= 16,
                 D= 2,
                 num_classes= 20,
                 kernel_1= 64,
                 kernel_2= 16,
                 dropout = 0.25,
                 activation = False)

        self.gcn_0 = GraphAttentionLayer(15, 20,
                                               dropout=0.3, alpha=0.1, concat=True)
        self.gcn_1 = GraphAttentionLayer(22, 20,
                                         dropout=0.3, alpha=0.1, concat=True)

        self.gcn_2 = GraphAttentionLayer(22, 20,
                                         dropout=0.3, alpha=0.1, concat=True)

        self.gcn_3 = GraphAttentionLayer(22, 20,
                                         dropout=0.3, alpha=0.1, concat=True)

        self.gcn_4 = GraphAttentionLayer(22, 20,
                                         dropout=0.3, alpha=0.1, concat=True)
        # self.gcn_2 = GraphAttentionLayer(66, 10,
        #                                  dropout=0.3, alpha=0.1, concat=True)

        self.LSTM = nn.LSTM(10,10)

        # self.encoder = TransformerEncoder(depth=2, emb_size=50)

        self.classifier = nn.Linear(20, 4)

        self.gating0 = GatingNetwork0(80, num_experts=4)
        self.gating1 = GatingNetwork1(40, num_experts=2)

        self.mlp0 = nn.Linear(10*12 + 12*40, 256)
        self.mlp1 = nn.Linear(256, 32)
        self.mlp2 = nn.Linear(32, 4)


        # common
        # self.layer_norm = nn.LayerNorm([30])
        self.bn = nn.BatchNorm1d(32)
        self.lrelu = nn.LeakyReLU(1e-4)
        self.dropout = nn.Dropout(0.5)
        # Output layer
        self.dense = Linear(128 * 2, 4)

    def forward(self, data_erd, data_cmc, batch_index=None):


        batch_erd = torch.stack(
            [torch.tensor([i] * (30)) for i in range(data_erd.shape[0])]).view(-1).cuda()

        batch_cmc = torch.stack(
            [torch.tensor([i] * (30)) for i in range(data_erd.shape[0])]).view(-1).cuda()


        data_cmc = data_cmc.reshape(data_cmc.shape[0], 30,
                                    4, -1)


        tmp_erd_out = self.gcn_0(data_erd, torch.ones(30,30).cuda())
        tmp_erd_out =tmp_erd_out.view(-1, tmp_erd_out.shape[-1])

        data_cmc1 = data_cmc.reshape(data_cmc.shape[0], 30*4, -1)

        tmp_cmc_out1 = self.gcn_1(data_cmc[:, :, 0, :], torch.ones(30,30).cuda())
        tmp_cmc_out2 = self.gcn_2(data_cmc[:, :, 1, :], torch.ones(30,30).cuda())
        tmp_cmc_out3 = self.gcn_3(data_cmc[:, :, 2, :], torch.ones(30,30).cuda())
        tmp_cmc_out4 = self.gcn_4(data_cmc[:, :, 3, :], torch.ones(30,30).cuda())


        tmp_cmc_out1 =tmp_cmc_out1.view(-1, tmp_cmc_out1.shape[-1])
        tmp_cmc_out2 =tmp_cmc_out2.view(-1, tmp_cmc_out2.shape[-1])
        tmp_cmc_out3 =tmp_cmc_out3.view(-1, tmp_cmc_out3.shape[-1])
        tmp_cmc_out4 =tmp_cmc_out4.view(-1, tmp_cmc_out4.shape[-1])


        # 进行全局平均池化（按图的 batch 维度）
        erd_out = global_mean_pool(tmp_erd_out, batch_erd)
        cmc_out1 = global_mean_pool(tmp_cmc_out1, batch_cmc)
        cmc_out2 = global_mean_pool(tmp_cmc_out2, batch_cmc)
        cmc_out3 = global_mean_pool(tmp_cmc_out3, batch_cmc)
        cmc_out4 = global_mean_pool(tmp_cmc_out4, batch_cmc)

        cmc_features = torch.cat([cmc_out1, cmc_out2, cmc_out3, cmc_out4], axis=1)
        expert_outputs_cmc = torch.stack([cmc_out1, cmc_out2, cmc_out3, cmc_out4], dim=1)

        gate_weights_cmc = self.gating0(cmc_features)  # [B, 2]
        weights_cmc = gate_weights_cmc.unsqueeze(-1)

        fused_cmc = torch.sum(expert_outputs_cmc * weights_cmc, dim=1)



        all_feature = torch.cat([erd_out, fused_cmc], axis=1)
        # x = local_x1.view(local_x1.size(0), -1)

        # MoE模型
        expert_outputs = torch.stack([erd_out, fused_cmc], dim=1)

        gate_weights = self.gating1(all_feature)  # [B, 2]
        weights = gate_weights.unsqueeze(-1)

        fused = torch.sum(expert_outputs * weights, dim=1)


        out = self.classifier(fused)

        return out