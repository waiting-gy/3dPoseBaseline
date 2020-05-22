#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        ######################################################################
        self.dilat2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        ######################################################################
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        ######################################################################
        self.dilat3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        ######################################################################
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)

        # ######################################################################
        # d = self.dilat2(y.unsqueeze(1))
        # y = d.squeeze(1)
        # ######################################################################

        y = self.batch_norm1(y)
        ######################################################################
        d = self.dilat2(y.unsqueeze(1))
        y = d.squeeze(1)
        ######################################################################
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        ######################################################################
        d = self.dilat3(y.unsqueeze(1))
        y = d.squeeze(1)
        ######################################################################
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class Linear2(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear2, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  16 * 2
        # 3d joints
        self.output_size = 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)

        ######################################################################
        #kernel_size   : 2 3 4
        #padding       : 1 2 3
        self.dilat1 = nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3, dilation=2, bias=False, padding=2)
        self.dilat2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        self.dilat3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        self.dilat4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)

        ######################################################################
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

      ######################
        # 3d joints
        self.input_size_2 =  16 * 3
        # 3d joints
        self.output_size_2 = 16 * 3

        # process input to linear size
        self.w3 = nn.Linear(self.input_size_2, self.linear_size)
        self.batch_norm2 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_2 = []
        for l in range(num_stage):
            self.linear_stages_2.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages_2 = nn.ModuleList(self.linear_stages_2)

        # post processing
        self.w4 = nn.Linear(self.linear_size, self.output_size_2)

        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(self.p_dropout)
     ##########################



    def forward(self, x):
        # pre-processing

        ######################################################################
        # d = self.dilat1(x.unsqueeze(1))
        # x = d.squeeze(1)
        #
        # print(x)
        # print(x)
        # print(self.dilat.weight)
        #
        # print(d)
        # print(type(d))
        # print(d.shape)
        # print(d.squeeze(1).shape)
        # print(x.unsqueeze(1).shape)
        # print(x.unsqueeze(1))

        ######################################################################
        # print(x.shape)
        # if x.shape == self.input_size:
        #     x = x.unsqueeze(0)
        ######################################################################
        # print(x.shape)
        y = self.w1(x)
        # print(y.shape)

        ######################################################################
        d = self.dilat1(y.unsqueeze(1))
        y = d.squeeze(1)
        ######################################################################

        y = self.batch_norm1(y)
        # print(y.shape)
        y = self.relu(y)
        # print(y.shape)
        y = self.dropout(y)
        # print(y.shape)

        ###########################
        # y3d = y
        ############################

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        ######################################################################
        d = self.dilat4(y.unsqueeze(1))
        y = d.squeeze(1)
        ######################################################################

        y = self.w2(y)

        ##########################

        # result = [y]
        #
        # y = self.w3(y)
        # y = self.batch_norm2(y)
        # y = self.relu2(y)
        # y = self.dropout2(y)
        #
        # # linear layers
        # for i in range(self.num_stage):
        #     y = self.linear_stages_2[i](y)
        #
        # #y = y + y3d
        #
        # y = self.w4(y)
        #
        #
        # result.append(y)

        # return result
        return y
