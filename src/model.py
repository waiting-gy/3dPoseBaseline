#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn

###########################
import torch
###########################


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

class Linear_pw(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear_pw, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.p1w1 = nn.Linear(self.l_size, self.l_size)
        self.p2w1 = nn.Linear(self.l_size, self.l_size)
        self.p3w1 = nn.Linear(self.l_size, self.l_size)
        self.p4w1 = nn.Linear(self.l_size, self.l_size)
        self.p5w1 = nn.Linear(self.l_size, self.l_size)

        self.p1w1_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p2w1_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p3w1_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p4w1_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p5w1_batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.p1w2 = nn.Linear(self.l_size, self.l_size)
        self.p2w2 = nn.Linear(self.l_size, self.l_size)
        self.p3w2 = nn.Linear(self.l_size, self.l_size)
        self.p4w2 = nn.Linear(self.l_size, self.l_size)
        self.p5w2 = nn.Linear(self.l_size, self.l_size)

        self.p1w2_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p2w2_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p3w2_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p4w2_batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.p5w2_batch_norm1 = nn.BatchNorm1d(self.l_size)


    def forward(self, x_p1, x_p2, x_p3, x_p4, x_p5):

        y_p1 = self.p1w1(x_p1)
        y_p2 = self.p1w1(x_p2)
        y_p3 = self.p1w1(x_p3)
        y_p4 = self.p1w1(x_p4)
        y_p5 = self.p1w1(x_p5)

        y_p1 = self.p1w1_batch_norm1(y_p1)
        y_p2 = self.p2w1_batch_norm1(y_p2)
        y_p3 = self.p3w1_batch_norm1(y_p3)
        y_p4 = self.p4w1_batch_norm1(y_p4)
        y_p5 = self.p5w1_batch_norm1(y_p5)

        y_p1 = self.relu(y_p1)
        y_p2 = self.relu(y_p2)
        y_p3 = self.relu(y_p3)
        y_p4 = self.relu(y_p4)
        y_p5 = self.relu(y_p5)

        y_p1 = self.dropout(y_p1)
        y_p2 = self.dropout(y_p2)
        y_p3 = self.dropout(y_p3)
        y_p4 = self.dropout(y_p4)
        y_p5 = self.dropout(y_p5)

        y_p1 = self.p1w2(y_p1)
        y_p2 = self.p1w2(y_p2)
        y_p3 = self.p1w2(y_p3)
        y_p4 = self.p1w2(y_p4)
        y_p5 = self.p1w2(y_p5)

        y_p1 = self.p1w2_batch_norm1(y_p1)
        y_p2 = self.p2w2_batch_norm1(y_p2)
        y_p3 = self.p3w2_batch_norm1(y_p3)
        y_p4 = self.p4w2_batch_norm1(y_p4)
        y_p5 = self.p5w2_batch_norm1(y_p5)

        y_p1 = self.relu(y_p1)
        y_p2 = self.relu(y_p2)
        y_p3 = self.relu(y_p3)
        y_p4 = self.relu(y_p4)
        y_p5 = self.relu(y_p5)

        y_p1 = self.dropout(y_p1)
        y_p2 = self.dropout(y_p2)
        y_p3 = self.dropout(y_p3)
        y_p4 = self.dropout(y_p4)
        y_p5 = self.dropout(y_p5)

        out1 = x_p1 + y_p1
        out2 = x_p2 + y_p2
        out3 = x_p3 + y_p3
        out4 = x_p4 + y_p4
        out5 = x_p5 + y_p5


        return out1, out2, out3, out4, out5


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

        # # process input to linear size
        # self.w1 = nn.Linear(self.input_size, self.linear_size)
        #
        # ######################################################################
        # #kernel_size   : 2 3 4
        # #padding       : 1 2 3
        # self.dilat1 = nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3, dilation=2, bias=False, padding=2)
        # self.dilat2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        # self.dilat3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        # self.dilat4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)

        ######################################################################



        ######################################################################

        self.p1w1 = nn.Linear(3 * 2, self.linear_size)
        self.p2w1 = nn.Linear(3 * 2, self.linear_size)
        self.p3w1 = nn.Linear(4 * 2, self.linear_size)
        self.p4w1 = nn.Linear(3 * 2, self.linear_size)
        self.p5w1 = nn.Linear(3 * 2, self.linear_size)

        # self.pw1_dilat1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        # self.pw2_dilat1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        # self.pw3_dilat1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        # self.pw4_dilat1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)
        # self.pw5_dilat1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False, padding=2)

        self.p1w1_batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.p2w1_batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.p3w1_batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.p4w1_batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.p5w1_batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.p1w1_relu = nn.ReLU(inplace=True)
        self.p2w1_relu = nn.ReLU(inplace=True)
        self.p3w1_relu = nn.ReLU(inplace=True)
        self.p4w1_relu = nn.ReLU(inplace=True)
        self.p5w1_relu = nn.ReLU(inplace=True)

        self.p1w1_dropout = nn.Dropout(self.p_dropout)
        self.p2w1_dropout = nn.Dropout(self.p_dropout)
        self.p3w1_dropout = nn.Dropout(self.p_dropout)
        self.p4w1_dropout = nn.Dropout(self.p_dropout)
        self.p5w1_dropout = nn.Dropout(self.p_dropout)


        self.pw_linear_stages = []
        for l in range(num_stage):
            self.pw_linear_stages.append(Linear_pw(self.linear_size, self.p_dropout))
        self.pw_linear_stages = nn.ModuleList(self.pw_linear_stages)


        self.p1w2 = nn.Linear(self.linear_size, self.linear_size)
        self.p2w2 = nn.Linear(self.linear_size, self.linear_size)
        self.p3w2 = nn.Linear(self.linear_size, self.linear_size)
        self.p4w2 = nn.Linear(self.linear_size, self.linear_size)
        self.p5w2 = nn.Linear(self.linear_size, self.linear_size)

        self.p1w2_1 = nn.Linear(self.linear_size, 6)
        self.p2w2_1 = nn.Linear(self.linear_size, 6)
        self.p3w2_1 = nn.Linear(self.linear_size, 8)
        self.p4w2_1 = nn.Linear(self.linear_size, 6)
        self.p5w2_1 = nn.Linear(self.linear_size, 6)


        # self.pw3 = nn.Linear(self.linear_size*5, self.linear_size)
        self.pw3 = nn.Linear(self.input_size, self.linear_size)
        self.pw3_batch_norm = nn.BatchNorm1d(self.linear_size)
        self.pw3_relu = nn.ReLU(inplace=True)
        self.pw3_dropout = nn.Dropout(self.p_dropout)

        for k in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w3_1 = nn.Linear(self.linear_size, self.output_size)

        self.relu3_1 = nn.ReLU(inplace=True)
        self.dropout3_1 = nn.Dropout(self.p_dropout)


        self.pw4 = nn.Linear(self.linear_size , self.output_size)



        ######################################################################


     #    self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
     #
     #    self.linear_stages = []
     #    for l in range(num_stage):
     #        self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
     #    self.linear_stages = nn.ModuleList(self.linear_stages)
     #
     #    # post processing
     #    self.w2 = nn.Linear(self.linear_size, self.output_size)
     #
     #    self.relu = nn.ReLU(inplace=True)
     #    self.dropout = nn.Dropout(self.p_dropout)
     #
     #  ######################
     #    # 3d joints
     #    self.input_size_2 =  16 * 3
     #    # 3d joints
     #    self.output_size_2 = 16 * 3
     #
     #    # process input to linear size
     #    self.w3 = nn.Linear(self.input_size_2, self.linear_size)
     #    self.batch_norm2 = nn.BatchNorm1d(self.linear_size)
     #
     #    self.linear_stages_2 = []
     #    for l in range(num_stage):
     #        self.linear_stages_2.append(Linear(self.linear_size, self.p_dropout))
     #    self.linear_stages_2 = nn.ModuleList(self.linear_stages_2)
     #
     #    # post processing
     #    self.w4 = nn.Linear(self.linear_size, self.output_size_2)
     #
     #    self.relu2 = nn.ReLU(inplace=True)
     #    self.dropout2 = nn.Dropout(self.p_dropout)
     # ##########################



    def forward(self, x):
        # # pre-processing
        #
        # ######################################################################
        # # d = self.dilat1(x.unsqueeze(1))
        # # x = d.squeeze(1)
        # #
        # # print(x)
        # # print(x)
        # # print(self.dilat.weight)
        # #
        # # print(d)
        # # print(type(d))
        # # print(d.shape)
        # # print(d.squeeze(1).shape)
        # # print(x.unsqueeze(1).shape)
        # # print(x.unsqueeze(1))
        #
        # ######################################################################
        # # print(x.shape)
        # # if x.shape == self.input_size:
        # #     x = x.unsqueeze(0)
        # ######################################################################
        # # print(x.shape)
        # y = self.w1(x)
        # # print(y.shape)
        #
        # ######################################################################
        # d = self.dilat1(y.unsqueeze(1))
        # y = d.squeeze(1)
        # ######################################################################
        #
        # y = self.batch_norm1(y)
        # # print(y.shape)
        # y = self.relu(y)
        # # print(y.shape)
        # y = self.dropout(y)
        # # print(y.shape)
        #
        # ###########################
        # # y3d = y
        # ############################
        #
        # # linear layers
        # for i in range(self.num_stage):
        #     y = self.linear_stages[i](y)
        #
        # ######################################################################
        # d = self.dilat4(y.unsqueeze(1))
        # y = d.squeeze(1)
        # ######################################################################
        #
        # y = self.w2(y)
        #
        # ##########################
        #
        # # result = [y]
        # #
        # # y = self.w3(y)
        # # y = self.batch_norm2(y)
        # # y = self.relu2(y)
        # # y = self.dropout2(y)
        # #
        # # # linear layers
        # # for i in range(self.num_stage):
        # #     y = self.linear_stages_2[i](y)
        # #
        # # #y = y + y3d
        # #
        # # y = self.w4(y)
        # #
        # #
        # # result.append(y)
        #
        # # return result
        # return y
        x_p1 = x[:, 0:6]
        x_p2 = x[:, 6:12]
        x_p3 = x[:, 12:20]
        x_p4 = x[:, 20:26]
        x_p5 = x[:, 26:32]

        y_p1 = self.p1w1(x_p1)
        y_p2 = self.p2w1(x_p2)
        y_p3 = self.p3w1(x_p3)
        y_p4 = self.p4w1(x_p4)
        y_p5 = self.p5w1(x_p5)

        y_p1 = self.p1w1_batch_norm1(y_p1)
        y_p2 = self.p2w1_batch_norm1(y_p2)
        y_p3 = self.p3w1_batch_norm1(y_p3)
        y_p4 = self.p4w1_batch_norm1(y_p4)
        y_p5 = self.p5w1_batch_norm1(y_p5)

        y_p1 = self.p1w1_relu(y_p1)
        y_p2 = self.p2w1_relu(y_p2)
        y_p3 = self.p3w1_relu(y_p3)
        y_p4 = self.p4w1_relu(y_p4)
        y_p5 = self.p5w1_relu(y_p5)

        y_p1 = self.p1w1_dropout(y_p1)
        y_p2 = self.p2w1_dropout(y_p2)
        y_p3 = self.p3w1_dropout(y_p3)
        y_p4 = self.p4w1_dropout(y_p4)
        y_p5 = self.p5w1_dropout(y_p5)

        for i in range(self.num_stage):
            y_p1, y_p2, y_p3, y_p4, y_p5  = self.pw_linear_stages[i](y_p1, y_p2, y_p3, y_p4, y_p5)

        y_p1 = self.p1w2_1_dropout(y_p1)
        y_p2 = self.p2w2_1_dropout(y_p2)
        y_p3 = self.p3w2_1_dropout(y_p3)
        y_p4 = self.p4w2_1_dropout(y_p4)
        y_p5 = self.p5w2_1_dropout(y_p5)


        y_p1 = y_p1 + x_p1
        y_p2 = y_p2 + x_p2
        y_p3 = y_p3 + x_p3
        y_p4 = y_p4 + x_p4
        y_p5 = y_p5 + x_p5


        y = torch.cat((y_p1, y_p2, y_p3, y_p4, y_p5), 1)

        y = self.pw3(y)
        y = self.pw3_batch_norm(y)
        y = self.pw3_relu(y)
        y = self.pw3_dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        result = self.pw4(y)

        return result














