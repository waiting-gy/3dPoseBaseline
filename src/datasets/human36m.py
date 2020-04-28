#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset

#
import json
import numpy as np
from opt import Options
#

import re
def sort_key(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)
def strsort(alist):
    alist.sort(key=sort_key,reverse=False)
    return alist
#A=['000000003426.jpg', '000000000899.jpg', '000000000179.jpg', '000000002518.jpg']
#print(strsort(A))


#
opt = Options().parse()
#

TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]


class Human36M(Dataset):
    def __init__(self, actions, data_path, use_hg=True, is_train=True):
        """
        :param actions: list of actions to use
        :param data_path: path to dataset
        :param use_hg: use stacked hourglass detections
        :param is_train: load train/test dataset
        """

        self.actions = actions
        self.data_path = data_path

        self.is_train = is_train
        self.use_hg = use_hg

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_meta, self.test_meta = [], []

        # loading data
        if self.use_hg:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
#            test_2d_file = '000000000001_keypoints.json'
        else:
            train_2d_file = 'train_2d.pth.tar'
            test_2d_file = 'test_2d.pth.tar'
#            test_2d_file = '000000000001_keypoints.json'

        if self.is_train:
            # load train data
            self.train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            self.train_2d = torch.load(os.path.join(data_path, train_2d_file))
            for k2d in self.train_2d.keys():
                (sub, act, fname) = k2d
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, _ = self.train_2d[k2d].shape
                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[0], '(training) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[k2d][i])
                    self.train_out.append(self.train_3d[k3d][i])

        else:
            # load test data
            self.test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))

            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                if act not in self.actions:
                    continue
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, _ = self.test_2d[k2d].shape
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[k2d][i])
                    self.test_out.append(self.test_3d[k3d][i])


           # org_path = r"./data/jsonAlpha_one2/"
           # filelist = strsort(os.listdir(org_path))
           # print(len(filelist))

           # for i in range(len(filelist)):
           #     frame = filelist[i].split('.')[0]
           #     with open(os.path.join(org_path,filelist[i]),encoding='utf8')as fp:
           #         json_data = json.load(fp)
           #     self.test_inp.append(np.array(json_data['people'][0]['pose_keypoints_2d']))
           # #self.spine_x = self.test_inp[]

            

    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()

        else:
        #    _data = self.test_inp[index]
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()


#            print(_data)
#            print(index)
#            list_input=[]
#            for i in range(2):
#                with open(os.path.join('/home/ubuntu/gaoyu/Projects/3d_pose_baseline_pytorch/data/json/','00000000000{}_keypoints.json'.format(i+1)),'r',encoding='utf8') as fp0:
#                    json_data0 = json.load(fp0)
#                list_input.append(np.array(json_data0['people'][0]['pose_keypoints_2d']))
#            print((np.array(list_input)).flatten().reshape(-1,64))
#            _data = list_input[index]
#            _data = np.array(json_data0['people'][0]['pose_keypoints_2d'])
        #   stat_2d = torch.load(os.path.join(opt.data_dir, 'stat_2d.pth.tar'))

        #    _data = _data[stat_2d['dim_use']]
            #print(stat_2d['dim_use'])
        #    mu = stat_2d['mean'][stat_2d['dim_use']]
        #    stddev = stat_2d['std'][stat_2d['dim_use']]
#            print(_data)
         #   inputs = torch.from_numpy(np.divide((_data - mu), stddev)).float()
#            data_mean = np.mean(_data, axis=0)
#            data_std  = np.std(_data, axis=0)
#            inputs = torch.from_numpy(np.divide((_data - data_mean), data_std)).float()

        #    outputs = torch.from_numpy(np.array([0 for i in range(48)])).float()
            #print(inputs)
        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)
#            return len(range(2))
