#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys
import time
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options
from src.procrustes import get_transformation
import src.data_process as data_process
from src import Bar
import src.utils as utils
import src.misc as misc
import src.log as log

from src.model import LinearModel, weight_init
from src.datasets.human36m import Human36M

#
from tqdm import tqdm
import matplotlib.pyplot as plt
import viz
import json
import matplotlib.image as imgplt
import re
from opt import Options

#
opt = Options().parse()
#
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
#


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = LinearModel()
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")

    # data loading
    print(">>> loading data")
    # load statistics data
    stat_3d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.tar'))
    stat_2d = torch.load(os.path.join(opt.data_dir, 'stat_2d_ft.pth.tar'))
    # test
    if opt.test:
        err_set = []
        for action in actions:
            print (">>> TEST on _{}_".format(action))
            test_loader = DataLoader(
                dataset=Human36M(actions=action, data_path=opt.data_dir, use_hg=opt.use_hg, is_train=False),
                batch_size=opt.test_batch,
                shuffle=False,
                num_workers=opt.job,
                pin_memory=True)
#
            with open('/home/ubuntu/gaoyu/Projects/3d_pose_baseline_pytorch/data/json/000000000002_keypoints.json','r',encoding='utf8') as fp0:
                json_data0 = json.load(fp0)

            _data0 = np.array(json_data0['people'][0]['pose_keypoints_2d'])
            data_mean0 = np.mean(_data0, axis=0)
            data_std0  = np.std(_data0, axis=0)

            _data0 = torch.from_numpy(np.divide((_data0 - data_mean0), data_std0)).float()

#            print(json_data0['people'][0]['pose_keypoints_2d'])
#            _data = torch.Tensor([json_data0['people'][0]['pose_keypoints_2d']])
#            print(_data0)

#
#            print(test_loader)
#
            _, err_test = test(test_loader, model, criterion, stat_3d, procrustes=opt.procrustes)
            err_set.append(err_test)

        print (">>>>>> TEST results:")
        for action in actions:
            print ("{}".format(action), end='\t')
        print ("\n")
        for err in err_set:
            print ("{:.4f}".format(err), end='\t')
        print (">>>\nERRORS: {}".format(np.array(err_set).mean()))
        sys.exit()

    # load dadasets for training
    test_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, use_hg=opt.use_hg, is_train=False),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, use_hg=opt.use_hg),
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> data loaded !")

    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train = train(
            train_loader, model, criterion, optimizer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        loss_test, err_test = test(test_loader, model, criterion, stat_3d, procrustes=opt.procrustes)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
                      ['int', 'float', 'float', 'flaot', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)
        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()


def train(train_loader, model, criterion, optimizer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):
    losses = utils.AverageMeter()

    model.train()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

#
    train_loader = tqdm(train_loader, dynamic_ncols=True)
#

    for i, (inps, tars) in enumerate(train_loader):
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        outputs, outputs_inputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()

        # ###########
        # alpha = 0.0
        # loss1 = criterion(outputs[0], targets)
        # loss2 = criterion(outputs[1], targets)
        # loss = alpha*loss1 + (1.0-alpha)*loss2
        # ########

        loss = criterion(outputs, targets)
        loss_input  = criterion(outputs_inputs, inputs)
        loss = loss + loss_input

        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

#        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
#            .format(batch=i + 1,
#                    size=len(train_loader),
#                    batchtime=batch_time * 10.0,
#                    ttl=bar.elapsed_td,
#                    eta=bar.eta_td,
#                    loss=losses.avg)
#        bar.next()
#
        train_loader.set_description(
                    '({batch}/{size}) | batch: {batchtime:.4}ms | loss: {loss:.6f}'.format(
                        batch=i + 1,
                        size=len(train_loader),
                        batchtime=batch_time * 10.0,
                        loss=losses.avg)
                    )
    train_loader.close()
#

#    bar.finish()
    return glob_step, lr_now, losses.avg


def test(test_loader, model, criterion, stat_3d, procrustes=False):
    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
#    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))
#
    test_loader = tqdm(test_loader, dynamic_ncols=True)
    fig = plt.figure(figsize=(9.6, 5.4))#1920:1080
    stat_2d = torch.load(os.path.join(opt.data_dir, 'stat_2d.pth.tar'))
#
    for i, (inps, tars) in enumerate(test_loader):
        start = time.time()
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        outputs, outputs_inputs = model(inputs)

#        print('input:',((inputs)))#16*2
#        print('input:',((inputs[0])))#16*2
#        print('output:',(len(outputs[0])))#16*3
#        print('targets:',(len(targets[0])))#16*3

        # calculate loss

        # ###########
        # alpha = 0.0
        # loss1 = criterion(outputs[0], targets)
        # loss2 = criterion(outputs[1], targets)
        # loss = alpha * loss1 + (1.0 - alpha) * loss2
        # ########
        outputs_coord = outputs

        loss = criterion(outputs, targets)
        loss_input = criterion(outputs_inputs, inputs)
        loss = loss + loss_input

        losses.update(loss.item(), inputs.size(0))

        tars = targets

        # calculate erruracy
        #inputs_unnorm = data_process.unNormalizeData(inps.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
        targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

#        print(outputs_unnorm.shape[0])
#
#        print('outputs_unnorm:',((outputs_unnorm)))#16*2

        #_max = 0
        #_min = 10000

        #org_path = r"./data/jsonAlpha_one2/"
        #filelist = strsort(os.listdir(org_path))
        ##print(len(filelist))


        #with open(os.path.join(org_path,filelist[i]),encoding='utf8')as fp:
        #    json_data = json.load(fp)
        #spine_x = json_data['people'][0]['pose_keypoints_2d'][24]
        #spine_y = json_data['people'][0]['pose_keypoints_2d'][25]
        #spine_x = spine_x
        #spine_y = spine_y

        #print(spine_x)

        #for k in range(outputs_unnorm.shape[0]):
        #    for j in range(32):
        #       tmp = outputs_unnorm[k][j * 3 + 2] # tmp = z
        #       outputs_unnorm[k][j * 3 + 2] = outputs_unnorm[k][j * 3 + 1]# z = y
        #       outputs_unnorm[k][j * 3 + 1] = tmp # y = z
        #       if outputs_unnorm[k][j * 3 + 2] > _max:
        #           _max = outputs_unnorm[k][j * 3 + 2]# z max
        #       if outputs_unnorm[k][j * 3 + 2] < _min:
        #           _min = outputs_unnorm[k][j * 3 + 2]# z min
        # #plot出的姿态是倒立的,通过该方法将其校正
        #for k in range(outputs_unnorm.shape[0]):
        #    for j in range(32):
        #        outputs_unnorm[k][j * 3 + 2] = _max - outputs_unnorm[k][j * 3 + 2] + _min# z = max-z
        #        outputs_unnorm[k][j * 3] += (spine_x - 630)# x
        #        outputs_unnorm[k][j * 3 + 2] += (500 - spine_y)# z

        #for k in range(inputs_unnorm.shape[0]):
        #    for j in range(32):
        #       #tmp1 = inputs_unnorm[k][j * 2 + 2] # tmp = z
        #       #inputs_unnorm[k][j * 2 + 2] = inputs_unnorm[k][j * 3 + 1]# z = y
        #       #inputs_unnorm[k][j * 2 + 1] = tmp1 # y = z
        #       if inputs_unnorm[k][j * 2 + 1] > _max:
        #           _max = inputs_unnorm[k][j * 2 + 1]# z max
        #       if inputs_unnorm[k][j * 2 + 1] < _min:
        #           _min = inputs_unnorm[k][j * 2 + 1]# z min
        # #plot出的姿态是倒立的,通过该方法将其校正
        #for k in range(inputs_unnorm.shape[0]):
        #    for j in range(32):
        #        inputs_unnorm[k][j * 2 + 1] = _max - inputs_unnorm[k][j * 2 + 1] + _min# z = max-z
        #        #inputs_unnorm[k][j * 3] += (spine_x - 630)# x
        #        #inputs_unnorm[k][j * 3 + 2] += (500 - spine_y)# z

#        for k in range(len(outputs_unnorm)):
#           for j in range(32):
#              tmp0 = targets_unnorm[k][j * 3 + 2]# tmp = z
#              targets_unnorm[k][j * 3 + 2] = targets_unnorm[k][j * 3 + 1]# z = y
#              targets_unnorm[k][j * 3 + 1] = tmp0 # y = z
#
#              tmp = outputs_unnorm[k][j * 3 + 2]# tmp = z
#              outputs_unnorm[k][j * 3 + 2] = outputs_unnorm[k][j * 3 + 1]# z = y
#              outputs_unnorm[k][j * 3 + 1] = tmp # y = z
#
#        for k in range(len(outputs_unnorm)):
#           hip_z0 = targets_unnorm[k][3]
#           for j in range(32):
#              targets_unnorm[k][j * 3 + 2] = targets_unnorm[k][j * 3 + 2] - 2*(targets_unnorm[k][j * 3 + 2] - hip_z0)
#
#           hip_z = outputs_unnorm[k][3]
#           for j in range(32):
#              outputs_unnorm[k][j * 3 + 2] = outputs_unnorm[k][j * 3 + 2] - 2*(outputs_unnorm[k][j * 3 + 2] - hip_z)

        #for pp in range(len(outputs_unnorm)):

#
           #ax2 = fig.add_subplot(131)
           #ax2.get_xaxis().set_visible(False)
           #ax2.get_yaxis().set_visible(False)
           #ax2.set_axis_off()
           #ax2.set_title('Input')
           #org_path = r"/home/ubuntu/gaoyu/alphapose/Video3D3_cmu/sep-json/"
           #filelist = strsort(os.listdir(org_path))
           #print(filelist[i])
           #img2d = imgplt.imread(os.path.join('/home/ubuntu/gaoyu/alphapose/Video3D3_cmu/vis/', '{0}.jpg'.format((filelist[i].split('.')[0]).zfill(12))))
           #ax2.imshow(img2d, aspect='equal')
#
#           ax0 = fig.add_subplot(131, projection='3d')
#           ax0.view_init(0, 300)
#           viz.show3Dpose( targets_unnorm[pp], ax0, add_labels=True, title = 'GroundTruth')
           #ax0 = fig.add_subplot(132)
           #ax0.view_init(0, 300)
           #viz.show2Dpose( inputs_unnorm[pp]*2.4, ax0, add_labels=True, title = '2DPose Input')

           #Reconstruction = 1/((time.time() - start))
           #start = time.time()
           #ax1 = fig.add_subplot(133, projection='3d')
#          # print(len(outputs_unnorm[pp])) #96
           #ax1.view_init(0, 300) #默认值30，120
           #viz.show3Dpose( outputs_unnorm[pp], ax1, add_labels=True, title = 'Reconstruction:  {}FPS'.format(int(Reconstruction)))

           #plt.pause(0.0000001)
           #plt.clf()


#        print('targets_unnorm:',len(outputs_unnorm[0]))#96=32*3
#        print('outputs_unnorm:',len(outputs_unnorm[0]))#96=32*3
        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        outputs_use = outputs_unnorm[:, dim_use]
        targets_use = targets_unnorm[:, dim_use]

#        print('targets_unnorm:',len(outputs_use[0]))#51=17*3
#        print('outputs_unnorm:',len(outputs_use[0]))#51=17*3


        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3)
                _, Z, T, b, c = get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                outputs_use[ba, :] = out.reshape(1, 51)

        sqerr = (outputs_use - targets_use) ** 2

        distance = np.zeros((sqerr.shape[0], 17))
        dist_idx = 0
        for k in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

        # update summary
        if (i + 1) % 100 == 0:
#            batch_time = time.time() - start
            batch_time = 1
#            start = time.time()

        #bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
        #    .format(batch=i + 1,
        #            size=len(test_loader),
        #            batchtime=batch_time * 10.0,
        #            ttl=bar.elapsed_td,
        #            eta=bar.eta_td,
        #            loss=losses.avg)
        #bar.next()

#
        test_loader.set_description(
                    '({batch}/{size}) | batch: {batchtime:.4}ms | loss: {loss:.6f}'.format(
                        batch=i + 1,
                        size=len(test_loader),
                        batchtime=batch_time * 10.0,
                        loss=losses.avg)
                    )
    test_loader.close()
#

    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
#    bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":
    option = Options().parse()
    main(option)
