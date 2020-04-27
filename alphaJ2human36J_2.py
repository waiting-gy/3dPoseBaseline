# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import viz

import re

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]
enc_in = np.zeros((1, 64))
enc_in[0] = [0 for i in range(64)]

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

def cmu2human36m(json_data, frame):
    _data = json_data['bodies'][0]['joints']
    xy = []

    for o in range(0,len(_data),3):   #alphapose 54 3为步长
        xy.append(_data[o])
        xy.append(_data[o+1]) 

    joints_array = np.zeros((1, 36))
    joints_array[0] = [0 for i in range(36)]
    for k in range(len(joints_array[0])):
        joints_array[0][k] = xy[k]
    _data = joints_array[0]

    # mapping all body parts or 3d-pose-baseline format
    for i in range(len(order)):#len(order) =14
        for j in range(2):
        # create encoder input
            enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]  
    for j in range(2):  
# Hip
        enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
# Neck/Nose
#        enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
        enc_in[0][14 * 2 + j] = (_data[0 * 2 + j] + _data[1 * 2 + j]) / 2
# Thorax
#        enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]
        enc_in[0][13 * 2 + j] =  (_data[2 * 2 + j] + _data[5 * 2 + j]) / 2
# Spine
        enc_in[0][12 * 2 + j] = (enc_in[0][13 * 2 + j] + enc_in[0][0 * 2 + j]) /2
# set spine
    spine_x = enc_in[0][24]
    spine_y = enc_in[0][25]

#    human36 =  {'pose_keypoints_2d':[0 for i in range(64)]}
    human36 = {}
    people = {}
    human36['pose_keypoints_2d'] =  enc_in[0].tolist()

#    print(human36)
    people['people'] = [human36]
    print(people)
    #save as another json
    with open(os.path.join('./data/jsonAlpha_one2/','{}_keypoints.json').format(str(frame).zfill(12)),'w') as outfile:
        json.dump(people,outfile)


if __name__ == '__main__':
#    org_path = r"/home/ubuntu/gaoyu/alphapose/Video3D3_cmu/multi_person/multi2/sep-json/"
#    save_path = r"./data/jsonAlpha_multi/"
    org_path = r"/home/ubuntu/gaoyu/alphapose/Video3D3_cmu/sep-json/"
    save_path = r"./data/jsonAlpha_one/"
    filelist = strsort(os.listdir(org_path))
    print(len(filelist))

    for i in range(len(filelist)):
        frame = filelist[i].split('.')[0]
        with open(os.path.join(org_path,filelist[i]),encoding='utf8')as fp:
            json_data = json.load(fp)
        cmu2human36m(json_data, frame)






'''
#save as another json
with open(os.path.join('./data/json/','{}_keypoints.json').format(str(2).zfill(12)),'w') as outfile:
    json.dump(people,outfile)

fig = plt.figure(figsize=(9.6, 5.4))#1920:1080
ax = plt.subplot(111)
viz.show2Dpose(np.array(human36['pose_keypoints_2d']), ax)

plt.show()
'''



