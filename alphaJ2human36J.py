# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import viz

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

def cmu2human36m(json_data, frame):
    json_data2intlist = np.array(json_data['bodies'][0]['joints']).astype(int).tolist()

    people = {'people':[]}
    pose_keypooints_2d = {'pose_keypoints_2d':[0 for i in range(16*2)]}
#    print(pose_keypooints_2d)
    #get 16 bodies from 18bodis(alphapose_cmu)
    #R-foot
    pose_keypooints_2d['pose_keypoints_2d'][0] =  json_data2intlist[10*3]
    pose_keypooints_2d['pose_keypoints_2d'][1] =  json_data2intlist[10*3 + 1]
    #R-Knee
    pose_keypooints_2d['pose_keypoints_2d'][2] =  json_data2intlist[9*3]
    pose_keypooints_2d['pose_keypoints_2d'][3] =  json_data2intlist[9*3 + 1]
    #R-Hip
    pose_keypooints_2d['pose_keypoints_2d'][4] =  json_data2intlist[8*3]
    pose_keypooints_2d['pose_keypoints_2d'][5] =  json_data2intlist[8*3 + 1]
    #L-Hip
    pose_keypooints_2d['pose_keypoints_2d'][6] =  json_data2intlist[11*3]
    pose_keypooints_2d['pose_keypoints_2d'][7] =  json_data2intlist[11*3 + 1]
    #L-Knee
    pose_keypooints_2d['pose_keypoints_2d'][8] =  json_data2intlist[12*3]
    pose_keypooints_2d['pose_keypoints_2d'][9] =  json_data2intlist[12*3 + 1]
    #L-Foot
    pose_keypooints_2d['pose_keypoints_2d'][10] =  json_data2intlist[13*3]
    pose_keypooints_2d['pose_keypoints_2d'][11] =  json_data2intlist[13*3 + 1]
    #R-Wist
    pose_keypooints_2d['pose_keypoints_2d'][20] =  json_data2intlist[4*3]
    pose_keypooints_2d['pose_keypoints_2d'][21] =  json_data2intlist[4*3 + 1]
    #R-Elbow
    pose_keypooints_2d['pose_keypoints_2d'][22] =  json_data2intlist[3*3]
    pose_keypooints_2d['pose_keypoints_2d'][23] =  json_data2intlist[3*3 + 1]
    #R-Shoulder
    pose_keypooints_2d['pose_keypoints_2d'][24] =  json_data2intlist[2*3]
    pose_keypooints_2d['pose_keypoints_2d'][25] =  json_data2intlist[2*3 + 1]
    #L-Shoulder
    pose_keypooints_2d['pose_keypoints_2d'][26] =  json_data2intlist[5*3]
    pose_keypooints_2d['pose_keypoints_2d'][27] =  json_data2intlist[5*3 + 1]
    #L-Elbow
    pose_keypooints_2d['pose_keypoints_2d'][28] =  json_data2intlist[6*3]
    pose_keypooints_2d['pose_keypoints_2d'][29] =  json_data2intlist[6*3 + 1]
    #L-Wrist
    pose_keypooints_2d['pose_keypoints_2d'][30] =  json_data2intlist[7*3]
    pose_keypooints_2d['pose_keypoints_2d'][31] =  json_data2intlist[7*3 + 1]

    #Hip * ("{11, " “"LHip" ”"} + {8, "RHip"} " )/2
    pose_keypooints_2d['pose_keypoints_2d'][12] =  int((json_data2intlist[8*3] + json_data2intlist[11*3])/2)
    pose_keypooints_2d['pose_keypoints_2d'][13] =  int((json_data2intlist[8*3 + 1] + json_data2intlist[11*3 +1])/2)
    #Throax * ("{5, "LShoulder"}  +{2, "RShoulder"}") /2
    pose_keypooints_2d['pose_keypoints_2d'][14] =  int((json_data2intlist[2*3] + json_data2intlist[5*3])/2)
    pose_keypooints_2d['pose_keypoints_2d'][15] =  int((json_data2intlist[2*3 + 1] + json_data2intlist[5*3 + 1])/2)  
    #Spine *  ("‘Hip‘ +‘Thorax‘") /2  
    pose_keypooints_2d['pose_keypoints_2d'][16] = int((pose_keypooints_2d['pose_keypoints_2d'][12] + pose_keypooints_2d['pose_keypoints_2d'][14])/2)
    pose_keypooints_2d['pose_keypoints_2d'][17] = int((pose_keypooints_2d['pose_keypoints_2d'][13] + pose_keypooints_2d['pose_keypoints_2d'][15])/2)
    #Head * = ("{16, " “"REye" ”"}  +  {17, "LEye"}, " )/2
    pose_keypooints_2d['pose_keypoints_2d'][18] = int((json_data2intlist[14*3] + json_data2intlist[15*3])/2)
    pose_keypooints_2d['pose_keypoints_2d'][19] = int((json_data2intlist[14*3 + 1] + json_data2intlist[15*3 + 1])/2)

    #people['people'] = [pose_keypooints_2d]
    #print(people)

    #neck/nose = (json_data2intlist[1/0*3]

    #save as another json
    #with open(os.path.join('./data/json/','{}_keypoints.json').format(str(1).zfill(12)),'w') as outfile:
    #    json.dump(people,outfile)


    #human2.6m
    human36 =  {'pose_keypoints_2d':[0 for i in range(64)]}
#    print(human36)
    #Hip
    human36['pose_keypoints_2d'][0*2] = pose_keypooints_2d['pose_keypoints_2d'][6*2]
    human36['pose_keypoints_2d'][0*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][6*2 + 1]
    #RHip
    human36['pose_keypoints_2d'][1*2] = pose_keypooints_2d['pose_keypoints_2d'][2*2]
    human36['pose_keypoints_2d'][1*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][2*2 + 1]
    #RKnee
    human36['pose_keypoints_2d'][2*2] = pose_keypooints_2d['pose_keypoints_2d'][1*2]
    human36['pose_keypoints_2d'][2*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][1*2 + 1]
    #RFoot
    human36['pose_keypoints_2d'][3*2] = pose_keypooints_2d['pose_keypoints_2d'][0*2]
    human36['pose_keypoints_2d'][3*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][0*2 + 1]
    #LHip
    human36['pose_keypoints_2d'][6*2] = pose_keypooints_2d['pose_keypoints_2d'][3*2]
    human36['pose_keypoints_2d'][6*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][3*2 + 1]
    #LKnee
    human36['pose_keypoints_2d'][7*2] = pose_keypooints_2d['pose_keypoints_2d'][4*2]
    human36['pose_keypoints_2d'][7*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][4*2 + 1]
    #LFoot
    human36['pose_keypoints_2d'][8*2] = pose_keypooints_2d['pose_keypoints_2d'][5*2]
    human36['pose_keypoints_2d'][8*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][5*2 + 1]
    #Spine
    human36['pose_keypoints_2d'][12*2] = pose_keypooints_2d['pose_keypoints_2d'][7*2]
    human36['pose_keypoints_2d'][12*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][7*2 + 1]
    #Thorax
    human36['pose_keypoints_2d'][13*2] = pose_keypooints_2d['pose_keypoints_2d'][8*2]
    human36['pose_keypoints_2d'][13*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][8*2 + 1]
    #Neck/Nose #neck/nose = (json_data2intlist[1/0*3]
    human36['pose_keypoints_2d'][14*2] = json_data2intlist[3*3]
    human36['pose_keypoints_2d'][14*2 + 1 ] = json_data2intlist[3*3 + 1]
    #Head 
    human36['pose_keypoints_2d'][15*2] = pose_keypooints_2d['pose_keypoints_2d'][9*2]
    human36['pose_keypoints_2d'][15*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][9*2 + 1]
    #LShoulder
    human36['pose_keypoints_2d'][17*2] = pose_keypooints_2d['pose_keypoints_2d'][13*2]
    human36['pose_keypoints_2d'][17*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][13*2 + 1]
    #LElbow
    human36['pose_keypoints_2d'][18*2] = pose_keypooints_2d['pose_keypoints_2d'][14*2]
    human36['pose_keypoints_2d'][18*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][14*2 + 1]
    #LWrist
    human36['pose_keypoints_2d'][19*2] = pose_keypooints_2d['pose_keypoints_2d'][15*2]
    human36['pose_keypoints_2d'][19*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][15*2 + 1]
    #RShoulder
    human36['pose_keypoints_2d'][25*2] = pose_keypooints_2d['pose_keypoints_2d'][12*2]
    human36['pose_keypoints_2d'][25*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][12*2 + 1]
    #RElbow
    human36['pose_keypoints_2d'][26*2] = pose_keypooints_2d['pose_keypoints_2d'][11*2]
    human36['pose_keypoints_2d'][26*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][11*2 + 1]
    #RWrist
    human36['pose_keypoints_2d'][27*2] = pose_keypooints_2d['pose_keypoints_2d'][10*2]
    human36['pose_keypoints_2d'][27*2 + 1] = pose_keypooints_2d['pose_keypoints_2d'][10*2 + 1]

    print(human36)
    people['people'] = [human36]
    #save as another json
    with open(os.path.join('./data/jsonAlpha_one/','{}_keypoints.json').format(str(frame).zfill(12)),'w') as outfile:
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



