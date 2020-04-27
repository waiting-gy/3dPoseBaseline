# -*- coding: utf-8 -*-

import json
import os
import numpy as np

with open('/home/ubuntu/gaoyu/Projects/3d_pose_baseline_pytorch/1696alphapose.json',encoding='utf8') as fp:
    json_data = json.load(fp)
print(np.array(json_data['bodies'][0]['joints']).astype(int))

json_data2intlist = np.array(json_data['bodies'][0]['joints']).astype(int).tolist()

people = {'people':[]}
pose_keypooints_2d = {'pose_keypoints_2d':[0 for i in range(16*2)]}
print(pose_keypooints_2d)
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

people['people'] = [pose_keypooints_2d]
print(people)

#save as another json
with open(os.path.join('./data/json/','{}_keypoints.json').format(str(1).zfill(12)),'w') as outfile:
    json.dump(people,outfile)




