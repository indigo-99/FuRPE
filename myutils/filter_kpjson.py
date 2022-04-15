import numpy as np
import os
import os.path as osp
import json


body_thresh=0.1
hand_thresh=0.2
face_thresh=0.4
min_valid_keypoints=12

# read comp_fns and del_indexes
data_path='/data/panyuqing/expose_experts/data/curated_fits'
all_image_name_path=osp.join(data_path,'comp_img_fns.npy')
comp_img_fns=list(np.load(all_image_name_path, allow_pickle=True))

del_index_path=osp.join(data_path,'del_indexes.npy')
del_indexes=np.load(del_index_path, allow_pickle=True)
del_indexes=list(del_indexes)
indices = [i for i in range(len(comp_img_fns)) if (i not in del_indexes)]

add_del_indexes=[]
for img_index in indices:
    img_fn = comp_img_fns[img_index]
    img_name=img_fn.split('/')[-1].split('\\')[-1].split('.')[0] #im03019

    # check keypoint2d.json
    if 'add_data' in img_fn:
        #is_add_data=True
        img_dir_tmp=osp.dirname(img_fn)#/xxx/images
        img_dir=img_dir_tmp.replace('images', '')#/xxx strip has problems!
        cont=json.load(open(osp.join(img_dir,'keypoints2d',img_name+'_keypoints.json'),encoding='utf-8'))           
        body_kp=np.array(cont['people'][0]['pose_keypoints_2d']).reshape(25,3)
        face_kp=np.array(cont['people'][0]['face_keypoints_2d']).reshape(70,3)
        left_hand_kp=np.array(cont['people'][0]['hand_left_keypoints_2d']).reshape(21,3)
        right_hand_kp=np.array(cont['people'][0]['hand_right_keypoints_2d']).reshape(21,3)
        
        body_conf = body_kp[:, -1]
        face_conf = face_kp[:, -1]
        left_hand_conf = left_hand_kp[:, -1]
        right_hand_conf = right_hand_kp[:, -1]

        body_conf[body_conf < body_thresh] = 0.0
        left_hand_conf[left_hand_conf < hand_thresh] = 0.0
        right_hand_conf[right_hand_conf < hand_thresh] = 0.0
        face_conf[face_conf < face_thresh] = 0.0

        body_kp[:, -1] = body_conf
        left_hand_kp[:, -1] = left_hand_conf
        right_hand_kp[:, -1] = right_hand_conf
        face_kp[:, -1] = face_conf

        # body_conf = (
        #     body_conf >= body_thresh).astype(
        #         body_kp.dtype)
        # left_hand_conf = (
        #     left_hand_conf >= hand_thresh).astype(
        #         body_kp.dtype)
        # right_hand_conf = (
        #     right_hand_conf >= hand_thresh).astype(
        #         body_kp.dtype)
        # face_conf = (
        #     face_conf >= face_thresh).astype(
        #         body_kp.dtype)
        
        
        keypoints2d=np.concatenate((body_kp,face_kp,left_hand_kp,right_hand_kp))
        
        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]
        valid_keypoints = keypoints[conf > 0]
        if len(valid_keypoints) < min_valid_keypoints:
            # add del_indexes
            add_del_indexes.append(img_index)

print('num of error kp2d.json files: ',len(add_del_indexes))
del_indexes+=add_del_indexes 
# save new del_indexes
delres=np.array(del_indexes)
np.save(osp.join(data_path,'del_indexes_thresh12.npy'), delres)

'''
for img_index in indices:
    img_fn = comp_img_fns[img_index]
    img_name=img_fn.split('/')[-1].split('\\')[-1].split('.')[0] #im03019
    if img_name=='video_5_frame_001324':
        print('data not deleted')
        img_dir_tmp=osp.dirname(img_fn)#/xxx/images
        img_dir=img_dir_tmp.replace('images', '')#/xxx strip has problems!
        cont=json.load(open(osp.join(img_dir,'keypoints2d',img_name+'_keypoints.json'),encoding='utf-8'))           
        body_kp=np.array(cont['people'][0]['pose_keypoints_2d']).reshape(25,3)
        face_kp=np.array(cont['people'][0]['face_keypoints_2d']).reshape(70,3)
        left_hand_kp=np.array(cont['people'][0]['hand_left_keypoints_2d']).reshape(21,3)
        right_hand_kp=np.array(cont['people'][0]['hand_right_keypoints_2d']).reshape(21,3)
        
        body_conf = body_kp[:, -1]
        face_conf = face_kp[:, -1]
        left_hand_conf = left_hand_kp[:, -1]
        right_hand_conf = right_hand_kp[:, -1]

        body_conf[body_conf < body_thresh] = 0.0
        left_hand_conf[left_hand_conf < hand_thresh] = 0.0
        right_hand_conf[right_hand_conf < hand_thresh] = 0.0
        face_conf[face_conf < face_thresh] = 0.0

        body_kp[:, -1] = body_conf
        left_hand_kp[:, -1] = left_hand_conf
        right_hand_kp[:, -1] = right_hand_conf
        face_kp[:, -1] = face_conf

        keypoints2d=np.concatenate((body_kp,face_kp,left_hand_kp,right_hand_kp))
        print(keypoints2d)'''