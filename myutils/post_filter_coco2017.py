import json
import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm

from expose.data.targets.keypoints import dset_to_body_model, get_part_idxs
'''Filter data after the first filter (according to keypoint conf),
This time more strictly constraining the keypoints from raw 144 keypoints to the first 25 keypoints, 
because these are body keypoints, to avoid an img with hands detected but no body (useless in training)'''
# same as expose-master\expose\data\utils\bbox.py
body_thresh=0.1
hand_thresh=0.2
face_thresh=0.4
#min_valid_keypoints=12 # other datasets use 12 before change this code to transform kpts order here

coco_source_idxs, smplx_target_idxs = dset_to_body_model(
    dset='coco',
    model_type='smplx', use_hands=True, use_face=True,
    use_face_contour=True,
    keyp_format='coco25')
coco_source_idxs = np.asarray(coco_source_idxs, dtype=np.int64) #wrong! only 19 indexes!
smplx_target_idxs = np.asarray(smplx_target_idxs, dtype=np.int64)

# for i in range(len(coco_source_idxs)):
#     print('coco_source_idxs: ',coco_source_idxs[i],' , smplx_target_idxs: ',smplx_target_idxs[i])



def check_kpt_conf(cont,min_valid_keypoints=6):
    face_valid = cont['face_valid']
    foot_valid = cont['foot_valid']
    left_hand_valid = cont['lefthand_valid']
    right_hand_valid = cont['righthand_valid']

    body_kp=np.array(cont['keypoints']+cont['foot_kpts']).reshape(23,3)
    face_kp=np.array(cont['face_kpts']).reshape(68,3)
    left_hand_kp=np.array(cont['lefthand_kpts']).reshape(21,3)
    right_hand_kp=np.array(cont['righthand_kpts']).reshape(21,3)
    
    # body_conf = cont['score']
    # face_conf = cont['face_score']
    # foot_conf = cont['foot_score']
    # left_hand_conf = cont['lefthand_score']
    # right_hand_conf = cont['righthand_score']

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
    
    # transform coco kpt indexes to smplx format to ensure the data able to be used in crop img according to kpt (at least 6 kpts>threshold)
    output_keypoints2d = np.zeros([127 + 17, 3], dtype=np.float32)
    output_keypoints2d[smplx_target_idxs] = keypoints2d[coco_source_idxs]

    keypoints = output_keypoints2d[:, :-1]
    conf = output_keypoints2d[:, -1]
    # valid_keypoints = keypoints[conf > 0]
    # if len(valid_keypoints) < min_valid_keypoints:
    #     return False
    keypoints_first25=keypoints[:25]
    conf_first25=conf[:25]
    if len(keypoints_first25[conf_first25 > 0]) < min_valid_keypoints:
        return False
    else:
        # if ((not face_valid) and (not foot_valid) and (not left_hand_valid) and (not right_hand_valid)):
        #     return False 
        return True


divide='train'

dataset_pre='/data/panyuqing/expose_experts/add_data/coco2017/'+divide
img_dir=dataset_pre + '/images'
kpt_dir=dataset_pre + '/keypoints2d/'
new_kpt_dir=dataset_pre + '/keypoints2d_strict/'
os.makedirs(new_kpt_dir,exist_ok=True)
# annotations_path='/data/panyuqing/expose_experts/add_data/coco2017/coco_wholebody_train_v1.0.json'
# annotations=json.load(open(annotations_path,encoding='utf-8'))
# anno=annotations['annotations']
kpt_files=os.listdir(kpt_dir) 
kpt_paths=[osp.join(kpt_dir,f) for f in kpt_files]

data_path='/data/panyuqing/expose_experts/data/curated_fits'
add_coco2017_path=osp.join(data_path,'coco2017_train_comp_img_fns_del05.npy') 
#add_coco2017_path=osp.join(data_path,'coco2017_train_comp_img_fns.npy') 
raw_comp_fn_list=list(np.load(add_coco2017_path,allow_pickle=True))
print('raw train data len: ',len(raw_comp_fn_list))#46054

comp_fn_list=[]
# kpt_paths=kpt_paths[:2]
for kp in tqdm(kpt_paths):
    pic_name=kp.split('/')[-1].split('\\')[-1].split('.')[0]#[:12]#_keypoints
    pic_path=osp.join(img_dir,pic_name[:12]+'.jpg')
    # print(pic_name)
    # print(pic_path)
    cont=json.load(open(kp,encoding='utf-8'))
    if check_kpt_conf(cont):
        savekpt_path=new_kpt_dir+pic_name#+'_keypoints.json'
        with open(savekpt_path,'w') as f_obj:
            json.dump(cont,f_obj)
        if pic_path in raw_comp_fn_list:
            comp_fn_list.append(pic_path)
print('strict selection of body conf higher points > 6 data len: ',len(comp_fn_list))#46054
np.save(osp.join(data_path,'coco2017_train_comp_img_fns_del05_bodyconf6.npy') ,np.array(comp_fn_list))