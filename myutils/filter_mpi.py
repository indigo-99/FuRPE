import json
import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm
from loguru import logger
'''
Filter out the images with few credible keypoints, and copy the valie images' keypoints of MPI/3DPW datasets to the target_dir.
The filtering threshold is defined in EXPOSE code to crop bbox according to keypoints. Fewer than the threshold will lead to cropping error.
'''

body_thresh=0.1
hand_thresh=0.2
face_thresh=0.4
min_valid_keypoints=12

def check_kpt_conf(jname):
    try:
        cont=json.load(open(jname,encoding='utf-8'))   
    except Exception:
        #traceback.print_exc()
        return False
    if (len(cont['people'])>0) and (len(cont['people'][0]['pose_keypoints_2d'])==75):
        body_kp=np.array(cont['people'][0]['pose_keypoints_2d'])
        if 0 in body_kp.shape:
            return False
        body_kp=body_kp.reshape(25,3)
        face_kp=np.array(cont['people'][0]['face_keypoints_2d'])
        if 0 in face_kp.shape:
            face_kp=np.zeros(shape=(70,3))
        else:
            face_kp=face_kp.reshape(70,3)
        left_hand_kp=np.array(cont['people'][0]['hand_left_keypoints_2d'])
        if 0 in left_hand_kp.shape:
            left_hand_kp=np.zeros(shape=(21,3))
        else:
            left_hand_kp=left_hand_kp.reshape(21,3)
        right_hand_kp=np.array(cont['people'][0]['hand_right_keypoints_2d'])
        if 0 in right_hand_kp.shape:
            right_hand_kp=np.zeros(shape=(21,3))
        else:
            right_hand_kp=right_hand_kp.reshape(21,3)
        
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
            
        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]
        valid_keypoints = keypoints[conf > 0]
        if len(valid_keypoints) < min_valid_keypoints:
            return False
        else:
            return True
    else:
        return False

dset='posetrack'#'ochuman'#'posetrack'#'h36m'#'3dpw'#'mpi'#CMU
if dset=='mpi':
    indexes=[0,1,2,4,5,6,7,8]#S1:[1,2,4,5,6,7,8]
    S_num='S6'#['S2','S3','S4','S5','S6','S7','S8']
    Seq_num='Seq2'
    dataset_pre='/data/common/mpi_inclu_2d_keypoint/'+S_num+'/'+Seq_num+'/imageFrames/video_'
    #'/data/common/mpi_inclu_2d_keypoint/S2/Seq1/imageFrames/video_'
    target_dir='/data/panyuqing/expose_experts/add_data/mpi/'+S_num+'_'+Seq_num 
elif dset=='3dpw':
    dataset_pre='/data/common/3DPW_inclu_2dkp/imageFiles/'
    #'/data/common/3DPW_inclu_2dkp/imageFiles/outdoors_slalom_01/image_00370_keypoints.json'
    target_dir='/data/panyuqing/expose_experts/add_data/3dpw/train/'
    train_dirs=[
        'courtyard_arguing_00', 'courtyard_laceShoe_00', 'courtyard_backpack_00',
        'courtyard_rangeOfMotions_00','courtyard_basketball_00', 'courtyard_relaxOnBench_00',
        'courtyard_bodyScannerMotions_00', 'courtyard_relaxOnBench_01', 'courtyard_box_00', 
        'courtyard_shakeHands_00','courtyard_capoeira_00', 'courtyard_warmWelcome_00',
        'courtyard_captureSelfies_00', 'outdoors_climbing_00', 'courtyard_dancing_01', 
        'outdoors_climbing_01', 'courtyard_giveDirections_00', 'outdoors_climbing_02',
        'courtyard_golf_00', 'outdoors_freestyle_00', 'courtyard_goodNews_00',
        'outdoors_slalom_00', 'courtyard_jacket_00', 'outdoors_slalom_01',
    ]
else: #'ochuman' 'posetrack' 'h36m' 'CMU'
    S_num='train'#'all' #['eft','train','val','test'] #['S1','S5','S6','S7','S8','S9','S11']
    dataset_pre='/data/panyuqing/expose_experts/add_data/'+dset+'/'
    raw_images_dir=dataset_pre+'images/'+S_num
    raw_keypoints_dir=dataset_pre+'keypoints2d_openpose/'+S_num+'_kpt'
    target_dir='/data/panyuqing/expose_experts/add_data/'+dset+'/images_keypoints2d_filter/'+S_num
    #logger.info('raw_images_dir: {}',raw_images_dir)
    #logger.info('raw_keypoints_dir: {}',raw_keypoints_dir)


os.makedirs(target_dir,exist_ok=True)
kpt_dir=target_dir + '/keypoints2d/'
img_dir=target_dir + '/images/'
os.makedirs(kpt_dir,exist_ok=True)
os.makedirs(img_dir,exist_ok=True)

if dset=='mpi':
    for i in indexes:
        files=os.listdir(dataset_pre+str(i))
        pic_files=[osp.join(dataset_pre+str(i),f) for f in files if 'jpg' in f]
        for pf in tqdm(pic_files):
            pic_name=pf.split('/')[-1].split('\\')[-1].split('.')[0]
            jname=osp.join(dataset_pre+str(i),pic_name+'_keypoints.json')
            if (osp.exists(jname) and check_kpt_conf(jname)):
                shutil.copyfile(jname, kpt_dir+'video_'+str(i)+'_'+pic_name+'_keypoints.json')
                shutil.copyfile(pf , img_dir+'video_'+str(i)+'_'+pic_name+'.jpg' )
elif dset=='3dpw':
    train_dirs=train_dirs[18:]
    for t_d in tqdm(train_dirs):
        files=os.listdir(dataset_pre+t_d)
        pic_files=[osp.join(dataset_pre+t_d,f) for f in files if 'jpg' in f]
        for pf in tqdm(pic_files):
            pic_name=pf.split('/')[-1].split('\\')[-1].split('.')[0]
            jname=osp.join(dataset_pre+t_d,pic_name+'_keypoints.json')
            if (osp.exists(jname) and check_kpt_conf(jname)):
                shutil.copyfile(jname, kpt_dir+t_d+'_'+pic_name+'_keypoints.json')
                shutil.copyfile(pf , img_dir+t_d+'_'+pic_name+'.jpg' )
else: # dset=='h36m' or dset=='posetrack'
    pic_files=os.listdir(raw_images_dir)
    for pf in tqdm(pic_files):
        pf_path=osp.join(raw_images_dir,pf)
        pic_name=pf.split('/')[-1].strip('.jpg') # S1_Directions_1.54138969_000001.jpg
        jname=osp.join(raw_keypoints_dir,pic_name+'_keypoints.json') # S1_Directions_1.54138969_000001_keypoints.json
        #logger.info('pic_name: {}',pic_name)
        #logger.info('jname: {}',jname)
        if (osp.exists(jname) and check_kpt_conf(jname)):
            shutil.copyfile(jname, kpt_dir+pic_name+'_keypoints.json') 
            shutil.copyfile(pf_path , img_dir+pic_name+'.jpg' )


