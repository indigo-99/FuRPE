# -*- coding: utf-8 -*-
import os.path as osp
import os
import time

import torch
import torch.utils.data as dutils
import numpy as np
import torch.nn.functional as F
#from smplx import build_layer as build_body_model
#from fvcore.common.config import CfgNode as CN

from loguru import logger
from tqdm import tqdm

import sys
#sys.path.append('/data/panyuqing/experts')
#from deca_api_v2 import DecaApi
#from minimal_hand_pytorch_api import MinimalHandAPI
#from spin_api import SpinApi
#sys.path.append('/data/panyuqing/expose_experts')



data_path='/data/panyuqing/expose_experts/data/curated_fits'
save_3dparam_vertices_dir='/data/panyuqing/expose_experts/data/params3d_v'
save_path='/data/panyuqing/experts/res'
#*********************************************
#dset='cf_mpi_s1'
#dset='mpi_S2_Seq2'
#dset='coco2017'
#dset='3dpw_train'
#*********************************************
dset_list=['cf_mpi_s1', 'coco2017', 'mpi_S2_Seq2', 'mpi_S3_Seq2', 'mpi_S4_Seq2', 'mpi_S5_Seq2', 'mpi_S6_Seq2', 'mpi_S7_Seq2', 'mpi_S8_Seq2',]
for dset in tqdm(dset_list):
    frank_path='/data/panyuqing/frankmocap/hand_label_feat_right/'+ dset+'_frankhand'
    comp_img_fns=[]
    if dset=='cf_mpi_s1':
        #save_path='/data/panyuqing/experts/res/'#132 cf, psyai cf+mpi s1
        all_image_name_path=osp.join(data_path,'comp_img_fns.npy')
        indices=list(np.load(osp.join(data_path,'indices_del12.npy'), allow_pickle=True))
        
    elif dset[:3]=='mpi':
        S_num=dset[4:]
        all_image_name_path=osp.join(data_path,dset+'_comp_img_fns.npy')
    if dset=='coco2017':
        all_image_name_path=osp.join(data_path,dset+'_train_comp_img_fns.npy')
    if dset=='3dpw_train':
        all_image_name_path=osp.join(data_path,dset+'_comp_img_fns.npy')
    raw_comp_img_fns=list(np.load(all_image_name_path, allow_pickle=True))
    if dset=='cf_mpi_s1':
        comp_img_fns=[raw_comp_img_fns[ind] for ind in indices]
    else:
        comp_img_fns=raw_comp_img_fns
    logger.info('Finish loading img file names and deleting del_indexes.')
    '''
    del_index_path=osp.join(data_path,'del_indexes_12.npy')
    del_indexes=np.load(del_index_path, allow_pickle=True)
    del_indexes=list(del_indexes)
    indices = [i for i in range(len(comp_img_fns)) if (i not in del_indexes)]

    #np.save(osp.join(data_path,'indices_del12.npy'),np.array(indices))
    indices=list(np.load(osp.join(data_path,'indices_del12.npy'), allow_pickle=True))

    spin_api=SpinApi()
    device="cuda"
    deca_api=DecaApi(device)
    os.chdir('/data/panyuqing/expose_experts')

    #indices=indices[32444:]
    # for img_index in tqdm(indices):

    for i in tqdm(range(len(indices))):
        img_index=indices[i]
        img_fn = comp_img_fns[img_index]
        img_name=img_fn.split('/')[-1].split('\\')[-1].split('.')[0] #im03019
        if 'add_data' in img_fn:
            save_3dparam_vertices_path=osp.join(save_3dparam_vertices_dir,'mpi','S1', img_name + '.npy')
        else:
            save_3dparam_vertices_path=osp.join(save_3dparam_vertices_dir, 'curated_fits',img_name + '.npy')
        param3d_vertices_data=np.load(save_3dparam_vertices_path, allow_pickle=True).item()
        if 'save_feature_body' in param3d_vertices_data:
            continue #already has feature
            
        #logger.info('Start getting spin feature.')
        #if i<32444:
        #body_img_path=osp.join(save_path,img_name,img_name+'_body_crop.jpg')#for curated_fits train data, body crop image both in 132 and pysai server
        #else:
        #save_add_path='/data/panyuqing/experts/add_data_crop_res'
        save_add_path='/data/panyuqing/experts/res'
        body_img_path=osp.join(save_add_path,img_name,img_name+'_body_crop.jpg')# for added mpi_s2 data, crop body image not in 132 server
        save_feature_body=spin_api.return_feature(body_img_path)

        param3d_vertices_data['save_feature_body']=save_feature_body
        #logger.info('Finish getting spin feature.')
        #logger.info('spin feature shape: {}',save_feature_body.shape)

        if 'jaw_pose' in param3d_vertices_data:
            #logger.info('Start getting deca feature.')     
            #if i<32444:
            #face_img_path=osp.join(save_path,img_name,img_name+'_face_crop.jpg')
            #else:
            face_img_path=osp.join(save_add_path,img_name,img_name+'_face_crop.jpg')
            save_feature_face=deca_api.return_feature(face_img_path)
            param3d_vertices_data['save_feature_face']=save_feature_face
            logger.info('Change jaw pose in param3d_v npys, img_name: {}', img_fn)
            #logger.info('Finish getting deca feature.')  
            
        
        if 'left_hand_pose' in param3d_vertices_data:
            if i<32444:
                left_hand_img_path=osp.join(save_path,img_name,img_name+'_Left_hand_crop.jpg')
            else:
                left_hand_img_path=osp.join(save_add_path,img_name,img_name+'_Left_hand_crop.jpg')
            save_feature_lh=_api.return_feature(left_hand_img_path)
            param3d_vertices_data['save_feature_face_left_hand']=
            logger.info('Change left hand pose in param3d_v npys: {}', img_fn)
            
        if 'right_hand_pose' in param3d_vertices_data:
            if i<32444:
                right_hand_img_path=osp.join(save_path,img_name,img_name+'_Right_hand_crop.jpg')
            else:
                left_hand_img_path=osp.join(save_add_path,img_name,img_name+'_Left_hand_crop.jpg')
            
            save_feature_rh=_api.return_feature(right_hand_img_path)
            param3d_vertices_data['save_feature_face_right_hand']=
            logger.info('Change right hand pose in param3d_v npys: {}',img_fn)
        
        #np.save('/data/panyuqing/expose_experts/newparam3d_'+img_name+'_.npy', param3d_vertices_data)
        np.save(save_3dparam_vertices_path, param3d_vertices_data)
        '''
    left_num=0
    right_num=0
    #comp_img_fns=comp_img_fns[:1000]
    for cfn in tqdm(comp_img_fns):  
        img_name=cfn.split('/')[-1].split('.')[0]

        if dset[:3]=='mpi':
            save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'mpi',S_num,img_name + '.npy')
        elif dset=='cf_mpi_s1':
            if 'video' in img_name:
                S_num='S1'
                save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'mpi',S_num, img_name + '.npy')
            else:
                save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'curated_fits', img_name + '.npy')
            
        else:
            save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, dset,img_name + '.npy')
        param3d_vertices_data=np.load(save_3dparam_vertices_path, allow_pickle=True).item()

        if 'right_hand_pose' in param3d_vertices_data:
            right_hand_frank_param=osp.join(frank_path,'mocap',img_name+'_Right_hand_crop_prediction_result.pkl') 
            if osp.exists(right_hand_frank_param):
                #logger.info('right exists! {}',right_hand_frank_param)
                cont=np.load(right_hand_frank_param,allow_pickle=True)
                if ('pred_output_list' in cont) and len(cont['pred_output_list'])>0:
                    hands=cont['pred_output_list'][0]
                    if ('right_hand' in hands) and len(hands['right_hand'])>0:
                        hand_feat=hands['right_hand']['save_feat']#.cpu().numpy.reshape(16,3)
                        param3d_vertices_data['save_feature_right_hand']=hand_feat
                        np.save(save_3dparam_vertices_path, param3d_vertices_data)
                        right_num+=1

        
        if 'left_hand_pose' in param3d_vertices_data:
            left_hand_frank_param=osp.join(frank_path,'mocap',img_name+'_Left_hand_crop_prediction_result.pkl') 
            if osp.exists(left_hand_frank_param):
                #logger.info('left exists!')
                cont=np.load(left_hand_frank_param,allow_pickle=True)
                if ('pred_output_list' in cont) and len(cont['pred_output_list'])>0:
                    hands=cont['pred_output_list'][0]
                    if ('left_hand' in hands) and len(hands['left_hand'])>0:
                        hand_feat=hands['left_hand']['save_feat']#.cpu().numpy.reshape(16,3)
                        param3d_vertices_data['save_feature_left_hand']=hand_feat
                        np.save(save_3dparam_vertices_path, param3d_vertices_data)
                        left_num+=1
            
    print('dset: ',dset)
    print('left_num: ',left_num)
    print('right_num: ',right_num)

    # cont=np.load('/data/panyuqing/expose_experts/newparam3d_'+img_name+'.npy', allow_pickle=True).item()
    # for k in cont:
    #     logger.info('{} shape : {}',k,cont[k].shape)
    '''
    global_orient shape : (3,)
    body_pose shape : (63,)
    betas shape : (1, 10)
    expression shape : (1, 10)
    jaw_pose shape : (1, 3)
    left_hand_pose shape : (15,3)
    right_hand_pose shape : (15,3)
    vertices (10475, 3)
    keypoints3d (144, 3)
    save_feature_body shape : torch.Size([1, 1024])
    save_feature_face shape : torch.Size([1, 1024])
    save_feature_left_hand torch.Size([1, 1024])
    save_feature_right hand torch.Size([1, 1024])
    scale torch.Size([1])
    translation torch.Size([1, 2])
    crop_body_bbox len:4
    '''