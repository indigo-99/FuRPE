# -*- coding: utf-8 -*-
import os.path as osp
import os
import time

import torch
import torch.utils.data as dutils
import numpy as np
import torch.nn.functional as F
from smplx import build_layer as build_body_model
from fvcore.common.config import CfgNode as CN
from loguru import logger
import cv2
from tqdm import tqdm

import sys
sys.path.append('/data/panyuqing/experts')
from blazehand.handJointCalRotate import handJointCalRotate_api
from deca_api_v2 import DecaApi
#from blazehand.blazehand_hand_coord import blazehand_hand_coord_transfer
sys.path.append('/data/panyuqing/expose_experts')

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat




if __name__ == '__main__':
    device = torch.device('cpu')
    # build smplify-x model to get pseudo gt vertices
    _C = CN()

    _C.body_model = CN()

    _C.body_model.j14_regressor_path = '/data/panyuqing/expose_experts/data/SMPLX_to_J14.pkl'
    _C.body_model.mean_pose_path = '/data/panyuqing/expose_experts/data/all_means.pkl'
    _C.body_model.shape_mean_path = '/data/panyuqing/expose_experts/data/shape_mean.npy'
    _C.body_model.type = 'smplx'
    _C.body_model.model_folder = '/data/panyuqing/expose_experts/data/models'
    _C.body_model.use_compressed = True
    _C.body_model.gender = 'neutral'
    _C.body_model.num_betas = 10
    _C.body_model.num_expression_coeffs = 10
    _C.body_model.use_feet_keypoints = True
    _C.body_model.use_face_keypoints = True
    _C.body_model.use_face_contour = True

    _C.body_model.global_orient = CN()
    # The configuration for the parameterization of the body pose
    _C.body_model.global_orient.param_type = 'cont_rot_repr'

    _C.body_model.body_pose = CN()
    # The configuration for the parameterization of the body pose
    _C.body_model.body_pose.param_type = 'cont_rot_repr'
    _C.body_model.body_pose.finetune = False

    _C.body_model.left_hand_pose = CN()
    # The configuration for the parameterization of the left hand pose
    _C.body_model.left_hand_pose.param_type = 'pca'
    _C.body_model.left_hand_pose.num_pca_comps = 12
    _C.body_model.left_hand_pose.flat_hand_mean = False
    # The type of prior on the left hand pose

    _C.body_model.right_hand_pose = CN()
    # The configuration for the parameterization of the left hand pose
    _C.body_model.right_hand_pose.param_type = 'pca'
    _C.body_model.right_hand_pose.num_pca_comps = 12
    _C.body_model.right_hand_pose.flat_hand_mean = False

    _C.body_model.jaw_pose = CN()
    _C.body_model.jaw_pose.param_type = 'cont_rot_repr'
    _C.body_model.jaw_pose.data_fn = 'clusters.pkl'
    model_path='/data/panyuqing/expose_experts/data/models'
    model_type='smplx'
    body_model_cfg=_C.get('body_model', {})
    body_model = build_body_model(
        model_path,
        model_type=model_type,
        dtype=torch.float32,
        **body_model_cfg)

    #device="cuda"
    deca_api=DecaApi(device)

    data_path='/data/panyuqing/expose_experts/data/curated_fits'
    save_3dparam_vertices_dir='/data/panyuqing/expose_experts/data/params3d_v'
    new_save_3dparam_vertices_dir='/data/panyuqing/expose_experts/data/params3d_v_newmh'
    os.makedirs(new_save_3dparam_vertices_dir, exist_ok=True)
    os.makedirs(new_save_3dparam_vertices_dir+'/mpi', exist_ok=True)
    os.makedirs(new_save_3dparam_vertices_dir+'/curated_fits', exist_ok=True)
    #save_path='/data/panyuqing/experts/add_data_crop_res'#132 mpi s1
    #save_path='/data/panyuqing/experts/res'#132 cf, psyai cf+mpi s1
    save_path='/data/panyuqing/experts/mpi'#psyai mpi s2/3/4/../8

    all_image_name_path=osp.join(data_path,'comp_img_fns.npy')
    #comp_img_fns=list(np.load(all_image_name_path, allow_pickle=True))
    comp_img_fns=[]
    #indices=list(np.load(osp.join(data_path,'indices_del12.npy'), allow_pickle=True))
    indices=[]

    #S_num_list=[]
    S_num_list=['S2_Seq2','S3_Seq2',]#'S4_Seq2','S5_Seq2','S6_Seq2','S7_Seq2','S8_Seq2',
    # add_path = '/data/panyuqing/expose_experts/add_data/mpi/'+S_num+'/images'
    for S_num in S_num_list:
        add_fn_path=osp.join(data_path,'mpi_'+S_num+'_comp_img_fns_del12.npy')
        if osp.exists(add_fn_path):
            raw_len=len(comp_img_fns)
            comp_img_fns+=list(np.load(add_fn_path, allow_pickle=True))
            indices+=[i for i in range(raw_len,len(comp_img_fns))]

    logger.info('Finish loading img file names, num of img: {}',len(indices))

    left_num=0
    right_num=0
    #indices=indices[:3]
    for idx in tqdm(indices):
        cfn=comp_img_fns[idx]    
        img_name=cfn.split('/')[-1].split('.')[0]
        #logger.info('cfn: {}',cfn)
        if 'Seq' in cfn:
            img_dir_tmp=osp.dirname(cfn)#add_data/mpi/S2/images or /data/panyuqing/expose_experts/add_data/mpi/S2_Seq2/images
            img_dir=img_dir_tmp.replace('images', '')#/xxx strip has problems! #add_data/mpi/S2_Seq2/
            S_num=img_dir.split('/')[-2]#S2_Seq2
            left_hand_crop_path=osp.join(save_path,S_num,img_name,img_name+'_Left_hand_crop.jpg') 
            right_hand_crop_path=osp.join(save_path,S_num,img_name,img_name+'_Right_hand_crop.jpg') 
            body_crop_path=osp.join(save_path,S_num,img_name,img_name+'_body_crop.jpg')
            #logger.info('left_hand_crop_path: {}',left_hand_crop_path)
            #logger.info('right_hand_crop_path: {}',right_hand_crop_path)
        else:
            left_hand_crop_path=osp.join(save_path,img_name,img_name+'_Left_hand_crop.jpg') 
            right_hand_crop_path=osp.join(save_path,img_name,img_name+'_Right_hand_crop.jpg') 
            body_crop_path=osp.join(save_path,img_name,img_name+'_body_crop.jpg')
        has_left=False
        has_right=False
        # if not osp.exists(body_crop_path):
        #     logger.info('no body crop img? {}',cfn)
        #     continue
        if osp.exists(left_hand_crop_path):
            frame = cv2.imread(left_hand_crop_path)
            #logger.info('frame shape: {}',frame.shape)
            left_axisangle = handJointCalRotate_api(frame)
            if left_axisangle is not None:
                has_left=True
            else:
                logger.info('mediapipe cannot detect hand again! {}',left_hand_crop_path)
        if osp.exists(right_hand_crop_path):
            frame = cv2.imread(right_hand_crop_path)
            frame = np.flip(frame, axis=1).copy()
            #xyz_blaze= blazehand_hand_coord_transfer(frame)
            right_axisangle = handJointCalRotate_api(frame)
            if right_axisangle is not None:
                has_right=True
                right_axisangle[:, [1, 2]] = right_axisangle[:, [1, 2]] * -1.0  # flip, x不变, yz相反
            else:
                logger.info('mediapipe cannot detect hand again! {}',right_hand_crop_path)
        
        if has_left or has_right:
            if 'Seq' in cfn:
                # img_dir_tmp=osp.dirname(img_fn)#add_data/mpi/S1/images
                # img_dir=img_dir_tmp.replace('images', '')#/xxx strip has problems! #add_data/mpi/S1
                # S_num=img_dir.split('/')[-2] #S1/S2_Seq1,  [-1] is empty
                #S_num='S1'
                save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'mpi',S_num,img_name + '.npy')
                new_save_3dparam_vertices_path=os.path.join(new_save_3dparam_vertices_dir, 'mpi',S_num,img_name + '.npy')
            else:
                save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'curated_fits',img_name + '.npy')
                new_save_3dparam_vertices_path=os.path.join(new_save_3dparam_vertices_dir, 'curated_fits',img_name + '.npy')
                
            param3d_vertices_data=np.load(save_3dparam_vertices_path, allow_pickle=True).item()
            new_param3d_vertices_data={}
            for k in param3d_vertices_data:
                if torch.is_tensor(param3d_vertices_data[k]):
                    new_param3d_vertices_data[k]=param3d_vertices_data[k].to(torch.float32)
                else:#numpy
                    new_param3d_vertices_data[k]=param3d_vertices_data[k].astype(np.float32)
            # new_param3d_vertices_data={
            #     'global_orient': param3d_vertices_data['global_orient'], 
            #     'body_pose': param3d_vertices_data['body_pose'],
            #     'betas': param3d_vertices_data['betas'],
            #     'save_feature_body':param3d_vertices_data['save_feature_body']}
            
            global_pose_mat=batch_rodrigues(torch.from_numpy(param3d_vertices_data['global_orient']).reshape(1, 3)).reshape(-1, 1, 3, 3)
            body_pose_mat=batch_rodrigues(torch.from_numpy(param3d_vertices_data['body_pose'].reshape(21,3))).reshape(-1, 21, 3, 3)
            betas_for_vert_model=torch.from_numpy(param3d_vertices_data['betas']).reshape(1, -1)

            global_pose_mat=global_pose_mat.to(device)
            body_pose_mat=body_pose_mat.to(device)
            betas_for_vert_model=betas_for_vert_model.to(device)
            final_body_parameters = {
                'global_orient': global_pose_mat, 
                'body_pose': body_pose_mat,
                'betas': betas_for_vert_model,
            }
            
            if 'jaw_pose' in param3d_vertices_data:
                if 'expression' not in param3d_vertices_data:
                    logger.info('has jaw but without expression: {}',cfn)
                    if 'Seq' in cfn:
                        face_img_path=osp.join(save_path,S_num,img_name,img_name+'_face_crop.jpg')
                    else:
                        face_img_path=osp.join(save_path,img_name,img_name+'_face_crop.jpg')
                    if not osp.exists(face_img_path):
                        logger.info('no face crop but has jaw?: {}',cfn)
                    crop_face=cv2.imread(face_img_path)
                    face_params=deca_api.deca_detect(crop_face)
                    expression=face_params['exp'].detach().cpu().numpy().astype(np.float32)
                    expression=expression[:,:10].copy()
                    new_param3d_vertices_data['expression']=expression.reshape((1, 10))
                # else:
                #     new_param3d_vertices_data['expression']= param3d_vertices_data['expression']
                expression_np=torch.from_numpy(new_param3d_vertices_data['expression']).reshape(1,-1)
                expression_np=expression_np.to(device)
                final_body_parameters['expression']= expression_np
                    
                
                #new_param3d_vertices_data['jaw_pose']=param3d_vertices_data['jaw_pose']
                jaw_pose=torch.from_numpy(new_param3d_vertices_data['jaw_pose'])
                jaw_pose_mat=batch_rodrigues(jaw_pose).reshape(-1, 1, 3, 3)
                jaw_pose_mat=jaw_pose_mat.to(device)
                final_body_parameters['jaw_pose']= jaw_pose_mat.to(torch.float32)

                #new_param3d_vertices_data['save_feature_face']=param3d_vertices_data['save_feature_face']

            if has_left:
                new_param3d_vertices_data['left_hand_pose']=left_axisangle.astype(np.float32)
                left_hand_pose_mat = batch_rodrigues(torch.from_numpy(left_axisangle.reshape(15,3))).reshape(-1, 15, 3, 3)
                left_hand_pose_mat=left_hand_pose_mat.to(device)
                final_body_parameters['left_hand_pose']=left_hand_pose_mat.to(torch.float32)
                left_num+=1
            if has_right:
                new_param3d_vertices_data['right_hand_pose']=right_axisangle.astype(np.float32)
                right_hand_pose_mat = batch_rodrigues(torch.from_numpy(right_axisangle.reshape(15,3))).reshape(-1, 15, 3, 3)
                right_hand_pose_mat=right_hand_pose_mat.to(device)
                final_body_parameters['right_hand_pose']=right_hand_pose_mat.to(torch.float32)
                right_num+=1

            final_body_model_output = body_model(
                get_skin=True, return_shaped=True, **final_body_parameters)
            
            vertices=final_body_model_output['vertices'].detach().cpu().numpy()
            vertices=vertices.reshape((vertices.shape[1],vertices.shape[2]))
        
            new_param3d_vertices_data['vertices']=vertices

            keypoints3d=final_body_model_output['joints'].detach().cpu().numpy()
            keypoints3d=keypoints3d.reshape((keypoints3d.shape[1],keypoints3d.shape[2]))
        
            new_param3d_vertices_data['keypoints3d']=keypoints3d#144,3

            np.save(save_3dparam_vertices_path, new_param3d_vertices_data)
    logger.info('total left hand num: {}',left_num)
    logger.info('total right hand num: {}',right_num)