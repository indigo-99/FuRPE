# -*- coding: utf-8 -*-
'''
When realize the jaw pose is not the 6 dimension vectors outputted by the face expert DECA, but the last 3 dims instead,
only change the jaw poses in pseudo_gt parameter files, avoiding reproducing all labels.
read distiled information, change jaw poses and regenerate vertices by the smplx model, then re-save results.
'''
import os.path as osp
import os

import torch
import numpy as np
import torch.nn.functional as F
from smplx import build_layer as build_body_model
from fvcore.common.config import CfgNode as CN
from loguru import logger


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

    data_path='/data/panyuqing/expose_experts/data/curated_fits'
    save_3dparam_vertices_dir=osp.join(data_path, 'params3d_v')   
    filenames=os.listdir(save_3dparam_vertices_dir)
    comp_filenames=[osp.join(save_3dparam_vertices_dir,fn) for fn in filenames]
    save_path='/data/panyuqing/experts/res'

    for i in range(len(comp_filenames)):
        cfn=comp_filenames[i]
        img_n=cfn.split('/')[-1].split('.')[0]
        param3d_vertices_data=np.load(cfn, allow_pickle=True).item()
        new_param3d_vertices_data={
            'global_orient': param3d_vertices_data['global_orient'], 
            'body_pose': param3d_vertices_data['body_pose'],
            'betas': param3d_vertices_data['betas']}

        if 'jaw_pose' in param3d_vertices_data:     
            face_data_path=osp.join(save_path,img_n,'distill',img_n+'_deca_parameter.npy')
            #if osp.exists(face_data_path):      
            face_data = np.load(face_data_path, allow_pickle=True).item()
            pose_6d=face_data['pose'].detach().cpu().numpy().astype(np.float32).reshape(1,6)
            new_jaw_pose=pose_6d[:,3:]
            new_param3d_vertices_data['jaw_pose']=new_jaw_pose
            new_param3d_vertices_data['expression']=param3d_vertices_data['expression']
            logger.info('Change number {} jaw pose in param3d_v npys: {}',i, cfn)
            #check if betas has error:
            #body_data_path=osp.join(save_path,img_n,'distill',img_n+'_body_crop_spin_parameter.npy')
            #body_data=np.load(body_data_path, allow_pickle=True).item()
            #betasinbody=body_data['betas'].detach().cpu().numpy().astype(np.float32)
            #logger.info('shape betas in body_data: {}',betasinbody)
            #logger.info('shape betas in old param3d_vertices_data: {}',param3d_vertices_data['betas'])
            #the above two is the same, no error in shape, but gt vertices are all fat
  
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
            expression_np=torch.from_numpy(param3d_vertices_data['expression']).reshape(1,-1)
            expression_np=expression_np.to(device)
            final_body_parameters['expression']= expression_np

            jaw_pose_mat=batch_rodrigues(torch.from_numpy(new_jaw_pose)).reshape(-1, 1, 3, 3)
            jaw_pose_mat=jaw_pose_mat.to(device)
            final_body_parameters['jaw_pose']= jaw_pose_mat

        if 'left_hand_pose' in param3d_vertices_data:
            new_param3d_vertices_data['left_hand_pose']=param3d_vertices_data['left_hand_pose']
            left_hand_pose_mat = batch_rodrigues(torch.from_numpy(param3d_vertices_data['left_hand_pose'].reshape(15,3))).reshape(-1, 15, 3, 3)
            left_hand_pose_mat=left_hand_pose_mat.to(device)
            final_body_parameters['left_hand_pose']=left_hand_pose_mat
        if 'right_hand_pose' in param3d_vertices_data:
            new_param3d_vertices_data['right_hand_pose']=param3d_vertices_data['right_hand_pose']
            right_hand_pose_mat = batch_rodrigues(torch.from_numpy(param3d_vertices_data['right_hand_pose'].reshape(15,3))).reshape(-1, 15, 3, 3)
            right_hand_pose_mat=right_hand_pose_mat.to(device)
            final_body_parameters['right_hand_pose']=right_hand_pose_mat

        final_body_model_output = body_model(
            get_skin=True, return_shaped=True, **final_body_parameters)
        
        vertices=final_body_model_output['vertices'].detach().cpu().numpy()
        vertices=vertices.reshape((vertices.shape[1],vertices.shape[2]))
    
        new_param3d_vertices_data['vertices']=vertices

        keypoints3d=final_body_model_output['joints'].detach().cpu().numpy()
        keypoints3d=keypoints3d.reshape((keypoints3d.shape[1],keypoints3d.shape[2]))
    
        new_param3d_vertices_data['keypoints3d']=keypoints3d
        #logger.info('keypoints3d shape: {}',keypoints3d.shape)#144,3

        np.save(cfn, new_param3d_vertices_data)
