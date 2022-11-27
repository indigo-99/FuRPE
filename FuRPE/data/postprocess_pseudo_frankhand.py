# -*- coding: utf-8 -*-
'''
Because the bad effect of the hand experts Minimal-Hand-torch and Blazehand, replace it with Frankmocap.
only change the hand poses in pseudo_gt parameter files, avoiding reproducing all labels.
read distiled information, change hand poses and regenerate vertices by the smplx model, then re-save results.
BEING USED FINALLY.
'''
import os.path as osp
import os

import torch
import numpy as np
import torch.nn.functional as F
from smplx import build_layer as build_body_model
from fvcore.common.config import CfgNode as CN
from loguru import logger
from tqdm import tqdm

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

    _C.body_model.j14_regressor_path = '/data1/panyuqing/expose_experts/data/SMPLX_to_J14.pkl'
    _C.body_model.mean_pose_path = '/data1/panyuqing/expose_experts/data/all_means.pkl'
    _C.body_model.shape_mean_path = '/data1/panyuqing/expose_experts/data/shape_mean.npy'
    _C.body_model.type = 'smplx'
    _C.body_model.model_folder = '/data1/panyuqing/expose_experts/data/models'
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
    model_path='/data1/panyuqing/expose_experts/data/models'
    model_type='smplx'
    body_model_cfg=_C.get('body_model', {})
    body_model = build_body_model(
        model_path,
        model_type=model_type,
        dtype=torch.float32,
        **body_model_cfg)

    data_path='/data1/panyuqing/expose_experts/data/curated_fits'
    save_3dparam_vertices_dir='/data1/panyuqing/expose_experts/data/params3d_v'
    #new_save_3dparam_vertices_dir='/data1/panyuqing/expose_experts/data/params3d_v_newmh'
    #os.makedirs(new_save_3dparam_vertices_dir, exist_ok=True)
    # os.makedirs(new_save_3dparam_vertices_dir+'/mpi', exist_ok=True)
    # os.makedirs(new_save_3dparam_vertices_dir+'/curated_fits', exist_ok=True)
    # os.makedirs(new_save_3dparam_vertices_dir+'/coco2017', exist_ok=True)
    # os.makedirs(new_save_3dparam_vertices_dir+'/3dpw_train', exist_ok=True)
    dset_list=['posetrack_train']#['posetrack_val']#['posetrack_test']#['CMU_all']#['ochuman_all'] #['posetrack_eft'] #['h36m_S1']#['coco2017', '3dpw_train', 'mpi_S2_Seq2', 'mpi_S3_Seq2', 'mpi_S4_Seq2', 'mpi_S5_Seq2', 'mpi_S6_Seq2', 'mpi_S7_Seq2', 'mpi_S8_Seq2','cf_mpi_s1']
    for dset in tqdm(dset_list):
        frank_path='/data/panyuqing/frankmocap/hand_label_feat_right/'+ dset+'_frankhand'
        comp_img_fns=[]
        #indices=list(np.load(osp.join(data_path,'indices_del12.npy'), allow_pickle=True))
        #indices=[]
        if dset=='cf_mpi_s1':
            #save_path='/data1/panyuqing/experts/res/'#132 cf, psyai cf+mpi s1
            all_image_name_path=osp.join(data_path,'comp_img_fns.npy')
        elif dset[:3]=='mpi':
            S_num=dset[4:]
            all_image_name_path=osp.join(data_path,dset+'_comp_img_fns.npy')
        elif dset=='coco2017':
            all_image_name_path=osp.join(data_path,dset+'_train_comp_img_fns.npy')
        elif dset=='3dpw_train':
            all_image_name_path=osp.join(data_path,dset+'_comp_img_fns.npy')
        else:
            res=dset.split('_')
            if len(res)==2:
                S_num=res[1]
                dset=res[0]
            all_image_name_path=osp.join(data_path,dset+'_'+S_num+'_comp_img_fns.npy')
        
        comp_img_fns=list(np.load(all_image_name_path, allow_pickle=True))

        # S_num_list=['S2_Seq2','S3_Seq2',]#'S4_Seq2','S5_Seq2','S6_Seq2','S7_Seq2','S8_Seq2',
        # for S_num in S_num_list:
        #     add_fn_path=osp.join(data_path,'mpi_'+S_num+'_comp_img_fns_del12.npy')
        #     if osp.exists(add_fn_path):
        #         raw_len=len(comp_img_fns)
        #         comp_img_fns+=list(np.load(add_fn_path, allow_pickle=True))
        #         indices+=[i for i in range(raw_len,len(comp_img_fns))]
        #logger.info('Finish loading img file names, num of img: {}',len(indices))

        left_num=0
        right_num=0
        #comp_img_fns=comp_img_fns[32420:]
        for cfn in tqdm(comp_img_fns):
            #cfn=comp_img_fns[idx]    
            img_name=cfn.split('/')[-1].replace('.jpg','')#.split('.')[0]
            #logger.info('cfn: {}',cfn)

            # left_hand_crop_path=osp.join(save_path,img_name,img_name+'_Left_hand_crop.jpg') 
            # right_hand_crop_path=osp.join(save_path,img_name,img_name+'_Right_hand_crop.jpg') 
            # body_crop_path=osp.join(save_path,img_name,img_name+'_body_crop.jpg')
            left_hand_frank_param=osp.join(frank_path,'mocap',img_name+'_Left_hand_crop_prediction_result.pkl') 
            right_hand_frank_param=osp.join(frank_path,'mocap',img_name+'_Right_hand_crop_prediction_result.pkl') 
            #logger.info('left_hand_frank_param: {}',left_hand_frank_param)

            left_hand_pose=None
            left_hand_pose_mat=None
            right_hand_pose=None
            right_hand_pose_mat=None

            if osp.exists(left_hand_frank_param):
                #logger.info('left exists!')
                cont=np.load(left_hand_frank_param,allow_pickle=True)
                if ('pred_output_list' in cont) and len(cont['pred_output_list'])>0:
                    hands=cont['pred_output_list'][0]
                    if ('left_hand' in hands) and len(hands['left_hand'])>0:
                        globe_left_hand_pose=hands['left_hand']['pred_hand_pose'].reshape(16,3)
                        left_hand_pose=globe_left_hand_pose[1:]
                        left_hand_feat=hands['left_hand']['save_feat']
                
            if osp.exists(right_hand_frank_param):
                #logger.info('right exists!')
                cont=np.load(right_hand_frank_param,allow_pickle=True)
                if ('pred_output_list' in cont) and len(cont['pred_output_list'])>0:
                    hands=cont['pred_output_list'][0]
                    if ('right_hand' in hands) and len(hands['right_hand'])>0:
                        globe_right_hand_pose=hands['right_hand']['pred_hand_pose'].reshape(16,3)
                        right_hand_pose=globe_right_hand_pose[1:]   
                        right_hand_feat=hands['right_hand']['save_feat']

            #if (left_hand_pose is not None) or (right_hand_pose is not None):
            if True: # generate vertices for each images of Human3.6m because no vertice generated in pseudo_gt.py
                if dset[:3]=='mpi':
                    save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'mpi',S_num,img_name + '.npy')
                    #new_save_3dparam_vertices_path=os.path.join(new_save_3dparam_vertices_dir, 'mpi',S_num,img_name + '.npy')
                elif dset=='cf_mpi_s1':
                    if 'video' in img_name:
                        S_num='S1'
                        save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'mpi',S_num, img_name + '.npy')
                    else:
                        save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, 'curated_fits', img_name + '.npy')
                else: #dset=='h36m' 'posetrack' 'ochuman' /curated_fittings
                    save_3dparam_vertices_path=os.path.join(save_3dparam_vertices_dir, dset,img_name + '.npy')
                    #new_save_3dparam_vertices_path=os.path.join(new_save_3dparam_vertices_dir, 'curated_fits',img_name + '.npy')
                    
                param3d_vertices_data=np.load(save_3dparam_vertices_path, allow_pickle=True).item()
                if 'vertices' in param3d_vertices_data:
                    if 'left_hand_pose' in param3d_vertices_data:
                        left_num+=1
                    if 'right_hand_pose' in param3d_vertices_data:
                        right_num+=1
                    continue #already finished this process, no need to duplicate
                new_param3d_vertices_data={}
                for k in param3d_vertices_data:
                    if torch.is_tensor(param3d_vertices_data[k]):
                        new_param3d_vertices_data[k]=param3d_vertices_data[k].to(torch.float32).detach().cpu().numpy()
                    elif type(param3d_vertices_data[k]) is np.ndarray:
                        new_param3d_vertices_data[k]=param3d_vertices_data[k].astype(np.float32)
                    else:
                        new_param3d_vertices_data[k]=param3d_vertices_data[k]
                #if len(param3d_vertices_data['global_orient'])==4:
                    #continue
                global_pose_mat=batch_rodrigues(torch.from_numpy(new_param3d_vertices_data['global_orient']).reshape(1, 3)).reshape(-1, 1, 3, 3)
                #else:
                    #global_pose_mat=batch_rodrigues(torch.from_numpy(param3d_vertices_data['global_orient']).reshape(1, 3)).reshape(-1, 1, 3, 3)
                body_pose_mat=batch_rodrigues(torch.from_numpy(new_param3d_vertices_data['body_pose'].reshape(21,3))).reshape(-1, 21, 3, 3)
                betas_for_vert_model=torch.from_numpy(new_param3d_vertices_data['betas']).reshape(1, -1)

                global_pose_mat=global_pose_mat.to(device)
                body_pose_mat=body_pose_mat.to(device)
                betas_for_vert_model=betas_for_vert_model.to(device)
                final_body_parameters = {
                    'global_orient': global_pose_mat, 
                    'body_pose': body_pose_mat,
                    #'betas': betas_for_vert_model,
                }
                
                if 'jaw_pose' in new_param3d_vertices_data:
                    expression_np=torch.from_numpy(new_param3d_vertices_data['expression']).reshape(1,-1)
                    expression_np=expression_np.to(device)
                    final_body_parameters['expression']= expression_np
                    
                    jaw_pose=torch.from_numpy(new_param3d_vertices_data['jaw_pose'])
                    jaw_pose_mat=batch_rodrigues(jaw_pose).reshape(-1, 1, 3, 3)
                    jaw_pose_mat=jaw_pose_mat.to(device)
                    final_body_parameters['jaw_pose']= jaw_pose_mat.to(torch.float32)

                if left_hand_pose is not None:
                    new_param3d_vertices_data['left_hand_pose']=left_hand_pose.astype(np.float32)
                    left_hand_pose_mat = batch_rodrigues(torch.from_numpy(left_hand_pose.reshape(15,3))).reshape(-1, 15, 3, 3)
                    left_hand_pose_mat=left_hand_pose_mat.to(device)
                    final_body_parameters['left_hand_pose']=left_hand_pose_mat.to(torch.float32)
                    left_num+=1
                    new_param3d_vertices_data['save_feature_left_hand']=left_hand_feat
                if right_hand_pose is not None:
                    new_param3d_vertices_data['right_hand_pose']=right_hand_pose.astype(np.float32)
                    right_hand_pose_mat = batch_rodrigues(torch.from_numpy(right_hand_pose.reshape(15,3))).reshape(-1, 15, 3, 3)
                    right_hand_pose_mat=right_hand_pose_mat.to(device)
                    final_body_parameters['right_hand_pose']=right_hand_pose_mat.to(torch.float32)
                    right_num+=1
                    new_param3d_vertices_data['save_feature_right_hand']=right_hand_feat

                final_body_model_output = body_model(
                    get_skin=True, return_shaped=True, **final_body_parameters)
                
                vertices=final_body_model_output['vertices'].detach().cpu().numpy()
                vertices=vertices.reshape((vertices.shape[1],vertices.shape[2]))
            
                new_param3d_vertices_data['vertices']=vertices

                keypoints3d=final_body_model_output['joints'].detach().cpu().numpy()
                keypoints3d=keypoints3d.reshape((keypoints3d.shape[1],keypoints3d.shape[2]))
            
                new_param3d_vertices_data['keypoints3d']=keypoints3d#144,3

                np.save(save_3dparam_vertices_path, new_param3d_vertices_data)
        logger.info('dset: {}',dset)
        logger.info('total left hand num: {}',left_num)
        logger.info('total right hand num: {}',right_num)