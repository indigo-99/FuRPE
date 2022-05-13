# -*- coding: utf-8 -*-
'''
get distiled information from 3 part experts, save the generated pseudo_ground_truth parameters as training data's label.
'''
import os.path as osp
import os

import torch
import torch.utils.data as dutils
import numpy as np
import torch.nn.functional as F
from smplx import build_layer as build_body_model
from fvcore.common.config import CfgNode as CN
from loguru import logger
from tqdm import tqdm

import sys
sys.path.append('/data1/panyuqing/experts')
# seg_experts_batch.py is located in '/data1/panyuqing/experts'
# it's intended to input hand/body/face cropped images from detectors into 3 experts separatedly, 
# and get distiled information, i.e. pseudo parameters as training data's label.
from seg_experts_batch import distil_from_body_face_hands
sys.path.append('/data1/panyuqing/expose_experts')

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters:
            rot_vecs: torch.tensor Nx3, array of N axis-angle vectors
        Returns:
            R: torch.tensor Nx3x3, The rotation matrices for the given axis-angle parameters
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

def batch_rot2aa(Rs, epsilon=1e-7):
    ''' Calculates the axis-angle rotation vectors for a batch of rotation matrices
        Parameters:
            Rs: torch.tensor Nx3x3, array of rotation matrices
        Returns:
            torch.tensor Nx3, the transformed axis-angle parameters
    '''
    cos = 0.5 * (torch.einsum('bii->b', [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

if __name__ == '__main__':
    device = torch.device('cpu')
    
    # build smplx model to get pseudo gt vertices from pose parameters (refer to expose/models/attention/predictor.py)
    # configs
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
    _C.body_model.global_orient.param_type = 'cont_rot_repr'
    # The configuration for the parameterization of the body pose
    _C.body_model.body_pose = CN()
    _C.body_model.body_pose.param_type = 'cont_rot_repr'
    _C.body_model.body_pose.finetune = False
#    The configuration for the parameterization of the left hand pose
    _C.body_model.left_hand_pose = CN()
    _C.body_model.left_hand_pose.param_type = 'pca'
    _C.body_model.left_hand_pose.num_pca_comps = 12
    _C.body_model.left_hand_pose.flat_hand_mean = False
    # The configuration for the parameterization of the right hand pose
    _C.body_model.right_hand_pose = CN()
    _C.body_model.right_hand_pose.param_type = 'pca'
    _C.body_model.right_hand_pose.num_pca_comps = 12
    _C.body_model.right_hand_pose.flat_hand_mean = False
    # The configuration for the parameterization of the head
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
    
    # react to import sys (change to another path, so change back here) 
    os.chdir('/data1/panyuqing/expose_experts') 
    #************* dataset name, should be changed when dealing with differenct datasets
    dset= 'h36m' #'mpi' # 'coco2017' # '3dpw'
    #*************
    # the path to save cropped part images of body/hand/face
    save_crop_path = '/data1/panyuqing/experts/'#res
    # the path to save pseudo_gt label files generated by 3 experts
    save_param_path = '/data1/panyuqing/expose_experts/data/params3d_v'
    # the path to save raw training images
    data_path='/data1/panyuqing/expose_experts/data/curated_fits/'

    if dset=='coco2017':
        # sub dir name
        S_num='train'
        # the path for added data (coco2017 and 3dpw_train are not used by expose)
        add_path = '/data/panyuqing/expose_experts/add_data/coco2017/'+S_num+'/images'
        # the path for added data's 2d keypoints info (generated by Openpose)
        add_ktp_path = '/data/panyuqing/expose_experts/add_data/coco2017/'+S_num+'/keypoints2d'
        # the list of data's filenames
        save_comp_path=osp.join(data_path,'coco2017_'+S_num+'_comp_img_fns.npy')
    elif dset=='mpi':
        S_num='S3_Seq2'
        add_path = '/data/panyuqing/expose_experts/add_data/mpi/'+S_num+'/images'
        save_comp_path=osp.join(data_path,'mpi_'+S_num+'_comp_img_fns.npy')
        save_comp_path=osp.join(data_path,'mpi_'+S_num+'_comp_img_fns.npy')
    elif dset=='3dpw':
        S_num='train'
        add_path = '/data/panyuqing/expose_experts/add_data/3dpw/'+S_num+'/images'
        add_ktp_path = '/data/panyuqing/expose_experts/add_data/3dpw/'+S_num+'/keypoints2d'
        save_comp_path=osp.join(data_path,'3dpw_'+S_num+'_comp_img_fns.npy')
    elif dset=='h36m':
        S_num='S5' # ['S1','S5','S6','S7','S8','S9','S11']
        add_path = '/data1/panyuqing/expose_experts/add_data/h36m/images_keypoints2d_filter/'+S_num+'/images'
        add_ktp_path = '/data1/panyuqing/expose_experts/add_data/h36m/images_keypoints2d_filter/'+S_num+'/keypoints2d'
        save_comp_path=osp.join(data_path,'h36m_'+S_num+'_comp_img_fns.npy')
    
    # if the data's namelist hasn't be generated, it means that the distiling process hasn't been completed
    if not osp.exists(save_comp_path):
        img_fns_kpt=os.listdir(add_ktp_path)
        # img_fns=[ikpt.split('_')[0]+'.jpg' for ikpt in img_fns_kpt]
        #/data1/panyuqing/expose_experts/add_data/3dpw/train/keypoints2d/
        #courtyard_warmWelcome_00_image_00512_keypoints.json
        img_fns=[ikpt.replace('_keypoints.json','')+'.jpg' for ikpt in img_fns_kpt]
        comp_img_fns=[osp.join(add_path,fn) for fn in img_fns]
        del_indexes=distil_from_body_face_hands(comp_img_fns,save_crop_path,save_param_path)
        # deleted indexes are the indexes of data whose body crops donnot exist (invalid data)
        logger.info('Distil_from_body_face_hands finishes!')
        os.chdir('/data1/panyuqing/expose_experts')
        new_comp_img_fns=[comp_img_fns[i] for i in range(len(comp_img_fns)) if i not in del_indexes]
        logger.info('num of del_indexes: {}',len(del_indexes))
        logger.info('num of add training data: {}',len(new_comp_img_fns))
        # save the valid data's namelist 
        np.save(save_comp_path, np.array(new_comp_img_fns))
    else:
        new_comp_img_fns=np.load(save_comp_path,allow_pickle=True)
    
    #img_names=[new_comp_img_fns[i].split('/')[-1].split('\\')[-1].split('.')[0] for i in range(len(new_comp_img_fns))]
    img_names=[new_comp_img_fns[i].split('/')[-1].split('\\')[-1].replace('.jpg','') for i in range(len(new_comp_img_fns))] # if . in img names, cannot split by ., use replace instead
    logger.info('img_names head -n 3: {}',img_names[:3])
    
    # read pseudo_gt parameters generated by 3 experts and transform them to vertices by the smplx model
    logger.info('Processing param & get vertice, kpt3d starts!')
    for i in tqdm(range(len(img_names))):
        img_name=img_names[i]
        image_path=new_comp_img_fns[i]
        if dset=='coco2017':
            save_3dparam_vertices_path=os.path.join(save_param_path, 'coco2017',img_name + '.npy')
        elif dset=='mpi': 
            save_3dparam_vertices_path=os.path.join(save_param_path, 'mpi',S_num,img_name + '.npy')
        elif dset=='3dpw':
            save_3dparam_vertices_path=os.path.join(save_param_path, '3dpw_train',img_name + '.npy')
        elif dset=='h36m':
            save_3dparam_vertices_path=os.path.join(save_param_path, 'h36m',img_name + '.npy') #S_num already in img_name

        # read 3d params and save vertices
        if not osp.exists(save_3dparam_vertices_path):
            logger.info('no expert param saved? {}',image_path)
            continue
     
        params_data = np.load(save_3dparam_vertices_path, allow_pickle=True).item()
        betas_for_vert_model=params_data['betas'].reshape(1,-1)
        betas = betas_for_vert_model.detach().cpu().numpy().astype(np.float32)

        raw_body_pose_mat = params_data['body_pose'].detach().cpu().numpy().astype(np.float32) # 3*3 mat need to convert to 3
        body_pose_mat=raw_body_pose_mat[:,:21,:,:].copy()
        body_pose_mat=torch.from_numpy(body_pose_mat.reshape(21, 3, 3))
        body_pose=batch_rot2aa(body_pose_mat).reshape(-1).detach().cpu().numpy()
        body_pose_mat=body_pose_mat.reshape(-1, 21, 3, 3)
        
        global_pose_mat = params_data['global_orient'].reshape(-1, 3, 3) # 3*3 mat need to convert to 3
        global_pose=batch_rot2aa(global_pose_mat).reshape(-1).detach().cpu().numpy()
        global_pose_mat=global_pose_mat.reshape(-1, 1, 3, 3)
        
        # change key names, because the saved parameters name are not fully the same as the smplx model's requirements
        if 'pose' in params_data:
            expression = params_data['exp'].detach().cpu().numpy().astype(np.float32)
            expression_10d=expression[:,:10].copy()
            jaw_pose=params_data['pose'][:,3:]
            # the first 3 is neck_pose, the last 3 is jaw_pose
            jaw_pose_mat=batch_rodrigues(jaw_pose).reshape(-1, 1, 3, 3)#.detach().cpu().numpy()
            jaw_pose=jaw_pose.detach().cpu().numpy().astype(np.float32)
            expression_np=torch.from_numpy(expression_10d).reshape(1,-1)
            
        else:
            jaw_pose=None
            jaw_pose_mat=None
            expression_10d=None
            expression_np=None
        
        # if 'left_hand_45' in params_data:      
        #     left_hand_pose=params_data['left_hand_45'] #torch tensor
        #     if left_hand_pose is None:
        #         left_hand_pose_mat=None
        #     else:
        #         left_hand_pose=left_hand_pose.astype(np.float32).reshape(15,3)
        #         left_hand_pose_mat = batch_rodrigues(torch.from_numpy(left_hand_pose)).reshape(-1, 15, 3, 3)
        #         left_hand_pose=left_hand_pose.reshape(-1)          
        # else: 
        #     left_hand_pose=None
        #     left_hand_pose_mat=None
        
        # if 'right_hand_45' in params_data:      
        #     right_hand_pose=params_data['right_hand_45']
        #     if right_hand_pose is None:
        #         right_hand_pose_mat=None
        #     else:
        #         right_hand_pose=right_hand_pose.astype(np.float32).reshape(15,3)
        #         right_hand_pose_mat = batch_rodrigues(torch.from_numpy(right_hand_pose)).reshape(-1, 15, 3, 3)
        #         right_hand_pose=right_hand_pose.reshape(-1)       
        # else: 
        #     right_hand_pose=None
        #     right_hand_pose_mat=None

      
        global_pose_mat=global_pose_mat.to(device)
        body_pose_mat=body_pose_mat.to(device)
        betas_for_vert_model=betas_for_vert_model.to(device)
        
        # the parameters feeded into the smplx model need to be rotation matrixes, not axis-angle vectors, so transform them
        final_body_parameters = {
            'global_orient': global_pose_mat, 
            'body_pose': body_pose_mat,
            #'betas': betas_for_vert_model, # not use betas generated by SPIN because it's for SMPL, not complies with SMPLX
        }
        save3dparam={
            'global_orient': global_pose, 
            'body_pose': body_pose,
            'betas': betas,
        }
        # if left_hand_pose_mat is not None:
        #     left_hand_pose_mat=left_hand_pose_mat.to(device)
        #     final_body_parameters['left_hand_pose']=left_hand_pose_mat

        #     save3dparam['left_hand_pose']=left_hand_pose
        # if right_hand_pose_mat is not None:
        #     right_hand_pose_mat=right_hand_pose_mat.to(device)
        #     final_body_parameters['right_hand_pose']=right_hand_pose_mat

        #     save3dparam['right_hand_pose']=right_hand_pose
        if jaw_pose_mat is not None:
            jaw_pose_mat=jaw_pose_mat.to(device)
            final_body_parameters['jaw_pose']= jaw_pose_mat

            save3dparam['jaw_pose']=jaw_pose
        if expression_np is not None:
            expression_np=expression_np.to(device)
            final_body_parameters['expression']= expression_np
            
            save3dparam['expression']=expression_10d
        
        # # get vertices and 3d joints by the smplx model
        # final_body_model_output = body_model(
        #     get_skin=True, return_shaped=True, **final_body_parameters)
        # vertices=final_body_model_output['vertices'].detach().cpu().numpy()
        # vertices=vertices.reshape((vertices.shape[1],vertices.shape[2]))
        # save3dparam['vertices']=vertices
        
        # # joints 3d
        # keypoints3d=final_body_model_output['joints'].detach().cpu().numpy()
        # keypoints3d=keypoints3d.reshape((keypoints3d.shape[1],keypoints3d.shape[2]))
        # save3dparam['keypoints3d']=keypoints3d

        # the feature outputted by the 3 experts, used in feature distiling, should be saved as well
        save3dparam['save_feature_body']=params_data['save_feature_body']
        if jaw_pose_mat is not None:
            save3dparam['save_feature_face']=params_data['save_feature_face']
        # camera SRT info
        save3dparam['scale']=params_data['scale']
        save3dparam['translation']=params_data['translation']
        
        # crop_body_bbox, only needed for the COCO2017 dataset for cropping body images
        save3dparam['crop_body_bbox']=params_data['crop_body_bbox']

        np.save(save_3dparam_vertices_path, save3dparam)

    #logger.info('Crop & save experts and vertices information: {}', save_3dparam_vertices_path)
