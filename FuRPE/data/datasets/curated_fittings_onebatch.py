# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import os
import os.path as osp
import pickle
import time

import torch
import torch.utils.data as dutils
import numpy as np
import json

from loguru import logger

from ..targets import (Keypoints2D,Keypoints3D,
                       Betas, Expression, GlobalPose, BodyPose,
                       HandPose, JawPose, Vertices, Joints, BoundingBox)
from ..targets.keypoints import dset_to_body_model, get_part_idxs
from ..utils.bbox import keyps_to_bbox, bbox_to_center_scale, bbox_scale

from ...utils.img_utils import read_img
from ...utils import nand
from ...utils.rotation_utils import batch_rodrigues

from smplx import build_layer as build_body_model
from fvcore.common.config import CfgNode as CN


class CuratedFittings(dutils.Dataset):
    '''
    The class of the dataloader designed for the Curated_Fittings training dataset.
    Add MPI, 3DPW train, COCO 2017 datasets (only the first one contributes to a better model effect)
    '''
    def __init__(self, data_path='data/curated_fits',
                 split='train',
                 img_folder='',
                 use_face=True, use_hands=True, use_face_contour=False,
                 head_only=False,
                 hand_only=False,
                 model_type='smplx',
                 keyp_format='coco25',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 num_betas=10,
                 num_expression_coeffs=10,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 min_hand_keypoints=8,
                 min_head_keypoints=8,
                 binarization=True,
                 return_params=True,
                 vertex_folder='vertices',
                 vertex_flip_correspondences='',
                 **kwargs):
        super(CuratedFittings, self).__init__()
        assert nand(head_only, hand_only), (
            'Hand only and head only can\'t be True at the same time')
        
        # whether or not to set low/high conf keypoints' value to be 0/1
        self.binarization = binarization
        
        # NOT BEING USED
        if metrics is None:
            metrics = []
        self.metrics = metrics 
        # the minimal number of hand/head keypoints to be detected in a image for cropping part images and feeding into sub-networks
        self.min_hand_keypoints = min_hand_keypoints # 8
        self.min_head_keypoints = min_head_keypoints

        if 'test' in split:
            split = 'val'
        self.split = split
        self.is_train = 'train' in split
        self.num_betas = num_betas
        self.return_params = return_params # whether or not to return ground truth parameters of the datasets, set True for training

        self.head_only = head_only # False
        self.hand_only = hand_only # False

        data_path='data/curated_fits' # the path of datasets
        data_path = osp.expandvars(osp.expanduser(data_path))
        self.data_path = osp.join(data_path, f'{split}.npz')

        self.transforms = transforms # transforms of datasets set in config.yaml
        self.dtype = dtype
        
        # read the flipping correspondences of vertices, download from expose's website
        vertex_flip_correspondences='data/smplx_flip_correspondences.npz'
        err_msg = (
            'Vertex flip correspondences path does not exist:' +
            f' {vertex_flip_correspondences}'
        )
        assert osp.exists(vertex_flip_correspondences), err_msg
        flip_data = np.load(vertex_flip_correspondences)
        self.bc = flip_data['bc']
        self.closest_faces = flip_data['closest_faces']

        self.img_folder = osp.expandvars(osp.join(img_folder, split))# NOT BEING USED
        
        # mean hand pose, add to the ground truth target hand pose from frankmocap pseudo GT (which should add up the mean hand pose to get final pose results)
        self.target_left_hand_mean=np.array([ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
                            -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
                            0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
                            -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
                            -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
                            -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
                            -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
                            -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
                            0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803]).astype(np.float32).reshape(15,3)
        self.target_right_hand_mean=np.array([ 0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
                            0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
                            -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
                            -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
                            0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
                            0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
                            -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
                            0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
                            -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]).astype(np.float32).reshape(15,3)

        self.use_face = use_face #True
        self.use_hands = use_hands #True
        self.use_face_contour = use_face_contour #True
        self.model_type = model_type # smplx
        self.keyp_format = keyp_format #'coco25'
        self.num_expression_coeffs = num_expression_coeffs #10
        self.body_thresh = body_thresh #0.1, the threshold of the confidence of keypoints to be trusted
        self.hand_thresh = hand_thresh #0.2
        self.face_thresh = face_thresh #0.4

        data = np.load(self.data_path, allow_pickle=True) # load the raw ground truth data in .npz file
        data = {key: data[key] for key in data.keys()}
        self.keypoints2D = data['keypoints2D'].astype(np.float32)#(n,137,3) 
        # all img files' name list
        img_fns = np.asarray(data['img_fns'], dtype=np.string_)
        img_fns_d=[img_fns[i].decode('utf-8') for i in range(len(img_fns))]#range(3000)]
        self.comp_img_fns=[osp.join('data',ifn) for ifn in img_fns_d]
        # total number of data
        self.num_items = len(img_fns_d)
        self.indices = None
        del img_fns
        del img_fns_d
        
        # add the pseudo ground truth data generated by 3 experts and integrated them with final vertices, feature, etc. into this dir 
        self.save_3dparam_vertices=os.path.join('data', 'params3d_v')
        
        del_indexes=[] # the indexes of data to be deleted because they cannot be labeled by experts
        if self.split == 'train':
            # get the paths of the Curated_Fittings training data
            all_image_name_path=osp.join(data_path,'comp_img_fns.npy')
            
            # add MPI training data 
            S_num_list=['S2_Seq2','S3_Seq2','S4_Seq2','S5_Seq2','S6_Seq2','S7_Seq2','S8_Seq2']
            
            # get the paths of added data
            add_image_name_path=[osp.join(data_path,'mpi_'+S_num+'_comp_img_fns_del12.npy') for S_num in S_num_list]
            # try to add MPI data whose hands can be detected for training the model, with data which all contains hands 
            #add_image_name_path=[osp.join(data_path,'mpi_'+S_num+'_comp_img_fns_withhand.npy') for S_num in S_num_list]   
            if osp.exists(all_image_name_path):
                self.comp_img_fns=list(np.load(all_image_name_path, allow_pickle=True))
                raw_len=len(self.comp_img_fns)
                # for p in add_image_name_path:
                #     if osp.exists(p):
                #         # add new data filenames to the training data list
                #         self.comp_img_fns+=list(np.load(p, allow_pickle=True))
            
            # add human3.6m data
            S_num_list=['S1','S5','S6','S7','S8']#,'S9','S11'] 
            add_h36m_path=[osp.join(data_path,'h36m_'+S_num+'_comp_img_fns_del05.npy') for S_num in S_num_list]
            for i in range(len(add_h36m_path)):
                add_path=add_h36m_path[i]
                if osp.exists(add_path):
                    self.comp_img_fns+=list(np.load(add_path, allow_pickle=True))
                    logger.info('h36m_{} filenames npy exist~',S_num_list[i])
                else:
                    logger.info('No h36m_{} filenames npy exist!',S_num_list[i]) 

            # add coco2017 data             
            #add_coco2017_path=osp.join(data_path,'coco2017_train_comp_img_fns_del05.npy') 
            '''add_coco2017_path=osp.join(data_path,'coco2017_train_comp_img_fns_del05_bodyconf6.npy')
            if osp.exists(add_coco2017_path):
                self.comp_img_fns+=list(np.load(add_coco2017_path, allow_pickle=True))
                logger.info('coco2017 filenames npy exist~')
            else:
                logger.info('No coco2017 filenames npy exist!')
            '''
            # add 3dpw_train data
            '''add_3dpw_train_path=osp.join(data_path,'3dpw_train_comp_img_fns_del05.npy') 
            if osp.exists(add_3dpw_train_path):
                self.comp_img_fns+=list(np.load(add_3dpw_train_path, allow_pickle=True))
                logger.info('3dpw_train filenames npy exist~')
            else:
                logger.info('No 3dpw_train filenames npy exist!')'''
            
            '''
            del_index_path=osp.join(data_path,'del_indexes_12.npy')#'del_indexes.npy')
            #del_index_path=osp.join(data_path,'del_indexes_curafits_coco.npy') # for adding coco in 132 server
            if osp.exists(del_index_path):
                del_indexes=np.load(del_index_path, allow_pickle=True)
                del_indexes=list(del_indexes)
            '''

            #self.indices = [i for i in range(len(self.comp_img_fns)) if (i not in del_indexes)]
            #self.indices=list(np.load(osp.join(data_path,'indices_del12.npy'), allow_pickle=True))
            self.indices=[]
            if len(self.comp_img_fns)>raw_len:
                self.indices+=[i for i in range(raw_len,len(self.comp_img_fns))]
            self.num_items = len(self.indices) #len(self.comp_img_fns)#
            logger.info('self.num_items: {}',self.num_items)
            #self.indices=self.indices[:24] # for test whether each data is iterated
            #self.num_items = len(self.indices) 
        
        self.is_right = None
        if 'is_right' in data:
            self.is_right = np.asarray(data['is_right'], dtype=np.bool_)
        if 'dset_name' in data:
            self.dset_name = np.asarray(data['dset_name'], dtype=np.string_)

        # idxs correlations to transform the sequence of keypoints from openpose25+hands+face to smplx
        source_idxs, target_idxs = dset_to_body_model(
            dset='openpose25+hands+face',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=self.use_face_contour,
            keyp_format=self.keyp_format)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)
        # idxs correlations to transform the sequence of keypoints from coco to smplx
        coco_source_idxs, coco_target_idxs = dset_to_body_model(
            dset='coco',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=self.use_face_contour,
            keyp_format=self.keyp_format)
        self.coco_source_idxs = np.asarray(coco_source_idxs, dtype=np.int64)
        self.coco_target_idxs = np.asarray(coco_target_idxs, dtype=np.int64)

        # get the idxs of body/hand/face keypoints in whole body keypoints
        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        head_idxs = idxs_dict['head']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]
            head_idxs = head_idxs[:-17]

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.head_idxs = np.asarray(head_idxs)

        self.body_dset_factor = 1.2
        self.head_dset_factor = 2.0
        self.hand_dset_factor = 2.0

        # when evaluation, read the raw ground truth labels of Curated Fittings
        if split=='val':
            self.betas = data['betas'].astype(np.float32)
            self.expression = data['expression'].astype(np.float32)
            self.keypoints2D = data['keypoints2D'].astype(np.float32)#(137,3)
            self.pose = data['pose'].astype(np.float32)
            
            _C = CN()

            _C.body_model = CN()

            _C.body_model.j14_regressor_path = 'data/SMPLX_to_J14.pkl'
            _C.body_model.mean_pose_path = 'data/all_means.pkl'
            _C.body_model.shape_mean_path = 'data/shape_mean.npy'
            _C.body_model.type = 'smplx'
            _C.body_model.model_folder = 'data/models'
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
            model_path='data/models'
            model_type='smplx'

            body_model_cfg=_C.get('body_model', {})
            self.body_model = build_body_model(
                model_path,
                model_type=model_type,
                dtype=torch.float32,
                **body_model_cfg)
        
        data.clear()
        del data

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return 'Curated Fittings( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return 'Curated Fittings/{}'.format(self.split)

    def get_num_joints(self):
        return 25 + 2 * 21 + 51 + 17 * self.use_face_contour

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False
    
    def __getitem__(self, index):
        '''
        get a data item with targeted labels from the dataloader
        '''
        img_index = index
        if self.indices is not None:
            img_index = self.indices[index]

        is_right = None
        if self.is_right is not None:
            if img_index<len(self.is_right):
                is_right = self.is_right[img_index]
        
        # flag to distinguish the added data (MPI, 3DPW train, etc.) to process differently
        is_add_data=False
        img_fn = self.comp_img_fns[img_index] # data/lsp/lspet/images/im03019.png
        if not osp.exists(img_fn):
            logger.info('img not exist: {}',img_fn)
        img = read_img(img_fn)
        img_name=img_fn.split('/')[-1].split('\\')[-1].replace('.jpg','')#.split('.')[0] #im03019
        if 'add_data' in img_fn:
            is_add_data=True
            img_dir_tmp=osp.dirname(img_fn)#add_data/mpi/S2/images
            #img_dir=img_dir_tmp.replace('images', '')#/xxx strip has problems! #add_data/mpi/S2
            img_dir=img_dir_tmp[:-6]
        
        # during training, read pseudo ground truth saved in advance generated by 3 part experts
        if self.split == 'train':
            if is_add_data:
                if 'coco2017' in img_dir:
                    save_3dparam_vertices_path=os.path.join(self.save_3dparam_vertices, 'coco2017',img_name + '.npy')
                elif '3dpw' in img_dir:
                    save_3dparam_vertices_path=os.path.join(self.save_3dparam_vertices, '3dpw_train',img_name + '.npy')
                elif 'h36m' in img_dir:
                    save_3dparam_vertices_path=os.path.join(self.save_3dparam_vertices, 'h36m',img_name + '.npy')
                
                else:#mpi
                    S_num=img_dir.split('/')[-2] #S1/S2_Seq1,  [-1] is empty
                    save_3dparam_vertices_path=os.path.join(self.save_3dparam_vertices, 'mpi',S_num,img_name + '.npy')
            else:
                save_3dparam_vertices_path=os.path.join(self.save_3dparam_vertices, 'curated_fits',img_name + '.npy')
            
            if osp.exists(save_3dparam_vertices_path):
                param3d_vertices_data=np.load(save_3dparam_vertices_path, allow_pickle=True).item()
            else:
                logger.info('Error: not finishing pseudo-info-collecting process? img_fn: {}',img_fn)
                logger.info('param_path: {}',save_3dparam_vertices_path)
                logger.info('S_num: {}',S_num)
                return 
            
            if 'jaw_pose' in param3d_vertices_data:     
                expression=param3d_vertices_data['expression']#.detach().cpu().numpy()
                jaw_pose=param3d_vertices_data['jaw_pose']#.detach().cpu().numpy()
                has_head=True
            else:
                has_head=False
                jaw_pose=None
                expression=None
            
            if 'left_hand_pose' in param3d_vertices_data:                     
                has_left_hand=True
                left_hand_pose=param3d_vertices_data['left_hand_pose'].astype(np.float32).reshape(15,3) #.detach().cpu().numpy()
                left_hand_pose+=self.target_left_hand_mean
                if np.any(left_hand_pose>=3.15) or np.any(left_hand_pose<=-3.15):
                    has_left_hand=False
                    left_hand_pose=None
            else: 
                has_left_hand=False
                left_hand_pose=None
            
            if 'right_hand_pose' in param3d_vertices_data:      
                has_right_hand=True
                right_hand_pose=param3d_vertices_data['right_hand_pose'].astype(np.float32).reshape(15,3) #.detach().cpu().numpy()
                right_hand_pose+=self.target_right_hand_mean
                if np.any(right_hand_pose>=3.15) or np.any(right_hand_pose<=-3.15):
                    has_right_hand=False
                    right_hand_pose=None
            else: 
                has_right_hand=False
                right_hand_pose=None
         
            betas = param3d_vertices_data['betas']
            body_pose = param3d_vertices_data['body_pose']#.detach().cpu().numpy()#.astype(np.float32) 
            global_pose=param3d_vertices_data['global_orient']#.detach().cpu().numpy()

            # rearrange the keypoints order to comform to the smplx model
            output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                        3], dtype=np.float32)

            if is_add_data is False:
                keypoints2d = self.keypoints2D[img_index]#(137,3), SPIN joints(49,3)
                output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]
            elif 'coco2017' in img_dir:
                cont=json.load(open(osp.join(img_dir,'keypoints2d',img_name+'_keypoints.json'),encoding='utf-8'))
                whole_kpt=cont['keypoints']+cont['foot_kpts']+cont['face_kpts']+cont['lefthand_kpts']+cont['righthand_kpts']
                keypoints2d=np.array(whole_kpt).reshape(133,3)
                output_keypoints2d[self.coco_target_idxs] = keypoints2d[self.coco_source_idxs]
            else: # openpose format keypoints: mpi etc.
                # read 2d keypoints
                cont=json.load(open(osp.join(img_dir,'keypoints2d',img_name+'_keypoints.json'),encoding='utf-8'))
                body_kp=np.array(cont['people'][0]['pose_keypoints_2d']).reshape(25,3)
                face_kp=np.array(cont['people'][0]['face_keypoints_2d']).reshape(70,3)
                left_hand_kp=np.array(cont['people'][0]['hand_left_keypoints_2d']).reshape(21,3)
                right_hand_kp=np.array(cont['people'][0]['hand_right_keypoints_2d']).reshape(21,3)
                # concat keypoints to the order: body, hand, face, and reorder them to comform to the smplx model
                keypoints2d=np.concatenate((body_kp,left_hand_kp,right_hand_kp,face_kp))
                output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]

            # Remove joints with negative confidence
            output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
            if self.body_thresh > 0:
                # Only keep the points with confidence above a threshold
                body_conf = output_keypoints2d[self.body_idxs, -1]
                if self.head_only or self.hand_only:
                    body_conf[:] = 0.0

                left_hand_conf = output_keypoints2d[self.left_hand_idxs, -1]
                right_hand_conf = output_keypoints2d[self.right_hand_idxs, -1]
                if self.head_only:
                    left_hand_conf[:] = 0.0
                    right_hand_conf[:] = 0.0

                face_conf = output_keypoints2d[self.face_idxs, -1]
                if self.hand_only:
                    face_conf[:] = 0.0
                    if is_right:
                        left_hand_conf[:] = 0
                    else:
                        right_hand_conf[:] = 0

                body_conf[body_conf < self.body_thresh] = 0.0
                left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
                right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0
                face_conf[face_conf < self.face_thresh] = 0.0
                if self.binarization:
                    body_conf = (
                        body_conf >= self.body_thresh).astype(
                            output_keypoints2d.dtype)
                    left_hand_conf = (
                        left_hand_conf >= self.hand_thresh).astype(
                            output_keypoints2d.dtype)
                    right_hand_conf = (
                        right_hand_conf >= self.hand_thresh).astype(
                            output_keypoints2d.dtype)
                    face_conf = (
                        face_conf >= self.face_thresh).astype(
                            output_keypoints2d.dtype)

                output_keypoints2d[self.body_idxs, -1] = body_conf
                output_keypoints2d[self.left_hand_idxs, -1] = left_hand_conf
                output_keypoints2d[self.right_hand_idxs, -1] = right_hand_conf
                output_keypoints2d[self.face_idxs, -1] = face_conf
            
            # build the targeted label for this data item
            target = Keypoints2D(
                output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)

            # add 3d joints if exist in param3d_vertices_data:
            # keypoints3d=None
            # if 'keypoints3d' in param3d_vertices_data:
            #     keypoints3d=param3d_vertices_data['keypoints3d']#(144,3)
            #     # make all points' conf=1
            #     conf=np.ones(shape=(keypoints3d.shape[0],1))
            #     keypoints3d=np.concatenate([keypoints3d,conf],axis=-1)
            #     keypoints3d_field = Keypoints3D(
            #         keypoints3d, img.shape, flip_axis=0, dtype=self.dtype)
            #     target.add_field('keypoints3d', keypoints3d_field)
            
            # add pseudo ground truth of features into the target for feature distiling during training
            if 'save_feature_body' in param3d_vertices_data:
                save_feature_body=param3d_vertices_data['save_feature_body'].to(torch.float32)#torch.size[1, 1024]
                target.add_field('save_feature_body', save_feature_body)

            if has_head:
                if 'save_feature_face' in param3d_vertices_data:
                    save_feature_face=param3d_vertices_data['save_feature_face'].to(torch.float32)#torch.size[1, 1024]
                    target.add_field('save_feature_face', save_feature_face)

            if self.head_only:
                keypoints = output_keypoints2d[self.head_idxs, :-1]
                conf = output_keypoints2d[self.head_idxs, -1]
            elif self.hand_only:
                keypoints = output_keypoints2d[self.hand_idxs, :-1]
                conf = output_keypoints2d[self.hand_idxs, -1]
            else:
                keypoints = output_keypoints2d[:, :-1]
                conf = output_keypoints2d[:, -1]
            
            has_left_hand = (has_left_hand and (output_keypoints2d[self.left_hand_idxs, -1].sum() >
                            self.min_hand_keypoints))
            
            # add ground truth hand info to the target
            if has_left_hand:
                left_hand_bbox = keyps_to_bbox(
                    output_keypoints2d[self.left_hand_idxs, :-1],
                    output_keypoints2d[self.left_hand_idxs, -1],
                    img_size=img.shape, scale=1.5)
                left_hand_bbox_target = BoundingBox(left_hand_bbox, img.shape)
                target.add_field('left_hand_bbox', left_hand_bbox_target)
                target.add_field(
                    'orig_left_hand_bbox',
                    BoundingBox(left_hand_bbox, img.shape, transform=False))
                
                if 'save_feature_left_hand' in param3d_vertices_data:
                    save_feature_left_hand=param3d_vertices_data['save_feature_left_hand'].to(torch.float32)#.cpu().numpy()
                    #save_feature_left_hand=torch.from_numpy(save_feature_left_hand).to(torch.float32)#torch.size[1, 1024]
                    
                    if save_feature_left_hand.numel():
                        target.add_field('save_feature_left_hand', save_feature_left_hand)
                    else:
                        logger.info('empty npy? {}',save_3dparam_vertices_path)
                else:
                    logger.info('index {} has no left hand feature? {}',img_index,img_fn)
                    logger.info('no left hand npy? {}',save_3dparam_vertices_path)

            has_right_hand= (has_right_hand and (output_keypoints2d[self.right_hand_idxs, -1].sum() >
                            self.min_hand_keypoints))

            if has_right_hand:
                right_hand_bbox = keyps_to_bbox(
                    output_keypoints2d[self.right_hand_idxs, :-1],
                    output_keypoints2d[self.right_hand_idxs, -1],
                    img_size=img.shape, scale=1.5)
                right_hand_bbox_target = BoundingBox(right_hand_bbox, img.shape)
                target.add_field('right_hand_bbox', right_hand_bbox_target)
                target.add_field(
                    'orig_right_hand_bbox',
                    BoundingBox(right_hand_bbox, img.shape, transform=False))

                if 'save_feature_right_hand' in param3d_vertices_data:
                    save_feature_right_hand=param3d_vertices_data['save_feature_right_hand'].to(torch.float32)#.cpu().numpy()
                    #save_feature_right_hand=torch.from_numpy(save_feature_right_hand).to(torch.float32)#torch.size[1, 1024]
                    if save_feature_right_hand.numel():
                        target.add_field('save_feature_right_hand', save_feature_right_hand)
                    else:
                        logger.info('empty npy? {}',save_3dparam_vertices_path)
                else:
                    logger.info('index {} has no right hand feature? {}',img_index,img_fn)
                    logger.info('no right hand npy? {}',save_3dparam_vertices_path)
            
            has_head = (has_head and (output_keypoints2d[self.head_idxs, -1].sum() >
                        self.min_head_keypoints))

            if has_head:
                head_bbox = keyps_to_bbox(
                    output_keypoints2d[self.head_idxs, :-1],
                    output_keypoints2d[self.head_idxs, -1],
                    img_size=img.shape, scale=1.2)
                head_bbox_target = BoundingBox(head_bbox, img.shape)
                target.add_field('head_bbox', head_bbox_target)
                target.add_field(
                    'orig_head_bbox',
                    BoundingBox(head_bbox, img.shape, transform=False))

            if self.head_only:
                dset_scale_factor = self.head_dset_factor
            elif self.hand_only:
                dset_scale_factor = self.hand_dset_factor
            else:
                dset_scale_factor = self.body_dset_factor
            
            center, scale, bbox_size = bbox_to_center_scale(
                keyps_to_bbox(keypoints, conf, img_size=img.shape),
                dset_scale_factor=dset_scale_factor,
            )
            if (center is None) or (scale is None) or (bbox_size is None):
                logger.info('none error img_fn: {}', img_fn)
                if is_add_data:
                    tmp=osp.join(img_dir,'keypoints2d',img_name+'_keypoints.json')
                    logger.info('none error kp json path {}', tmp)

            target.add_field('center', center)
            target.add_field('scale', scale)
            target.add_field('bbox_size', bbox_size)
            target.add_field('keypoints_hd', output_keypoints2d)
            target.add_field('orig_center', center)
            target.add_field('orig_bbox_size', bbox_size)

            # return the ground truth params by adding them into the target, for training use
            # create several classes to wrap all the parameters (defined in expose/data/targets/)
            if self.return_params:
                betas_field = Betas(betas=betas)
                target.add_field('betas', betas_field)
                global_pose_field = GlobalPose(global_pose=global_pose)
                target.add_field('global_pose', global_pose_field)
                body_pose_field = BodyPose(body_pose=body_pose)
                target.add_field('body_pose', body_pose_field)
                
                if has_head:
                    expression_field = Expression(expression=expression)
                    target.add_field('expression', expression_field)
                    jaw_pose_field = JawPose(jaw_pose=jaw_pose)
                    target.add_field('jaw_pose', jaw_pose_field)
                
                if has_left_hand or has_right_hand:
                    if not has_left_hand:
                        hand_pose_field = HandPose(left_hand_pose=None,
                                                right_hand_pose=right_hand_pose)
                    elif not has_right_hand:
                        hand_pose_field = HandPose(left_hand_pose=left_hand_pose,
                                                right_hand_pose=None)
                    else:
                        hand_pose_field = HandPose(left_hand_pose=left_hand_pose,
                                                right_hand_pose=right_hand_pose)
                    target.add_field('hand_pose', hand_pose_field)
                else:
                    if output_keypoints2d[self.right_hand_idxs, -1].sum() > self.min_hand_keypoints:
                        logger.info('has right openpose (but no right pseudo label) img_fn: {}',img_fn)
                
            # add pseudo ground truth vertices to the target
            if hasattr(self, 'dset_name'):
                #dset_name = self.dset_name[img_index].decode('utf-8')
                H, W, _ = img.shape
                vertices_data=param3d_vertices_data['vertices']
                intrinsics = np.array([[5000, 0, 0.5 * W],
                                    [0, 5000, 0.5 * H],
                                    [0, 0, 1]], dtype=np.float32)
                target.add_field('intrinsics', intrinsics)
                
                vertex_field = Vertices(
                    vertices_data, bc=self.bc, closest_faces=self.closest_faces)
                target.add_field('vertices', vertex_field)
            
            # add the img name to the target
            target.add_field('fname', f'{img_index:05d}.jpg')
            # add the dataset name to the target
            target.add_field('name', 'curated_fits')
            cropped_image = img
            # do transformation to imgs according to config.yaml
            if (self.transforms is not None):
                force_flip = False
                if is_right is not None:
                    force_flip = not is_right and self.hand_only

                img, cropped_image, cropped_target = self.transforms(
                    img, target, force_flip=force_flip)

            return img, cropped_image, cropped_target, img_index
        
        elif self.split == 'val':
            # during evaluation, use the raw code of expose (read the raw ground truth label, not pseudo ones)
            return self.get_item_val(index)
        else:
            logger.info('invalid split value: {}',self.split)
        
    def get_item_val(self,index):
        '''
        get a data item during evaluation, 
        use the raw code of expose (read the raw ground truth label, not pseudo ones)
        '''
        is_right = None
        if self.is_right is not None:
            is_right = self.is_right[index]

        pose = self.pose[index].copy() #(55,3)
        betas = self.betas[index, :self.num_betas]
        expression = self.expression[index]

        eye_offset = 0 if pose.shape[0] == 53 else 2
        global_pose = pose[0].reshape(-1)
        body_pose = pose[1:22, :].reshape(-1)
        jaw_pose = pose[22].reshape(-1)
        left_hand_pose = pose[
            23 + eye_offset:23 + eye_offset + 15].reshape(-1)
        right_hand_pose = pose[23 + 15 + eye_offset:].reshape(-1)

        # translate GT pose parameters from axis-angle to rotation matrix, to inputting into the smplx model and get vertices
        pose_torch=torch.from_numpy(pose)
        pose_mat=batch_rodrigues(pose_torch)
        global_pose_mat = pose_mat[0].reshape(-1, 1, 3, 3)
        body_pose_mat = pose_mat[1:22, :,:].reshape(-1, 21, 3, 3)
        jaw_pose_mat = pose_mat[22].reshape(-1, 1, 3, 3)
        left_hand_pose_mat = pose_mat[23 + eye_offset:23 + eye_offset + 15].reshape(-1, 15, 3, 3)
        right_hand_pose_mat = pose_mat[23 + 15 + eye_offset:].reshape(-1, 15, 3, 3)

        #  start = time.perf_counter()
        keypoints2d = self.keypoints2D[index]
        #  logger.info('Reading keypoints: {}', time.perf_counter() - start)

        img_fn = self.comp_img_fns[index]
        #  start = time.perf_counter()
        img = read_img(img_fn)
        #  logger.info('Reading image: {}'.format(time.perf_counter() - start))

        # Pad to compensate for extra keypoints
        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]

        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            body_conf = output_keypoints2d[self.body_idxs, -1]
            if self.head_only or self.hand_only:
                body_conf[:] = 0.0

            left_hand_conf = output_keypoints2d[self.left_hand_idxs, -1]
            right_hand_conf = output_keypoints2d[self.right_hand_idxs, -1]
            if self.head_only:
                left_hand_conf[:] = 0.0
                right_hand_conf[:] = 0.0

            face_conf = output_keypoints2d[self.face_idxs, -1]
            if self.hand_only:
                face_conf[:] = 0.0
                if is_right:
                    left_hand_conf[:] = 0
                else:
                    right_hand_conf[:] = 0

            body_conf[body_conf < self.body_thresh] = 0.0
            left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
            right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0
            face_conf[face_conf < self.face_thresh] = 0.0
            if self.binarization:
                body_conf = (
                    body_conf >= self.body_thresh).astype(
                        output_keypoints2d.dtype)
                left_hand_conf = (
                    left_hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                right_hand_conf = (
                    right_hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                face_conf = (
                    face_conf >= self.face_thresh).astype(
                        output_keypoints2d.dtype)

            output_keypoints2d[self.body_idxs, -1] = body_conf
            output_keypoints2d[self.left_hand_idxs, -1] = left_hand_conf
            output_keypoints2d[self.right_hand_idxs, -1] = right_hand_conf
            output_keypoints2d[self.face_idxs, -1] = face_conf

        target = Keypoints2D(
            output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)

        if self.head_only:
            keypoints = output_keypoints2d[self.head_idxs, :-1]
            conf = output_keypoints2d[self.head_idxs, -1]
        elif self.hand_only:
            keypoints = output_keypoints2d[self.hand_idxs, :-1]
            conf = output_keypoints2d[self.hand_idxs, -1]
        else:
            keypoints = output_keypoints2d[:, :-1]
            conf = output_keypoints2d[:, -1]

        left_hand_bbox = keyps_to_bbox(
            output_keypoints2d[self.left_hand_idxs, :-1],
            output_keypoints2d[self.left_hand_idxs, -1],
            img_size=img.shape, scale=1.5)
        left_hand_bbox_target = BoundingBox(left_hand_bbox, img.shape)
        has_left_hand = (output_keypoints2d[self.left_hand_idxs, -1].sum() >
                         self.min_hand_keypoints)
        if has_left_hand:
            target.add_field('left_hand_bbox', left_hand_bbox_target)
            target.add_field(
                'orig_left_hand_bbox',
                BoundingBox(left_hand_bbox, img.shape, transform=False))

        right_hand_bbox = keyps_to_bbox(
            output_keypoints2d[self.right_hand_idxs, :-1],
            output_keypoints2d[self.right_hand_idxs, -1],
            img_size=img.shape, scale=1.5)
        right_hand_bbox_target = BoundingBox(right_hand_bbox, img.shape)
        has_right_hand = (output_keypoints2d[self.right_hand_idxs, -1].sum() >
                          self.min_hand_keypoints)
        if has_right_hand:
            target.add_field('right_hand_bbox', right_hand_bbox_target)
            target.add_field(
                'orig_right_hand_bbox',
                BoundingBox(right_hand_bbox, img.shape, transform=False))

        head_bbox = keyps_to_bbox(
            output_keypoints2d[self.head_idxs, :-1],
            output_keypoints2d[self.head_idxs, -1],
            img_size=img.shape, scale=1.2)
        head_bbox_target = BoundingBox(head_bbox, img.shape)
        has_head = (output_keypoints2d[self.head_idxs, -1].sum() >
                    self.min_head_keypoints)
        if has_head:
            target.add_field('head_bbox', head_bbox_target)
            target.add_field(
                'orig_head_bbox',
                BoundingBox(head_bbox, img.shape, transform=False))

        if self.head_only:
            dset_scale_factor = self.head_dset_factor
        elif self.hand_only:
            dset_scale_factor = self.hand_dset_factor
        else:
            dset_scale_factor = self.body_dset_factor
        center, scale, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=dset_scale_factor,
        )
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        #logger.info('bbox_size: {}',bbox_size)

        target.add_field('keypoints_hd', output_keypoints2d)
        target.add_field('orig_center', center)
        target.add_field('orig_bbox_size', bbox_size)

        #  start = time.perf_counter()
        if self.return_params:
            betas_field = Betas(betas=betas)
            target.add_field('betas', betas_field)
            expression_field = Expression(expression=expression)
            target.add_field('expression', expression_field)

            global_pose_field = GlobalPose(global_pose=global_pose)
            target.add_field('global_pose', global_pose_field)
            body_pose_field = BodyPose(body_pose=body_pose)
            target.add_field('body_pose', body_pose_field)
            hand_pose_field = HandPose(left_hand_pose=left_hand_pose,
                                       right_hand_pose=right_hand_pose)
            target.add_field('hand_pose', hand_pose_field)
            jaw_pose_field = JawPose(jaw_pose=jaw_pose)
            target.add_field('jaw_pose', jaw_pose_field)

        if hasattr(self, 'dset_name'):
            #dset_name = self.dset_name[index].decode('utf-8') #NOT BEING USED
            final_body_parameters = {
                'global_orient': global_pose_mat, #3 cannot be reshaped to [1,3,3]
                'body_pose': body_pose_mat,
                'left_hand_pose': left_hand_pose_mat,
                'right_hand_pose': right_hand_pose_mat,
                'jaw_pose': jaw_pose_mat,
                'betas': torch.from_numpy(betas).reshape(1,-1),
                'expression': torch.from_numpy(expression).reshape(1,-1)
            }
            final_body_model_output = self.body_model(
                get_skin=True, return_shaped=True, **final_body_parameters)
            vertices=final_body_model_output['vertices'].detach().cpu().numpy()
            vertices=vertices.reshape((vertices.shape[1],vertices.shape[2]))

            H, W, _ = img.shape

            intrinsics = np.array([[5000, 0, 0.5 * W],
                                   [0, 5000, 0.5 * H],
                                   [0, 0, 1]], dtype=np.float32)

            target.add_field('intrinsics', intrinsics)
            vertex_field = Vertices(
                vertices, bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertex_field)
        # add the file name into the target:
        target.add_field('fname', f'{index:05d}.jpg')
        # add the dataset's name into the target:
        target.add_field('name', 'curated_fits')
        cropped_image = None
        # do the same transformation as that before training, to get close model effect
        if self.transforms is not None:
            force_flip = False
            if is_right is not None:
                force_flip = not is_right and self.hand_only
            img, cropped_image, cropped_target = self.transforms(
                img, target, force_flip=force_flip)
        return img, cropped_image, cropped_target, index
