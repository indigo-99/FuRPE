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

import sys

from typing import Dict, List, Optional

import time

import os.path as osp

from collections import defaultdict
from loguru import logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit


from smplx import build_layer

from ..backbone import build_backbone
from ..common.networks import MLP, IterativeRegression
from ..common.networks import FrozenBatchNorm2d
from ..common.pose_utils import build_pose_decoder
from ..nnutils import init_weights
from ..camera import CameraParams, build_cam_proj

from FuRPE.utils.rotation_utils import batch_rodrigues, batch_rot2aa
from FuRPE.data.targets.keypoints import KEYPOINT_NAMES
from FuRPE.data.utils import flip_pose
from FuRPE.data.targets.keypoints import FLIP_INDS

from FuRPE.utils.typing_utils import Tensor

# add mano model path for hand_only_training
from manopth import manolayer
_mano_root = '/data/panyuqing/MinimalHandPytorch/mano/models'

class HandPredictor(nn.Module):
    '''
    the class of the hand pose estimation model 
    '''
    def __init__(self, exp_cfg,
                 global_orient_desc,
                 hand_pose_desc,
                 camera_data,
                 wrist_pose_mean=None,
                 detach_mean=False,
                 mean_pose_path='',
                 dtype=torch.float32):
        super(HandPredictor, self).__init__()
        # get the hand sub-network configs
        network_cfg = exp_cfg.get('network', {})
        attention_net_cfg = network_cfg.get('attention', {})
        hand_net_cfg = attention_net_cfg.get('hand', {})

        self.hand_model_type = hand_net_cfg.get('type', 'mano')

        hand_model_cfg = exp_cfg.get('hand_model', {})
        self.hand_model_cfg = hand_model_cfg.copy()

        self.right_wrist_index = KEYPOINT_NAMES.index('right_wrist')
        self.left_wrist_index = KEYPOINT_NAMES.index('left_wrist')

        camera_cfg = hand_net_cfg.get('camera', {})
        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']

        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        #  self.camera_mean = camera_mean
        self.register_buffer('camera_mean', camera_mean)
        self.camera_scale_func = camera_data['scale_func']

        # The number of shape coefficients
        self.num_betas = self.hand_model_cfg['num_betas']
        shape_mean = torch.zeros([self.num_betas], dtype=dtype)
        self.register_buffer('shape_mean', shape_mean)

        # build the hand pose decoders
        self.global_orient_decoder = global_orient_desc.decoder
        cfg = {'param_type': global_orient_desc.decoder.get_type()}
        self.wrist_pose_decoder = build_pose_decoder(cfg, 1)
        wrist_pose_mean = self.wrist_pose_decoder.get_mean()
        wrist_pose_dim = self.wrist_pose_decoder.get_dim_size()
        self.register_buffer('wrist_pose_mean', wrist_pose_mean)

        self.register_buffer(
            'global_orient_mean', wrist_pose_mean.unsqueeze(dim=0))

        self.hand_pose_decoder = hand_pose_desc.decoder
        hand_pose_mean = hand_pose_desc.mean
        self.register_buffer('hand_pose_mean', hand_pose_mean)
        hand_pose_dim = hand_pose_desc.dim
        
        # get the mean parameters list and their idxs in the mean_lst
        mean_lst = []
        start = 0
        wrist_pose_idxs = list(range(start, start + wrist_pose_dim))
        self.register_buffer('wrist_pose_idxs',
                             torch.tensor(wrist_pose_idxs, dtype=torch.long))
        start += wrist_pose_dim
        mean_lst.append(wrist_pose_mean.view(-1))

        hand_pose_idxs = list(range(
            start, start + hand_pose_dim))
        self.register_buffer(
            'hand_pose_idxs', torch.tensor(hand_pose_idxs, dtype=torch.long))
        start += hand_pose_dim
        mean_lst.append(hand_pose_mean.view(-1))

        shape_idxs = list(range(start, start + self.num_betas))
        self.register_buffer(
            'shape_idxs', torch.tensor(shape_idxs, dtype=torch.long))
        start += self.num_betas
        mean_lst.append(shape_mean.view(-1))

        camera_idxs = list(range(
            start, start + camera_param_dim))
        self.register_buffer(
            'camera_idxs', torch.tensor(camera_idxs, dtype=torch.long))
        start += camera_param_dim
        mean_lst.append(camera_mean)

        self.register_buffer('camera_mean', camera_mean.unsqueeze(dim=0))

        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()
        self.param_dim = param_dim

        # Construct the hand feature extraction backbone
        backbone_cfg = hand_net_cfg.get('backbone', {})
        self.backbone, feat_dims = build_backbone(backbone_cfg) #resnet18, same as head_predictor.py

        self.append_params = hand_net_cfg.get('append_params', True)
        self.num_stages = hand_net_cfg.get('num_stages', 1)

        self.feature_key = hand_net_cfg.get('feature_key', 'avg_pooling') # avg_pooling in config
        feat_dim = feat_dims[self.feature_key]
        self.feat_dim = feat_dim
        
        # Construct the hand pose regressor
        regressor_cfg = hand_net_cfg.get('mlp', {})
        regressor = MLP(feat_dim + self.append_params * param_dim,
                        param_dim, **regressor_cfg)
        self.regressor = IterativeRegression(
            regressor, param_mean, detach_mean=detach_mean,
            num_stages=self.num_stages)
        
        # whether or not to stop updating the hand sub-network (backbone and regressor)
        freeze_hand_backbone=False#True
        if freeze_hand_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # for param in self.regressor.parameters():
            #     param.requires_grad = False
            self.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(
                self.backbone)
            # self.regressor = FrozenBatchNorm2d.convert_frozen_batchnorm(
            #     self.regressor)
        
        # add the initialization of the mano model to generate hand vertices from hand parameters
        self.mano = manolayer.ManoLayer(flat_hand_mean=False, #True, # don't need mean hand params, start from flat hand
                        side="right",
                        mano_root=_mano_root,
                        use_pca=False,
                        root_rot_mode='rotmat',
                        joint_rot_mode='rotmat')

    def get_feat_dim(self) -> int:
        ''' Returns the dimension of the expected feature vector '''
        return self.feat_dim

    def get_param_dim(self) -> int:
        ''' Returns the dimension of the predicted parameter vector '''
        return self.param_dim

    def get_num_stages(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        return self.num_stages

    def get_shape_mean(self, batch_size: int = 1) -> Tensor:
        ''' Returns the mean shape for the hands '''
        return self.shape_mean.reshape(1, -1).expand(batch_size, -1)

    def get_camera_mean(self, batch_size: int = 1) -> Tensor:
        ''' Returns the camera mean '''
        return self.camera_mean.reshape(1, -1).expand(batch_size, -1)

    def get_wrist_pose_mean(self, batch_size=1) -> Tensor:
        ''' Returns wrist pose mean '''
        return self.wrist_pose_mean.reshape(1, -1).expand(batch_size, -1)

    def get_finger_pose_mean(self, batch_size=1) -> Tensor:
        ''' Returns neck pose mean '''
        return self.hand_pose_mean.reshape(1, -1).expand(batch_size, -1)

    def get_param_mean(self,
                       batch_size: int = 1,
                       add_shape_noise: bool = False,
                       shape_mean: Tensor = None,
                       shape_std: float = 0.0,
                       shape_prob: float = 0.0,
                       num_hand_components: int = 3,
                       add_hand_pose_noise: bool = False,
                       hand_pose_mean: Tensor = None,
                       hand_pose_std: float = 1.0,
                       hand_noise_prob: float = 0.0,
                       targets: List = None,
                       randomize_global_orient: bool = False,
                       global_rot_noise_prob: float = 0.0,
                       global_rot_min: bool = 0.0,
                       global_rot_max: bool = 0.0,
                       ) -> Tensor:
        ''' Returns the mean vector given to the iterative regressor
        '''
        mean = self.regressor.get_mean().clone().reshape(1, -1).expand(
            batch_size, -1).clone()
        if not self.training:
            return mean

        raise NotImplementedError

    def param_tensor_to_dict(self, param_tensor):
        ''' Converts a flattened tensor to a dictionary of hand parameters '''
        wrist_pose = torch.index_select(param_tensor, 1, self.wrist_pose_idxs)
        hand_pose = torch.index_select(param_tensor, 1, self.hand_pose_idxs)

        betas = torch.index_select(param_tensor, 1, self.shape_idxs)

        return dict(wrist_pose=wrist_pose, hand_pose=hand_pose, betas=betas)

    def forward(self,
                hand_imgs: Tensor,
                hand_mean: Optional[Tensor] = None,
                global_orient_from_body_net: Optional[Tensor] = None,
                body_pose_from_body_net: Optional[Tensor] = None,
                parent_rots: Optional[Tensor] = None,
                num_hand_imgs: int = 0,
                device: torch.device = None,
                ) -> Dict[str, Dict[str, Tensor]]:
        ''' 
        processing of the predictor
        input:
            hand_imgs: training/testing data
            num_hand_imgs: the number of training/testing hand specific data (not being used during training)
            hand_mean: training/testing hand specific ground truth labels (not being used during training)
            parent_rots: the rotation matrix of hands' parent joints
        output:
            a dict of predicted hand parameters
        '''
        batch_size = hand_imgs.shape[0]
        num_body_data = batch_size - num_hand_imgs
        if batch_size == 0:
            return {}

        if device is None:
            device = hand_imgs.device
        dtype = hand_imgs.dtype

        if parent_rots is None:
            parent_rots = torch.eye(3, dtype=dtype, device=device).reshape(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).clone()

        right_hand_idxs = torch.arange(
            0, num_body_data // 2, dtype=torch.long, device=device)
        left_hand_idxs = torch.arange(
            num_body_data // 2, num_body_data, dtype=torch.long, device=device)

        hand_features = self.backbone(hand_imgs)
        hand_parameters, hand_deltas = self.regressor(
            hand_features[self.feature_key], cond=hand_mean)
            # hand_mean=None, hand_features all Nan but hand_imgs not all black/white?

        hand_model_parameters = []
        model_parameters = []
        for stage_idx, parameters in enumerate(hand_parameters):
            parameters_dict = self.param_tensor_to_dict(parameters)

            # Decode the predicted wrist pose as a rotation matrix
            dec_wrist_pose_abs = self.wrist_pose_decoder(
                parameters_dict['wrist_pose'])

            # Undo the rotation of the parent joints to make the wrist rotation
            # relative again
            dec_wrist_pose = torch.matmul(
                parent_rots.reshape(-1, 3, 3).transpose(1, 2),
                dec_wrist_pose_abs.reshape(-1, 3, 3)
            )
            raw_right_wrist_pose, raw_left_wrist_pose = None, None
            if len(right_hand_idxs) > 0:
                raw_right_wrist_pose = self.global_orient_decoder.encode(
                    dec_wrist_pose[right_hand_idxs].unsqueeze(dim=1)).reshape(
                        num_body_data // 2, -1)

            if len(left_hand_idxs) > 0:
                left_wrist_poses = flip_pose(
                    dec_wrist_pose[left_hand_idxs], pose_format='rot-mat')
                raw_left_wrist_pose = self.global_orient_decoder.encode(
                    left_wrist_poses.unsqueeze(dim=1)).reshape(
                        num_body_data // 2, -1)

            dec_hand_pose = self.hand_pose_decoder(
                parameters_dict['hand_pose'])
            model_betas = parameters_dict['betas']

            model_parameters.append(
                dict(right_hand_pose=dec_hand_pose,
                     betas=model_betas,
                     wrist_pose=dec_wrist_pose_abs,
                     hand_pose=dec_hand_pose,
                     raw_right_wrist_pose=raw_right_wrist_pose,#none
                     raw_left_wrist_pose=raw_left_wrist_pose,#none
                     raw_right_hand_pose=parameters_dict['hand_pose'],
                     )
            )# stage len: 3
            

            if self.hand_model_type == 'mano':
                hand_model_parameters.append(
                    dict(
                        betas=model_betas,
                        wrist_pose=dec_wrist_pose_abs,
                        hand_pose=dec_hand_pose,
                    )
                )
            else:
                raise RuntimeError(
                    f'Invalid hand model type: {self.hand_model_type}')

        # add for training hand-sub-networks
        pred_wrist_pose = model_parameters[-1]['wrist_pose'] #[64, 1, 3, 3]
        pred_hand_pose = model_parameters[-1]['hand_pose'] #[64, 15, 3, 3]
        total_hand_pose = torch.cat((pred_wrist_pose, pred_hand_pose),1)#[64, 16, 3, 3]
        shape = model_parameters[-1]['betas']
        #self.mano = self.mano.to(device=device)
        hand_vertices, hand_joints = self.mano(total_hand_pose, shape)#.float())
        #logger.info('hand_joints: {}',hand_joints)
        #logger.info('hand_vertices: {}',hand_vertices)
        #hand_joints = hand_joints.detach().cpu().numpy()#.tolist() 
        joints3d = hand_joints

        # Extract the camera parameters estimated by the hand only image
        camera_params = torch.index_select(hand_parameters[-1], 1, self.camera_idxs)
        scale = camera_params[:, 0].view(-1, 1)
        translation = camera_params[:, 1:3]
        # Pass the predicted scale through exp() to make sure that the scale values are always positive
        scale = self.camera_scale_func(scale)
        # Project the joints on the image plane
        proj_joints = self.projection(
            joints3d,
            scale=scale, translation=translation)
        
        
        output = {'num_stages': self.num_stages,
                  'features': hand_features[self.feature_key], # 
                  'proj_joints':proj_joints, # 
                  'joints':joints3d, # 
                  #'vertices':hand_vertices, 
                  }
        output['camera_parameters'] = CameraParams(
            translation=translation, scale=scale) 

        for stage in range(self.num_stages):
            # Only update the current stage if the parameters exist
            key = f'stage_{stage:02d}'
            output[key] = model_parameters[stage]
        # add vertices to the output dict
        output['stage_02']['vertices']=hand_vertices
        return output
