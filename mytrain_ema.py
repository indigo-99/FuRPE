# -*- coding: utf-8 -*-
import sys
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import matplotlib.pyplot as plt
import PIL.Image as pil_img
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import time
import argparse
from collections import defaultdict
from loguru import logger
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import cv2
import math

import resource

from expose.utils.plot_utils import (
    create_skel_img, OverlayRenderer, GTRenderer,
    blend_images,
)
from expose.config.cmd_parser import set_face_contour
from expose.config import cfg
from expose.models.smplx_net import SMPLXNet
from expose.data import make_all_data_loaders
#from expose.data.pseudo_gt import save_pseudo_gt
from expose.utils.checkpointer import Checkpointer
from expose.data.targets.image_list import to_image_list
from expose.data.targets.keypoints import KEYPOINT_NAMES

from expose.optimizers import build_optimizer, build_scheduler

# limit the max num of files which can be opened (refer to EXPOSE's inference.py)
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

# whether to use feature_distil of each sub-network(body/face/hand) or not
# feature_distil: add feature loss during training computed by KLDivLoss, 
# using MLP_feat class to get the same dimensions 
use_hand_feature_distil=False
use_face_feature_distil=False#True
use_body_feature_distil=False#True
feature_loss_amplify_factor=100000

# whether to freeze a sub-network(body/face/hand) or not
# when you don't want to change the parameters of a sub-network, set freeze to be True
# the loss will not be counted on to the total loss or backwarded.
freeze_body=False
freeze_face=False
freeze_hand=False # True


class MLP_feat(nn.Module):
    # feature transformation from input_size to output_size
    def __init__(self, input_size, output_size,hidden_size):
        super(MLP_feat, self).__init__()
        self.mlpmodel = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.mlpmodel(x)
        return out

# class Predictor(nn.Module): 
#     def __init__(self, projection_size, mlp_hidden_size):
#         super(Predictor, self).__init__()
#         self.predictor = nn.Sequential(
#             nn.Linear(projection_size, mlp_hidden_size, bias=False),
#             nn.BatchNorm1d(mlp_hidden_size),
#             nn.ReLU(True),
#             nn.Linear(mlp_hidden_size, projection_size, bias=False)
#         )
    
#     def forward(self, x):
#         out = self.predictor(x)
#         return out


class ExTrainer(object):
    """ class for Trainer of the expressive motion capture model, 
        (distiling from EXPOSE's architecture).
    """
    def __init__(self, exp_cfg):
        '''
        The initialization of the trainer.
        input: 
            exp_cfg: configs of the expressive model, read from data/conf*.yaml
        '''
        self.exp_cfg = exp_cfg
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            logger.error('CUDA is not available!')
            sys.exit(3)
        
        # add convertor of feature dimensions for feature distiling
        #self.mlp_feat = MLP_feat(input_size = 1024, output_size = 2048, hidden_size = 2048)
        #self.mlp_feat_face = MLP_feat(input_size = 1024, output_size = 512, hidden_size = 512)
        #self.mlp_feat_hand = MLP_feat(input_size = 1024, output_size = 512, hidden_size = 512)
        #self.klloss = nn.KLDivLoss(reduction = 'mean')

        self.self_supervised_EMA_model = SMPLXNet(self.exp_cfg)
        
        try:
            self.self_supervised_EMA_model = self.self_supervised_EMA_model.to(device=self.device)
            #self.teacher_model = self.teacher_model.to(device=self.device)
            #self.online_model = self.online_model.to(device=self.device)
            #self.mlp_feat = self.mlp_feat.to(device=self.device)
            #self.mlp_feat_face = self.mlp_feat_face.to(device=self.device)
            #self.mlp_feat_hand = self.mlp_feat_hand.to(device=self.device)
        except RuntimeError:
            sys.exit(3)
        
        self.online_model = self.self_supervised_EMA_model.get_online_model()#.online_model()
        self.teacher_model = self.self_supervised_EMA_model.get_teacher_model()#.teacher_model()

        # set the path of checkpoints' folder according to exp_cfg, 'data/checkpoints' as default
        self.checkpoint_folder = osp.join(self.exp_cfg.output_folder, self.exp_cfg.checkpoint_folder)
        # initialize checkpoint with the current model
        self.ckpt=Checkpointer(self.online_model, save_dir=self.checkpoint_folder,
                                pretrained=self.exp_cfg.pretrained)
    
        # train from start if not exists
        self.epoch_count = 0
        self.iter_count = 0

        # if the latest checkpoint already exists (training based on existing models), load it to the current ckpt.
        save_fn = osp.join(self.checkpoint_folder, 'latest_checkpoint')
        if osp.exists(save_fn):
            # load checkpoint weights into self.model
            extra_checkpoint_data = self.ckpt.load_checkpoint(removesmplx = True) #self.ckpt.load_checkpoint_change()
            if 'epoch_count' in extra_checkpoint_data:
                self.epoch_count = extra_checkpoint_data['epoch_count'] 
                logger.info('already trained model epoch_number: {}',self.epoch_count )
            if 'iter_count' in extra_checkpoint_data:
                self.iter_count = extra_checkpoint_data['iter_count'] 
                logger.info('already trained model iter_count: {}',self.iter_count )
            
        self.update_module(decay_rate=0)
        self.decay_rate = 0.996
        self.tau = self.decay_rate
        
        # set the path to save summaries (losses & visualizations, if any), 'data/summaries' as default
        self.summary_folder = osp.join(self.exp_cfg.output_folder,
                                       self.exp_cfg.summary_folder)
        os.makedirs(self.summary_folder, exist_ok=True)
        self.summary_steps = self.exp_cfg.summary_steps
        self.filewriter = SummaryWriter(self.summary_folder, max_queue=1)
        # set the degrees of body vertices during visualization (summary)
        self.body_degrees = exp_cfg.get('degrees', {}).get(
            'body', [90, 180, 270])
        self.hand_degrees = exp_cfg.get('degrees', {}).get(
            'hand', [90, 180, 270])
        self.head_degrees = exp_cfg.get('degrees', {}).get(
            'head', [90, 180, 270])
        self.imgs_per_row = exp_cfg.get('imgs_per_row', 2)

        # transform images back to normal during visualization (summary)
        self.means = np.array(self.exp_cfg.datasets.body.transforms.mean)
        self.std = np.array(self.exp_cfg.datasets.body.transforms.std)

        # render predicted truth meshes
        body_crop_size = exp_cfg.get('datasets', {}).get('body', {}).get(
            'crop_size', 256)
        self.body_renderer = None #OverlayRenderer(img_size=body_crop_size)

        hand_crop_size = exp_cfg.get('datasets', {}).get('hand', {}).get(
            'crop_size', 256)
        self.hand_renderer = None #OverlayRenderer(img_size=hand_crop_size)

        head_crop_size = exp_cfg.get('datasets', {}).get('head', {}).get(
            'crop_size', 256)
        self.head_renderer = None #OverlayRenderer(img_size=head_crop_size)

        # render ground truth meshes
        self.render_gt_meshes = False#exp_cfg.get('render_gt_meshes', True)
        if self.render_gt_meshes:
            self.gt_body_renderer = GTRenderer(img_size=body_crop_size)
            self.gt_hand_renderer = GTRenderer(img_size=hand_crop_size)
            self.gt_head_renderer = GTRenderer(img_size=head_crop_size)
        else:
            self.gt_body_renderer = None#GTRenderer(img_size=body_crop_size)
            self.gt_hand_renderer = None#GTRenderer(img_size=hand_crop_size)
            self.gt_head_renderer = None#GTRenderer(img_size=head_crop_size)

        # multi_tasks_loss: training the weights of each part of loss
        log_var_body = torch.zeros(()).to(device=self.device)
        log_var_hand = torch.zeros(()).to(device=self.device)
        log_var_face = torch.zeros(()).to(device=self.device)
        log_var_body.requires_grad=True
        log_var_hand.requires_grad=True
        log_var_face.requires_grad=True
        self.train_loss_weight_list=[log_var_body, log_var_hand, log_var_face]
        self.precisions=[]
        for i in range(len(self.train_loss_weight_list)):
            self.precisions.append(torch.exp(-self.train_loss_weight_list[i]))
        # Initialized standard deviations (ground truth is 10 and 1):
        # std_body = torch.exp(log_var_body)**0.5
        # std_hand = torch.exp(log_var_hand)**0.5
        # std_face = torch.exp(log_var_face)**0.5
        #logger.info([std_body.item(), std_hand.item(), std_face.item()])  
        #[1,1,1] will change after training and close to gt 10 and 1(where gt comes from???)

        # build the optimizer
        optim_cfg = self.exp_cfg.optim
        self.optimizer = build_optimizer(model = self.online_model, optim_cfg = optim_cfg, train_loss_weight_list = self.train_loss_weight_list)
        # if mutli_task_loss weights' traing not needed: #torch.optim.Adam(params=,lr=self.exp_cfg.optim.lr,weight_decay=0)
        
        # build the scheduler
        sched_cfg = optim_cfg.scheduler
        self.scheduler = build_scheduler(self.optimizer,sched_cfg)

        self.total_epochs = self.exp_cfg.optim.num_epochs
    
    def online_teacher_output_loss(self, online_params, teacher_params):
        # pose loss
        # keypoint loss
        ot_loss = 0.0
        ot_loss += self.regression_loss(online_params['final']['global_orient'],teacher_params['final']['global_orient'], factor=5 , item_name='global_orient')
        ot_loss += self.regression_loss(online_params['final']['body_pose'],teacher_params['final']['body_pose'], factor=5 , item_name='body_pose')
        ot_loss += self.regression_loss(online_params['final']['left_hand_pose'],teacher_params['final']['left_hand_pose'], factor=5 , item_name='left_hand_pose')
        ot_loss += self.regression_loss(online_params['final']['right_hand_pose'],teacher_params['final']['right_hand_pose'], factor=5 , item_name='right_hand_pose')
        ot_loss += self.regression_loss(online_params['final']['jaw_pose'],teacher_params['final']['jaw_pose'], factor=5 , item_name='jaw_pose')
        ot_loss += self.regression_loss(online_params['final']['betas'],teacher_params['final']['betas'], factor=5 , item_name='betas')
        ot_loss += self.regression_loss(online_params['final']['expression'],teacher_params['final']['expression'], factor=5 , item_name='expression')
        ot_loss += self.regression_loss(online_params['final']['proj_joints'],teacher_params['final']['proj_joints'], factor=5 , item_name='proj_joints')
         
        '''
        ['final']['global_orient']
        ['final']['body_pose']
        ['final']['left_hand_pose']
        ['final']['right_hand_pose']
        ['final']['jaw_pose']
        ['final']['betas']
        ['final']['expression']
        ['final']['proj_joints']'''
        return ot_loss
        
    def regression_loss(self, x, y, factor = 1, item_name = None):
        if x is not None and y is not None:
            norm_x = F.normalize(x, dim=1)
            norm_y = F.normalize(y, dim=1)
            loss = 2 - 2 * (norm_x * norm_y).sum() / x.numel()#x.size(0)
            loss = loss * factor
            #logger.info('item_name: {}, regression_loss: {}',item_name, loss)
        else:
            loss = 0.0
        return loss

    def get_current_decay_rate(self, base_tau):
        tau = 1 - (1 - base_tau) * (math.cos(math.pi * self.iter_count / (self.epoch_steps * self.total_epochs)) + 1) / 2
        return tau
    
    def update_module(self, decay_rate):
        online_dict = self.online_model.state_dict()
        teacher_dict = self.teacher_model.state_dict()
        for key in teacher_dict:
            teacher_dict[key] = decay_rate * teacher_dict[key] + (1 - decay_rate) * online_dict[key]
        self.teacher_model.load_state_dict(teacher_dict)
    
    def gt_loss(self, out_losses, body_targets, feats = None):
        loss = 0
        # body related losses
        if not freeze_body:           
            # add weighted loss (trained together)
            loss_bd_pose = out_losses['body_loss']['body_pose_loss']
            loss += torch.sum(self.precisions[0].data * loss_bd_pose + self.train_loss_weight_list[0].data, -1)
            
            l_bd_2dkpt = out_losses['keypoint_loss']['body_joints_2d_loss']
            loss_bd_2dkpt = torch.sum(self.precisions[0].data * l_bd_2dkpt + self.train_loss_weight_list[0].data, -1)
            loss += loss_bd_2dkpt

            if use_body_feature_distil:
                # get the ground truth feature
                save_feature_body_batch = [
                    t.get_field('save_feature_body').reshape((1024,)).to(self.device) for t in body_targets]
                # add mlp to convert feature dimensions
                feature_body_convert = torch.stack([
                    self.mlp_feat(f) for f in save_feature_body_batch]) # each is 2048 dim
                gt_feat = F.softmax(feature_body_convert, dim = -1)

                # get the predicted feature
                pred_feat = F.log_softmax(feats['body_feat'], dim=-1) # soft_log on each rows
                # compute KL-loss, multipled by 100000 because its small value
                f_body_loss = self.klloss(pred_feat,gt_feat) * feature_loss_amplify_factor
                
                feature_body_loss = torch.sum(self.precisions[0].data * f_body_loss + self.train_loss_weight_list[0].data, -1)
                loss += feature_body_loss
        
        # global_orient loss
        if 'global_orient_loss' in out_losses['body_loss']:
            l_global_orient = out_losses['body_loss']['global_orient_loss']
            loss_global_orient = torch.sum(self.precisions[0].data * l_global_orient + self.train_loss_weight_list[0].data, -1)
            loss += loss_global_orient
                
        #face related losses
        if not freeze_face:
            f_joints_2d_loss = out_losses['keypoint_loss']['face_joints_2d_loss']
            face_joints_2d_loss = torch.sum(self.precisions[2].data * f_joints_2d_loss + self.train_loss_weight_list[2].data, -1)
            loss += face_joints_2d_loss
            
            # add cropped keypoints loss
            if 'head_crop_kpt_loss' in out_losses:
                loss += 0.002 * out_losses['head_crop_kpt_loss']

            if ('expression_loss' in out_losses['body_loss']):
                l_expression = out_losses['body_loss']['expression_loss']
                loss_expression = torch.sum(self.precisions[2].data * l_expression + self.train_loss_weight_list[2].data, -1)
                loss += loss_expression
            
            if 'jaw_pose_loss' in out_losses['body_loss']:
                l_jaw_pose = out_losses['body_loss']['jaw_pose_loss']
                loss_jaw_pose = torch.sum(self.precisions[2].data * l_jaw_pose + self.train_loss_weight_list[2].data, -1)
                loss += loss_jaw_pose
                
                if use_face_feature_distil:
                    # if none of images in this batch contains faces, not count into this part of loss 
                    has_feature_face_idxes = [
                        i for i in range(len(body_targets)) if body_targets[i].has_field('save_feature_face')]
                    save_feature_face_batch = [
                        body_targets[i].get_field('save_feature_face').reshape((1024,)).to(self.device) for i in range(len(body_targets)) if i in has_feature_face_idxes]
                    
                    # add mlp to convert GT feature dimensions
                    feature_face_convert = torch.stack([
                        self.mlp_feat_face(f) for f in save_feature_face_batch]) # each is 512 dim
                    gt_feat = F.softmax(feature_face_convert,dim=-1)

                    feature_face_pred = torch.stack([
                        feats['face_feat'][i] for i in range(len(feats['face_feat'])) if i in has_feature_face_idxes]) 
                    pred_feat = F.log_softmax(feature_face_pred, dim=-1) # the 0 dim is batch
                    
                    f_face_loss = self.klloss(pred_feat,gt_feat) * feature_loss_amplify_factor
                    feature_face_loss = torch.sum(self.precisions[2].data * f_face_loss + self.train_loss_weight_list[2].data, -1)
                    loss += feature_face_loss
                else:
                    feature_face_loss=0
        
        # hand related losses
        if not freeze_hand:
            h_joints_2d_loss = 2*out_losses['keypoint_loss']['hand_joints_2d_loss']
            hand_joints_2d_loss = torch.sum(self.precisions[1].data * h_joints_2d_loss + self.train_loss_weight_list[1].data, -1)
            loss += hand_joints_2d_loss

            # add cropped keypoints loss
            if 'left_hand_crop_kpt_loss' in out_losses:
                loss += 0.002 * out_losses['left_hand_crop_kpt_loss']
            if 'right_hand_crop_kpt_loss' in out_losses:
                loss += 0.002 * out_losses['right_hand_crop_kpt_loss']

            feature_left_hand_loss=None
            if 'left_hand_pose_loss' in out_losses['body_loss']:
                l_lh_pose = 10*out_losses['body_loss']['left_hand_pose_loss']
                loss_lh_pose = torch.sum(self.precisions[1].data * l_lh_pose + self.train_loss_weight_list[1].data, -1)
                loss += loss_lh_pose
                
                if use_hand_feature_distil and loss_lh_pose!=0:
                    # if none of images in this batch contains faces, not count into this part of loss 
                    has_feature_left_hand_idxes = [
                        i for i in range(len(body_targets)) if body_targets[i].has_field('save_feature_left_hand')]
                    if len(has_feature_left_hand_idxes)>0:
                        save_feature_left_hand_batch = [
                            body_targets[i].get_field('save_feature_left_hand').reshape((1024,)).to(self.device) for i in range(len(body_targets)) if i in has_feature_left_hand_idxes]       
                        # add mlp to convert feature dimensions
                        feature_left_hand_convert = torch.stack([
                            self.mlp_feat_hand(f) for f in save_feature_left_hand_batch]) # each is 512 dim
                        feature_left_hand_pred = torch.stack([
                            feats['left_hand_feat'][i] for i in range(len(feats['left_hand_feat'])) if i in has_feature_left_hand_idxes]) 
                        pred_left_hand = F.log_softmax(feature_left_hand_pred, dim=-1) # the 0 dim is batch
                        gt_feat = F.softmax(feature_left_hand_convert,dim=-1)
                        f_left_hand_loss = self.klloss(pred_left_hand,gt_feat) * feature_loss_amplify_factor
                        feature_left_hand_loss = torch.sum(self.precisions[1].data * f_left_hand_loss + self.train_loss_weight_list[1].data, -1)
                        loss += feature_left_hand_loss

            feature_right_hand_loss=None
            if 'right_hand_pose_loss' in out_losses['body_loss']:
                lrp = 10*out_losses['body_loss']['right_hand_pose_loss']
                #loss_rh_pose = torch.sum(self.precisions[1] * loss_rh_pose + self.train_loss_weight_list[1], -1)
                loss_rh_pose = self.precisions[1].data * lrp + self.train_loss_weight_list[1].data-1 
                loss += loss_rh_pose
                
                if use_hand_feature_distil and loss_rh_pose!=0:
                    # if none of images in this batch contains faces, not count into this part of loss 
                    has_feature_right_hand_idxes = [
                        i for i in range(len(body_targets)) if body_targets[i].has_field('save_feature_right_hand')]
                    if len(has_feature_right_hand_idxes)>0:
                        save_feature_right_hand_batch = [
                            body_targets[i].get_field('save_feature_right_hand').reshape((1024,)).to(self.device) for i in range(len(body_targets)) if i in has_feature_right_hand_idxes]       
                        # add mlp to convert feature dimensions
                        feature_right_hand_convert = torch.stack([
                            self.mlp_feat_hand(f) for f in save_feature_right_hand_batch]) # each is 512 dim
                        feature_right_hand_pred = torch.stack([
                            feats['right_hand_feat'][i] for i in range(len(feats['right_hand_feat'])) if i in has_feature_right_hand_idxes]) 
                        pred_right_hand = F.log_softmax(feature_right_hand_pred, dim=-1) # the 0 dim is batch
                        gt_feat = F.softmax(feature_right_hand_convert,dim=-1)
                        f_right_hand_loss = self.klloss(pred_right_hand,gt_feat) * feature_loss_amplify_factor
                        feature_right_hand_loss = torch.sum(self.precisions[1].data * f_right_hand_loss + self.train_loss_weight_list[1].data, -1)
                        loss += feature_right_hand_loss
        
        lossdict={}
        if not freeze_hand:
            lossdict['hand_joints_2d_loss'] = hand_joints_2d_loss
            if 'hand_joints_3d_loss' in out_losses['keypoint_loss']:
                lossdict['hand_joints_3d_loss'] = out_losses['keypoint_loss']['hand_joints_3d_loss']
            
            if 'left_hand_pose_loss' in out_losses['body_loss']:
                lossdict['loss_lh_pose'] = loss_lh_pose
                if feature_left_hand_loss is not None:
                    lossdict['feature_left_hand_loss'] = feature_left_hand_loss
                if 'left_hand_crop_kpt_loss' in out_losses:
                    lossdict['left_hand_crop_kpt_loss'] = out_losses['left_hand_crop_kpt_loss']
                
            if 'right_hand_pose_loss' in out_losses['body_loss']:
                lossdict['loss_rh_pose'] = loss_rh_pose
                if feature_right_hand_loss is not None:
                    lossdict['feature_right_hand_loss'] = feature_right_hand_loss
                if 'right_hand_crop_kpt_loss' in out_losses:
                    lossdict['right_hand_crop_kpt_loss'] = out_losses['right_hand_crop_kpt_loss']
            
        if not freeze_body:
            lossdict['body_joints_2d_loss'] = loss_bd_2dkpt
            lossdict['loss_bd_pose'] = loss_bd_pose
            if 'body_joints_3d_loss' in out_losses['keypoint_loss']:
                lossdict['body_joints_3d_loss'] = out_losses['keypoint_loss']['body_joints_3d_loss']
            if use_body_feature_distil:
                lossdict['feature_body_loss'] = feature_body_loss  
            
        if 'global_orient_loss' in out_losses['body_loss']:
            lossdict['loss_global_orient'] = loss_global_orient
        
        if not freeze_face:
            lossdict['face_joints_2d_loss'] = face_joints_2d_loss
            if 'jaw_pose_loss' in out_losses['body_loss']:
                lossdict['loss_jaw_pose'] = loss_jaw_pose
                if use_face_feature_distil:
                    lossdict['feature_face_loss'] = feature_face_loss
            if 'expression_loss' in out_losses['body_loss']:
                lossdict['loss_expression'] = loss_expression
            if 'face_joints_3d_loss' in out_losses['keypoint_loss']:
                lossdict['face_joints_3d_loss'] = out_losses['keypoint_loss']['face_joints_3d_loss']
            
            if 'head_crop_kpt_loss' in out_losses:
                    lossdict['head_crop_kpt_loss'] = out_losses['head_crop_kpt_loss']

        return loss, lossdict


    def train(self):
        ''' Training process.
        '''
        # crop and get experts' pseudo ground truth before load data to train the model, 
        # save results in dir: ../experts/res. Only need to be executed once.
        # save_pseudo_gt(self.exp_cfg)

        # Run training for several epochs
        for epoch in tqdm(range(self.epoch_count, self.total_epochs)):
            # Create new DataLoader every epoch
            train_dataloaders = make_all_data_loaders(self.exp_cfg, split='train')
            body_dataloader=train_dataloaders['body']#['hand']['head'] # can train hand and head after body training
            self.epoch_steps = len(body_dataloader)
            # dset = body_dataloader[0].dataset # if set multiple datasets in config.yaml, body_dataloader will be a list
            # dset_name = dset.name()

            # Iterate batches in an epoch
            for step, batch in enumerate(tqdm(body_dataloader, desc='Epoch '+str(epoch)+' iteration')):
                # train each step, output predicted parameters and losses
                out_params1, out_params2, losses = self.train_step(batch)
                # update the decay_rate according to epoch nums
                self.tau = self.get_current_decay_rate(self.decay_rate)
                self.update_module(decay_rate=self.tau)
                self.iter_count += 1
                
                '''# Save summaries every summary_steps, displayed in Tensorboard
                if self.iter_count % self.summary_steps == 0:
                    self.create_summaries(input_batch=batch,out_params=out_params,
                                        losses=losses,
                                        renderer=self.body_renderer,
                                        gt_renderer=self.gt_body_renderer,
                                        degrees=self.body_degrees)#dset_name=dset_name,'''
                
                # Save checkpoint every checkpoint_steps 
                if self.iter_count % self.exp_cfg.checkpoint_steps == 0:
                    # stop gradient before saving checkpoint
                    self.online_model.eval()
                    self.ckpt.save_checkpoint('myckpt_e'+str(epoch)+'_i'+str(self.iter_count),epoch_count=epoch, batch_idx=step+1, batch_size=self.exp_cfg.datasets.body.batch_size, iter_count=self.iter_count)        
                    # restart gradient after saving checkpoint
                    self.online_model.train()
                    logger.info('Checkpoint saved: ','myckpt_e'+str(epoch)+'_i'+str(self.iter_count))
                    
                    # multi-task losses' weight after training
                    std_body = torch.exp(self.train_loss_weight_list[0])**0.5
                    std_hand = torch.exp(self.train_loss_weight_list[1])**0.5
                    std_face = torch.exp(self.train_loss_weight_list[2])**0.5
                    std_list = [std_body.item(), std_hand.item(), std_face.item()]
                    logger.info('std_list (loss weight): {}', std_list)   
        return 

    def train_step(self, input_batch):
        ''' Training step.
            input: a data batch of the current step
        '''
        # set model's parameters to be trained with gradient
        self.online_model.train()
        self.optimizer.zero_grad() 
        # Get data from the batch
        batch1, batch2 = input_batch
        images1, cropped_images1, cropped_targets1 = batch1 
        images2, cropped_images2, cropped_targets2 = batch2 
        
        # keys of cropped_targets:
        # gt_keypoints_2d = cropped_targets['keypoints_hd'] # 2D keypoints
        # ['body_pose'] ['hand_pose'] ['jaw_pose'] ['global_pose'] ['betas'] ['expression'] ['vertices']
        # 'left_hand_bbox','orig_left_hand_bbox','right_hand_bbox','orig_right_hand_bbox', 'head_bbox','orig_head_bbox'
        # 'center','scale','bbox_size','orig_center','orig_bbox_size','intrinsics',fname
        
        if cropped_images1 is None or cropped_images2 is None:
            logger.error('train_step: this batch of cropped_images is none!')
            return
        if images1 is not None:
            # package raw input images into datatype image_list, for model's use (expose/models/attention/predictor.py)
            full_imgs1 = to_image_list(images1).to(device=self.device)
        else:
            full_imgs1 = None
        
        if images2 is not None :
            # package raw input images into datatype image_list, for model's use (expose/models/attention/predictor.py)
            full_imgs2 = to_image_list(images2).to(device=self.device)
        else:
            full_imgs2 = None

        body_imgs1 = cropped_images1.to(device=self.device)
        body_targets1 = [target.to(self.device) for target in cropped_targets1]

        body_imgs2 = cropped_images2.to(device=self.device)
        body_targets2 = [target.to(self.device) for target in cropped_targets2]

        # for accuratedly computing time elapsed during model prediction
        torch.cuda.synchronize()
        
        # Feed images in the network to predict camera and SMPLX parameters (expose/models/smplx_net.py)
        model_output_online1 = self.online_model(body_imgs1, targets=body_targets1, full_imgs=full_imgs1)
                             #device=self.device)
        model_output_online2 = self.online_model(body_imgs2, targets=body_targets2, full_imgs=full_imgs2)
                             #device=self.device)
        
        with torch.no_grad():
            model_output_teacher1 = self.teacher_model(body_imgs1, targets=body_targets1, full_imgs=full_imgs1)
                                #device=self.device)
            model_output_teacher2 = self.teacher_model(body_imgs2, targets=body_targets2, full_imgs=full_imgs2)
                                #device=self.device)

        torch.cuda.synchronize()

        out_params_online1 = model_output_online1['body']
        out_params_online2 = model_output_online2['body']
        out_params_teacher1 = model_output_teacher1['body']
        out_params_teacher2 = model_output_teacher2['body']

        # add for feature distil: predicted features
        feats_online1 = model_output_online1['feats']
        feats_online2 = model_output_online1['feats']
        ##logger.info('pred body feat shape: {}',feats['body_feat'].shape)#torch.Size([32, 2048])
        ##logger.info('pred face feat shape: {}',feats['face_feat'].shape)#torch.Size([32, 512])
        ##logger.info('out_params key: {}',out_params.keys())


        # Compute EMA losses between the online and teacher model
        # initialize the loss
        loss = 0
        ot_loss12 = self.online_teacher_output_loss(out_params_online1, out_params_teacher2)
        ot_loss21 = self.online_teacher_output_loss(out_params_online2, out_params_teacher1)
        loss += ot_loss12
        loss += ot_loss21

        # shape regularization
        # change shape loss to L2 loss without target limitation because SPIN' shape is not compatible to SMPLX, according to frankmocap paper
        shapenp1 = out_params_online1['final']['betas']#.detach().cpu().numpy()
        shapenp2 = out_params_online2['final']['betas']
        loss_shape = 0.2*((shapenp1 ** 2 ).mean()+(shapenp2 ** 2 ).mean())
        loss += loss_shape
        
        # Compute losses between model outputs and the ground truth labels (pseudo)
        out_losses1 = model_output_online1['losses']
        out_losses2 = model_output_online2['losses']
        gt_loss_online1, lossdict_online1 = self.gt_loss(out_losses1, body_targets1, feats_online1)
        gt_loss_online2, lossdict_online2 = self.gt_loss(out_losses2, body_targets2, feats_online2)
        loss += gt_loss_online1
        loss += gt_loss_online2
        
        # camera loss, without ground truth (useless)
        # loss += ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()

        # loss backprop
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # output losses for summary
        lossdict_integrate={'loss' : loss.detach().item(),
                'loss_shape' : loss_shape.detach().item(),
                'ot_loss12' : ot_loss12.detach().item(),
                'ot_loss21' : ot_loss21.detach().item()}
        
        # integrate the 2 input losses of the online_model
        for key in lossdict_online1:
            if key not in lossdict_integrate:
                lossdict_integrate[key] = lossdict_online1[key].detach().item()
            else:
                lossdict_integrate[key] += lossdict_online1[key].detach().item()
        for key in lossdict_online2:
            if key not in lossdict_integrate:
                lossdict_integrate[key] = lossdict_online2[key].detach().item()
            else:
                lossdict_integrate[key] += lossdict_online2[key].detach().item()
        
        logger.info('lossdict_integrate: {}',lossdict_integrate)
        
        return out_params_online1, out_params_online2, lossdict_integrate


    def create_summaries(self, input_batch, out_params, losses,
                         renderer=None, gt_renderer=None,
                         degrees=None, prefix='',dset_name=None):
        ''' Training step.
            input: a data batch of the current step
        '''

        if not hasattr(self, 'filewriter'):
            return
        if degrees is None:
            degrees = []

        full_imgs, cropped_images, cropped_targets = input_batch
        images = cropped_images.to(device = self.device).detach().cpu().numpy()
        targets = [target.to(self.device) for target in cropped_targets]
        camera_parameters = out_params.get('camera_parameters')
        # the final output parameters of the model
        body_stage_n_out = out_params.get('final', {}) 

        crop_size = images.shape[-1]
        
        # transform images back to normal
        imgs = (images * self.std[np.newaxis, :, np.newaxis, np.newaxis] +
                   self.means[np.newaxis, :, np.newaxis, np.newaxis])
        
        # content saved in summary
        summary_imgs = OrderedDict()
        summary_imgs['rgb'] = imgs # the first column is rgb imgs

        # ground truth 2d keypoints imgs
        gt_keyp_imgs = []
        for img_idx in range(imgs.shape[0]):
            input_img = np.ascontiguousarray(
                np.transpose(imgs[img_idx], [1, 2, 0]))
            gt_keyp2d = targets[img_idx].smplx_keypoints.detach().cpu().numpy()
            gt_conf = targets[img_idx].conf.detach().cpu().numpy()

            gt_keyp2d[:, 0] = (
                gt_keyp2d[:, 0] * 0.5 + 0.5) * crop_size
            gt_keyp2d[:, 1] = (
                gt_keyp2d[:, 1] * 0.5 + 0.5) * crop_size

            gt_keyp_img = create_skel_img(
                input_img, gt_keyp2d,
                targets[img_idx].CONNECTIONS,
                gt_conf > 0,
                names=KEYPOINT_NAMES)

            gt_keyp_img = np.transpose(gt_keyp_img, [2, 0, 1])
            gt_keyp_imgs.append(gt_keyp_img)
        
        gt_keyp_imgs = np.stack(gt_keyp_imgs)
        summary_imgs['gt_keypoints'] = gt_keyp_imgs

        # predicted 2d keypoints imgs
        proj_joints = body_stage_n_out.get('proj_joints', None)
        if proj_joints is not None:
            proj_points = body_stage_n_out[
                'proj_joints'].detach().cpu().numpy()
            proj_points = (proj_points * 0.5 + 0.5) * crop_size

            reproj_joints_imgs = []
            for img_idx in range(imgs.shape[0]):
                gt_conf = targets[img_idx].conf.detach().cpu().numpy()

                input_img = np.ascontiguousarray(
                    np.transpose(imgs[img_idx], [1, 2, 0]))

                reproj_joints_img = create_skel_img(
                    input_img,
                    proj_points[img_idx],
                    targets[img_idx].CONNECTIONS,
                    valid=gt_conf > 0, names=KEYPOINT_NAMES)

                reproj_joints_img = np.transpose(
                    reproj_joints_img, [2, 0, 1])
                reproj_joints_imgs.append(reproj_joints_img)

            # Add the the projected keypoints
            reproj_joints_imgs = np.stack(reproj_joints_imgs)
            summary_imgs['proj_joints'] = reproj_joints_imgs

        camera_scale = camera_parameters.scale.detach()
        camera_transl = camera_parameters.translation.detach()
        
        # visualize ground truth meshes
        render_gt_meshes = (self.render_gt_meshes and
                            any([t.has_field('vertices') for t in targets]))
        if render_gt_meshes:
            gt_mesh_imgs = []
            faces = body_stage_n_out['faces']
            for bidx, t in enumerate(targets):
                if not (t.has_field('vertices') and t.has_field('intrinsics')):
                    gt_mesh_imgs.append(np.zeros_like(imgs[bidx]))
                    logger.info('empty gt mesh with no vertices: {}',bidx)
                    continue

                curr_gt_vertices = t.get_field(
                    'vertices').vertices.detach().cpu().numpy().squeeze()
                #intrinsics = t.get_field('intrinsics')
                
                mesh_img = renderer(
                    curr_gt_vertices[np.newaxis], faces,
                    camera_scale, camera_transl,
                    bg_imgs=imgs[[bidx]],
                    return_with_alpha=False
                )
                gt_mesh_imgs.append(mesh_img.squeeze())

            gt_mesh_imgs = np.stack(gt_mesh_imgs)
            B, C, H, W = gt_mesh_imgs.shape

            row_pad = (crop_size - H) // 2
            gt_mesh_imgs = np.pad(
                gt_mesh_imgs,
                [[0, 0], [0, 0], [row_pad, row_pad], [row_pad, row_pad]])
            summary_imgs['gt_meshes'] = gt_mesh_imgs

        # visualize predicted meshes
        vertices = body_stage_n_out.get('vertices', None)
        if vertices is not None:
            body_imgs = []
            vertices = vertices.detach().cpu().numpy()
            faces = body_stage_n_out['faces']
            body_imgs = renderer(
                vertices, faces,
                camera_scale, camera_transl,
                bg_imgs=imgs,
                return_with_alpha=False,
            )
            # Add the rendered meshes
            summary_imgs['overlay'] = body_imgs.copy()

            # render meshes from different degrees
            for deg in degrees:
                body_imgs = renderer(
                    vertices, faces,
                    camera_scale, camera_transl,
                    deg=deg,
                    return_with_alpha=False,
                )
                summary_imgs[f'{deg:03d}'] = body_imgs.copy()

        summary_imgs = np.concatenate(
            list(summary_imgs.values()), axis=3)
        img_grid = make_grid(
            torch.from_numpy(summary_imgs), nrow=self.imgs_per_row)
        
        if dset_name is not None:
            img_tab_name = (f'{dset_name}/{prefix}/Images' if len(prefix) > 0 else
                            f'{dset_name}/Images')
        else:
            img_tab_name = (f'{prefix}/Images' if len(prefix) > 0 else
                            f'Images')
        self.filewriter.add_image(img_tab_name, img_grid, self.iter_count)

        # save losses in summary
        for loss_name, val in losses.items():
            self.filewriter.add_scalar(loss_name, val, self.iter_count)
        return


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # training with several processes
    torch.multiprocessing.set_start_method('spawn')

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch Exmodel Trainer'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)
    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    # parser.add_argument('--datasets', nargs='+',
    #                     default=['openpose'], type=str,
    #                     help='Datasets to process')
    
    cmd_args = parser.parse_args()
    cfg.merge_from_file(cmd_args.exp_cfg)

    cfg.is_training = True
    # whether to use 17 face contour points or not (default to be true)
    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)
    
    trainer = ExTrainer(cfg)
    trainer.train()
    
