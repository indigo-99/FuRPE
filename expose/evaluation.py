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

from copy import deepcopy
from collections import defaultdict, OrderedDict

import time
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.utils import make_grid
import pickle

import tqdm

from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from .utils.metrics import (mpjpe, vertex_to_vertex_error,
                            ProcrustesAlignmentMPJPE,
                            PelvisAlignmentMPJPE,
                            RootAlignmentMPJPE)
from .data.targets.image_list import to_image_list
from .data.targets.keypoints import KEYPOINT_NAMES, get_part_idxs
from .utils.plot_utils import (
    create_skel_img, OverlayRenderer, GTRenderer,
    blend_images,
)

# Joints selectors for computing metrics (eg: mpjpe)
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]


def make_filter(name):
    ''' set a filter for records of the "name" metric during logging
    '''
    def filter(record):
        return record['extra'].get('key_name') == name
    return filter


class Evaluator(object):
    '''the class for evaluation the current model's effect on certain test datasets
    '''
    def __init__(self, exp_cfg, distributed=False):
        super(Evaluator, self).__init__()
        # distribute the model on different devices or not (default: False)
        self.distributed = distributed 
        # transparency of visualization, defined in config.yaml
        self.alpha_blend = exp_cfg.get('alpha_blend', 0.7)
        
        # 14 joints regressor's stored path, to regress the predicted / ground truth vertices to L14 joints
        j14_regressor_path = exp_cfg.j14_regressor_path
        with open(j14_regressor_path, 'rb') as f:
            self.J14_regressor = pickle.load(f, encoding='latin1')
        
        # add: SPIN J14 regressor, to regress SMPL (not SMPLX) vertices to L14 joints
        # only used in evaluation on the 3DPW testset
        j14_regressor_path = 'data/J_regressor_h36m.npy'
        self.J14_regressor_SMPL = np.load(j14_regressor_path, encoding='latin1',allow_pickle=True)

        # read body vertex ids in the whole vertices, for computing the body_v2v_error
        body_vertex_ids_path = osp.expandvars(
            exp_cfg.get('body_vertex_ids_path', ''))
        body_vertex_ids = None
        if osp.exists(body_vertex_ids_path):
            body_vertex_ids = np.load(body_vertex_ids_path).astype(np.int32)
        self.body_vertex_ids = body_vertex_ids

        # read face vertex ids in the whole vertices, for computing the face_v2v_error
        face_vertex_ids_path = osp.expandvars(
            exp_cfg.get('face_vertex_ids_path', ''))
        face_vertex_ids = None
        if osp.exists(face_vertex_ids_path):
            face_vertex_ids = np.load(face_vertex_ids_path).astype(np.int32)#(5023,)
        self.face_vertex_ids = face_vertex_ids

        # read hand vertex ids in the whole vertices, for computing the left/right_hand_v2v_error
        hand_vertex_ids_path = osp.expandvars(
            exp_cfg.get('hand_vertex_ids_path', ''))
        left_hand_vertex_ids, right_hand_vertex_ids = None, None
        if osp.exists(hand_vertex_ids_path):
            with open(hand_vertex_ids_path, 'rb') as f:
                vertex_idxs_data = pickle.load(f, encoding='latin1')
            left_hand_vertex_ids = vertex_idxs_data['left_hand']#(778,)
            right_hand_vertex_ids = vertex_idxs_data['right_hand']#(778,)
        self.left_hand_vertex_ids = left_hand_vertex_ids
        self.right_hand_vertex_ids = right_hand_vertex_ids

        # if body_vertex_ids not exist, body PA-V2V error cannot be computed, so generate it according to hand/face ids (downloaded from github)
        if self.body_vertex_ids is None:
            hand_face_ids = list(self.face_vertex_ids) + list(self.left_hand_vertex_ids) + list(self.right_hand_vertex_ids)
            body_vertex_ids = [i for i in range(10475) if i not in hand_face_ids]
            self.body_vertex_ids = np.array(body_vertex_ids)
            save_body_vertex_ids_path = 'data/body_vertex_ids.npy'
            # save the results for ensuing use
            if not osp.exists(save_body_vertex_ids_path):
                np.save(save_body_vertex_ids_path, self.body_vertex_ids)

        # the number of imgs showed per row in the evaluation summary
        self.imgs_per_row = exp_cfg.get('imgs_per_row', 2)
        self.exp_cfg = exp_cfg.clone()

        # the output folder of the evaluation summary
        self.output_folder = osp.expandvars(exp_cfg.output_folder)
        self.summary_folder = osp.join(self.output_folder,
                                       exp_cfg.summary_folder)
        os.makedirs(self.summary_folder, exist_ok=True)
        self.summary_steps = exp_cfg.summary_steps

        # the output folder of logs
        self.results_folder = osp.join(self.output_folder,
                                       exp_cfg.results_folder)
        os.makedirs(self.results_folder, exist_ok=True)
        self.loggers = defaultdict(lambda: None)

        # the degrees of visualization imgs saved in summary
        self.body_degrees = exp_cfg.get('degrees', {}).get(
            'body', [90, 180, 270])

        self.body_alignments = {'procrustes': ProcrustesAlignmentMPJPE(),
                                'pelvis': PelvisAlignmentMPJPE(),
                                'root':RootAlignmentMPJPE(), # not used in expose's experiments
                                }
        '''hand_fscores_thresh = exp_cfg.get('fscores_thresh', {}).get(
            'hand', [5.0 / 1000, 15.0 / 1000])
        self.hand_fscores_thresh = hand_fscores_thresh

        self.hand_alignments = {
            'procrustes': ProcrustesAlignmentMPJPE(
                fscore_thresholds=hand_fscores_thresh),
        }
        head_fscores_thresh = exp_cfg.get('fscores_thresh', {}).get(
            'head', [5.0 / 1000, 15.0 / 1000])
        self.head_fscores_thresh = head_fscores_thresh
        self.head_alignments = {
            'procrustes': ProcrustesAlignmentMPJPE(
                fscore_thresholds=head_fscores_thresh)}'''

        # self.plot_conf_thresh = exp_cfg.plot_conf_thresh # 0.3 in config.yaml, but not be used in the following code
        # get the idxs of each part in the whole body vertices
        idxs_dict = get_part_idxs()
        self.body_idxs = idxs_dict['body']
        self.hand_idxs = idxs_dict['hand']
        self.left_hand_idxs = idxs_dict['left_hand']
        self.right_hand_idxs = idxs_dict['right_hand']
        #self.flame_idxs = idxs_dict['flame']

        # used in data preprocessing, so need to transform imgs back to normal according to them as well
        self.means = np.array(self.exp_cfg.datasets.body.transforms.mean)
        self.std = np.array(self.exp_cfg.datasets.body.transforms.std)

        # renderer for each part (show predicted vertices on the raw imgs)
        body_crop_size = exp_cfg.get('datasets', {}).get('body', {}).get(
            'crop_size', 256)
        self.body_renderer = OverlayRenderer(img_size=body_crop_size)
        '''
        hand_crop_size = exp_cfg.get('datasets', {}).get('hand', {}).get(
            'crop_size', 256)
        self.hand_renderer = OverlayRenderer(img_size=hand_crop_size)

        head_crop_size = exp_cfg.get('datasets', {}).get('head', {}).get(
            'crop_size', 256)
        self.head_renderer = OverlayRenderer(img_size=head_crop_size)'''
        
        # ground truth renderer 
        self.render_gt_meshes = exp_cfg.get('render_gt_meshes', True)
        if self.render_gt_meshes:
            self.gt_body_renderer = GTRenderer(img_size=body_crop_size)
            #self.gt_hand_renderer = GTRenderer(img_size=hand_crop_size)
            #self.gt_head_renderer = GTRenderer(img_size=head_crop_size)

    @torch.no_grad()
    def __enter__(self):
        ''' save summary by the SummaryWriter '''
        self.filewriter = SummaryWriter(self.summary_folder, max_queue=1)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        ''' close the filewriter after evaluation '''
        self.filewriter.close()

    def create_summaries(self, step, dset_name, images, targets,
                         model_output, camera_parameters,
                         renderer=None, gt_renderer=None,
                         degrees=None, prefix=''):
        if not hasattr(self, 'filewriter'):
            return
        if degrees is None:
            degrees = []

        crop_size = images.shape[-1]
        
        # transform images back to normal according to the preprocessing params
        imgs = (images * self.std[np.newaxis, :, np.newaxis, np.newaxis] +
                   self.means[np.newaxis, :, np.newaxis, np.newaxis])
        
        # contents to save in summary (the same as mytrain.py)
        summary_imgs = OrderedDict()
        summary_imgs['rgb'] = imgs # the first column to be raw imgs

        # ground truth 2d keypoints imgs
        gt_keyp_imgs = []
        for img_idx in range(imgs.shape[0]):
            input_img = np.ascontiguousarray(
                np.transpose(imgs[img_idx], [1, 2, 0]))
            gt_keyp2d = targets[img_idx].smplx_keypoints.detach(
            ).cpu().numpy()
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
        proj_joints = model_output.get('proj_joints', None)
        if proj_joints is not None:
            proj_points = model_output[
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

        # render ground truth meshes on imgs
        render_gt_meshes = (self.render_gt_meshes and
                            any([t.has_field('vertices') for t in targets]))
        if render_gt_meshes:
            gt_mesh_imgs = []
            faces = model_output['faces'] # not human's face, but the facets of meshes
            for bidx, t in enumerate(targets):
                if not (t.has_field('vertices') and t.has_field('intrinsics')):
                    gt_mesh_imgs.append(np.zeros_like(imgs[bidx]))
                    continue

                curr_gt_vertices = t.get_field(
                    'vertices').vertices.detach().cpu().numpy().squeeze()
                intrinsics = t.get_field('intrinsics') # intrinsic camera parameters

                mesh_img = gt_renderer(
                    curr_gt_vertices[np.newaxis], faces=faces,
                    intrinsics=intrinsics[np.newaxis],
                    bg_imgs=imgs[[bidx]])
                
                gt_mesh_imgs.append(mesh_img.squeeze())

            gt_mesh_imgs = np.stack(gt_mesh_imgs)
            B, C, H, W = gt_mesh_imgs.shape
            row_pad = (crop_size - H) // 2
            gt_mesh_imgs = np.pad(
                gt_mesh_imgs,
                [[0, 0], [0, 0], [row_pad, row_pad], [row_pad, row_pad]])
            summary_imgs['gt_meshes'] = gt_mesh_imgs

        # render predicted meshes on imgs
        vertices = model_output.get('vertices', None)
        if vertices is not None:
            body_imgs = []

            camera_scale = camera_parameters.scale.detach()
            camera_transl = camera_parameters.translation.detach()

            vertices = vertices.detach().cpu().numpy()
            faces = model_output['faces']
            body_imgs = renderer(
                vertices, faces,
                camera_scale, camera_transl,
                bg_imgs=imgs,
                return_with_alpha=False,
            )
            summary_imgs['overlay'] = body_imgs.copy()

            # add rendered images in different digrees
            for deg in degrees:
                body_imgs = renderer(
                    vertices, faces,
                    camera_scale, camera_transl,
                    deg=deg,
                    return_with_alpha=False,
                )
                summary_imgs[f'{deg:03d}'] = body_imgs.copy()

        # save summary_imgs
        summary_imgs = np.concatenate(
            list(summary_imgs.values()), axis=3)
        img_grid = make_grid(
            torch.from_numpy(summary_imgs), nrow=self.imgs_per_row)
        img_tab_name = (f'{dset_name}/{prefix}/Images' if len(prefix) > 0 else
                        f'{dset_name}/Images')
        self.filewriter.add_image(img_tab_name, img_grid, step)
        return

    def build_metric_logger(self, name):
        '''build a recorder of the "name" metric, savve results to output_fn
        '''
        output_fn = osp.join(
            self.results_folder, name + '.log')
        if self.loggers[name] is None:
            logger.add(output_fn, filter=make_filter(name))
            self.loggers[name] = logger.bind(key_name=name)

    def compute_mpjpe(self, model_joints, targets,
                      alignments,
                      gt_joint_idxs=None,
                      joint_idxs=None):
        '''compute the mpjpg metric for evaluation
            input: 
                model_joints: the predicted 3d keypoints from out model
                targets: ground truth data
                alignments: the alignment methods used to compute the mpjpe metric
                gt_joint_idxs: the idxs of ground truth keypoints needed in mpjpe computation
                joint_idxs: the idxs of predicted keypoints needed in mpjpe computation
        '''  
        # ground truth 3d keypoints
        gt_keyps = [target.get_field(
            'keypoints3d'). smplx_keypoints.detach().cpu().numpy()
            for target in targets
            if target.has_field('keypoints3d')]
        gt_conf = [target.get_field('keypoints3d').conf.detach().cpu().numpy()
                   for target in targets
                   if target.has_field('keypoints3d')]
        idxs = [idx
                for idx, target in enumerate(targets)
                if target.has_field('keypoints3d')]
        
        if len(gt_keyps) < 1:
            out_array = {
                key: np.zeros(model_joints.shape[:2], dtype=np.float32)
                for key in alignments
            }
            return {'error': defaultdict(lambda: 0.0),
                    'valid': 0, 'array': out_array}
        if model_joints is None:
            return {'error': defaultdict(lambda: 0.0),
                    'valid': 0, 'array': out_array}

        if torch.is_tensor(model_joints):
            model_joints = model_joints.detach().cpu().numpy()
        if joint_idxs is None:
            joint_idxs = np.arange(0, model_joints.shape[1])

        gt_keyps = np.asarray(gt_keyps)
        gt_conf = np.asarray(gt_conf)
        if gt_joint_idxs is not None:
            gt_keyps = gt_keyps[:, gt_joint_idxs]
            gt_conf = gt_conf[:, gt_joint_idxs]
        if joint_idxs is not None:
            model_joints = model_joints[:, joint_idxs]
        num_valid_joints = (gt_conf > 0).sum()
        idxs = np.asarray(idxs)

        # compute the mpjpe value for each alignment method
        mpjpe_err = {}
        for alignment_name, alignment in alignments.items():
            mpjpe_err[alignment_name] = []
            for bidx in range(gt_keyps.shape[0]):
                align_out = alignment(
                    model_joints[bidx, :],
                    gt_keyps[bidx, :])
                mpjpe_err[alignment_name].append(
                    align_out['point'])
            mpjpe_err[alignment_name] = np.stack(mpjpe_err[alignment_name])

        return {
            'valid': num_valid_joints,
            'array': mpjpe_err
        }

    def compute_v2v(self, model_vertices, targets, alignments, vids=None):
        '''compute the v2v error metric for evaluation
            input: 
                model_vertices: the predicted vertices from out model
                targets: ground truth data
                alignments: the alignment methods used to compute the mpjpe metric
                vids: vertex ids needed to computing the v2v error, if None, use all vertexs
        '''  
        if model_vertices is None:
            return {'valid': 0,
                    'fscore': {},
                    'point': {}}

        gt_vertices = [target.get_field('vertices').
                       vertices.detach().cpu().numpy()
                       for target in targets
                       if target.has_field('vertices')]
        if len(gt_vertices) < 1:
            out_array = {
                key: np.zeros(
                    model_vertices.shape[:2], dtype=np.float32)
                for key in alignments
            }
            return {'fscore': {},
                    'valid': 0, 'point': out_array}
        gt_vertices = np.array(gt_vertices)

        if torch.is_tensor(model_vertices):
            model_vertices = model_vertices.detach().cpu().numpy()

        if vids is not None:
            gt_vertices = gt_vertices[:, vids]
            model_vertices = model_vertices[:, vids]

        v2v_err = {}
        fscores = {}
        # compute v2v errors of different alignment methods
        for alignment_name, alignment in alignments.items():
            v2v_err[alignment_name] = []
            fscores[alignment_name] = defaultdict(lambda: [])

            for bidx in range(gt_vertices.shape[0]):
                align_out = alignment(
                    model_vertices[bidx], gt_vertices[bidx])
                v2v_err[alignment_name].append(align_out['point'])
                
                for thresh, val in align_out['fscore'].items():
                    fscores[alignment_name][thresh].append(
                        val['fscore'].copy())

            v2v_err[alignment_name] = np.stack(v2v_err[alignment_name])
            for thresh in fscores[alignment_name]:
                fscores[alignment_name][thresh] = np.stack(
                    fscores[alignment_name][thresh])

        return {'point': v2v_err, 'fscore': fscores}

    def run_body_eval(self, dataloaders, model, step, alignments=None,
                      device=None):
        '''run body evaluation
            input: 
                dataloaders: the predicted vertices from out model
                model: the motion capture model that we trained
                step: the training step of the current model (just to show info in evaluation results)
                alignments: the alignment methods used to compute the mpjpe metric
                device: gpu or cpu
        '''  
        if alignments is None:
            alignments = {'procrustes': ProcrustesAlignmentMPJPE(),
                          #  'root': RootAlignmentMPJPE(),
                          }
        if device is None:
            device = torch.device('cpu')

        for dataloader in dataloaders:
            dset = dataloader.dataset

            dset_name = dset.name()
            dset_metrics = dset.metrics

            compute_body_mpjpe = 'body_mpjpe' in dset_metrics
            if compute_body_mpjpe:
                body_valid = 0
                body_mpjpe_err = {
                    alignment_name: [] for alignment_name in alignments}
                self.build_metric_logger(f'{dset_name}_body_mpjpe')
            
            # not computed in expose's experiments
            compute_hand_mpjpe = 'hand_mpjpe' in dset_metrics
            if compute_hand_mpjpe:
                left_hand_valid = 0
                left_hand_mpjpe_err = {
                    alignment_name: [] for alignment_name in alignments}

                right_hand_valid = 0
                right_hand_mpjpe_err = {
                    alignment_name: [] for alignment_name in alignments}
                self.build_metric_logger(f'{dset_name}_left_hand_mpjpe')
                self.build_metric_logger(f'{dset_name}_right_hand_mpjpe')

            compute_head_mpjpe = 'head_mpjpe' in dset_metrics
            if compute_head_mpjpe:
                head_valid = 0
                head_mpjpe_err = {
                    alignment_name: [] for alignment_name in alignments}
                self.build_metric_logger(f'{dset_name}_head_mpjpe')
            
            # only be computed when evaluating 3DPW
            compute_mpjpe14 = 'mpjpe14' in dset_metrics
            if compute_mpjpe14:
                mpjpe14_err = {
                    alignment_name: [] for alignment_name in alignments}
                self.build_metric_logger(f'{dset_name}_mpjpe14')
            
            # be computed when evaluating EHF
            compute_v2v = 'v2v' in dset_metrics
            if compute_v2v:
                v2v_err = {key: [] for key in alignments}
                self.build_metric_logger(f'{dset_name}_v2v')

                body_v2v_err = {key: [] for key in alignments}
                left_hand_v2v_err = {key: [] for key in alignments}
                right_hand_v2v_err = {key: [] for key in alignments}
                face_v2v_err = {key: [] for key in alignments}

            if not any([compute_mpjpe14, compute_body_mpjpe, compute_v2v]):
                continue

            desc = f'Evaluating dataset: {dset_name}'

            for idx, batch in enumerate(
                    tqdm.tqdm(dataloader, desc=desc, dynamic_ncols=True)):

                full_imgs_list, body_imgs, body_targets = batch
                full_imgs = to_image_list(full_imgs_list)

                hand_imgs, hand_targets = None, None # if None, the hand/head imgs will be cropped according to the model's predicted keypoints
                head_imgs, head_targets = None, None

                if full_imgs is not None:
                    full_imgs = full_imgs.to(device=device)
                body_imgs = body_imgs.to(device=device)
                body_targets = [target.to(device) for target in body_targets]

                model_output = model(
                    body_imgs, body_targets,
                    hand_imgs=hand_imgs, hand_targets=hand_targets,
                    head_imgs=head_imgs, head_targets=head_targets,
                    full_imgs=full_imgs,
                    device=device)

                body_vertices = None
                body_output = model_output.get('body')
                # get the output of the model's final stage
                body_stage_n_out = body_output.get('final', {})
                if body_stage_n_out is not None:
                    body_vertices = body_stage_n_out.get('vertices', None)
                    body_joints = body_stage_n_out.get('joints', None)
                if body_vertices is None:
                    num_stages = body_output.get('num_stages', 1)
                    body_stage_n_out = body_output.get(
                        f'stage_{num_stages - 1:02d}', {})
                    if body_stage_n_out is not None:
                        body_vertices = body_stage_n_out.get('vertices', None)
                        body_joints = body_stage_n_out.get('joints', None)

                out_params = {}
                for key, val in body_stage_n_out.items():
                    if not torch.is_tensor(val):
                        #logger.info('not tensor output: k: {}, v shape: {}',key,val.shape)
                        continue
                    out_params[key] = val.detach().cpu().numpy()

                if compute_body_mpjpe:
                    body_mpjpe_out = self.compute_mpjpe(
                        body_joints, body_targets,
                        gt_joint_idxs=self.body_idxs,
                        joint_idxs=self.body_idxs,
                        alignments=alignments)
                    body_valid += body_mpjpe_out['valid']

                    computed_errors = body_mpjpe_out['array']
                    for alignment_name, val in computed_errors.items():
                        # pelvis alignment is not suitable for the mpjpe computation
                        if alignment_name == 'pelvis':
                            continue
                        body_mpjpe_err[alignment_name].append(
                            val)

                if compute_head_mpjpe:
                    head_mpjpe_out = self.compute_mpjpe(
                        body_joints, head_targets,
                        gt_joint_idxs=self.head_idxs,
                        joint_idxs=self.head_idxs,
                        alignments=alignments)
                    head_valid += head_mpjpe_out['valid']

                    computed_errors = head_mpjpe_out['array']
                    for alignment_name, val in computed_errors.items():
                        if alignment_name == 'pelvis':
                            continue
                        head_mpjpe_err[alignment_name].append(val)

                if compute_hand_mpjpe:
                    left_hand_mpjpe_out = self.compute_mpjpe(
                        body_joints, body_targets,
                        gt_joint_idxs=self.left_hand_idxs,
                        joint_idxs=self.left_hand_idxs,
                        alignments=alignments)
                    left_hand_valid += left_hand_mpjpe_out['valid']

                    computed_errors = left_hand_mpjpe_out['array']
                    for alignment_name, val in computed_errors.items():
                        if alignment_name == 'pelvis':
                            continue
                        left_hand_mpjpe_err[alignment_name].append(val)

                    right_hand_mpjpe_out = self.compute_mpjpe(
                        body_joints, body_targets,
                        gt_joint_idxs=self.right_hand_idxs,
                        joint_idxs=self.right_hand_idxs,
                        alignments=alignments)
                    right_hand_valid += right_hand_mpjpe_out['valid']

                    computed_errors = right_hand_mpjpe_out['array']
                    for alignment_name, val in computed_errors.items():
                        if alignment_name == 'pelvis':
                            continue
                        right_hand_mpjpe_err[alignment_name].append(val)

                # compute v2v error metrics for experiments
                if compute_v2v:
                    v2v_output = self.compute_v2v(
                        body_vertices, body_targets, alignments)
                    for alignment_name, val in v2v_output['point'].items():
                        if alignment_name == 'pelvis':
                            continue
                        v2v_err[alignment_name].append(val)
                    # compute body specific v2v error
                    if self.body_vertex_ids is not None:
                        body_v2v_output = self.compute_v2v(
                            body_vertices, body_targets,
                            alignments, vids=self.body_vertex_ids
                        )
                        for alignment_name, val in body_v2v_output['point'].items():
                            if alignment_name == 'pelvis':
                                continue
                            body_v2v_err[alignment_name].append(val)
                    # compute hands specific v2v error
                    if self.left_hand_vertex_ids is not None:
                        left_hand_v2v_output = self.compute_v2v(
                            body_vertices, body_targets,
                            alignments, vids=self.left_hand_vertex_ids
                        )
                        iterator = left_hand_v2v_output['point'].items()
                        for alignment_name, val in iterator:
                            if alignment_name == 'pelvis':
                                continue
                            left_hand_v2v_err[alignment_name].append(val)
                    if self.right_hand_vertex_ids is not None:
                        right_hand_v2v_output = self.compute_v2v(
                            body_vertices, body_targets,
                            alignments, vids=self.right_hand_vertex_ids
                        )
                        iterator = right_hand_v2v_output['point'].items()
                        for alignment_name, val in iterator:
                            if alignment_name == 'pelvis':
                                continue
                            right_hand_v2v_err[alignment_name].append(val)
                    # compute face specific v2v error
                    if self.face_vertex_ids is not None:
                        face_v2v_output = self.compute_v2v(
                            body_vertices, body_targets,
                            alignments, vids=self.face_vertex_ids
                        )
                        for alignment_name, val in face_v2v_output['point'].items():
                            if alignment_name == 'pelvis':
                                continue
                            face_v2v_err[alignment_name].append(val)
                
                # compute the mpjpe14 metric (only used in evaluation on 3DPW)
                if compute_mpjpe14 and body_vertices is not None:
                    
                    # gt_joints14 = [target.get_field('joints14').
                    #                joints.detach().cpu().numpy()
                    #                for target in body_targets
                    #                if target.has_field('joints14')]
                    # if len(gt_joints14) > 0:
                    #     gt_joints14 = np.asarray(gt_joints14)

                    # get the gt_joints14 by regress gt_vertices to the 14 joints
                    gt_vertices = [target.get_field('vertices').
                                vertices.detach().cpu().numpy()
                                for target in body_targets
                                if target.has_field('vertices')]
                    gt_vertices = np.array(gt_vertices) # (32, 10475, 3)
                    if len(gt_vertices) > 0:
                    
                        if torch.is_tensor(body_vertices):
                            body_vertices = body_vertices.detach(
                            ).cpu().numpy()

                        pred_joints = np.einsum(
                            'jv,bvm->bjm', self.J14_regressor, body_vertices)
                        
                        # when evaluating EHF, quit this commentting
                        # gt_joints14 = np.einsum(
                            # 'jv,bvm->bjm', self.J14_regressor, gt_vertices) 
                        # when evaluating 3DPW, quit this commentting
                        gt_joints14 = np.einsum(
                            'jv,bvm->bjm', self.J14_regressor_SMPL, gt_vertices)[:,H36M_TO_J14,:] # get 14 joints from 17 H36m joints
                        #error = np.sqrt(((pred_joints - gt_joints14) ** 2).sum(axis=-1)).mean(axis=-1)
                        #logger.info('mpjpe from spin: {}',error)

                        for alignment_name, alignment in alignments.items():
                            for bidx in range(gt_joints14.shape[0]):
                                mpjpe14_err[alignment_name].append(
                                    alignment(
                                        pred_joints[bidx],
                                        gt_joints14[bidx])['point'])
                ''' # save visualizations to summary
                if idx == 0:
                    camera_parameters = body_output.get('camera_parameters')
                    self.create_summaries(
                        step, dset_name,
                        body_imgs.detach().cpu().numpy(),
                        body_targets,
                        body_stage_n_out,
                        camera_parameters=camera_parameters,
                        renderer=self.body_renderer,
                        gt_renderer=self.gt_body_renderer,
                        degrees=self.body_degrees,
                    )'''

            # Compute Body Mean per Joint point error
            if compute_body_mpjpe:
                for key, val in body_mpjpe_err.items():
                    val = np.concatenate(val)
                    logger.info(f'{key}: {val.shape}')
                    # Compute the mean over the dataset and convert to
                    # millimeters
                    logger.info(f'body valid: {body_valid}')
                    metric_value = val.sum() / body_valid * 1000
                    alignment_name = key.title()

                    # Store the Procrustes aligned MPJPE
                    self.loggers[f'{dset_name}_body_mpjpe'].info(
                        '[{:06d}] {}: {}  3D Keypoint error: {:.4f} mm',
                        step, dset_name,
                        alignment_name,
                        metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/MPJPE'
                    self.filewriter.add_scalar(
                        metric_name, metric_value, step)

            # Compute Hand Mean per Joint point error
            if compute_hand_mpjpe:
                for key, val in left_hand_mpjpe_err.items():
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    # Compute the mean over the dataset and convert to
                    # millimeters
                    metric_value = val.sum() / left_hand_valid * 1000
                    alignment_name = key.title()
                    # Store the Procrustes aligned MPJPE
                    #  self.loggers[f'{dset_name}_hand_mpjpe'].info(
                    logger.info(
                        '[{:06d}] {}: {} 3D Left Hand Keypoint error: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)
                    metric_name = f'{dset_name}/{alignment_name}/LeftHand'
                    self.filewriter.add_scalar(
                        metric_name, metric_value, step)
                for key, val in right_hand_mpjpe_err.items():
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    # Compute the mean over the dataset and convert to
                    # millimeters
                    metric_value = val.sum() / right_hand_valid * 1000
                    alignment_name = key.title()
                    # Store the Procrustes aligned MPJPE
                    #  self.loggers[f'{dset_name}_hand_mpjpe'].info(
                    logger.info(
                        '[{:06d}] {}: {} 3D Right Hand Keypoint error: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)
                    metric_name = f'{dset_name}/{alignment_name}/RightHand'
                    self.filewriter.add_scalar(
                        metric_name, metric_value, step)

            # Compute Head Mean per Joint point error
            if compute_head_mpjpe:
                for key, val in head_mpjpe_err.items():
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    metric_value = val.sum() / head_valid * 1000
                    alignment_name = key.title()

                    # Store the Procrustes aligned MPJPE
                    self.loggers[f'{dset_name}_head_mpjpe'].info(
                        '[{:06d}] {}: {}  3D Head Keypoint error: {:.4f} mm',
                        step,
                        dset_name,
                        alignment_name,
                        metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/Head'
                    self.filewriter.add_scalar(metric_name, metric_value, step)

            # Compute Mean per Joint point error
            if compute_mpjpe14:
                for key, val in mpjpe14_err.items():
                    if len(val) < 1:
                        continue
                    val = np.asarray(val)
                    metric_value = np.mean(val) * 1000
                    alignment_name = key.title()

                    # Store the Procrustes aligned MPJPE
                    self.loggers[f'{dset_name}_mpjpe14'].info(
                        '[{:06d}] {}: {}  MPJPE: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/MPJPE'
                    self.filewriter.add_scalar(metric_name, metric_value, step)

            if compute_v2v:
                summary_dict = {}
                for key, val in v2v_err.items():
                    # Divide by the number of items in the dataset and the
                    # number of vertices
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    metric_value = np.mean(val) * 1000 # convert to millemeter
                    alignment_name = key.title()

                    self.loggers[f'{dset_name}_v2v'].info(
                        '[{:06d}] {}: Vertex-To-Vertex/{}: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/V2V'
                    summary_dict[metric_name] = val
                    self.filewriter.add_scalar(metric_name, metric_value, step)
                for key, val in body_v2v_err.items():
                    # Divide by the number of items in the dataset and the
                    # number of vertices
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    metric_value = np.mean(val) * 1000
                    alignment_name = key.title()

                    self.loggers[f'{dset_name}_v2v'].info(
                        '[{:06d}] {}: Body Vertex-To-Vertex/{}: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/BodyV2V'
                    summary_dict[metric_name] = val
                    self.filewriter.add_scalar(metric_name, metric_value, step)
                for key, val in left_hand_v2v_err.items():
                    # Divide by the number of items in the dataset and the
                    # number of vertices
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    metric_value = np.mean(val) * 1000
                    alignment_name = key.title()

                    self.loggers[f'{dset_name}_v2v'].info(
                        '[{:06d}] {}: Left Hand Vertex-To-Vertex/{}: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/LeftHandV2V'
                    summary_dict[metric_name] = val
                    self.filewriter.add_scalar(metric_name, metric_value, step)

                for key, val in right_hand_v2v_err.items():
                    # Divide by the number of items in the dataset and the
                    # number of vertices
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    metric_value = np.mean(val) * 1000
                    alignment_name = key.title()

                    self.loggers[f'{dset_name}_v2v'].info(
                        '[{:06d}] {}: Right Hand Vertex-To-Vertex/{}: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/RightHandV2V'
                    summary_dict[metric_name] = val
                    self.filewriter.add_scalar(metric_name, metric_value, step)

                for key, val in face_v2v_err.items():
                    # Divide by the number of items in the dataset and the
                    # number of vertices
                    if len(val) < 1:
                        continue
                    val = np.concatenate(val, axis=0)
                    metric_value = np.mean(val) * 1000
                    alignment_name = key.title()

                    self.loggers[f'{dset_name}_v2v'].info(
                        '[{:06d}] {}: Face Vertex-To-Vertex/{}: {:.4f} mm',
                        step, dset_name, alignment_name, metric_value)

                    metric_name = f'{dset_name}/{alignment_name}/FaceV2V'
                    summary_dict[metric_name] = val
                    self.filewriter.add_scalar(metric_name, metric_value, step)

        return

    @torch.no_grad()
    def run(self, model, dataloaders, device, step=0):
        '''run evaluation
            input: 
                dataloaders: the evaluating dataloader
                model: the motion capture model that we trained
                step: the training step of the current model (just to show info in evaluation results)
                device: gpu or cpu
        '''  
        model.eval()
        assert not (model.training), 'Model is in training mode!'

        # whole body dataloader (hand/head specific datasets' evaluation code was deleted, because they don't support evaluation on testsets by ourselves, submitting to websites/emails required)
        body_dloader = dataloaders.get('body', None)

        if self.distributed:
            eval_model = deepcopy(model.module)
        else:
            eval_model = deepcopy(model)

        eval_model.eval()
        assert not (eval_model.training), 'Model is in training mode!'
        if body_dloader is not None:
            self.run_body_eval(body_dloader, eval_model,
                               alignments=self.body_alignments,
                               step=step, device=device)
        