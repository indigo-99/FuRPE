# -*- coding: utf-8 -*-
import sys
import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
import torch.utils.data as dutils
import numpy as np
import cv2
import yaml
import json

from ..targets.keypoints import (Keypoints2D, Keypoints3D,
                                 get_part_idxs,
                                 dset_to_body_model)
from ..targets import Vertices, Joints
from ..utils.bbox import keyps_to_bbox, bbox_to_center_scale
from ..utils import read_keypoints
from ...utils.img_utils import read_img
from loguru import logger

from plyfile import PlyData
#add for pixie
from skimage.transform import estimate_transform, warp
from ..utils import detectors#copy pixielib/detectors to expose/data/utils

class EHF(dutils.Dataset):

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 alignments_folder='alignments',
                 num_betas=10, num_expr_coeffs=10,
                 use_face_contour=False,
                 dtype=torch.float32,
                 transforms=None,
                 split='train',
                 keyp_format='coco25',
                 metrics=None,
                 use_joint_conf=True,
                 head_only=False,
                 hand_only=False,
                 is_right=True,
                 binarization=True,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 **kwargs):
        super(EHF, self).__init__()
        if metrics is None:
            metrics = ['v2v']
        self.metrics = metrics

        self.dtype = dtype
        self.data_folder = osp.expandvars(data_folder)
        self.img_folder = img_folder
        self.keyp_folder = keyp_folder
        self.alignments_folder = alignments_folder
        self.use_joint_conf = use_joint_conf
        self.use_face_contour=use_face_contour
        #  keypoint_fname = osp.join(self.data_folder, 'gt_keyps.npy')
        #keypoint_fname = osp.join(self.data_folder, 'gt_keyps.npz')
        #keypoint_data = np.load(keypoint_fname)
        #self.keypoints = keypoint_data['gt_keypoints_2d']
        #self.keypoints3d = keypoint_data['gt_keypoints_3d']
        #self.joints14 = keypoint_data['gt_joints14']
        #if not use_face_contour:
            #self.keypoints = self.keypoints[:, :-17]

        self.is_train = 'train' in split
        self.split = split
        self.keyp_format = keyp_format
        self.is_right = is_right
        self.head_only = head_only
        self.hand_only = hand_only
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization

        #annot_fn = osp.join(self.data_folder, 'annotations.yaml')
        #with open(annot_fn, 'r') as annot_file:
            #annotations = yaml.load(annot_file)
        #self.annotations = annotations
        #self.annotations = (self.annotations['train'] +
        #                    self.annotations['test'])
        #P add:
        self.annotations=[]
        for i in range(1,101):
            if i < 10:
                self.annotations.append('0'+str(i)+'_img')
            else:
                self.annotations.append(str(i)+'_img')

        self.transforms = transforms

        self.num_betas = num_betas
        self.num_expr_coeffs = num_expr_coeffs
        self.use_face_contour = use_face_contour

        self.img_fns = sorted(
            os.listdir(osp.join(self.data_folder, self.img_folder)))
        
        # filename_cont_path='/data/panyuqing/expose_experts/data/curated_fits/EHF_comp_img_fns.npy'
        # tmp=np.load(filename_cont_path)
        # self.img_fns = sorted(
        #     os.listdir('/data/panyuqing/expose_experts/data/params3d_v/EHF'))

        source_idxs, target_idxs = dset_to_body_model(
             dset='openpose25+hands+face',
             model_type='smplx', use_hands=True, use_face=True,
             use_face_contour=self.use_face_contour,
             keyp_format=self.keyp_format)

        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)
        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)

        self.body_dset_factor = 1.2
        self.head_dset_factor = 2.0
        self.hand_dset_factor = 2.0

        #add detector for pixie test
        self.detector = detectors.FasterRCNN()#device=device)

    def __repr__(self):
        return 'EHF'

    def name(self):
        return 'EHF/Test'

    def get_num_joints(self):
        return 14

    def __len__(self):
        return len(self.img_fns)

    def get_elements_per_index(self):
        return 1

    def __getitem__(self, index):
        fn = self.annotations[index]
        img_path = osp.join(self.data_folder, self.img_folder,
                            fn + '.png')
        img = read_img(img_path)

        _, fn = os.path.split(fn)
        
        #fn:66_img.png -> keypoint_fn:66_2Djnt.json
        keypoint_fn=fn[:-3]+'2Djnt.json'
        #print('fn: ',fn)
        #print('keypoint_fn: ',keypoint_fn)
        keypoint_cont=open(osp.join(self.data_folder,self.keyp_folder,keypoint_fn),encoding='utf-8').read()
        keypoint_dict=json.loads(keypoint_cont)
        tmp=keypoint_dict["people"][0]
        tmpbd=np.array(tmp["pose_keypoints_2d"]).reshape((25,3))
        tmpfc=np.array(tmp["face_keypoints_2d"]).reshape((70,3))
        tmplh=np.array(tmp["hand_left_keypoints_2d"]).reshape((21,3))
        tmprh=np.array(tmp["hand_right_keypoints_2d"]).reshape((21,3))
        keypoints2d=np.concatenate((tmpbd,tmplh,tmprh,tmpfc)) #(137,3)

        # Copy keypoints from the GT data
        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                        3], dtype=np.float32)
        #output_keypoints2d[:, :-1] = self.keypoints[index].copy()
        #output_keypoints2d[:, :-1] = keypoint.copy()
        #output_keypoints2d[:, -1] = 1.0
        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]
        
        # output_keypoints3d = np.zeros(
        #     [127 + 17 * self.use_face_contour, 4], dtype=np.float32)
        # output_keypoints3d[:, :-1] = self.keypoints3d[index].copy()
        # output_keypoints3d[:, -1] = 1.0

        is_right = self.is_right
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
        keypoints = output_keypoints2d[:, :-1]
        conf = output_keypoints2d[:, -1]
        if self.head_only:
            dset_scale_factor = self.head_dset_factor
        elif self.hand_only:
            dset_scale_factor = self.hand_dset_factor
        else:
            dset_scale_factor = self.body_dset_factor

        center, scale, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.size),
            dset_scale_factor=dset_scale_factor,
        )
        if center is None:
            return None, None, None, None

        if self.hand_only:
            target.add_field('is_right', is_right)
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('keypoints_hd', output_keypoints2d)

        # target.add_field(
        #     'keypoints3d',
        #     Keypoints3D(output_keypoints3d, img.shape, flip_axis=0)
        # )

        orig_center, _, orig_bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.size),
            dset_scale_factor=1.0,
        )
        target.add_field('orig_center', orig_center)
        target.add_field('orig_bbox_size', bbox_size)

        # alignment_path = osp.join(self.data_folder, self.alignments_folder,
        #                           fn.replace('.07_C', '') + '.pkl')
        # with open(alignment_path, 'rb') as alignment_file:
        #     alignment_data = pickle.load(alignment_file, encoding='latin1')
        
        alignment_path = osp.join(self.data_folder, self.alignments_folder,fn[:-3]+'align.ply')
        with open(alignment_path,'rb') as f:
            alignment_data = PlyData.read(f)
            #vertices = alignment_data['v']
            vx=np.array(alignment_data['vertex'].data['x'])
            vy=np.array(alignment_data['vertex'].data['y'])
            vz=np.array(alignment_data['vertex'].data['z'])
            vx=vx.reshape((vx.shape[0],1))
            vy=vy.reshape((vy.shape[0],1))
            vz=vz.reshape((vz.shape[0],1))
            vertices=np.concatenate((vx,vy,vz),axis=1)
        del alignment_data
        del vx
        del vy
        del vz
        transl = np.array([-0.03609917, 0.43416458, 2.37101226])
        camera_pose = np.array([-2.9874789618512025, 0.011724572107320893,
                                -0.05704686818955933])
        camera_pose = cv2.Rodrigues(camera_pose)[0]

        
        cam_vertices = vertices.dot(camera_pose.T) + transl.reshape(1, 3)
        #print('vertices shape: ',vertices.shape)#(10475, 3)
        #print('cam_vertices shape: ',cam_vertices.shape)#(10475, 3)
        vertices_field = Vertices(cam_vertices)
        target.add_field('vertices', vertices_field)

        H, W, _ = img.shape
        intrinsics = np.array([[1498.22426237, 0, 790.263706],
                               [0, 1498.22426237, 578.90334],
                               [0, 0, 1]], dtype=np.float32)
        target.add_field('intrinsics', intrinsics)
        target.add_field('fname', fn)
        # if self.transforms is not None:
        #     force_flip = False
        #     if self.hand_only and not is_right:
        #         force_flip = True
        #     img, cropped_image, target = self.transforms(
        #         img, target, dset_scale_factor=1.2, force_flip=force_flip)
        
        ###change to pixie dataloader in body_datasets.py Testdata
        img=img/255.
        image_tensor = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)[None, ...]
        bbox = self.detector.run(image_tensor)
        if bbox is None:
            logger.info('no person detected! run original image')
            left = 0; right = W-1; top=0; bottom=H-1
        else:
            left = bbox[0]; right = bbox[2]; top = bbox[1]; bottom = bbox[3]
        old_size = max(right - left, bottom - top)
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        pixie_scale=1.1
        size = int(old_size*pixie_scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        
        # left = 0; right = W-1; top=0; bottom=H-1
        # bbox = [left, top, right, bottom]
        # src_pts = np.array([[0, 0], [0, H-1], [W-1, 0]])
        
        # crop image
        crop_size=224 # same as pixie
        hd_size=1024
        DST_PTS = np.array([[0,0], [0,crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(img, tform.inverse, output_shape=(crop_size, crop_size))
        dst_image = dst_image.transpose(2,0,1)
        # hd image
        DST_PTS = np.array([[0,0], [0,hd_size - 1], [hd_size - 1, 0]])
        tform_hd = estimate_transform('similarity', src_pts, DST_PTS)
        hd_image = warp(img, tform_hd.inverse, output_shape=(hd_size, hd_size))
        hd_image = hd_image.transpose(2,0,1)
        image = torch.tensor(dst_image).float()
        image_hd = torch.tensor(hd_image).float()
        target.to_tensor()
        #return img, cropped_image, target, index
        return image_hd, image, target, index
