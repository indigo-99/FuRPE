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
import os
import os.path as osp

import pickle

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..targets import Keypoints2D, Joints, Vertices

from ..targets.keypoints import dset_to_body_model
from ...utils.img_utils import read_img

#p add:
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import colors
import open3d as o3d
import trimesh

Mesh = o3d.geometry.TriangleMesh
Vector3d = o3d.utility.Vector3dVector
Vector3i = o3d.utility.Vector3iVector

FOLDER_MAP_FNAME = 'folder_map.pkl'

# copy from SPIN/constant.py 
# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

class ThreeDPW(dutils.Dataset):
    def __init__(self, data_path='data/3dpw',
                 img_folder='',
                 seq_folder='sequenceFiles',
                 param_folder='smplx_npz_data',
                 split='val',
                 use_face=True, use_hands=True, use_face_contour=False,
                 model_type='smplx',
                 dtype=torch.float32,
                 vertex_folder='smplx_vertices',
                 return_vertices=True,
                 joints_to_ign=None,
                 use_joint_conf=True,
                 metrics=None,
                 transforms=None,
                 body_thresh=0.3,
                 binarization=True,
                 min_visible=6,
                 **kwargs):
        super(ThreeDPW, self).__init__()

        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.binarization = binarization
        self.return_vertices = return_vertices

        self.split = split
        self.is_train = 'train' in split

        self.data_path = osp.expandvars(osp.expanduser(data_path))
        #seq_path = osp.join(self.data_path, 'imageFiles')#seq_folder)
        if self.split == 'train':
            #seq_split_path = osp.join(seq_path, 'train')
            npz_fn = osp.join(self.data_path, param_folder, '3dpw_train.npz')
        elif self.split == 'val':
            #seq_split_path = osp.join(seq_path, 'validation')
            npz_fn = osp.join(
                self.data_path, param_folder, '3dpw_validation.npz')
        elif self.split == 'test':
            #seq_split_path = osp.join(seq_path, 'test')
            npz_fn = osp.join(self.data_path, param_folder, '3dpw_test.npz')

        self.vertex_folder = osp.join(
            self.data_path, 'smpl_vertices_test')#vertex_folder)#, self.split)

        self.img_folder = osp.join(self.data_path, img_folder)
        folder_map_fname = osp.expandvars(
            osp.join(self.img_folder, split, FOLDER_MAP_FNAME))
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)
            self.items_per_folder = max(data_dict.values())
            self.img_folder = osp.join(self.img_folder, split)

        data_dict = np.load(npz_fn, allow_pickle=True)
        #  data_dict = {key: data[key] for key in data.keys()}

        if 'cam_intrinsics' in data_dict:
            self.cam_intrinsics = data_dict['cam_intrinsics']

        self.img_paths = np.asarray(data_dict['imgname']) #['img_paths'])
        logger.info('length of raw testing images: {}',self.img_paths.shape[0])

        #idxs = [ii for ii, path in enumerate(self.img_paths)
                #if 'downtown_walking_00' in path]
        #idxs = np.array(idxs)
        #logger.info('length of selected testing idxs: {}',idxs.shape[0])
        #logger.info('idxs: {}',idxs)

        idxs = np.arange(len(self.img_paths))
        self.idxs = idxs
        self.img_paths = self.img_paths[idxs]

        if 'keypoints2d' in data_dict:
            self.keypoints2d = np.asarray(
                data_dict['keypoints2d']).astype(np.float32)[idxs]
        elif 'keypoints2D' in data_dict:
            self.keypoints2d = np.asarray(
                data_dict['keypoints2D']).astype(np.float32)[idxs]
        else:
            raise KeyError(f'Keypoints2D not in 3DPW {split} dictionary')
        self.joints3d = np.asarray(
            data_dict['joints3d']).astype(np.float32)[idxs]
        #  self.v_shaped = np.asarray(data_dict['v_shaped']).astype(np.float32)
        self.num_items = len(self.img_paths)
        logger.info('length of selected testing idxs: {}',self.num_items)

        #  self.pids = np.asarray(data_dict['person_ids'], dtype=np.int32)
        self.pids = np.asarray(data_dict['pid'], dtype=np.int32)[idxs]
        self.center = np.asarray(
            data_dict['center'], dtype=np.float32)[idxs]
        self.scale = np.asarray(
            data_dict['scale'], dtype=np.float32)[idxs]
        self.bbox_size = np.asarray(
            data_dict['bbox_size'], dtype=np.float32)[idxs]
        
        # p add:
        self.pose = np.asarray(data_dict['pose'], dtype=np.float32)[idxs]
        self.betas = np.asarray(data_dict['shape'], dtype=np.float32)[idxs]
        #self.trans = np.asarray(data_dict['trans'], dtype=np.float32)[idxs]
        # logger.info('self.img_paths: {}',self.img_paths)
        # logger.info('self.pose: {}',self.pose)
        # logger.info('self.betas: {}',self.betas)
        # logger.info('self.pids: {}',self.pids)

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.model_type = model_type
        self.use_joint_conf = use_joint_conf
        self.body_thresh = body_thresh

        source_idxs, target_idxs = dset_to_body_model(
            dset='3dpw', model_type='smplx',
            use_face_contour=self.use_face_contour)
        self.source_idxs = np.asarray(source_idxs)
        self.target_idxs = np.asarray(target_idxs)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return '3DPW( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return '3DPW/{}'.format(self.split)

    def get_num_joints(self):
        return 14

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def __getitem__(self, index):
                #  start = time.perf_counter()
        img_fn = self.img_paths[index]#data/3DPW/imageFiles/downtown_bus_00/image_00143.jpg
        # logger.info('index: {}',index)
        # logger.info('img_fn: {}',img_fn)
        # logger.info('self.pids[index]: {}',self.pids[index])
        # logger.info('self.pose[index]: {}',self.pose[index])
        # logger.info('self.betas[index]: {}',self.betas[index])
        if self.use_folder_split:
            folder_idx = (index + self.idxs[0]) // self.items_per_folder
            img_fn = osp.join(self.img_folder,
                              'folder_{:010d}'.format(folder_idx),
                              f'{index + self.idxs[0]:010d}.jpg')
        img_fn = osp.join('data/3DPW',img_fn)
        img = read_img(img_fn)
        #  print('read img:', time.perf_counter() - start)

        #keypoints2d = self.keypoints2d[index, :]
        keypoints2d = self.keypoints2d[index, :,:]#(18,3)
        #  print('read data:', time.perf_counter() - start)
        #  start = time.perf_counter()
        #  logger.info('V + J: {}'.format(time.perf_counter() - start))

        #  # Pad to compensate for extra keypoints
        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]

        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            output_keypoints2d[
                output_keypoints2d[:, -1] < self.body_thresh, -1] = 0

        # If we don't want to use the confidence scores as weights for the loss
        if self.binarization:
            # then set those above the conf thresh to 1
            output_keypoints2d[:, -1] = (
                output_keypoints2d[:, -1] >= self.body_thresh).astype(
                output_keypoints2d.dtype)

        center = self.center[index]
        scale = self.scale[index]
        bbox_size = self.bbox_size[index]

        #  keypoints = output_keypoints2d[:, :-1]
        #  conf = output_keypoints2d[:, -1]
        target = Keypoints2D(
            output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)
        target.add_field('center', center)
        target.add_field('orig_center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('keypoints_hd', output_keypoints2d)

        target.add_field('filename', self.img_paths[index])

        head, fname = osp.split(self.img_paths[index])
        _, seq_name = osp.split(head)
        target.add_field('fname', f'{seq_name}/{fname}_{self.pids[index]}')
        #target.add_field('fname', f'{seq_name}/{fname}')
        
        if self.return_vertices:#smplx_vertices/
            vertex_dir=osp.join(self.vertex_folder,img_fn.split('/')[3])#data/3DPW/smplx_vertices/downtown_bus_00
            os.makedirs(vertex_dir, exist_ok=True)
            if 'smpl' in vertex_dir:
                vertex_fname = osp.join(vertex_dir,img_fn.split('/')[4].split('.')[0]+str(self.pids[index])+'.ply')
            elif vertex_dir=='smplx_vertices':
                vertex_fname = osp.join(vertex_dir,img_fn.split('/')[4].split('.')[0]+'.pkl')
            elif vertex_dir=='smplx_vertices_raw':
                vertex_fname = osp.join(vertex_dir,img_fn.split('/')[4].split('.')[0]+'.npy')
            
            #data/3DPW/smplx_vertices/downtown_bus_00/image_00143.ply

            if osp.exists(vertex_fname):
                if 'smpl' in vertex_dir:
                    mesh = trimesh.load(vertex_fname, process=False)
                    vertices=np.asarray(mesh.vertices, dtype=np.float32)
                elif vertex_dir=='smplx_vertices':
                    meshdict = np.load(vertex_fname, allow_pickle=True)
                    vertices=meshdict['vertices'].detach().cpu().numpy()
                elif vertex_dir=='smplx_vertices_raw':
                    vertices = np.load(vertex_fname, allow_pickle=True).detach().cpu().numpy()
            else:
                pose=self.pose[index]
                betas=self.betas[index]
                #trans=self.trans[index]
                #support_dir='/data/panyuqing/amass/support_data'
                #bm_fname=osp.join(support_dir, 'body_models/smplx/SMPLX_NEUTRAL.npz')
                #bm_fname=osp.join(support_dir, 'body_models/smplh/neutral/model.npz')
                bm_fname='/data/panyuqing/SPIN/data/cleansmpl_result/SMPL_NEUTRAL.pkl'
                device=torch.device('cuda')
                #bm=BodyModel(bm_fname=bm_fname, num_betas=10,num_expressions=10).to(device)
                bm=BodyModel(bm_fname=bm_fname, num_betas=10).to(device)
                bdata={'root_orient':pose[:3].reshape(1,3).astype(np.float32),
                    'pose_body':pose[3:66].reshape(1,63).astype(np.float32),
                    'betas':betas.reshape(1,10).astype(np.float32),}
                    #'trans':trans.reshape(1,3).astype(np.float32)}

                modelres = bm.forward(**{k:torch.from_numpy(v).to(device) for k,v in bdata.items() if k in ['root_orient','pose_body', 'betas']})
                vertices=modelres.v
                #joints=modelres.Jtr
                
                #add trans
                # bdata={'pose':pose[:].reshape(1,66).astype(np.float32),
                #     'betas':betas.reshape(1,10).astype(np.float32),
                #     'trans':}
                # bm = load_model(bm_fname)
                # body_model = build_body_model(
                #     model_path=bm_fname,
                #     model_type='SMPL',
                # )

                vertices=vertices.detach().cpu().numpy()
                #np.save(vertex_fname,vertices)
                # with open(vertex_fname, 'wb') as writefile:
                #     pickle.dump(vertices, writefile)
                # mesh = trimesh.Trimesh(vertices=vertices, faces=bm.f.detach().cpu().numpy(),
                #     vertex_colors=np.tile(colors['grey'], (6890, 1)),process=False)
                #open3dmesh = mesh.as_open3d
                open3dmesh=Mesh()
                open3dmesh.vertices = Vector3d(vertices[0])
                open3dmesh.triangles = Vector3i(modelres.f.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(vertex_fname, open3dmesh)

            #img_name='_'.join([img_fn.split('/')[1],img_fn.split('/')[2].split('.')[0]])
            
            vertex_field = Vertices(vertices.reshape(-1, 3))
            target.add_field('vertices', vertex_field)

            intrinsics = self.cam_intrinsics[index]
            target.add_field('intrinsics', intrinsics)
            
            # joints = Joints(joints[:14])
            # target.add_field('joints14', joints)
        
        if not self.is_train:
            # joints3d = self.joints3d[index].reshape(24,3)
            # #joints = Joints(joints3d[:14])
            # joints = Joints(joints3d[J24_TO_J14,:])
            # target.add_field('joints14', joints)

            if hasattr(self, 'v_shaped'):
                v_shaped = self.v_shaped[index]
                target.add_field('v_shaped', Vertices(v_shaped))
        #  print('SMPL-HF Field {}'.format(time.perf_counter() - start))

        #  start = time.perf_counter()
        if self.transforms is not None:
            img, cropped_image, target = self.transforms(
                img, target, dset_scale_factor=1.2, force_flip=False)
        #  logger.info('Transforms: {}'.format(time.perf_counter() - start))

        return img, cropped_image, target, index
