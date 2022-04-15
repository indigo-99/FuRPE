# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from tqdm import tqdm

#import open3d as o3d

import time
import argparse
from collections import defaultdict, OrderedDict
from loguru import logger
import numpy as np

import torch
import json

import resource

from expose.config.cmd_parser import set_face_contour
from expose.config import cfg
from expose.models.smplx_net import SMPLXNet
from expose.data import make_all_data_loaders
from expose.utils.checkpointer import Checkpointer
from expose.data.targets.image_list import to_image_list

from expose.evaluation import Evaluator
from pytorch3d.io import load_obj

from flamemodel.FLAME import FLAME
from yacs.config import CfgNode as CN
flamecfg = CN()
flamecfg.deca_dir='/data/panyuqing/DECA'
flamecfg.topology_path = os.path.join(flamecfg.deca_dir, 'data', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
#flamecfg.dense_template_path = os.path.join(flamecfg.deca_dir, 'data', 'texture_data_256.npy')
#flamecfg.fixed_displacement_path = os.path.join(flamecfg.deca_dir, 'data', 'fixed_displacement_256.npy')
flamecfg.flame_model_path = os.path.join(flamecfg.deca_dir, 'data', 'generic_model.pkl') 
flamecfg.flame_lmk_embedding_path = os.path.join(flamecfg.deca_dir, 'data', 'landmark_embedding.npy') 
#flamecfg.face_mask_path = os.path.join(flamecfg.deca_dir, 'data', 'uv_face_mask.png') 
#flamecfg.face_eye_mask_path = os.path.join(flamecfg.deca_dir, 'data', 'uv_face_eye_mask.png') 
#flamecfg.mean_tex_path = os.path.join(flamecfg.deca_dir, 'data', 'mean_texture.jpg') 
#flamecfg.tex_path = os.path.join(flamecfg.deca_dir, 'data', 'FLAME_albedo_from_BFM.npz') 
#flamecfg.tex_type = 'BFM' # BFM, FLAME, albedoMM
#flamecfg.uv_size = 256
#flamecfg.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
flamecfg.n_shape = 100
#flamecfg.n_tex = 50
flamecfg.n_exp = 50
#flamecfg.n_pose = 6
#flamecfg.use_tex = True
#flamecfg.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

def undo_img_normalization(image, mean, std, add_alpha=True):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().squeeze()

    out_img = (image * std[np.newaxis, :, np.newaxis, np.newaxis] +
               mean[np.newaxis, :, np.newaxis, np.newaxis])
    if add_alpha:
        out_img = np.pad(
            out_img, [[0, 0], [0, 1], [0, 0], [0, 0]],
            mode='constant', constant_values=1.0)
    return out_img

def batch_rot2aa(Rs, epsilon=1e-7):
    #Rs is B x 3 x 3
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

def save_obj(filename, vertices):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    i = 0
    vertices = vertices.cpu().numpy()
    verts_no_use, faces, aux_no_use = load_obj(flamecfg.topology_path)
    faces = faces[0].cpu().numpy()

    if filename.split('.')[-1] != 'obj':
        filename = filename + '.obj'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1

    # write obj
    with open(filename, 'w') as f:
        # write vertices
        for i in range(vertices.shape[0]):
            f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        for i in range(faces.shape[0]):
            f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
    

@torch.no_grad()
def main(
    exp_cfg,
    show=False,
    demo_output_folder='demo_output',
    pause=-1,
    focal_length=5000, sensor_width=36,
    save_vis=True,
    save_params=False,
    save_mesh=False,
    degrees=[],
):

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               colorize=True)

    demo_output_folder = osp.expanduser(osp.expandvars(demo_output_folder))
    logger.info(f'Saving results to: {demo_output_folder}')
    print('saving results to:',demo_output_folder)
    os.makedirs(demo_output_folder, exist_ok=True)

    model = SMPLXNet(exp_cfg)
    try:
        model = model.to(device=device)
    except RuntimeError:
        # Re-submit in case of a device error
        sys.exit(3)

    checkpoint_folder = osp.join(
        exp_cfg.output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder,
                                pretrained=exp_cfg.pretrained)

    arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]

    model = model.eval()
    head_model = model.get_head_model()

    dataloaders = make_all_data_loaders(exp_cfg, split='test')


    #body_dloader = dataloaders['body'][0]
    head_dloader = dataloaders['head'][0]
    
    flame = FLAME(flamecfg).to(device)
    index_7=[36,39,42,45,33,48,54]# extract 7 landmark from 68 3d face keypoints
    # mano = manolayer.ManoLayer(flat_hand_mean=True,
    #                     side="left",#side="right",
    #                     mano_root=_mano_root,
    #                     use_pca=False,
    #                     root_rot_mode='rotmat',
    #                     joint_rot_mode='rotmat').to(device)
    for idx, batch in enumerate(tqdm(head_dloader)):
        _, head_imgs, head_targets = batch
        
        head_imgs = head_imgs.to(device=device)
        batch_size=len(head_imgs) #64

        model_output = head_model(head_imgs=head_imgs,num_head_imgs=len(head_imgs))
        final_model_output=model_output['stage_02']

        global_orient=torch.squeeze(final_model_output['head_pose'],dim=1)  #[64, 1, 3, 3]->[64,3,3]
        jaw_pose=torch.squeeze(final_model_output['jaw_pose'],dim=1)
        global_orient_aa=batch_rot2aa(global_orient)#[64,3]
        jaw_pose_aa=batch_rot2aa(jaw_pose)#[64,3]
        all_pose=torch.cat((global_orient_aa,jaw_pose_aa),1)#[64,6]
        #logger.info('all_pose.shape: {}',all_pose.shape)
            
        verts, landmarks2d, landmarks3d = flame(
                        shape_params=final_model_output['betas'],
                        expression_params=final_model_output['expression'],
                        pose_params=all_pose)

        # extract 7 landmark from 68 3d face keypoints
        landmarks3d_7=landmarks3d[:, index_7, :].cpu().numpy()



        if idx==0:
            logger.info('verts.shape: {}',verts.shape)# batch_size, 5023, 3
            logger.info('all_pose.shape: {}',all_pose.shape)# batch_size, 6
            logger.info('landmarks3d_7.shape: {}',landmarks3d_7.shape)# batch_size, 7, 3
            logger.info('landmarks3d.shape: {}',landmarks3d.shape)# batch_size, 68, 3
            
        for bid in range(batch_size):
            img_path=head_targets[bid].get_field('fname')
            #data/now/NoW_Dataset/final_release_version/iphone_pictures/FaMoS_*/multiview_neutral/IMG_*.jpg
            subject_name= img_path.split('/')[-3]#FaMoS_*
            challenge=img_path.split('/')[-2]#multiview_neutral
            img_name=img_path.split('/')[-1].split('.')[0]#IMG_*.jpg
            
            # if idx==0:
            #     logger.info('img_path: {}',img_path)
            #     logger.info('subject_name: {}',subject_name)
            #     logger.info('challenge: {}',challenge)
            #     logger.info('img_name: {}',img_name)
            # save 3d landmark(7*3)
            savelmk_path=os.path.join('data/now/predicted_meshes', subject_name) 
            os.makedirs(savelmk_path, exist_ok=True)
            savelmk_path=os.path.join(savelmk_path,challenge)
            os.makedirs(savelmk_path, exist_ok=True)
            savemesh_path=os.path.join(savelmk_path,img_name + '.obj')
            savelmk_path=os.path.join(savelmk_path,img_name + '.npy')
            #savelmk_path=os.path.join('data/now/predicted_meshes', subject_name, challenge ,img_name + '.npy')
            np.save(savelmk_path, landmarks3d_7[bid])
            # save mesh
            #savemesh_path=os.path.join('data/now/predicted_meshes', subject_name, challenge ,img_name + '.obj')
            save_obj(savemesh_path, verts[bid])
            



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.multiprocessing.set_start_method('spawn')

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be' +
                        ' saved')
    parser.add_argument('--datasets', nargs='+',default=[], type=str,
                        #default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--show', default=False,
                        type=lambda arg: arg.lower() in ['true'],
                        help='Display the results')
    parser.add_argument('--pause', default=-1, type=float,
                        help='How much to pause the display')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*', help='Extra command line arguments')
    parser.add_argument('--focal-length', dest='focal_length', type=float,
                        default=5000,
                        help='Focal length')
    parser.add_argument('--degrees', type=float, nargs='*', default=[],
                        help='Degrees of rotation around the vertical axis')
    parser.add_argument('--save-vis', dest='save_vis', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save visualizations')
    parser.add_argument('--save-mesh', dest='save_mesh', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save meshes')
    parser.add_argument('--save-params', dest='save_params', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save parameters')

    cmd_args = parser.parse_args()

    show = cmd_args.show
    output_folder = cmd_args.output_folder
    pause = cmd_args.pause
    focal_length = cmd_args.focal_length
    save_vis = cmd_args.save_vis
    save_params = cmd_args.save_params
    save_mesh = cmd_args.save_mesh
    degrees = cmd_args.degrees

    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)

    cfg.is_training = False
    cfg.datasets.body.splits.test = cmd_args.datasets
    use_face_contour = cfg.datasets.use_face_contour
    logger.info('use_face_contour: {}',use_face_contour)
    logger.info('cmd_datasets: {}',cmd_args.datasets)
    if 'threedpw' in cmd_args.datasets:#list
        use_face_contour=False
    set_face_contour(cfg, use_face_contour=use_face_contour)

    with threadpool_limits(limits=1):
        main(cfg, show=show, demo_output_folder=output_folder, pause=pause,
             focal_length=focal_length,
             save_vis=save_vis,
             save_mesh=save_mesh,
             save_params=save_params,
             degrees=degrees,
             )
