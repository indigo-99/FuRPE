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
from manopth import manolayer
_mano_root = '/data/panyuqing/MinimalHandPytorch/mano/models'

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
target_right_hand_mean=np.array([ 0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
                    0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
                    -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
                    -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
                    0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
                    0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
                    -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
                    0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
                    -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]).astype(np.float32).reshape(15,3)

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

def dump(xyz_pred_list, verts_pred_list,pred_out_path='pred.json'):
    """ Save predictions into a json file. """
    # make sure its only lists
    #xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    #verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


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

    # demo_output_folder = osp.expanduser(osp.expandvars(demo_output_folder))
    # logger.info(f'Saving results to: {demo_output_folder}')
    # print('saving results to:',demo_output_folder)
    # os.makedirs(demo_output_folder, exist_ok=True)

    self_supervised_EMA_model = SMPLXNet(exp_cfg)
    try:
        self_supervised_EMA_model = self_supervised_EMA_model.to(device=device)
    except RuntimeError:
        # Re-submit in case of a device error
        sys.exit(3)
    model = self_supervised_EMA_model.get_online_model()#.online_model()
    checkpoint_folder = osp.join(
        exp_cfg.output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder,
                                pretrained=exp_cfg.pretrained)

    #arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    # for key in arguments:
    #     if key in extra_checkpoint_data:
    #         arguments[key] = extra_checkpoint_data[key]

    model = model.eval()
    hand_model = model.get_hand_model()

    dataloaders = make_all_data_loaders(exp_cfg, split='test')

    xyz_pred_list, verts_pred_list = list(), list()

    #body_dloader = dataloaders['body'][0]
    hand_dloader = dataloaders['hand'][0]
    mano = manolayer.ManoLayer(flat_hand_mean=True, # don't need mean hand params, start from flat hand
                        side="right",#side="left",
                        mano_root=_mano_root,
                        use_pca=False,
                        root_rot_mode='rotmat',
                        joint_rot_mode='rotmat')
    mano = mano.to(device=device)
    for idx, batch in enumerate(tqdm(hand_dloader)):
        _, hand_imgs, hand_targets = batch

        hand_imgs = hand_imgs.to(device=device)
        #hand_targets = [t.to(device=device) for t in hand_targets]

        model_output = hand_model(hand_imgs=hand_imgs,num_hand_imgs=len(hand_imgs))
        
        pred_wrist_pose=model_output['stage_02'].get('wrist_pose')#[64, 1, 3, 3]
        pred_hand_pose=model_output['stage_02'].get('hand_pose')#[64, 15, 3, 3]
        #add mean pose
        #torch_mean_hand=torch.from_numpy(target_right_hand_mean).to(device)
        #orch_mean_hand_rotmat=batch_rodrigues(torch_mean_hand).reshape(1,15,3,3)
        #pred_hand_pose=pred_hand_pose+torch_mean_hand_rotmat
        
        #logger.info('output keys: {}',model_output.keys())
        #['num_stages', 'features', 'stage_00', 'stage_01', 'stage_02']
        # logger.info('stage_02 keys: {}',model_output['stage_02'].keys())
        #'right_hand_pose', 'betas', 'wrist_pose', 'hand_pose', 'raw_right_wrist_pose', 'raw_left_wrist_pose', 'raw_right_hand_pose'
        total_hand_pose=torch.cat((pred_wrist_pose, pred_hand_pose),1)#[64, 16, 3, 3]
        shape = model_output['stage_02'].get('betas')

        hand_vertices, hand_joints = mano(total_hand_pose, shape)#.float())
        hand_vertices=hand_vertices.detach().cpu().numpy().tolist()
        hand_joints=hand_joints.detach().cpu().numpy().tolist()
        #hand_vertices = model_output.get('vertices').detach().cpu().numpy()
        #hand_joints = model_output.get('joints').detach().cpu().numpy()
        
        # xyz_pred_list.append(hand_joints)
        # verts_pred_list.append(hand_vertices)
        xyz_pred_list+=hand_joints
        verts_pred_list+=hand_vertices

        if idx==0:
            logger.info('shape.shape: {}',shape.shape)
            logger.info('total_hand_pose.shape: {}',total_hand_pose.shape)
            logger.info('hand_joints.shape: {}',len(hand_joints))
            logger.info('hand_vertices.shape: {}',len(hand_vertices))
    
    dump(xyz_pred_list, verts_pred_list, pred_out_path='pred.json')

    

    # with Evaluator(exp_cfg) as evaluator:
    #     evaluator.run(model, dataloaders, exp_cfg, device)


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