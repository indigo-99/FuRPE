# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
from typing import List, Optional
import functools
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import resource
import numpy as np
from collections import OrderedDict, defaultdict
from loguru import logger
import cv2
import argparse
import time
import open3d as o3d
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import PIL.Image as pil_img
import matplotlib.pyplot as plt

import torch
import torch.utils.data as dutils
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, Normalize, ToTensor

from FuRPE.data.datasets import ImageFolder, ImageFolderWithBoxes

from FuRPE.data.targets.image_list import to_image_list
from FuRPE.utils.checkpointer import Checkpointer

from FuRPE.data.build import collate_batch
from FuRPE.data.transforms import build_transforms

from FuRPE.models.smplx_net import SMPLXNet
from FuRPE.config import cfg
from FuRPE.config.cmd_parser import set_face_contour
from FuRPE.utils.plot_utils import HDRenderer

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector

# p add: read video and store each frame as a jpg
def save_img(save_imgs_path='/data/panyuqing/expose_experts/testvideo_img/',video_path='/data/panyuqing/expose_experts/testvideo/'):
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name = osp.join(save_imgs_path, file_name)
        os.makedirs(folder_name,exist_ok=True)
        vc = cv2.VideoCapture(osp.join(video_path,video_name)) #读入视频文件
        c=0
        rval=vc.isOpened()
        # logger.info('rval: {}',rval)
        # logger.info('path: {}',video_path+video_name)
        while rval:   #循环读取视频帧
            c = c + 1
            rval, frame = vc.read()
            pic_save_path=osp.join(folder_name,file_name + '_' + str(c) + '.jpg')
            if rval:
                cv2.imwrite(pic_save_path, frame) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)

#P add: read output overlay imgs and generate a demo video
def gen_video_out_ffmpeg(in_dir, out_dir, video_name):
    in_dir = osp.abspath(in_dir)
    os.makedirs(out_dir,exist_ok=True)
    out_path = osp.abspath(osp.join(out_dir, video_name+'.mp4'))
    logger.info(">> Generating video in {}",out_path)
    #ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{in_dir}/*.png"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path}'
    ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate 25 -i "{in_dir}/{video_name}_%d.png" {out_path}'
    os.system(ffmpeg_cmd)

def gen_video_out(in_dir, out_dir, video_name):
    in_dir = osp.abspath(in_dir)
    os.makedirs(out_dir,exist_ok=True)
    out_path = osp.abspath(osp.join(out_dir, video_name+'.mp4'))
    logger.info(">> Generating video in {}",out_path)
    
    pic_names=os.listdir(in_dir)
    pic_names=sorted(pic_names)
    pic_paths=[osp.join(in_dir,pn) for pn in pic_names]
    first_frame=cv2.imread(pic_paths[0])
    
    fps=25#20
    width = first_frame.shape[1]  #1200
    height = first_frame.shape[0]  #1200
    if max(width,height)>1080:
        scale=2
    else: 
        scale=1
    size = (int(width/scale), int(height/scale))
    videowriter = cv2.VideoWriter(out_path,
                                  cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                  fps, size)
    for i in range(len(pic_paths)):
        fr = cv2.imread(pic_paths[i])
        # attention! width, height
        img = cv2.resize(fr, size, interpolation=cv2.INTER_LINEAR)
        #print('img size: ',img.shape)
        #print(int(fr.shape[0]/2), int(fr.shape[1]/2))
        #videowriter.write(fr)
        videowriter.write(img)
    print("Finish generating video: ",out_path)
    videowriter.release()

def collate_fn(batch):
    output_dict = dict()

    for d in batch:
        for key, val in d.items():
            if key not in output_dict:
                output_dict[key] = []
            output_dict[key].append(val)
    return output_dict


def preprocess_images(
    image_folder: str,
    exp_cfg,
    num_workers: int = 8, batch_size: int = 1,
    min_score: float = 0.5,
    scale_factor: float = 1.2,
    device: Optional[torch.device] = None
) -> dutils.DataLoader:

    if device is None:
        device = torch.device('cuda')
        if not torch.cuda.is_available():
            logger.error('CUDA is not available!')
            sys.exit(3)

    rcnn_model = keypointrcnn_resnet50_fpn(pretrained=True)
    rcnn_model.eval()
    rcnn_model = rcnn_model.to(device=device)

    transform = Compose(
        [ToTensor(), ]
    )

    # Load the images
    dataset = ImageFolder(image_folder, transforms=transform)
    rcnn_dloader = dutils.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn
    )

    out_dir = osp.expandvars('$HOME/Dropbox/boxes')
    os.makedirs(out_dir, exist_ok=True)

    img_paths = []
    bboxes = []
    for bidx, batch in enumerate(
            tqdm(rcnn_dloader, desc='Processing with R-CNN')):
        batch['images'] = [x.to(device=device) for x in batch['images']]

        output = rcnn_model(batch['images'])
        for ii, x in enumerate(output):
            img = np.transpose(
                batch['images'][ii].detach().cpu().numpy(), [1, 2, 0])
            img = (img * 255).astype(np.uint8)

            img_path = batch['paths'][ii]
            _, fname = osp.split(img_path)
            fname, _ = osp.splitext(fname)

            #  out_path = osp.join(out_dir, f'{fname}_{ii:03d}.jpg')
            for n, bbox in enumerate(output[ii]['boxes']):
                bbox = bbox.detach().cpu().numpy()
                if output[ii]['scores'][n].item() < min_score:
                    continue
                img_paths.append(img_path)
                bboxes.append(bbox)

                #  cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]),
                #  (255, 0, 0))
            #  cv2.imwrite(out_path, img[:, :, ::-1])

    dataset_cfg = exp_cfg.get('datasets', {})
    body_dsets_cfg = dataset_cfg.get('body', {})

    body_transfs_cfg = body_dsets_cfg.get('transforms', {})
    transforms = build_transforms(body_transfs_cfg, is_train=False)
    batch_size = body_dsets_cfg.get('batch_size', 64)

    expose_dset = ImageFolderWithBoxes(
        img_paths, bboxes, scale_factor=scale_factor, transforms=transforms)

    expose_collate = functools.partial(
        collate_batch, use_shared_memory=num_workers > 0,
        return_full_imgs=True)
    expose_dloader = dutils.DataLoader(
        expose_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=expose_collate,
        drop_last=False,
        pin_memory=True,
    )
    return expose_dloader


def weak_persp_to_blender(
        targets,
        camera_scale,
        camera_transl,
        H, W,
        sensor_width=36,
        focal_length=5000):
    ''' Converts weak-perspective camera to a perspective camera
    '''
    if torch.is_tensor(camera_scale):
        camera_scale = camera_scale.detach().cpu().numpy()
    if torch.is_tensor(camera_transl):
        camera_transl = camera_transl.detach().cpu().numpy()

    output = defaultdict(lambda: [])
    for ii, target in enumerate(targets):
        orig_bbox_size = target.get_field('orig_bbox_size')
        bbox_center = target.get_field('orig_center')
        z = 2 * focal_length / (camera_scale[ii] * orig_bbox_size)

        transl = [
            camera_transl[ii, 0].item(), camera_transl[ii, 1].item(),
            z.item()]
        shift_x = - (bbox_center[0] / W - 0.5)
        shift_y = (bbox_center[1] - 0.5 * H) / W
        focal_length_in_mm = focal_length / W * sensor_width
        output['shift_x'].append(shift_x)
        output['shift_y'].append(shift_y)
        output['transl'].append(transl)
        output['focal_length_in_mm'].append(focal_length_in_mm)
        output['focal_length_in_px'].append(focal_length)
        output['center'].append(bbox_center)
        output['sensor_width'].append(sensor_width)
    for key in output:
        output[key] = np.stack(output[key], axis=0)
    return output


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


@torch.no_grad()
def main(
    image_folder: str,
    exp_cfg,
    show: bool = False,
    demo_output_folder: str = 'demo_output',
    pause: float = -1,
    focal_length: float = 5000,
    rcnn_batch: int = 1,
    sensor_width: float = 36,
    save_vis: bool = True,
    comparing: bool =True,
) -> None:
    
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)
    
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               colorize=True)
    
    # Use rcnn to detect body/hand/face bbox, and package bboxes into dataloader for inputing into the model
    expose_dloader = preprocess_images(
        image_folder, exp_cfg, batch_size = rcnn_batch, device = device,num_workers=4)

    # path to save predicted results 
    demo_output_folder = osp.expanduser(osp.expandvars(demo_output_folder))
    logger.info(f'Saving results to: {demo_output_folder}')
    os.makedirs(demo_output_folder, exist_ok=True)

    # initialize the motion-capture model
    model = SMPLXNet(exp_cfg)
    try:
        model = model.to(device=device)
    except RuntimeError:
        # Re-submit in case of a device error
        sys.exit(3)

    # load checkpoints to the model
    output_folder = exp_cfg.output_folder
    checkpoint_folder = osp.join(output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(
        model, save_dir=checkpoint_folder, pretrained=exp_cfg.pretrained)

    # arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    # for key in arguments:
    #     if key in extra_checkpoint_data:
    #         arguments[key] = extra_checkpoint_data[key]

    model = model.eval()

    means = np.array(exp_cfg.datasets.body.transforms.mean)
    std = np.array(exp_cfg.datasets.body.transforms.std)

    render = save_vis or show
    body_crop_size = exp_cfg.get('datasets', {}).get('body', {}).get(
        'transforms').get('crop_size', 256)
    if render:
        hd_renderer = HDRenderer(img_size=body_crop_size)
    
    total_time = 0
    cnt = 0
    for bidx, batch in enumerate(tqdm(expose_dloader, dynamic_ncols=True)):

        full_imgs_list, body_imgs, body_targets = batch
        if full_imgs_list is None:
            continue

        full_imgs = to_image_list(full_imgs_list)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]
        full_imgs = full_imgs.to(device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        model_output = model(body_imgs, body_targets, full_imgs=full_imgs,
                             device=device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        cnt += 1
        total_time += elapsed

        hd_imgs = full_imgs.images.detach().cpu().numpy().squeeze()
        body_imgs = body_imgs.detach().cpu().numpy()
        body_output = model_output.get('body')

        _, _, H, W = full_imgs.shape
        #  H, W, _ = hd_imgs.shape
        if render:
            hd_imgs = np.transpose(undo_img_normalization(hd_imgs, means, std),
                                   [0, 2, 3, 1])
            hd_imgs = np.clip(hd_imgs, 0, 1.0)
            bg_hd_imgs = np.transpose(hd_imgs, [0, 3, 1, 2])
            right_hand_crops = body_output.get('right_hand_crops')
            left_hand_crops = torch.flip(
                body_output.get('left_hand_crops'), dims=[-1])
            head_crops = body_output.get('head_crops')
            bg_imgs = undo_img_normalization(body_imgs, means, std)

            right_hand_crops = undo_img_normalization(
                right_hand_crops, means, std)
            left_hand_crops = undo_img_normalization(
                left_hand_crops, means, std)
            head_crops = undo_img_normalization(head_crops, means, std)

        body_output = model_output.get('body', {})
        num_stages = body_output.get('num_stages', 3)
        stage_n_out = body_output.get(f'stage_{num_stages - 1:02d}', {})

        faces = stage_n_out['faces']

        out_img = OrderedDict()

        final_model_vertices = None
        stage_n_out = model_output.get('body', {}).get('final', {})
        if stage_n_out is not None:
            final_model_vertices = stage_n_out.get('vertices', None)

        if final_model_vertices is not None:
            final_model_vertices = final_model_vertices.detach().cpu().numpy()
            camera_parameters = model_output.get('body', {}).get(
                'camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()

        hd_params = weak_persp_to_blender(
            body_targets,
            camera_scale=camera_scale,
            camera_transl=camera_transl,
            H=H, W=W,
            sensor_width=sensor_width,
            focal_length=focal_length,
        )
        
        # Render the overlays of the final prediction
        if render:
            hd_overlays = hd_renderer(
                final_model_vertices,
                faces,
                focal_length=hd_params['focal_length_in_px'],
                camera_translation=hd_params['transl'],
                camera_center=hd_params['center'],
                bg_imgs=bg_hd_imgs,
                return_with_alpha=True,
                body_color=[0.4, 0.4, 0.7]
            )
            out_img['hd_overlay'] = hd_overlays

            for key in out_img.keys():
                out_img[key] = np.clip(
                    np.transpose(
                        out_img[key], [0, 2, 3, 1]) * 255, 0, 255).astype(
                            np.uint8)
        
            
            for idx in tqdm(range(len(body_targets)), 'Saving ...'):
                fname = body_targets[idx].get_field('fname')
                # if len(fname)>6:
                #     logger.info('too long video! frame name over 6 nums! {}',fname)
                # fname=int(fname)
                curr_img = out_img['hd_overlay']
                if comparing:
                    #logger.info('hd_img[idx] shape: {}',hd_imgs[idx].shape)#(1440,2560,4)
                    color255_img = hd_imgs[idx] * 255
                    cur_concat = np.concatenate((color255_img.astype(np.uint8), curr_img[idx]), axis=1)
                    # pil_img.fromarray(cur_concat).save(
                    #     osp.join(demo_output_folder, f'{fname:06d}.png'))
                    pil_img.fromarray(cur_concat).save(
                        osp.join(demo_output_folder, fname+'.png'))
                else:
                    # pil_img.fromarray(curr_img[idx]).save(
                    #     osp.join(demo_output_folder, f'{fname:06d}.png'))
                    pil_img.fromarray(curr_img[idx]).save(
                        osp.join(demo_output_folder, fname+'.png'))

    logger.info(f'Average inference time: {total_time / cnt}')
    
    logger.info(f'Generating demo videos.')
    out_dir = '/data/panyuqing/expose_experts/testvideo_res'
    video_name = image_folder.split('/')[-1]#'stand'#'sit'
    gen_video_out(in_dir = demo_output_folder, out_dir = out_dir, video_name = video_name)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'Expose_experts Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--image-folder', type=str, dest='image_folder',
                        help='The folder with images that will be processed')
    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be' +
                        ' saved')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*', help='Extra command line arguments')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--expose-batch',
                        dest='expose_batch',
                        default=1, type=int,
                        help='ExPose batch size')
    parser.add_argument('--rcnn-batch',
                        dest='rcnn_batch',
                        default=1, type=int,
                        help='R-CNN batch size')
    parser.add_argument('--pause', default=-1, type=float,
                        help='How much to pause the display')
    parser.add_argument('--focal-length', dest='focal_length', type=float,
                        default=5000,
                        help='Focal length')
    parser.add_argument('--save-vis', dest='save_vis', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save visualizations')

    cmd_args = parser.parse_args()

    image_folder = cmd_args.image_folder
    output_folder = cmd_args.output_folder
    pause = cmd_args.pause
    focal_length = cmd_args.focal_length
    save_vis = cmd_args.save_vis
    expose_batch = cmd_args.expose_batch
    rcnn_batch = cmd_args.rcnn_batch

    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)

    cfg.datasets.body.batch_size = expose_batch

    cfg.is_training = False
    cfg.datasets.body.splits.test = cmd_args.datasets
    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)

    # p add:
    # all_image_folder='/data/panyuqing/expose_experts/testvideo_img'
    # video_path='/data/panyuqing/expose_experts/testvideo'
    # save_img(save_imgs_path=all_image_folder,video_path=video_path)
    # image_folder=osp.join(all_image_folder,'stand')#'sit

    
    with threadpool_limits(limits=1):
        main(
            image_folder,
            cfg,
            demo_output_folder=output_folder,
            pause=pause,
            focal_length=focal_length,
            save_vis=save_vis,
            rcnn_batch=rcnn_batch,
        )
    
