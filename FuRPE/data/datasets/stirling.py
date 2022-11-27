# -*- coding: utf-8 -*-

# change from stirling.py

import os.path as osp

import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..targets import BoundingBox
from ..utils import bbox_to_center_scale

from ...utils.img_utils import read_img

FOLDER_MAP_FNAME = 'folder_map.pkl'


class Stirling3D(dutils.Dataset):
    '''
    The class of the dataloader designed for the NoW test-dataset (a face specific dataset)
    '''
    def __init__(self, data_path='data/now/NoW_Dataset',
                 head_only=True,
                 split='test',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 **kwargs):
        super(Stirling3D, self).__init__()
        assert head_only, 'Stirling3D/NoW can only be used as a head only dataset'

        self.split = split
        assert 'test' in split, (
            f'Stirling3D/NoW can only be used for testing, but got split: {split}'
        )
        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.data_path = osp.expandvars(osp.expanduser(data_path))
        self.transforms = transforms
        self.dtype = dtype
        
        # the absolute dir of images
        img_dir=osp.join('/data/panyuqing/expose_experts',self.data_path, 'final_release_version', 'iphone_pictures')
        bbox_dir=osp.join('/data/panyuqing/expose_experts',self.data_path, 'final_release_version', 'detected_face')
        evaluation_img_path='data/now/imagepathstest.txt' # all testing imgs' paths list

        with open(evaluation_img_path,'r',encoding='utf-8') as ef:
            filepaths=ef.readlines()
            self.img_paths = np.array(
                [osp.join(img_dir, fname.strip())
                for fname in filepaths]
            )
            self.bbox_paths= np.array(
                [osp.join(bbox_dir, fname.strip().replace('iphone_pictures','detected_face')[:-4]+'.npy')
                for fname in filepaths]
            )
        
        self.num_items = len(self.img_paths)
        logger.info('self.num_items: {}',self.num_items)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return 'Stirling3D( \n\t Split: {self.split}\n)'

    def get_num_joints(self):
        return 0

    def only_2d(self):
        return False

    def __len__(self):
        return self.num_items

    def name(self):
        return f'Stirling3D/{self.split}'

    def __getitem__(self, index):
        '''
        get a data item with targeted labels from the dataloader
        '''
        full_img = read_img(self.img_paths[index])
        bbox_path=self.bbox_paths[index]
        gt_bbox=np.load(bbox_path,allow_pickle=True, encoding='latin1')
        top = gt_bbox.item()['top']
        right = gt_bbox.item()['right']
        bottom = gt_bbox.item()['bottom']
        left = gt_bbox.item()['left']
        #logger.info('left {}, right {}, top {}, bottom {}',left,right,top,bottom)
        
        # try to crop a square
        height=bottom-top+1
        width=right-left+1
        # crop img from bbox[xmin,ymin,xmax,ymax]
        if height>=width:
            newleft=int(max(0,left-height/2))
            newright=int(min(full_img.shape[1],right+height/2))
            #logger.info('newleft {}, newright {}, top {}, bottom {}',newleft,newright,top,bottom)
            img=full_img[top:bottom+1,newleft:newright+1, :]
        else:
            newtop=int(max(0,top-width/2))
            newbottom=int(min(full_img.shape[0],bottom+width/2))
            #logger.info('left {}, right {}, newtop {}, newbottom {}',left,right,newtop,newbottom)
            img=full_img[newtop:newbottom+1,left:right+1, :]
        
        H, W, _ = img.shape
        bbox = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
                
        target = BoundingBox(bbox, size=img.shape)

        center = np.array([W, H], dtype=np.float32) * 0.5
        target.add_field('center', center)

        center, scale, bbox_size = bbox_to_center_scale(bbox)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('image_size', img.shape)

        if self.transforms is not None:
            img, cropped_image, target = self.transforms(img, target)

        target.add_field('name', self.name())
        #target.add_field('fname', osp.split(self.img_paths[index])[1])
        target.add_field('fname', self.img_paths[index])
        return img, cropped_image, target, index
