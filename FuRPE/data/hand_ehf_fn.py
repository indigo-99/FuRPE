# -*- coding: utf-8 -*-
'''
get filenames in the EHF testsets whose pseudo parameters contain hand poses.
store the filenames as an .npy
(Try to test the errors from pseudo parameters to ground truth parameters of EHF)
'''
import os.path as osp
import os

import numpy as np
from loguru import logger
from tqdm import tqdm

import sys

if __name__ == '__main__':
    save_param_path = '/data/panyuqing/expose_experts/data/params3d_v'

    data_path='/data/panyuqing/expose_experts/data/curated_fits/'
    save_comp_path=osp.join(data_path,'EHF_comp_img_fns.npy')#osp.join(data_path,'mpi_'+S_num+'_comp_img_fns.npy')
    save_hand_comp_path=osp.join(data_path,'hand_EHF_comp_img_fns.npy')
    new_comp_img_fns=np.load(save_comp_path,allow_pickle=True)
    img_names=[new_comp_img_fns[i].split('/')[-1].split('\\')[-1].split('.')[0] for i in range(len(new_comp_img_fns))]
    
    hand_ehf_fns=[]
    for i in tqdm(range(len(img_names))):
        img_name=img_names[i]
        image_path=new_comp_img_fns[i]
        save_3dparam_vertices_path=os.path.join(save_param_path, 'EHF',img_name + '.npy')
        
        # read 3d params and save vertices
        if not osp.exists(save_3dparam_vertices_path):
            logger.info('no expert param saved? {}',save_3dparam_vertices_path)
            continue
     
        params_data = np.load(save_3dparam_vertices_path, allow_pickle=True).item()
           
        if 'left_hand_pose' in params_data or 'right_hand_pose' in params_data:
            hand_ehf_fns.append(image_path)

    np.save(save_hand_comp_path, np.array(hand_ehf_fns))
