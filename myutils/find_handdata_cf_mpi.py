import json
import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm

data_path='/data/panyuqing/expose_experts/data/curated_fits'

# params3d_v_path='/data/panyuqing/expose_experts/data/params3d_v/curated_fit'
# all_image_name_path=osp.join(data_path,'comp_img_fns.npy')
# raw_comp_fn_list=list(np.load(all_image_name_path,allow_pickle=True))
# print('raw curated_fit&mpi_s1 train data len: ',len(raw_comp_fn_list))#


params3d_v_path='/data/panyuqing/expose_experts/data/params3d_v/mpi'
S_num_list=['S2_Seq2','S3_Seq2','S4_Seq2','S5_Seq2','S6_Seq2','S7_Seq2','S8_Seq2']
for divide in tqdm(S_num_list):
    param_dir=osp.join(params3d_v_path,divide)
    add_img_path=osp.join(data_path,'mpi_'+divide+'_comp_img_fns_del12.npy') 
    raw_comp_fn_list=list(np.load(add_img_path,allow_pickle=True))
    print(divide,' raw train data len: ',len(raw_comp_fn_list))#46054

    comp_fn_list=[]
    for fn in tqdm(raw_comp_fn_list):
        pic_name=fn.split('/')[-1].split('\\')[-1].split('.')[0]#[:12]#_keypoints
        param_file=osp.join(param_dir,pic_name+'.npy')
        param3d_vertices_data=np.load(param_file, allow_pickle=True).item()
        
        has_left=False
        has_right=False
        if 'left_hand_pose' in param3d_vertices_data:     
            left_hand_pose=param3d_vertices_data['left_hand_pose'].astype(np.float32).reshape(15,3)
            has_left=True

        if 'right_hand_pose' in param3d_vertices_data:   
            right_hand_pose=param3d_vertices_data['right_hand_pose'].astype(np.float32).reshape(15,3)
            has_right=True
        if has_left or has_right:
            comp_fn_list.append(fn)
    print(divide,' data with hand len: ',len(comp_fn_list))#46054
    np.save(osp.join(data_path,'mpi_'+divide+'_comp_img_fns_withhand.npy') ,np.array(comp_fn_list))