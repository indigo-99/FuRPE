import json
import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm
'''Filter the mpi data after the first filter (according to keypoint conf),
This time filtering out data whose keypoints number doesn't satisfy the belowing requirements to avoid error during training'''
def check_kpt_conf(cont,min_valid_keypoints=6):
    if (len(cont['people'])>0) and (len(cont['people'][0]['pose_keypoints_2d'])==75):
        body_kp=np.array(cont['people'][0]['pose_keypoints_2d'])
        if 0 in body_kp.shape:
            return False
        body_kp=body_kp.reshape(25,3)
        return True
    else:
        return False

S_num_list=['S2_Seq2','S3_Seq2','S4_Seq2','S5_Seq2','S6_Seq2','S7_Seq2','S8_Seq2']
for divide in tqdm(S_num_list):
    dataset_pre='/data/panyuqing/expose_experts/add_data/mpi/'+divide
    img_dir=dataset_pre + '/images'
    kpt_dir=dataset_pre + '/keypoints2d/'
    new_kpt_dir=dataset_pre + '/keypoints2d_strict/'
    os.makedirs(new_kpt_dir,exist_ok=True)

    kpt_files=os.listdir(kpt_dir) 
    kpt_paths=[osp.join(kpt_dir,f) for f in kpt_files]

    data_path='/data/panyuqing/expose_experts/data/curated_fits'
    add_img_path=osp.join(data_path,'mpi_'+divide+'_comp_img_fns.npy') 
    raw_comp_fn_list=list(np.load(add_img_path,allow_pickle=True))
    print('raw train data len: ',len(raw_comp_fn_list))#46054

    comp_fn_list=[]
    for fn in tqdm(raw_comp_fn_list):
        pic_name=fn.split('/')[-1].split('\\')[-1].split('.')[0]#[:12]#_keypoints
        kpt_path=osp.join(kpt_dir,pic_name+'_keypoints.json')
        # print(pic_name)
        cont=json.load(open(kpt_path,encoding='utf-8'))
        if check_kpt_conf(cont):
            savekpt_path=new_kpt_dir+pic_name+'_keypoints.json'
            shutil.copyfile(kpt_path, savekpt_path)
            comp_fn_list.append(fn)
    print(divide,' strict selection data len: ',len(comp_fn_list))#46054
    np.save(osp.join(data_path,'mpi_'+divide+'_comp_img_fns_del0kpt.npy') ,np.array(comp_fn_list))