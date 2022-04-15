import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm

param_dir='/data/panyuqing/expose_experts/data/curated_fits/params3d_v'
allparams=os.listdir(param_dir)
target_dir='/data/panyuqing/expose_experts/data/params3d_v'

S_num='S1'
#seq_num='Seq2'
comp_target_dir=osp.join(target_dir,'mpi',S_num)
os.makedirs(comp_target_dir,exist_ok=True)
for p in tqdm(allparams):
    comp_path=osp.join(param_dir,p)
    if 'video_' in comp_path:
        shutil.move(comp_path , osp.join(comp_target_dir,p))
    else:
        shutil.move(comp_path , osp.join(target_dir,'curated_fits',p))
