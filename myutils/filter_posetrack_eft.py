import numpy as np
import os
import os.path as osp
import json
from tqdm import tqdm
import shutil
'''
Filter out the images 
'''
data_dir='/data1/panyuqing/expose_experts/add_data/posetrack/train'


'''
#for eft(high quality images selected by eft)
target_dir='/data1/panyuqing/expose_experts/add_data/posetrack/images/eft'

json_fn='/data1/panyuqing/expose_experts/add_data/posetrack/PoseTrack_ver01.json'
json_cont=json.load(open(json_fn,encoding='utf-8'))           
eft_fns=[json_cont['data'][i]['imageName'] for i in range(len(json_cont['data']))]
print('len eft_fns: ',len(eft_fns))#28457

for fn in tqdm(eft_fns):
    comp_fn=osp.join(data_dir,fn)
    dirname=fn.split('/')[0]
    image_fn=fn.split('/')[1]
    #print('comp_fn: ',comp_fn)
    #print('dirname: ',dirname)
    #print('image_fn: ',image_fn)
    if osp.exists(comp_fn):
        new_comp_image_fn = osp.join(target_dir,dirname+'_'+image_fn) #015828_mpii_train_000054.jpg
        #print('new_comp_image_fn: ',new_comp_image_fn)
        shutil.copy(comp_fn, new_comp_image_fn)
    else:
        print('not exits: ',comp_fn)
'''
# for train/val/test
data_dir='/data1/panyuqing/expose_experts/add_data/posetrack/val'
target_dir='/data1/panyuqing/expose_experts/add_data/posetrack/images/val'
sub_dirs=os.listdir(data_dir)
comp_sub_dirs=[osp.join(data_dir,s) for s in sub_dirs]
for csdi in tqdm(range(len(comp_sub_dirs))):
    sub_dir=sub_dirs[csdi]
    comp_sub_dir=comp_sub_dirs[csdi]
    image_fns=os.listdir(comp_sub_dir)
    comp_image_fns=[osp.join(comp_sub_dir,fn) for fn in image_fns]
    for i in tqdm(range(len(comp_image_fns))):
        new_comp_image_fn = osp.join(target_dir,sub_dir+'_'+image_fns[i]) #015828_mpii_train_000054.jpg
        shutil.move(comp_image_fns[i] , new_comp_image_fn)