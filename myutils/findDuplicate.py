import os
import os.path as osp
from tqdm import tqdm

a_dir='/data1/panyuqing/expose_experts/add_data/posetrack/images/test'
b_dir='/data1/panyuqing/expose_experts/add_data/posetrack/images/val'
img_list_a=os.listdir(a_dir)
img_list_b=os.listdir(b_dir)
img_dict={}
for ip in tqdm(img_list_a):
    img_dict[ip]=1
print('Finish building dict.')

dup=[]
uni=[]
for ip in tqdm(img_list_b):
    if ip in img_dict:
        dup.append(ip)
    else:
        uni.append(ip)

print('total num: ',len(img_list_b))
print('duplicate num: ',len(dup))
print('unique num: ',len(uni))