# consecutive pictures in 3DPW testset contains many similarity
# therefore, pick out pictures on the interval of 10 pictures
import os
import os.path as osp
import shutil
from tqdm import tqdm

dirs='downtown_arguing_00,downtown_rampAndStairs_00,downtown_walkUphill_00,downtown_bar_00,downtown_runForBus_00,downtown_warmWelcome_00,downtown_bus_00,downtown_runForBus_01,downtown_weeklyMarket_00,downtown_cafe_00,downtown_sitOnStairs_00,downtown_windowShopping_00,downtown_car_00,downtown_stairs_00,flat_guitar_01,downtown_crossStreets_00,downtown_upstairs_00,flat_packBags_00,downtown_downstairs_00,downtown_walkBridge_01,office_phoneCall_00,downtown_enterShop_00,downtown_walking_00,outdoors_fencing_01'
dir_list=dirs.split(',')
#dir_list=[dir_list[0]]
root_dir='/data1/panyuqing/expose_experts/data/3DPW/imageFiles/'
target_dir='/data1/panyuqing/expose_experts/data/3DPW/pickTest'
for cur_dir in tqdm(dir_list):
    cur_pics=sorted(os.listdir(osp.join(root_dir,cur_dir)))
    pic_pics=[cur_pics[i] for i in range(len(cur_pics)) if i%10==0]
    print('pick pictures num of ',cur_dir,' : ',len(pic_pics))
    pic_pics_path=[osp.join(root_dir,cur_dir,pp) for pp in pic_pics]
    target_path=[osp.join(target_dir,cur_dir+'_'+pp) for pp in pic_pics]
    [shutil.copyfile(pic_pics_path[i], target_path[i]) for i in range(len(pic_pics_path))]