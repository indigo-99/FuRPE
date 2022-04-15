import numpy as np
import json
import os
import os.path as osp

img_dir='/data/panyuqing/expose_experts/add_data/coco2017/train/'
img_name='000000229295'#'000000138248'#'000000229295'

from expose.data.targets.keypoints import dset_to_body_model, get_part_idxs
# same as expose-master\expose\data\utils\bbox.py
body_thresh=0.1
hand_thresh=0.2
face_thresh=0.4
#min_valid_keypoints=12 # other datasets use 12 before change this code to transform kpts order here

coco_source_idxs, smplx_target_idxs = dset_to_body_model(
    dset='coco',
    model_type='smplx', use_hands=True, use_face=True,
    use_face_contour=True,
    keyp_format='coco25')
coco_source_idxs = np.asarray(coco_source_idxs, dtype=np.int64) #wrong! only 19 indexes!
smplx_target_idxs = np.asarray(smplx_target_idxs, dtype=np.int64)

print('coco_source_idxs: ',coco_source_idxs)
print('smplx_target_idxs: ',smplx_target_idxs)
print('len coco_source_idxs: ',len(coco_source_idxs))
print('len smplx_target_idxs: ',len(smplx_target_idxs))
'''
coco_source_idxs, smplx_target_idxs = dset_to_body_model(
    dset='openpose25+hands+face',
    model_type='smplx', use_hands=True, use_face=True,
    use_face_contour=True,
    keyp_format='coco25')
coco_source_idxs = np.asarray(coco_source_idxs, dtype=np.int64) #wrong! only 19 indexes!
smplx_target_idxs = np.asarray(smplx_target_idxs, dtype=np.int64)

print('coco_source_idxs: ',coco_source_idxs)
print('smplx_target_idxs: ',smplx_target_idxs)
print('len coco_source_idxs: ',len(coco_source_idxs))
print('len smplx_target_idxs: ',len(smplx_target_idxs))
'''
def check_kpt_conf(cont,min_valid_keypoints=6):
    face_valid = cont['face_valid']
    foot_valid = cont['foot_valid']
    left_hand_valid = cont['lefthand_valid']
    right_hand_valid = cont['righthand_valid']

    body_kp=np.array(cont['keypoints']+cont['foot_kpts']).reshape(23,3)
    face_kp=np.array(cont['face_kpts']).reshape(68,3)
    left_hand_kp=np.array(cont['lefthand_kpts']).reshape(21,3)
    right_hand_kp=np.array(cont['righthand_kpts']).reshape(21,3)

    body_conf = body_kp[:, -1]
    face_conf = face_kp[:, -1]
    left_hand_conf = left_hand_kp[:, -1]
    right_hand_conf = right_hand_kp[:, -1]

    body_conf[body_conf < body_thresh] = 0.0
    left_hand_conf[left_hand_conf < hand_thresh] = 0.0
    right_hand_conf[right_hand_conf < hand_thresh] = 0.0
    face_conf[face_conf < face_thresh] = 0.0

    body_kp[:, -1] = body_conf
    left_hand_kp[:, -1] = left_hand_conf
    right_hand_kp[:, -1] = right_hand_conf
    face_kp[:, -1] = face_conf
    keypoints2d=np.concatenate((body_kp,face_kp,left_hand_kp,right_hand_kp))
    
    # transform coco kpt indexes to smplx format to ensure the data able to be used in crop img according to kpt (at least 6 kpts>threshold)
    output_keypoints2d = np.zeros([127 + 17, 3], dtype=np.float32)
    output_keypoints2d[smplx_target_idxs] = keypoints2d[coco_source_idxs]
    # keypoints = keypoints2d[:, :-1]
    # conf = keypoints2d[:, -1]
    keypoints = output_keypoints2d[:, :-1]
    conf = output_keypoints2d[:, -1]
    valid_keypoints = keypoints[conf > 0]
    
    # print('keypoints2d coco: ',keypoints2d)
    # print('keypoints2d smplx: ',output_keypoints2d)
    keypoints_first25=keypoints[:25]
    conf_first25=conf[:25]
    print('keypoints[:25] num conf>0: ',len(keypoints_first25[conf_first25 > 0]))
    print('keypoints[:25] must have at least 6 conf>0: ',len(keypoints_first25[conf_first25 > 0]) > min_valid_keypoints)
    if len(valid_keypoints) < min_valid_keypoints:
        return False
    else:
        # if ((not face_valid) and (not foot_valid) and (not left_hand_valid) and (not right_hand_valid)):
        #     return False
        print(len(valid_keypoints))
        return True



cont=json.load(open(osp.join(img_dir,'keypoints2d',img_name+'_keypoints.json'),encoding='utf-8'))
print('is valid: ',check_kpt_conf(cont))


whole_kpt=cont['keypoints']+cont['foot_kpts']+cont['face_kpts']+cont['lefthand_kpts']+cont['righthand_kpts']
keypoints2d=np.array(whole_kpt).reshape(133,3)
# body_bbox=cont['bbox']
# face_bbox=cont['face_box']
# left_bbox=cont['lefthand_box']
# right_bbox=cont['righthand_box']

body_kpts=np.array(cont['keypoints']).reshape(-1,3)
# print('body_kpts: ',body_kpts)
# print('body_kpts shape: ',body_kpts.shape)
# print('lefthand_kpts: ',cont['lefthand_kpts'])
# print('rightthand_kpts: ',cont['righthand_kpts'])
'''
import cv2 as cv
imgpath=osp.join(img_dir,'images',img_name+'.jpg')
img=cv.imread(imgpath)				#读取图片
color=(0,0,255)						#红色
font = cv.FONT_HERSHEY_DUPLEX  # 设置字体
for i in range(len(body_kpts)):
    x=body_kpts[i,0]
    y=body_kpts[i,1]
    cv.circle(img,(x,y),2,color,-1)		#cv.circle(图片,元祖格式表示圆心,int类型半径,颜色,是否实心标志)
    # 图片对象、文本、位置像素、字体、字体大小、颜色、字体粗细
    cv.putText(img, str(i), (x, y), font, 1, (254, 67, 101), 2)
cv.imwrite('/data/panyuqing/expose_experts/coco2017_'+img_name+'_kpt.jpg',img)				#将画好的图片保存在指定路径

'''