# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch

from .predictor import SMPLXHead
from .predictor_online import SMPLXHead_online


def build_attention_head(cfg):
    '''
    build the motion capture model: create a SMPLXHead instance defined in predictor.py
    '''
    return SMPLXHead(cfg)

def build_attention_head_online(cfg):
    '''
    build the online motion capture model: create a SMPLXHead instance defined in predictor_online.py
    '''
    return SMPLXHead_online(cfg)

