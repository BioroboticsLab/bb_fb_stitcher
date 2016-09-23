import affine_stitcher.stitcher
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging.config
import affine_stitcher.helpers as helpers
from affine_stitcher.stitcher import Transformation

logging.config.fileConfig('logging_config.ini')

img_l = cv2.imread('data/stitcher/Input/fg_sub_imgs/'
                   'Cam_0_20140918120038_045539_TO_Cam_0_20140918120622_267825.jpg')
img_r = cv2.imread('data/stitcher/Input/fg_sub_imgs/'
                   'Cam_1_20140918120522_949626_TO_Cam_1_20140918121058_677877.jpg')

st = affine_stitcher.stitcher.FeatureBasedStitcher(400, 500, transformation=Transformation.AFFINE)
homo, result, vis = st((img_l, img_r), True)

helpers.display(result, 'result')
helpers.display(vis, 'matches')
