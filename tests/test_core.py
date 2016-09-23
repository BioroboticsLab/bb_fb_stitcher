import logging.config
import affine_stitcher.core as core
import cv2
import affine_stitcher.helpers as helpers

logging.config.fileConfig('logging_config.ini')

img_l =cv2.imread('data/core/Cam_0_20140918120038_045539_TO_Cam_0_20140918120622_267825.mkv.jpg')
img_r =cv2.imread('data/core/Cam_1_20140918120522_949626_TO_Cam_1_20140918121058_677877.mkv.jpg')
# img_l =cv2.imread('data/core/Cam_0_20161507130847_631282517.jpg')
# img_r =cv2.imread('data/core/Cam_1_20161507130847_631282517.jpg')
helpers.display(img_l,'left image', time=50)
helpers.display(img_r, 'right image', time=50)
bb_stitcher_fb = core.BB_FeatureBasedSticher()
result = bb_stitcher_fb.stitch((img_l, img_r))
helpers.display(result, 'left restult')
# helpers.display(img_r_res, 'right result')
