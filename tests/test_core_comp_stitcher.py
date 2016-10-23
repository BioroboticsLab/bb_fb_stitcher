import logging.config
import fb_stitcher.core as core
import cv2
import fb_stitcher.helpers as helpers

logging.config.fileConfig('logging_config.ini')

img_l =cv2.imread('data/core_comp_stitcher/Cam_0_2016-09-01T14:20:38.410765Z.jpg')
img_r =cv2.imread('data/core_comp_stitcher/Cam_1_2016-09-01T14:20:38.410794Z.jpg')

bb_sel_st = core.BB_SelectionStitcher()
bb_sel_st((img_l, img_r),(0,1))
result = bb_sel_st.overlay_images()
helpers.display(result, 'left restult')
