import logging.config
import fb_stitcher.core as core
import cv2
import fb_stitcher.helpers as helpers
from fb_stitcher.stitcher import Transformation

logging.config.fileConfig('logging_config.ini')

img_l =cv2.imread('data/core_comp_stitcher/Cam_0_2016-09-01T14:20:38.410765Z.jpg')
img_r =cv2.imread('data/core_comp_stitcher/Cam_1_2016-09-01T14:20:38.410794Z.jpg')

bb_sel_st = core.BB_SelectionStitcher()
bb_sel_st((img_l, img_r),(0,1))
result = bb_sel_st.overlay_images()
helpers.display(result, 'left restult')
#
# img_l_alpha = cv2.cvtColor(img_l, cv2.COLOR_BGR2BGRA)
# left = bb_stitcher_fb.transform_left_image(img_l_alpha)
# cv2.imwrite('data/core/Output/left.png', left)
#
# img_r_alpha = cv2.cvtColor(img_r, cv2.COLOR_BGR2BGRA)
# right = bb_stitcher_fb.transform_right_image(img_r_alpha)
# cv2.imwrite('data/core/Output/right.png', right)
#
# cv2.imwrite('data/core/Output/result.jpg', result)
# bb_stitcher_fb.save_data('data/core/Output/out')
#
# # helpers.display(res, 'right result')