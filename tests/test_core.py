import logging.config
import affine_stitcher.core as core
import cv2
import affine_stitcher.helpers as helpers
from affine_stitcher.stitcher import Transformation

logging.config.fileConfig('logging_config.ini')
#
img_l =cv2.imread('data/core/Cam_0_2016-07-19T12:41:22.353295Z--2016-07-19T12:47:02.199607Z.jpg',-1)
img_r =cv2.imread('data/core/Cam_1_2016-07-19T12:41:22.685374Z--2016-07-19T12:47:02.533678Z.jpg',-1)
# img_l =cv2.imread('data/core/Cam_0.jpg',0)
# img_r =cv2.imread('data/core/Cam_1.jpg',0)
# img_l =cv2.imread('data/core/Cam_0_20161507130847_631282517.jpg')
# img_r =cv2.imread('data/core/Cam_1_20161507130847_631282517.jpg')
# img_l =cv2.imread('/mnt/myStorage/bb_affine_stitcher/tests/data/core/dog_left.jpg')
# img_r =cv2.imread('/mnt/myStorage/bb_affine_stitcher/tests/data/core/dog_right.jpg')
helpers.display(img_l,'left image', time=50)
helpers.display(img_r, 'right image', time=50)
bb_stitcher_fb = core.BB_FeatureBasedStitcher()
__, __, __, result = bb_stitcher_fb((img_l, img_r), True)
helpers.display(result, 'left restult')

img_l_alpha = cv2.cvtColor(img_l, cv2.COLOR_BGR2BGRA)
left = bb_stitcher_fb.transform_left_image(img_l_alpha)
cv2.imwrite('left.png', left)

img_r_alpha = cv2.cvtColor(img_r, cv2.COLOR_BGR2BGRA)
right = bb_stitcher_fb.transform_right_image(img_r_alpha)
cv2.imwrite('right.png', right)

cv2.imwrite('result.jpg', result)

# helpers.display(res, 'right result')
