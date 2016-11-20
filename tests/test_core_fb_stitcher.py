import cv2
import fb_stitcher.core as core
import fb_stitcher.helpers as helpers
import logging.config

logging.config.fileConfig('logging_config.ini')

img_l = cv2.imread(
    'data/core/Input/Cam_0_2016-07-19T12:41:22.353295Z--2016-07-19T12:47:02.199607Z.jpg', -1)
img_r = cv2.imread(
    'data/core/Input/Cam_1_2016-07-19T12:41:22.685374Z--2016-07-19T12:47:02.533678Z.jpg', -1)

helpers.display(img_l, 'left image', time=50)
helpers.display(img_r, 'right image', time=50)
bb_stitcher_fb = core.BB_FeatureBasedStitcher()
bb_stitcher_fb((img_l, img_r), (0, 1))
result = bb_stitcher_fb.overlay_images()
helpers.display(result, 'left restult')

img_l_alpha = cv2.cvtColor(img_l, cv2.COLOR_BGR2BGRA)
left = bb_stitcher_fb.transform_left_image(img_l_alpha)
cv2.imwrite('data/core/Output/left.png', left)

img_r_alpha = cv2.cvtColor(img_r, cv2.COLOR_BGR2BGRA)
right = bb_stitcher_fb.transform_right_image(img_r_alpha)
cv2.imwrite('data/core/Output/right.png', right)

cv2.imwrite('data/core/Output/result.jpg', result)
bb_stitcher_fb.save_data('data/core/Output/out')

# helpers.display(res, 'right result')
