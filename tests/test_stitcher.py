import cv2
import fb_stitcher.stitcher
from fb_stitcher.stitcher import Transformation
import logging.config

logging.config.fileConfig('logging_config.ini')

img_l = cv2.imread('data/stitcher/Input/fg_sub_imgs/'
                   'Cam_0_20140918120038_045539_TO_Cam_0_20140918120622_267825.jpg', -1)
img_r = cv2.imread('data/stitcher/Input/fg_sub_imgs/'
                   'Cam_1_20140918120522_949626_TO_Cam_1_20140918121058_677877.jpg', -1)

st = fb_stitcher.stitcher.FeatureBasedStitcher(400, 500, transformation=Transformation.EUCLIDEAN)
st((img_l, img_r))

# helpers.display(result, 'result')
# helpers.display(vis, 'matches')
