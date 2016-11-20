import cv2
import fb_stitcher.composer.composer as composer

import logging.config


logging.config.fileConfig('logging_config.ini')

img_l = cv2.imread('data/stitcher/Input/fg_sub_imgs/'
                   'Cam_0_20140918120038_045539_TO_Cam_0_20140918120622_267825.jpg')
img_r = cv2.imread('data/stitcher/Input/fg_sub_imgs/'
                   'Cam_1_20140918120522_949626_TO_Cam_1_20140918121058_677877.jpg')

comp = composer.Composer()
comp(img_l, img_r)
