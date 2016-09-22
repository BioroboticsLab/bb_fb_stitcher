import affine_stitcher.rotator as rotator
import logging.config
import cv2
import affine_stitcher.helpers as helpers
import numpy as np
logging.config.fileConfig('logging_config.ini')
img = cv2.imread('data/rotator/Input/Cam_0_20161507130847_631282517.jpg')

rot = rotator.Rotator()
rot_img=rot.rotate_image(img,45)
new_rot_img=rot.rotate_image(img,80)
helpers.display(rot_img, 'result')
pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
    3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]).astype(np.float64)
print(rot.rotate_points(pts_right_org))
