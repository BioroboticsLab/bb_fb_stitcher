import fb_stitcher.rotator as rotator
import logging.config
import cv2
import fb_stitcher.helpers as helpers
import numpy as np
logging.config.fileConfig('logging_config.ini')
img_1 = cv2.imread('data/rotator/Input/1.jpg')
img_2 = cv2.imread('data/rotator/Input/2.jpg')

rot = rotator.Rotator()
rot_img_1=rot.rotate_image(img_1,-90)
rot_img_2=rot.rotate_image(img_2,90)
cv2.imwrite('data/rotator/output/1_t.jpg', rot_img_1)
cv2.imwrite('data/rotator/output/2_t.jpg', rot_img_2)
# helpers.display(rot_img, 'result')
# pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
#     3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]).astype(np.float64)
# print('rotatet corners = \n{}'.format(rot.rotate_points(pts_right_org, 90, img.shape[:2])))
# cv2.imwrite('data/rotator/outpu/)