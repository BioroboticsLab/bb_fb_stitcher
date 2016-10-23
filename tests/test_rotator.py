import fb_stitcher.rotator as rotator
import logging.config
import cv2
import numpy as np

logging.config.fileConfig('logging_config.ini')
img_1 = cv2.imread('data/rotator/Input/1.jpg')
img_2 = cv2.imread('data/rotator/Input/2.jpg')

rot = rotator.Rotator()
rot_img_1=rot.rotate_image(img_1,-90)
rot_img_2=rot.rotate_image(img_2,90)
cv2.imwrite('data/rotator/output/1_t.jpg', rot_img_1)
cv2.imwrite('data/rotator/output/2_t.jpg', rot_img_2)
pts_left_org = np.array([[[1000.666, 3000], [353, 400], [369, 2703], [3647, 155], [3647, 2737], [
                        1831, 1412], [361, 1522], [3650, 1208], [1750, 172]]]).astype(np.float64)
print(rot.rotate_points(pts_left_org, 90 , img_1.shape))