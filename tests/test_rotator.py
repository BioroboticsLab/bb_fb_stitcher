import fb_stitcher.rotator as rotator
import logging.config
import cv2

logging.config.fileConfig('logging_config.ini')
img_1 = cv2.imread('data/rotator/Input/1.jpg')
img_2 = cv2.imread('data/rotator/Input/2.jpg')

rot = rotator.Rotator()
rot_img_1=rot.rotate_image(img_1,-90)
rot_img_2=rot.rotate_image(img_2,90)
cv2.imwrite('data/rotator/output/1_t.jpg', rot_img_1)
cv2.imwrite('data/rotator/output/2_t.jpg', rot_img_2)
