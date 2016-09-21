import affine_stitcher.stitcher
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging.config

def display(im,title='image', jupyter=False):
    if jupyter:
        plt.figure(figsize=(20, 20))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.title(title)
        plt.show()
    else:
        left_h, left_w = im.shape[:2]
        sh = 800
        sw = int(left_w * sh / left_h)
        sm = cv2.resize(im, (sw, sh))
        cv2.imshow(title, sm)
        cv2.waitKey(0)

logging.config.fileConfig('logging_config.ini')
# img_l = cv2.imread('data/Cam_0_20160715130847_631282.jpg')
# img_r = cv2.imread('data/Cam_1_20160715130847_631282.jpg')
img_l = cv2.imread('data/ferris_1.png')
img_r = cv2.imread('data/ferris_2.png')
# img_l = cv2.imread('data/building1.JPG')
# img_r = cv2.imread('data/building2.JPG')
# img_l = cv2.imread('data/Cam_0_rect.jpg')
# img_r = cv2.imread('data/Cam_1_rect.jpg')
# cv2.imshow('test',img_l)
# cv2.waitKey(0)
st = affine_stitcher.stitcher.Stitcher(1000)
homo, result, vis = st((img_l, img_r), True)

display(result, 'result')
display(vis, 'matches')
homo, result = st((img_l, img_r))

display(result, 'result')
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.imshow('matches', vis)
# cv2.waitKey(0)
