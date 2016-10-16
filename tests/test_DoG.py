import cv2
import fb_stitcher.helpers as helpers


def dog(img):
    k = 10
    sigma1 = 0.5
    sigma2 = sigma1*k
    radius1 = cv2.GaussianBlur(img, (3, 3), sigma1).astype(float)
    radius2 = cv2.GaussianBlur(img, (51, 51), sigma2).astype(float)
    result = radius1 - radius2
    return result

img_l = cv2.imread('data/test_Edge_Detectors/Input/Cam_0_2016-07-19T12:41:22.353295Z--2016-07-19T12:47:02.199607Z.jpg',0)
img_r = cv2.imread('data/test_Edge_Detectors/Input/Cam_1_2016-07-19T12:41:22.685374Z--2016-07-19T12:47:02.533678Z.jpg',0)
img_r_dog = dog(img_r)

helpers.display(img_r_dog)
