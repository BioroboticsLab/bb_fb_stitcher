import cv2
import affine_stitcher.helpers as helpers
from skimage.feature import blob_dog
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import exposure

def dog(img):
    k = 10
    sigma1 = 0.5
    sigma2 = sigma1*k
    radius1 = cv2.GaussianBlur(img, (3, 3), sigma1).astype(float)

    radius2 = cv2.GaussianBlur(img, (51, 51), sigma2).astype(float)
    result = radius2 - radius1
    return result

img_l = cv2.imread('data/test_Edge_Detectors/Input/Cam_0_2016-07-19T12:41:22.353295Z--2016-07-19T12:47:02.199607Z.jpg',0)
img_r = cv2.imread('data/test_Edge_Detectors/Input/Cam_1_2016-07-19T12:41:22.685374Z--2016-07-19T12:47:02.533678Z.jpg',0)
# print(img_l.shape)
# res = dog(img_l)
img_r_dog = dog(img_r)
# img_r_dog = exposure.adjust_gamma(sobel(img_l),0.5)
# # img_l_dog = cv2.Canny(img_l, 10, 100)
# print(img_l_dog.shape)
# # cv2.normalize(img_r_dog, img_r_dog, 0, 255, cv2.NORM_MINMAX)
# cv2.imwrite('data/test_Edge_Detectors/dog_left.jpg', img_l_dog)
# # cv2.imwrite('data/test_Edge_Detectors/dog_right.jpg', img_r_dog)
# helpers.display(img_r_dog)
# k1 = cv2.getGaussianKernel(3, 0.6)
# k2 = cv2.getGaussianKernel(101, 4.2)
# # cv2.filter2D(img_l, cv2)
# # img = cv2.GaussianBlur(img_r,(51,51),0)
# f1 = cv2.filter2D(img_l,cv2.CV_32F, k1)
# f2 = cv2.filter2D(img_l,cv2.CV_32F, k2)
# res = f1-f2
helpers.display(img_r_dog)
