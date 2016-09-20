import affine_stitcher.stitcher
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging.config

logging.config.fileConfig('logging_config.ini')
img_l = cv2.imread('data/building1.JPG')
img_r = cv2.imread('data/building2.JPG')
# cv2.imshow('test',img_l)
# cv2.waitKey(0)
st = affine_stitcher.stitcher.Stitcher()
h,w = img_l.shape[:2]

mask = np.zeros((h,w), np.uint8)
# mask[:1000,:2000]=None
mask[:]=255
# print(mask[:10,:10])
result, vis = st((img_l, img_r), True)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.imshow('matches', vis)
cv2.waitKey(0)
(kps, des) = st.get_keypoints_and_descriptors(img_l, mask)

# print(kps)

left_features = cv2.drawKeypoints(
    img_l, kps, None, (0, 0, 255), 4)

fig, axes = plt.subplots(1, 2, figsize=(16,8), sharey=True)
axes[0].imshow(cv2.cvtColor(left_features, cv2.COLOR_BGR2RGB))
axes[1].imshow(cv2.cvtColor(left_features, cv2.COLOR_BGR2RGB))
plt.show()
