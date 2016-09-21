import numpy as np
from logging import getLogger
import cv2

log = getLogger(__name__)

def lowe_ratio_test(kps1, kps2, matches, ratio = 0.75):
    kps1_impr = []
    kps2_impr = []
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio*m[1].distance:
            m = m[0]
            kps1_impr.append(kps1[m.queryIdx])
            kps2_impr.append(kps2[m.trainIdx])
            good.append(m)
    pts1 = np.float32([kp.pt for kp in kps1_impr])
    pts2 = np.float32([kp.pt for kp in kps2_impr])
    return pts1, pts2, good

def lowe_ratio_test_affine(kps1, kps2, matches, ratio = 0.75):
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio*m[1].distance:
            good.append(m[0])
    best = sorted(good, key=lambda x: x.distance)[:3]
    pts1 = np.float32([kps1[b.queryIdx].pt for b in best]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[b.trainIdx].pt for b in best]).reshape(-1, 1, 2)
    return pts1, pts2, best

def get_Top_matches(kps1, kps2, matchesMask, head):
    better = []
    for i, val in enumerate(matchesMask):
        if val == 1:
            better.append(val)

def display(im,title='image'):
    left_h, left_w = im.shape[:2]
    sh = 800
    sw = int(left_w * sh / left_h)
    sm = cv2.resize(im, (sw, sh))
    cv2.imshow(title, sm)
    cv2.waitKey(500)
