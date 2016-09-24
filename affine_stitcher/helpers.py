import numpy as np
from logging import getLogger
import cv2

log = getLogger(__name__)


def lowe_ratio_test(matches, ratio=0.75):
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            m = m[0]
            good_matches.append(m)
    return good_matches


def get_matching_points(kps1, kps2, matches):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2


def get_top_matches(kps1, kps2, matches, num=None):
    top_matches = sorted(matches, key=lambda m: m.distance)[:num]
    pts1, pts2 = get_matching_points(kps1, kps2, top_matches)
    return pts1, pts2, top_matches


def get_points_n_matches(kps1, kps2, matches, ratio=0.75):
    good_matches = lowe_ratio_test(matches, ratio)
    pts1, pts2 = get_matching_points(kps1, kps2, good_matches)
    return pts1, pts2, good_matches


def get_points_n_matches_affine(kps1, kps2, matches, ratio=0.75):
    good_matches = lowe_ratio_test(matches, ratio)
    pts1, pts2, best_matches = get_top_matches(kps1, kps2, good_matches, 3)
    return pts1, pts2, best_matches


def get_mask_matches(matches, mask):
    good_matches = []
    for i, m in enumerate(matches):
        if mask[i] == 1:
            good_matches.append(m)
    return good_matches


def calculate_num_matches(mask):
    i = 0
    for m in mask:
        if m == 1:
            i += 1
    return i


def display(im, title='image', time=0):
    left_h, left_w = im.shape[:2]
    sh = 800
    sw = int(left_w * sh / left_h)
    sm = cv2.resize(im, (sw, sh))
    cv2.imshow(title, sm)
    cv2.waitKey(time)
    cv2.destroyWindow(title)


def subtract_foreground(cap, show=False):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    bgimg = None
    while 1:
        ret, frame = cap.read()

        if ret:
            fgbg.apply(frame)
            bgimg = fgbg.getBackgroundImage()

            if show:
                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('img', 800, 600)
                cv2.imshow('img', bgimg)
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break
        else:
            break

    return bgimg
