import numpy as np
from logging import getLogger
import cv2
import affine_stitcher.config as config
import math

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


# def get_points_n_matches(kps1, kps2, matches, ratio=0.75):
#     good_matches = lowe_ratio_test(matches, ratio)
#     pts1, pts2 = get_matching_points(kps1, kps2, good_matches)
#     return pts1, pts2, good_matches

def get_points_n_matches(kps1, kps2, matches, ratio=1, max_shift = config.SHIFT):
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            m = m[0]
            dist = abs(np.array(kps1[m.queryIdx].pt)-np.array(kps2[m.trainIdx].pt))
            log.debug(dist)
            if dist[1] < max_shift:
                log.debug(dist[1])
                good_matches.append(m)
    # good_matches = lowe_ratio_test(matches, ratio)
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


# def unit_vector(vector):
#     """ Returns the unit vector of the vector.  """
#     return vector / np.linalg.norm(vector)
#
# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::
#
#             >>> angle_between((1, 0, 0), (0, 1, 0))
#             1.5707963267948966
#             >>> angle_between((1, 0, 0), (1, 0, 0))
#             0.0
#             >>> angle_between((1, 0, 0), (-1, 0, 0))
#             3.141592653589793
#     """
#     v1_u = unit_vector(v1)
#     v2_u = unit_vector(v2)
#     return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def cart_2_pol(pt):
    rho = np.sqrt(pt[0]**2+pt[1]**2)
    phi = math.degrees(np.arctan2(pt[1],pt[0]))
    return rho, phi


def bla(center_l, center_r, pt_l, pt_r):

    # get the translation vector
    tr_vec = np.subtract(center_l, center_r)

    # translate the right point
    trans_pt_r = np.add(pt_r, tr_vec)

    # convert to polar coordinates
    pt_l = cart_2_pol(pt_l)
    trans_pt_r = cart_2_pol(trans_pt_r)

    # get the rotation angle of right point to be in one line with pt_l
    rot_angle = trans_pt_r[1] - pt_l[1]

    roation_mat = np.vstack([cv2.getRotationMatrix2D((0,0), rot_angle, 1.0),
                             [0, 0, 1]])



    log.debug('Euclidean rotation matrix = \n{}'.format(roation_mat))

    translation = np.array([
        [1, 0, tr_vec[0]],  # x
        [0, 1, tr_vec[1]],  # y
        [0, 0, 1]
    ], np.float64)

    euclidean = roation_mat.dot(translation)

    log.debug('Euclidean Transormation =\n{}'.format(euclidean))

    return euclidean




