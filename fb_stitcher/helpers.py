import numpy as np
from logging import getLogger
import cv2
import fb_stitcher.config as config
import math
import re
import os

log = getLogger(__name__)


def lowe_ratio_test(matches, ratio=0.75):
    """Execute Lowe's ration test."""
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            m = m[0]
            good_matches.append(m)
    return good_matches


def get_matching_points(kps1, kps2, matches):
    """Return the matching points of kps1, kps2"""
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2


def get_top_matches(kps1, kps2, matches, num=None):
    """Return the the <num> Top matches with the minimum distance to each other."""

    # Sort matches by the distance (hamming) of their points.
    top_matches = sorted(matches, key=lambda m: m.distance)[:num]
    pts1, pts2 = get_matching_points(kps1, kps2, top_matches)
    return pts1, pts2, top_matches


def get_points_n_matches(kps1, kps2, matches, ratio=1, max_shift=config.SHIFT):
    """Estimate the best matches with a max_shift in y-direction."""
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            m = m[0]

            # Checks the distance of the points in the y-direction.
            dist = abs(np.array(kps1[m.queryIdx].pt)-np.array(kps2[m.trainIdx].pt))
            if dist[1] < max_shift:
                good_matches.append(m)

    pts1, pts2 = get_matching_points(kps1, kps2, good_matches)
    return pts1, pts2, good_matches


def get_points_n_matches_affine(kps1, kps2, matches, ratio=0.75):
    """Get the best 3 points with the smallest distance to each other."""
    good_matches = lowe_ratio_test(matches, ratio)
    pts1, pts2, best_matches = get_top_matches(kps1, kps2, good_matches, 3)
    return pts1, pts2, best_matches


def get_mask_matches(matches, mask):
    """Return the masked matches."""
    good_matches = []
    for i, m in enumerate(matches):
        if mask[i] == 1:
            good_matches.append(m)
    return good_matches


def calculate_num_matches(mask):
    """Count the number of good matches from mask."""
    i = 0
    for m in mask:
        if m == 1:
            i += 1
    return i


def display(im, title='image', time=0):
    """Display image."""
    left_h, left_w = im.shape[:2]
    sh = 800
    sw = int(left_w * sh / left_h)
    sm = cv2.resize(im, (sw, sh))
    cv2.imshow(title, sm)
    cv2.waitKey(time)
    cv2.destroyWindow(title)


def subtract_foreground(cap, show=False):
    """Subtract the moving foreground of an video."""
    fgbg = cv2.createBackgroundSubtractorMOG2()
    bgimg = None
    try:

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
    except KeyboardInterrupt:
        pass
    return bgimg


def cart_2_pol(pt):
    """Convert cartesian coordinate to polar coordinate."""
    rho = np.sqrt(pt[0]**2+pt[1]**2)
    phi = math.degrees(np.arctan2(pt[1], pt[0]))
    return rho, phi


def get_euclid_transform_mat(center_l, center_r, pt_l, pt_r):
    """Calculate the euclidean transformation of two point pairs."""

    # get the translation vector
    tr_vec = np.subtract(center_l, center_r)

    # translate the right point
    trans_pt_r = np.add(pt_r, tr_vec)

    # convert to polar coordinates
    pt_l = cart_2_pol(pt_l)
    trans_pt_r = cart_2_pol(trans_pt_r)

    # get the rotation angle of right point to be in one line with pt_l
    rot_angle = trans_pt_r[1] - pt_l[1]

    # -rot_angle opencv measures angle counter clockwise
    rotation_mat = np.vstack([cv2.getRotationMatrix2D((0, 0), -rot_angle, 1.0),
                             [0, 0, 1]])

    log.debug('Euclidean rotation matrix = \n{}'.format(rotation_mat))

    translation = np.array([
        [1, 0, tr_vec[0]],  # x
        [0, 1, tr_vec[1]],  # y
        [0, 0, 1]
    ], np.float64)

    euclidean = rotation_mat.dot(translation)

    log.debug('Euclidean Transformation =\n{}'.format(euclidean))

    return euclidean


def get_translation(shape_l, shape_r, homo_mat_l, homo_mat_r):
    """Determine the translation matrix of two transformed images.
    When two images have been transformed by an homography, it's possible
    that they are not aligned with the displayed area anymore. So they need to
    be translated and the display area must be increased.
    """
    # get origin width and height of images
    w_l, h_l = shape_l
    w_r, h_r = shape_r
    log.debug('(h_l,w_l) = {}'.format(shape_l))

    # define the corners of the left and right image.
    corners_l = np.float32([
        [0, 0],
        [0, w_l],
        [h_l, w_l],
        [h_l, 0]
    ]).reshape(-1, 1, 2)
    corners_r = np.float32([
        [0, 0],
        [0, w_r],
        [h_r, w_r],
        [h_r, 0]
    ]).reshape(-1, 1, 2)

    # transform the corners of the images, to get the dimension of the
    # transformed images and stitched image
    corners_tr_l = cv2.perspectiveTransform(corners_l, homo_mat_l)
    corners_tr_r = cv2.perspectiveTransform(corners_r, homo_mat_r)
    pts = np.concatenate((corners_tr_l, corners_tr_r), axis=0)

    # measure the max values in x and y direction to get the translation vector
    # so that whole image will be shown
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    # define translation matrix
    trans_m = np.array(
        [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
    total_size = (xmax - xmin, ymax - ymin)

    return trans_m, total_size


def overlay_images(foreground_image, background_image):
    result = np.copy(foreground_image)
    for r, row in enumerate(result):
        for c, cell in enumerate(row):
            if cell[3] == 0:
                result[r][c] = background_image[r][c]
    return result


def check_filename(filename):
    """Check if the filename is a valid background image."""
    correct_pattern = re.compile(config.FILE_NAMES)
    return None is not re.match(correct_pattern, filename)


def get_CamIdx(filename):
    """Gets the cam idx from the the filename."""
    basename = os.path.basename(filename)
    if not check_filename(basename):
        return None
    camIdxStr = basename.split('_', 2)[1]
    return int(camIdxStr)


def get_start_datetime(filename):
    basename = os.path.basename(filename)
    datetime = re.split('--|_TO_', basename)[0]
    return datetime
