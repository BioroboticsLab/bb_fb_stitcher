import affine_stitcher.rectificator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging.config
import affine_stitcher.helpers as helpers

logging.config.fileConfig('logging_config.ini')


def draw_makers(img, pts, color=(0, 0, 255),
                marker_types=cv2.MARKER_TILTED_CROSS):
    img_m = np.copy(img)
    pts = pts[0].astype(int)
    for pt in pts:
        cv2.drawMarker(img_m, tuple(pt), color, markerType=marker_types,
                       markerSize=40, thickness=5)
    return img_m

img_l = cv2.imread('data/rectificator/Input/Cam_0_20161507130847_631282517.jpg')
img_r = cv2.imread('data/rectificator/Input/Cam_1_20161507130847_631282517.jpg')

pts_left_org = np.array([[[1000, 3000], [353, 400], [369, 2703], [3647, 155], [3647, 2737], [
                        1831, 1412], [361, 1522], [3650, 1208], [1750, 172]]]).astype(np.float64)
pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
    3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]).astype(np.float64)

img_l_m = draw_makers(img_l, pts_left_org, (255, 0, 0), cv2.MARKER_CROSS)
img_r_m = draw_makers(img_r, pts_right_org, (255, 0, 0), cv2.MARKER_CROSS)

rect = affine_stitcher.rectificator.Rectificator()

imgl, imgr = rect.rectify_images(img_l_m, img_r_m)
helpers.display(imgl)
helpers.display(imgr)

pts_left_rect = rect.rectify_points(pts_left_org, img_l.shape[:2])
pts_right_rect = rect.rectify_points(pts_right_org, img_r.shape[:2])

img_l_m = draw_makers(imgl, pts_left_rect, (0, 0, 255))
img_r_m = draw_makers(imgr, pts_right_rect, (0, 0, 255))

helpers.display(img_l_m)
helpers.display(img_r_m)
