import cv2
import logging.config
import fb_stitcher.core as core
import numpy as np


logging.config.fileConfig('logging_config.ini')


def draw_makers(img, pts, color=(0, 0, 255),
                marker_types=cv2.MARKER_TILTED_CROSS):
    img_m = np.copy(img)
    pts = pts[0].astype(int)
    for pt in pts:
        cv2.drawMarker(img_m, tuple(pt), color, markerType=marker_types,
                       markerSize=40, thickness=5)
    return img_m

img_l = cv2.imread(
    'data/test_map_coordinates/Input/'
    'Cam_0_2016-07-19T12:41:22.353295Z--2016-07-19T12:47:02.199607Z.jpg', -1)
img_r = cv2.imread(
    'data/test_map_coordinates/Input/'
    'Cam_1_2016-07-19T12:41:22.685374Z--2016-07-19T12:47:02.533678Z.jpg', -1)

pts_left_org = np.array([[[1000, 3000], [353, 400], [369, 2703], [3647, 155], [3647, 2737], [
                        1831, 1412], [361, 1522], [3650, 1208], [1750, 172]]]).astype(np.float64)
pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
    3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]).astype(np.float64)

img_l_m = draw_makers(img_l, pts_left_org, (255, 0, 0), cv2.MARKER_CROSS)
img_r_m = draw_makers(img_r, pts_right_org, (255, 0, 0), cv2.MARKER_CROSS)

st = core.BB_FeatureBasedStitcher()
st.load_data('data/test_map_coordinates/Input/out.npz')
pts_left_mapped = st.map_left_coordinates(pts_left_org)
pts_right_mapped = st.map_right_coordinates(pts_right_org)

print(pts_left_mapped)

img_left_mapped = st.transform_left_image(img_l_m)
img_right_mapped = st.transform_right_image(img_r_m)


img_left_marked = draw_makers(img_left_mapped, pts_left_mapped)
img_right_marked = draw_makers(img_right_mapped, pts_right_mapped)

cv2.imwrite('data/test_map_coordinates/Output/left_points_mapped.jpg', img_left_marked)
cv2.imwrite('data/test_map_coordinates/Output/right_points_mapped.jpg', img_right_marked)
