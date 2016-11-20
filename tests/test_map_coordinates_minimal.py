import fb_stitcher.core as core
import numpy as np

camIdx = 0

pts_left_org = np.array([[[1000, 3000], [353, 400], [369, 2703]]]).astype(np.float64)

stitcher = core.BB_FeatureBasedStitcher()
stitcher.load_data('data/test_map_coordinates_2/Input/out.npz')
pts_mapped = stitcher.map_cam_coordinates(camIdx, pts_left_org)

print(pts_mapped)
