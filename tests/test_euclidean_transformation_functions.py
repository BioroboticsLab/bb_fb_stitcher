import fb_stitcher.helpers as helpers
import logging.config
import numpy as np
print(helpers.cart_2_pol([1, 1]))
logging.config.fileConfig('logging_config.ini')

ptl_center = np.array([4, 1])
ptr_center = np.array([1,1])
ptl_rot = np.array([6,2])
ptr_rot = np.array([2,2])

print(helpers.get_euclid_transform_mat(ptl_center, ptr_center, ptl_rot, ptr_rot))

# print(helpers.angle_between(np.array([1,1]), np.array([0,1])))

