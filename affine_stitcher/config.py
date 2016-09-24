"""Specific parameters for the Beebsbook Project.

These are the camera parameter for the
# PointGrey Flea3 Camera with the objectiv RICOH FL-CC1214A-2M
# distance 30cm
"""
import numpy as np

#  camera Parameters
INTR_M = np.array([
    [7.779946033890540e+03, -8.272625320568793,     2.131922260976990e+03],
    [0,                     7.779892395260141e+03,  1.213789155361251e+03],
    [0,                     0,                      1]
    ])

RDIST_M = np.array([-0.092357757076643, 0.362707179300789, -0.442500973922752])
TDIST_M = np.array([-0.012841576857909, 0.001563484164365])
# TODO Check Distortion_Coeff
DIST_C = np.array(
    [RDIST_M[0], RDIST_M[1], TDIST_M[0], TDIST_M[1], RDIST_M[2]])


#  max vertical shift value of images
SHIFT = 200
# ______
#       |
#       |
# left  |
#       |
#_______|