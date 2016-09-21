"""Specific paramets for the Beebsbook Project.

These are the camera parameter for the
# PointGrey Flea3 Camera with the objectiv RICOH FL-CC1214A-2M
# distance 30cm
"""
import numpy as np


intr_m = np.array([
    [7.779946033890540e+03, -8.272625320568793,     2.131922260976990e+03],
    [0,                     7.779892395260141e+03,  1.213789155361251e+03],
    [0,                     0,                      1]
    ])

rdist_m = np.array([-0.092357757076643, 0.362707179300789, -0.442500973922752])
tdist_m = np.array([-0.012841576857909, 0.001563484164365])
# TODO Check Distortion_Coeff
dist_c = np.array(
    [rdist_m[0], rdist_m[1], tdist_m[0], tdist_m[1], rdist_m[2]])
