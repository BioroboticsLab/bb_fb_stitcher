"""Specific parameters for the Beesbook Project.

These are the camera parameter for the
# PointGrey Flea3 Camera with the objective RICOH FL-CC1214A-2M
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


#  max vertical shift value of images, defines how much the features of left and
#  right image can be differ from each other in the y-direction.
SHIFT = 200

# valid BeesBook videofilenames
FILE_NAMES = ( '^Cam_\d_\d{14}_\d{3,6}_TO_Cam_\d_\d{14}_\d{3,6}.jpg$|'
                '^Cam_\d_\d{14}_\d_TO_Cam_\d_\d{14}_\d.jpg$|'
                '^Cam_\d_\d{14}__\d_TO_Cam_\d_\d{14}__\d.jpg$|'
                '^Cam_\d_\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z--\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z.jpg$')
