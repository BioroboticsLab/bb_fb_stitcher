import affine_stitcher.helpers as helpers
from os.path import basename
import cv2
import os

left_video = ('data/foreground_subtractor/Input/'
              'Cam_0_20140918120038_045539_TO_Cam_0_20140918120622_267825.mkv')
right_video = ('data/foreground_subtractor/Input/'
               'Cam_1_20140918120522_949626_TO_Cam_1_20140918121058_677877.mkv')

if not os.path.exists('data/foreground_subtractor/Output'):
    os.makedirs('data/foreground_subtractor/Output')

left_fname = basename(left_video)
right_fname = basename(right_video)
left_cap = cv2.VideoCapture(left_video)
left_bg = helpers.subtract_foreground(left_cap, display=True)
left_cap.release()
right_cap = cv2.VideoCapture(right_video)
right_bg = helpers.subtract_foreground(right_cap, display=True)
right_cap.release()
helpers.display(left_bg)
helpers.display(right_bg)

cv2.imwrite('data/foreground_subtractor/Output/{}.jpg'.format(left_fname),
            left_bg)
cv2.imwrite('data/foreground_subtractor/Output/{}.jpg'.format(right_fname),
            right_bg)
