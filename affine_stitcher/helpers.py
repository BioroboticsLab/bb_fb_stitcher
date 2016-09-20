import numpy as np
from logging import getLogger

log = getLogger(__name__)

def lowe_ratio_test(kps1, kps2, matches, ratio = 0.75):
    kps1_impr = []
    kps2_impr = []
    good = []
    log.debug('stat lowe')
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio*m[1].distance:
            m = m[0]
            kps1_impr.append(kps1[m.queryIdx])
            kps2_impr.append(kps2[m.trainIdx])
            good.append(m)
    pts1 = np.float32([kp.pt for kp in kps1_impr])
    pts2 = np.float32([kp.pt for kp in kps2_impr])
    log.debug('#pts1 = {}'.format(len(pts1)))
    log.debug('#pts2 = {}'.format(len(pts2)))
    return pts1, pts2, good
