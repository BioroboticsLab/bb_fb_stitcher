import numpy as np
from logging import getLogger

log = getLogger(__name__)

def lowe_ratio_test(kps1, kps2, matches, ratio = 0.75):
    kps1_impr = []
    kps2_impr = []
    good = []
    log.debug('start lowe')
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

def lowe_ratio_test_affine(kps1, kps2, matches, ratio = 0.75):
    good = []
    log.debug('start lowe affine')
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio*m[1].distance:
            good.append(m[0])
    best = sorted(good, key=lambda x: x.distance)[:3]
    pts1 = np.float32([kps1[b.queryIdx].pt for b in best]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[b.trainIdx].pt for b in best]).reshape(-1, 1, 2)
    log.debug('#pts1 = {}'.format(len(pts1)))
    log.debug('#pts2 = {}'.format(len(pts2)))
    return pts1, pts2, best

def display(im,title='image', jupyter=False):
    if jupyter:
        plt.figure(figsize=(20, 20))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.title(title)
        plt.show()
    else:
        left_h, left_w = im.shape[:2]
        sh = 800
        sw = int(left_w * sh / left_h)
        sm = cv2.resize(im, (sw, sh))
        cv2.imshow(title, sm)
        cv2.waitKey(0)
