from logging import getLogger
import affine_stitcher.rectificator as rect
import affine_stitcher.rotator as rot
import affine_stitcher.stitcher as stitch
import affine_stitcher.helpers as helpers
from affine_stitcher.stitcher import Transformation
from skimage.filters import roberts, sobel
from skimage.feature import blob_dog
from skimage import exposure
import numpy as np
from skimage import img_as_ubyte
import cv2

log = getLogger(__name__)

class BB_FeatureBasedSticher(object):

    def stitch(self, images):
        (img_l, img_r) = images

        re = rect.Rectificator()
        print(img_l.dtype)
        img_l_re, img_r_re = re.rectify_images(img_l, img_r)
        # img_l_re =  exposure.adjust_log(sobel(img_l_re),0.5)
        # img_r_re =  exposure.adjust_log(sobel(img_r_re),0.5)
        # helpers.display(img_l_re, 'vorher')
        # img_l_re = img_as_ubyte(img_l_re)
        # img_r_re = img_as_ubyte(img_r_re)
        # helpers.display(img_l_re, 'nacher')
        # # img_r_re = sobel(img_r_re)
        ro = rot.Rotator()
        img_l_ro = ro.rotate_image(img_l_re, 90)
        img_r_ro = ro.rotate_image(img_r_re, -90)
        helpers.display(img_l_ro, 'img_l_ro', time=500)
        helpers.display(img_r_ro, 'img_r_ro', time=500)
        st =  stitch.FeatureBasedStitcher(overlap=400, border=450, transformation=Transformation.PROJECTIVE)
        homo, result, vis = st((img_l_ro, img_r_ro), True)
        helpers.display(vis)
        cv2.destroyAllWindows()
        return result

