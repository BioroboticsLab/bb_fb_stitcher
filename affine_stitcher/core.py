from logging import getLogger
import affine_stitcher.rectificator as rect
import affine_stitcher.rotator as rot
import affine_stitcher.stitcher as stitch
import affine_stitcher.helpers as helpers
from affine_stitcher.stitcher import Transformation


log = getLogger(__name__)

class BB_FeatureBasedSticher(object):

    def stitch(self, images):
        (img_l, img_r) = images

        re = rect.Rectificator()
        img_l_re, img_r_re = re.rectify_images(img_l, img_r)

        ro = rot.Rotator()
        img_l_ro = ro.rotate_image(img_l, 90)
        img_r_ro = ro.rotate_image(img_r, -90)
        helpers.display(img_l_ro, 'img_l_ro')
        helpers.display(img_r_ro, 'img_r_ro')
        st =  stitch.FeatureBasedStitcher(overlap=400, transformation=Transformation.PROJECTIVE)
        homo, result, vis = st((img_l_ro, img_r_ro), True)
        helpers.display(vis)
        return result

