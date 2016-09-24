from logging import getLogger
import affine_stitcher.rectificator as rect
import affine_stitcher.rotator as rot
import affine_stitcher.stitcher as stitch
from affine_stitcher.stitcher import Transformation
import affine_stitcher.helpers as helpers
import cv2

log = getLogger(__name__)

class BB_FeatureBasedStitcher(object):

    def __init__(self):
        self.whole_transform_left = None
        self.whole_transform_right = None
        self.new_size = None
        self.cached_img_l = None
        self.cached_img_r = None

    def __call__(self, images, pano=False):
        (self.cached_img_l, self.cached_img_r) = images

        re = rect.Rectificator()
        self.cached_img_l, self.cached_img_r = re.rectify_images(self.cached_img_l, self.cached_img_r)
        ro = rot.Rotator()
        img_l_ro, img_l_ro_mat = ro.rotate_image(self.cached_img_l, 90, True)
        img_r_ro, img_r_ro_mat = ro.rotate_image(self.cached_img_r, -90, True)

        st =  stitch.FeatureBasedStitcher(overlap=400, border=500, transformation=Transformation.AFFINE)
        homo = st((img_l_ro, img_r_ro))

        self.whole_transform_left = img_l_ro_mat
        self.whole_transform_right = homo.dot(img_r_ro_mat)

        trans_m , self.new_size = helpers.get_translation(self.cached_img_l.shape[:2], self.cached_img_r.shape[:2], self.whole_transform_left, self.whole_transform_right)

        self.whole_transform_left = trans_m.dot(self.whole_transform_left)
        self.whole_transform_right = trans_m.dot(self.whole_transform_right)

        if pano:
            pano_img = st.warp_images()
            return self.whole_transform_left, self.whole_transform_right, pano_img

        return self.whole_transform_left, self.whole_transform_right

    def transform_left_image(self, img=None):
        if img is None:
            img = self.cached_img_l
        re = rect.Rectificator()
        img_rect = re.rectify_images(img)
        trans_img = cv2.warpPerspective(img_rect,self.whole_transform_left, self.new_size )
        return trans_img

    def transform_right_image(self, img=None):
        if img is None:
            img = self.cached_img_r
        re = rect.Rectificator()
        img_rect = re.rectify_images(img)
        trans_img = cv2.warpPerspective(img_rect, self.whole_transform_right, self.new_size )
        return trans_img
