from logging import getLogger
import affine_stitcher.rectificator as rect
import affine_stitcher.rotator as rot
import affine_stitcher.stitcher as stitch
from affine_stitcher.stitcher import Transformation
import affine_stitcher.helpers as helpers
import cv2
import numpy as np

log = getLogger(__name__)

class BB_FeatureBasedStitcher(object):

    def __init__(self):
        self.img_l_size = None
        self.img_r_size = None
        self.whole_transform_left = None
        self.whole_transform_right = None
        self.pano_size = None
        self.cached_img_l = None
        self.cached_img_r = None

    def __call__(self, images, pano=False):
        (self.cached_img_l, self.cached_img_r) = images
        self.img_l_size = tuple([self.cached_img_l.shape[1],self.cached_img_l.shape[0]])
        self.img_r_size = tuple([self.cached_img_r.shape[1],self.cached_img_r.shape[0]])

        re = rect.Rectificator()
        self.cached_img_l, self.cached_img_r = re.rectify_images(self.cached_img_l, self.cached_img_r)
        ro = rot.Rotator()
        img_l_ro, img_l_ro_mat = ro.rotate_image(self.cached_img_l, 90, True)
        img_r_ro, img_r_ro_mat = ro.rotate_image(self.cached_img_r, -90, True)

        st =  stitch.FeatureBasedStitcher(overlap=400, border=500, transformation=Transformation.AFFINE)
        homo = st((img_l_ro, img_r_ro))

        self.whole_transform_left = img_l_ro_mat
        self.whole_transform_right = homo.dot(img_r_ro_mat)

        trans_m , self.pano_size = helpers.get_translation(self.cached_img_l.shape[:2], self.cached_img_r.shape[:2], self.whole_transform_left, self.whole_transform_right)
        log.debug('new_size =\n{}'.format(self.pano_size))

        self.whole_transform_left = trans_m.dot(self.whole_transform_left)
        self.whole_transform_right = trans_m.dot(self.whole_transform_right)

        if pano:
            pano_img = st.warp_images()
            return self.whole_transform_left, self.whole_transform_right, self.pano_size, pano_img

        return self.img_l_size, self.img_l_size, self.whole_transform_left, self.whole_transform_right, self.pano_size

    @staticmethod
    def transform_image(img, homography, pano_size):
        re = rect.Rectificator()
        img_rect = re.rectify_images(img)
        trans_img = cv2.warpPerspective(img_rect, homography, pano_size)
        return trans_img

    def transform_left_image(self, img=None):
        if img is None:
            img = self.cached_img_l
        trans_img = BB_FeatureBasedStitcher.transform_image(img, self.whole_transform_left, self.pano_size)
        return trans_img

    def transform_right_image(self, img=None):
        if img is None:
            img = self.cached_img_r
        trans_img = BB_FeatureBasedStitcher.transform_image(img, self.whole_transform_right, self.pano_size)
        return trans_img

    @staticmethod
    def map_coordinates(points, homography, org_size_img):
        log.info('Start mapping {} points.'.format(len(points)))
        re = rect.Rectificator()
        points_rect = re.rectify_points(points, org_size_img)
        return cv2.perspectiveTransform(points_rect, homography)

    def map_left_coordinates(self, points):
        return self.map_coordinates(points, self.whole_transform_left, self.img_l_size)

    def map_right_coordinates(self, points):
        return self.map_coordinates(points, self.whole_transform_right, self.img_r_size)

    def save_data(self, path):
        np.savez(path,
            img_l_size = self.img_l_size,
            img_r_size = self.img_r_size,
            whole_transform_left = self.whole_transform_left,
            whole_transform_right = self.whole_transform_right,
            pano_size = self.pano_size
        )
        log.info('Stitcher arguments saved to {}'.format(path))

    def load_data(self, path):
        with np.load(path) as data:
            self.img_l_size = tuple(data['img_l_size']) # savez doen't save the tuple info
            self.img_r_size = tuple(data['img_r_size'])
            self.whole_transform_left = data['whole_transform_left']
            self.whole_transform_right = data['whole_transform_right']
            self.pano_size = tuple(data['pano_size'])
        log.info('Stitcher arguments loaded from {}'.format(path))