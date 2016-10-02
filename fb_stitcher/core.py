"""Main Class of the feature based stitcher for the Beesbook Project."""
import cv2
import fb_stitcher.helpers as helpers
import fb_stitcher.rectificator as rect
import fb_stitcher.rotator as rot
import fb_stitcher.stitcher as stitch
from fb_stitcher.stitcher import Transformation
from logging import getLogger
import numpy as np

log = getLogger(__name__)


class BB_FeatureBasedStitcher(object):
    """Stitching the images of the BeesBook Project."""

    def __init__(self, transform=Transformation.AFFINE):
        """Initialize feature based Stitcher."""
        self.whole_transform_left = None
        self.whole_transform_right = None
        self.pano_size = None

        self.img_l_size = None
        self.img_r_size = None

        self.cached_img_l = None
        self.cached_img_r = None

        self.transform = transform

    def __call__(self, images, angles=(90, -90)):
        """Calculate Stitching data for further stitching."""
        (self.cached_img_l, self.cached_img_r) = images
        self.img_l_size = tuple([self.cached_img_l.shape[1], self.cached_img_l.shape[0]])
        self.img_r_size = tuple([self.cached_img_r.shape[1], self.cached_img_r.shape[0]])

        re = rect.Rectificator()
        img_l, img_r = re.rectify_images(self.cached_img_l, self.cached_img_r)
        ro = rot.Rotator()
        img_l_ro, img_l_ro_mat = ro.rotate_image(img_l, angles[0], True)
        img_r_ro, img_r_ro_mat = ro.rotate_image(img_r, angles[1], True)

        st =  stitch.FeatureBasedStitcher(overlap=400, border=500, transformation=self.transform)
        homo = st((img_l_ro, img_r_ro))

        self.whole_transform_left = img_l_ro_mat
        self.whole_transform_right = homo.dot(img_r_ro_mat)

        trans_m , self.pano_size = helpers.get_translation(img_l.shape[:2], img_r.shape[:2], self.whole_transform_left, self.whole_transform_right)
        log.debug('new_size =\n{}'.format(self.pano_size))

        self.whole_transform_left = trans_m.dot(self.whole_transform_left)
        self.whole_transform_right = trans_m.dot(self.whole_transform_right)

        return self.img_l_size, self.img_l_size, self.whole_transform_left, self.whole_transform_right, self.pano_size

    @staticmethod
    def transform_image(img, homography, pano_size):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
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

    def overlay_images(self, img_l = None, img_r = None):
        if img_l is None or img_r is None:
            img_l = self.cached_img_l
            img_r = self.cached_img_r
        trans_img_l = self.transform_left_image(img_l)
        trans_img_r = self.transform_right_image(img_r)
        return BB_FeatureBasedStitcher.blend_transparent(trans_img_l, trans_img_r)

    def blend_transparent(fg_img = None, bg_img = None, blur = True):
        # Split out the transparency mask from the colour info
        overlay_img = fg_img[:, :, :3] # Grab the BRG planes
        overlay_mask = fg_img[:, :, 3:]  # And the alpha plane

        if blur:
            # Let's shrink and blur it a little to make the transitions smoother...
            overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
            overlay_mask = cv2.blur(overlay_mask, (20, 20))

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out background image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        bg_part = (bg_img[:, :, :3]* (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(cv2.addWeighted(bg_part, 255.0, overlay_part, 255.0, 0.0))

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
