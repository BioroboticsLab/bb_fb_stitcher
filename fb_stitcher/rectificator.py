import cv2
import fb_stitcher.config as config
from logging import getLogger

log = getLogger(__name__)


class Rectificator(object):
    """Class to rectify and remove lens distortion from images."""

    def __init__(self, intrinsic_mat=config.INTR_M, distortion_coeff=config.DIST_C):
        """Initialize a rectificator with camera parameters."""
        self.intr_m = intrinsic_mat
        self.dist_c = distortion_coeff
        self.cached_new_cam_mat = None
        self.cached_dim = None

    def rectify_images(self, *images):
        """Remove Lens distortion from images"""
        log.info('Start rectification of {} images.'.format(len(images)))
        if not images:
            log.warning('List of images for rectification is empty.')
            return None

        rect_imgs = []
        for img in images:
            if self.cached_new_cam_mat is None or self.cached_dim != img.shape[:2]:
                self.cached_dim = img.shape[:2]
                h, w = img.shape[:2]
                self.cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(
                    self.intr_m, self.dist_c, (w, h), 1, (w, h), 0)
                log.debug('new_camera_mat = \n{}'.format(self.cached_new_cam_mat))
            rect_imgs.append(cv2.undistort(
                img, self.intr_m, self.dist_c, None, self.cached_new_cam_mat))

        if len(rect_imgs) == 1:
            return rect_imgs[0]

        return rect_imgs

    def rectify_points(self, points, size):
        """Maps points from distorted image to its pos in an undistorted img. """
        log.info(size)
        self.cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(self.intr_m,
                                                                    self.dist_c,
                                                                    size, 1,
                                                                    size, 0)
        log.debug('new_camera_mat = \n{}'.format(self.cached_new_cam_mat))
        return cv2.undistortPoints(points, self.intr_m, self.dist_c, None, self.cached_new_cam_mat)
