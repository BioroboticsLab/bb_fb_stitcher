import affine_stitcher.config as config
import cv2
from logging import getLogger

log = getLogger(__name__)

class Rectificator(object):
    def __init__(self, intrinsic_mat=config.INTR_M, distortion_coeff=config.DIST_C):
        self.intr_m = intrinsic_mat
        self.dist_c = distortion_coeff
        self.cached_new_cam_mat = None
        self.cached_dim = None

    def rectify_images(self, *images):
        log.info('Start rectification.')
        if not images:
            log.warning('List of images for rectification is empty.')
            return None

        rect_imgs = []
        for img in images:
            if self.cached_new_cam_mat is None or self.cached_dim != img.shape[:2]:
                self.cached_dim = img.shape[:2]
                h, w = img.shape[:2]
                self.cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(self.intr_m, self.dist_c, (w, h), 1, (w, h), 0)
                log.debug('new_camera_mat = \n{}'.format(self.cached_new_cam_mat))
            rect_imgs.append(cv2.undistort(img, self.intr_m, self.dist_c, None, self.cached_new_cam_mat))

        if len(rect_imgs) == 1:
            return rect_imgs[0]

        return rect_imgs

    def rectify_points(self, points, img_dim):
        h, w = img_dim
        self.cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(self.intr_m, self.dist_c, (w, h), 1, (w, h), 0)
        log.debug('new_camera_mat = \n{}'.format(self.cached_new_cam_mat))
        return cv2.undistortPoints(points, self.intr_m, self.dist_c, None, self.cached_new_cam_mat)
