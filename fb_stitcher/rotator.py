from logging import getLogger
import numpy as np
import cv2

log = getLogger(__name__)


class Rotator(object):
    """Class to rotate image and to adjust the displayed area."""

    def __init__(self, shape=None, angle=None):
        self.shape = shape
        self.angle = angle
        self.rotation_mat = None
        self.size_new = None
        self.affine_mat = None
        if shape is not None and angle is not None:
            self.__get_affine_mat(shape, angle)

    def __get_affine_mat(self, shape, angle):
        """Calculate the affine transformation to rote image by given angle."""
        self.angle = angle
        self.shape = shape
        log.info('Start searching rotation_mat for angle {} and shape {}'.format(angle, shape))
        # Get img size
        size = (shape[1], shape[0])
        center = tuple(np.array(size) / 2.0)
        (width_half, height_half) = center

        # Convert the 3x2 rotation matrix to 3x3 ''homography''
        self.rotation_mat = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0),
                                       [0, 0, 1]])

        # To get just the rotation
        rot_matrix_2x2 = self.rotation_mat[:2, :2]

        # Declare the corners of the image in relation to the center
        corners = np.array([
            [-width_half, height_half],
            [width_half, height_half],
            [-width_half, -height_half],
            [width_half, -height_half]
        ])

        # get the rotated corners
        corners_rotated = corners.dot(rot_matrix_2x2)
        corners_rotated = np.array(corners_rotated, np.float32)

        # get the rectangle which would surround the rotated image
        __, __, w, h = cv2.boundingRect(np.array(corners_rotated))

        # boundingRect is 1px bigger so remove it
        self.size_new = (w-1, h-1)
        log.debug('size_new = {}'.format(self.size_new))

        # matrix to center the rotated image
        translation_matrix = np.array([
            [1, 0, int(w / 2 - width_half)],
            [0, 1, int(h / 2 - height_half)],
            [0, 0, 1]
        ])

        # get the affine Matrix
        self.affine_mat = translation_matrix.dot(self.rotation_mat)
        log.debug('affine_mat = \n{}'.format(self.affine_mat))
        return self.affine_mat

    def rotate_image(self, image, angle, ret=False):
        """Rotate image by given angle."""
        not_cached = self.affine_mat is None or self.size_new is None
        changed_val = self.shape != image.shape[:2] or self.angle != angle

        # Checks if previous an image with same properties has been rotated.
        # if not_cached or changed_val:
        self.__get_affine_mat(image.shape[:2], angle)

        rot_image = cv2.warpPerspective(image, self.affine_mat, self.size_new)
        if ret:
            return rot_image, self.affine_mat
        return rot_image

    def rotate_points(self, pts, angle=None, shape=None):
        """Rotate points in relation to image."""
        log.debug('Start rotate points.')
        if shape is None:
            shape = self.shape
        if angle is None:
            angle = self.angle

        not_cached = self.affine_mat is None or self.size_new is None
        changed_val = self.shape != shape or self.angle != angle
        if not_cached or changed_val:
            self.__get_affine_mat(shape, angle)
        return cv2.transform(pts, self.affine_mat[0:2])
