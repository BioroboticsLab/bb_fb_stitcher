import affine_stitcher.helpers as helpers
import numpy as np
import cv2
from logging import getLogger
from enum import Enum

draw_params = dict(matchColor=(0, 255, 0),
                   outImg=None,
                   flags = 2
                   )

log = getLogger(__name__)

class Transformation(Enum):
    TRANSLATION = 0
    EUCLIDEAN = 1
    SIMILARITY = 2
    AFFINE = 3
    PROJECTIVE = 4

class FeatureBasedStitcher(object):

    def __init__(self, overlap=None, border=None, transformation = Transformation.AFFINE):
        # cached the homography
        self.cached_homo = None
        self.overlap = overlap
        self.border = border
        self.transformation = transformation
        self.cached_right_img = None
        self.cached_right_img = None

    def __call__(self, images, drawMatches=False):

        # get the images
        (self.cached_left_img, self.cached_right_img) = images

        if self.cached_homo is None:

            # calculates the mask which will mark the feature searching area.
            left_mask, right_mask = self.calc_feature_masks(self.cached_left_img.shape[:2], self.cached_right_img.shape[:2])

            # Searching for keypoints and descriptors in the images
            log.info('Start searching for features.')
            (left_kps, left_ds, left_features) = self.get_keypoints_and_descriptors(
                self.cached_left_img, left_mask, True)
            (right_kps, right_ds, right_features) = self.get_keypoints_and_descriptors(
                self.cached_right_img, right_mask, True)
            # helpers.display(right_features, time=500)
            log.debug('Features found: #left_kps = {} | #right_kps = {}'.format(
                len(left_kps), len(right_kps)))
            assert(len(left_kps)>0 and len(right_kps)>0)
            # selection of the planar transformation and searching for the right homography
            if self.transformation == Transformation.PROJECTIVE:
                (self.cached_homo, mask_good, good_matches) = self.transform_projective(left_kps, right_kps, left_ds, right_ds)
            elif self.transformation == Transformation.AFFINE:
                (self.cached_homo, mask_good, good_matches) = self.transform_affine(left_kps, right_kps, left_ds, right_ds)
            elif self.transformation == Transformation.EUCLIDEAN:
                (self.cached_homo, mask_good, good_matches) = self.transform_euclidean(left_kps, right_kps, left_ds, right_ds)
            elif self.transformation == Transformation.TRANSLATION:
                (self.cached_homo, mask_good, good_matches) = self.transform_translation(left_kps, right_kps, left_ds, right_ds)
            if self.cached_homo is None:
                log.warning('No Transformation matrix found.')
                return None
            log.info('Transformation matrix found.')

        # Creates Panorama from images
        log.debug('TM =\n{}'.format(self.cached_homo))

        if drawMatches:
            matchesMask = mask_good.ravel().tolist()
            result_matches = cv2.drawMatches(
                self.cached_left_img, left_kps, self.cached_right_img, right_kps, good_matches, matchesMask=None, **draw_params)
            return self.cached_homo, result_matches

        return self.cached_homo

    def calc_feature_masks(self, left_shape, right_shape):
        left_mask = np.ones(left_shape, np.uint8)*255
        right_mask = np.ones(right_shape, np.uint8)*255
        if self.overlap is not None:
            left_mask[:, :left_shape[1] - self.overlap] = 0
            right_mask[:, self.overlap:] = 0
        if self.border is not None:
            left_mask[:self.border, :] = 0
            left_mask[left_shape[0]-self.border:, :] = 0
            right_mask[:self.border, :] = 0
            right_mask[right_shape[0] - self.border:, :] = 0
        return left_mask, right_mask

    @staticmethod
    def get_keypoints_and_descriptors(img, mask=None, drawMatches=False):
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO opencv versions check

        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=2, nOctaves=4)
        surf.setUpright(True)
        surf.setExtended(128)

        kps, ds = surf.detectAndCompute(img, mask)

        if drawMatches:
            marked_matches = cv2.drawKeypoints(img, kps, None, (0, 0, 255), 4)
            return kps, ds, marked_matches

        return kps, ds

    @staticmethod
    def transform_projective(left_kps, right_kps, left_ds, right_ds):
        bf = cv2.BFMatcher()
        log.info('Start matching Features.')
        raw_matches = bf.knnMatch(left_ds, right_ds, k=2)
        log.debug('Matches found: #raw_matches = {}'.format(len(raw_matches)))
        left_pts, right_pts, good_matches = helpers.get_points_n_matches(
            left_kps, right_kps, raw_matches)
        log.debug('# Filtered Features = {}'.format(len(good_matches)))
        if len(left_pts) > 3:
            log.info('Start finding homography.')
            (homo, mask_good) = cv2.findHomography(
                right_pts, left_pts, cv2.RANSAC, 2.0)
            return homo, mask_good, good_matches
        return None

    def transform_affine(self, left_kps, right_kps, left_ds, right_ds):
        (homo, mask_good,
         good_matches) = FeatureBasedStitcher.transform_projective(left_kps,
                                                                   right_kps,
                                                                   left_ds,
                                                                   right_ds)
        if good_matches is None:
            log.warning('No homography matrix for further steps found.')
            return None
        better_matches = helpers.get_mask_matches(good_matches, mask_good)
        left_pts, right_pts, top_matches = helpers.get_top_matches(
            left_kps, right_kps, better_matches, 3)
        mask_top = np.ones((3, 1))
        if len(left_pts) > 2:
            log.info('Start finding affine transformation matrix.')
            affine = cv2.getAffineTransform(left_pts, right_pts)
            affine = cv2.invertAffineTransform(affine)
            affine = np.vstack([affine, [0, 0, 1]])
            return affine, mask_top, top_matches
        return None

    @staticmethod
    def transform_euclidean(left_kps, right_kps, left_ds, right_ds):
        (homo, mask_good,
         good_matches) = FeatureBasedStitcher.transform_projective(left_kps,
                                                                   right_kps,
                                                                   left_ds,
                                                                   right_ds)
        if good_matches is None:
            log.warning('No homography matrix for further steps found.')
            return None
        better_matches = helpers.get_mask_matches(good_matches, mask_good)
        left_pts, right_pts, top_matches = helpers.get_top_matches(
            left_kps, right_kps, better_matches, 2)
        mask_top = np.ones((2, 1))
        if len(left_pts) == 2:
            log.info('Start finding euclidean transformation matrix.')
            euclidean = helpers.get_euclid_transform_mat(left_pts[0][0], right_pts[0][0],
                                                         left_pts[1][0], right_pts[1][0])
            return euclidean, mask_top, top_matches
        return None

    def transform_translation(self, left_kps, right_kps, left_ds, right_ds):
        (homo, mask_good,
         good_matches) = FeatureBasedStitcher.transform_projective(left_kps,
                                                                   right_kps,
                                                                   left_ds,
                                                                   right_ds)
        if good_matches is None:
            log.warning('No homography matrix for further steps found.')
            return None
        better_matches = helpers.get_mask_matches(good_matches, mask_good)
        left_pts, right_pts, top_matches = helpers.get_top_matches(
            left_kps, right_kps, better_matches, 1)
        mask_top = np.ones((1, 1))
        if len(left_pts) == 1:
            log.info('Start finding translation transformation matrix.')
            tr_vec = np.subtract(left_pts[0][0], right_pts[0][0])
            log.debug('translation_vector{}'.format(tr_vec))
            translation = np.array([
                [1, 0, tr_vec[0]],  # x
                [0, 1, tr_vec[1]],  # y
                [0, 0, 1]
            ], np.float64)
            return translation, mask_top, top_matches
        return None

    def warp_images(self, left_img = None, right_img = None):
        if left_img is None:
            left_img = self.cached_left_img
        if right_img is None:
            right_img = self.cached_right_img
        result = cv2.warpPerspective(
            right_img, self.cached_homo,
            (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
        result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
        return result


    # deprecated
    @staticmethod
    def get_best_3_matches(left_kps, right_kps, left_ds, right_ds):
        bf = cv2.BFMatcher()
        log.info('Start matching Features.')
        raw_matches = bf.knnMatch(left_ds, right_ds, k=2)
        log.debug('Matches found: #raw_matches = {}'.format(len(raw_matches)))
        left_pts, right_pts, better_matches = helpers.get_points_n_matches_affine(
            left_kps, right_kps, raw_matches)
        log.debug('# Filtered Features = {}'.format(len(better_matches)))
        if len(left_pts) > 2:
            log.info('Start finding affine transformation matrix.')
            affine = cv2.getAffineTransform(left_pts, right_pts)
            affine = cv2.invertAffineTransform(affine)
            affine = np.vstack([affine, [0, 0, 1]])
            mask_better = np.ones((3, 1))
            return affine, mask_better, better_matches
        return None



