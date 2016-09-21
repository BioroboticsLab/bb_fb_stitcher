import affine_stitcher.helpers as helpers
import numpy as np
import cv2
from logging import getLogger

draw_params = dict(matchColor=(0, 255, 0),
                   outImg=None,
                   flags = 2
                   )

log = getLogger(__name__)


class Stitcher(object):

    def __init__(self, overlap=None, affine = True):
        # cached the homography
        self.cached_homo = None
        self.overlap = overlap
        self.affine = affine

    def __call__(self, images, drawMatches=False):

        # get the images
        (left_img, right_img) = images

        if self.cached_homo is None:
            if self.overlap is not None:
                left_mask = np.zeros(left_img.shape[:2], np.uint8)
                left_mask[:, left_img.shape[1] - self.overlap:] = 255
                right_mask = np.zeros(right_img.shape[:2], np.uint8)
                right_mask[:, :self.overlap] = 255
            else:
                left_mask = None
                right_mask = None
            log.info('Start searching for features.')
            (left_kps, left_ds, left_features) = self.get_keypoints_and_descriptors(
                left_img, left_mask, True)
            (right_kps, right_ds, right_features) = self.get_keypoints_and_descriptors(
                right_img, right_mask, True)
            log.debug('Features found: #left_kps = {} | #right_kps = {}'.format(
                len(left_kps), len(right_kps)))
            if self.affine:
                (homo, mask_good, good_matches) = self.match_features_and_affine(left_kps, right_kps, left_ds, right_ds)
            else:
                (homo, mask_good, good_matches) = self.match_features(left_kps, right_kps, left_ds, right_ds)
            if homo is None:
                log.warning('No Transformation matrix found.')
                return None
            log.info('Transformation matrix found.')
            self.cached_homo = homo

        log.debug('TM =\n{}'.format(self.cached_homo))
        result = self.warp_images(left_img, right_img)

        if drawMatches:
            matchesMask = mask_good.ravel().tolist()
            result_matches = cv2.drawMatches(
                left_img, left_kps, right_img, right_kps, good_matches, matchesMask=matchesMask, **draw_params)
            return (self.cached_homo, result, result_matches)

        return self.cached_homo, result

    def get_keypoints_and_descriptors(self, img, mask=None, drawMatches=False):
        # TODO Überprüfen was besser ist als grauers Bild oder ob es egal ist.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO opencv versions check

        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=4)
        surf.setUpright(True)
        surf.setExtended(128)
        kps, ds = surf.detectAndCompute(img_gray, mask)

        if drawMatches:
            marked_matches = cv2.drawKeypoints(img, kps, None, (0, 0, 255), 4)
            return (kps, ds, marked_matches)

        return (kps, ds)

    def match_features(self, left_kps, right_kps, left_ds, right_ds):
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
            return (homo, mask_good, good_matches)
        return None

    def get_best_3_matches(self, left_kps, right_kps, left_ds, right_ds, drawMatches=False):
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
            return (affine, mask_better, better_matches)
        return None

    def match_features_and_affine(self, left_kps, right_kps, left_ds, right_ds, drawMatches=False):
        (homo, mask_good, good_matches) = self.match_features(left_kps, right_kps, left_ds, right_ds)
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
            mask_top = np.ones((3, 1))
            return (affine, mask_top, top_matches)
        return None

    def warp_images(self, left_img, right_img):
        result = cv2.warpPerspective(
            right_img, self.cached_homo, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
        result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
        return result
