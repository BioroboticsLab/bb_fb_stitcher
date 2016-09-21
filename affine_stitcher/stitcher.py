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

    def __init__(self):
        # cached the homography
        self.cached_homo = None

    def __call__(self, images, drawMatches=False, overlap=None, affine=True):

        # get the images
        (left_img, right_img) = images

        if self.cached_homo is None:
            if overlap is not None:
                left_mask = np.zeros(left_img.shape[:2], np.uint8)
                left_mask[:, left_img.shape[1] - overlap:] = 255
                right_mask = np.zeros(right_img.shape[:2], np.uint8)
                right_mask[:, :overlap] = 255
            else:
                left_mask = None
                right_mask = None
            log.info('Start searching for features.')
            (left_kps, left_ds, left_features) = self.get_keypoints_and_descriptors(
                left_img, left_mask, True)
            (right_kps, right_ds, right_features) = self.get_keypoints_and_descriptors(
                right_img, right_mask, True)
            helpers.display(left_features)
            log.debug('Features found: #left_kps = {} | #right_kps = {}'.format(
                len(left_kps), len(right_kps)))
            if affine:
                (homo, mask, good_matches) = self.get_best_3_matches(left_kps, right_kps, left_ds, right_ds)
            else:
                (homo, mask, good_matches) = self.match_features(left_kps, right_kps, left_ds, right_ds)
            if homo is None:
                log.warning('No Transformation matrix found.')
                return None
            log.info('Transformation matrix found.')
            self.cached_homo = homo
            log.debug('TM =\n{}'.format(self.cached_homo))
            result = self.warp_images(left_img, right_img)

            if drawMatches:
                matchesMask = mask.ravel().tolist()
                result_matches = cv2.drawMatches(
                    left_img, left_kps, right_img, right_kps, good_matches, matchesMask=matchesMask, **draw_params)
                return (result, result_matches)

            return result

    def get_keypoints_and_descriptors(self, img, mask=None, drawMatches=False):
        # TODO Überprüfen was besser ist als grauers Bild oder ob es egal ist.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO opencv versions check

        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4)
        surf.setUpright(True)
        surf.setExtended(64)
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
        left_pts, right_pts, good_matches = helpers.lowe_ratio_test(
            left_kps, right_kps, raw_matches)
        log.debug('# Filtered Features = {}'.format(len(good_matches)))
        if len(left_pts) > 3:
            log.info('Start finding homography.')
            (homo, mask) = cv2.findHomography(
                right_pts, left_pts, cv2.RANSAC, 10.0)
            return (homo, mask, good_matches)
        return None

    def get_best_3_matches(self, left_kps, right_kps, left_ds, right_ds, drawMatches=False):
        bf = cv2.BFMatcher()
        log.info('Start matching Features.')
        raw_matches = bf.knnMatch(left_ds, right_ds, k=2)
        log.debug('Matches found: #raw_matches = {}'.format(len(raw_matches)))
        left_pts, right_pts, better = helpers.lowe_ratio_test_affine(
            left_kps, right_kps, raw_matches)
        log.debug('# Filtered Features = {}'.format(len(better)))
        if len(left_pts) > 2:
            log.info('Start finding affine transformation matrix.')
            affine = cv2.getAffineTransform(left_pts, right_pts)
            affine = cv2.invertAffineTransform(affine)
            affine = np.vstack([affine, [0, 0, 1]])
            mask = np.array([[1],[1],[1]])
            return (affine, mask, better)
        return None

    def warp_images(self, left_img, right_img):
        result = cv2.warpPerspective(
            right_img, self.cached_homo, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
        result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
        return result
