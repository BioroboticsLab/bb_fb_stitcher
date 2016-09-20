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
    def __call__(self, images, drawMatches=False):

        # get the images
        (right_img, left_img) = images
        if self.cached_homo is None:
            (left_kps, left_ds) = self.get_keypoints_and_descriptors(left_img)
            (right_kps, right_ds) = self.get_keypoints_and_descriptors(right_img)
            log.debug('#left_kps = {}'.format(len(left_kps)))
            log.debug('#right_kps = {}'.format(len(right_kps)))
            (homo, mask, good) = self.match_features(left_kps, right_kps, left_ds, right_ds)

            if homo is None:
                return None

            self.cached_homo = homo
            log.debug(self.cached_homo)
            result = cv2.warpPerspective(
                left_img, self.cached_homo, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
            result[0:right_img.shape[0], 0:right_img.shape[1]] = right_img

            if drawMatches:
                matchesMask = mask.ravel().tolist()
                result_matches = cv2.drawMatches(left_img, left_kps, right_img, right_kps, good,matchesMask=matchesMask,**draw_params)
                return (result, result_matches)

            return result

    def get_keypoints_and_descriptors(self, img, mask=None):
        # TODO Überprüfen was besser ist als grauers Bild oder ob es egal ist.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO opencv versions check

        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=300, nOctaves=3)
        surf.setUpright(False)
        surf.setExtended(128)

        kps, ds = surf.detectAndCompute(img_gray, mask)

        return (kps, ds)

    def match_features(self, left_kps, right_kps, left_ds, right_ds):
        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(left_ds, right_ds, k=2)
        log.debug('#left_kps = {}'.format(len(left_kps)))
        log.debug('#right_kps = {}'.format(len(right_kps)))
        log.debug('#raw_matches = {}'.format(len(raw_matches)))
        left_pts, right_pts, good = helpers.lowe_ratio_test(
            left_kps, right_kps, raw_matches)
        if len(left_pts) > 3:
            (homo, mask) = cv2.findHomography(
                left_pts, right_pts, cv2.RANSAC, 10.0)
            return (homo, mask, good)
        return None
