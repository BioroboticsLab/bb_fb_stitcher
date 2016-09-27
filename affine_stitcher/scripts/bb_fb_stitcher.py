import affine_stitcher.helpers as helpers
import affine_stitcher.core as core
from affine_stitcher.stitcher import Transformation
import argparse
import os
import cv2
from argparse import RawTextHelpFormatter

def process_images(args):
    img_l =cv2.imread(args.left,-1)
    img_r =cv2.imread(args.right,-1)
    bb_stitcher_fb = core.BB_FeatureBasedStitcher(Transformation(args.transform))
    __, __, __, result = bb_stitcher_fb((img_l, img_r), True)
    result = bb_stitcher_fb.overlay_images()
    cv2.imwrite(args.pano, result)
    pass


def main():
    parser = argparse.ArgumentParser(
        prog = 'BeesBook feature based Stitcher.',
        description = 'This will stich two images.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('left', help='Path of the left image.', type=str)
    parser.add_argument('right', help='Path of the left image.', type=str)
    parser.add_argument('pano', help='Output path of the panorama.', type=str)
    parser.add_argument('transform', help='Type of Transfromation: \n'
                                          '0 - Translation\n'
                                          '1 - EUCLIDEAN\n'
                                          '(2 - SIMILARITY)\n'
                                          '3 - AFFINE\n'
                                          '4 - PROJECTIVE', type=int, choices=[0,1,3,4])

    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()
