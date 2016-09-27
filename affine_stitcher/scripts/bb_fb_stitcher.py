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
    bb_stitcher_fb((img_l, img_r))
    bb_stitcher_fb.save_data(args.data)
    print(args.pano)
    if args.pano is not None:
        result = bb_stitcher_fb.overlay_images()
        cv2.imwrite(args.pano[0], result)
    print(args.pano)
    pass


def main():
    parser = argparse.ArgumentParser(
        prog = 'BeesBook feature based Stitcher.',
        description = 'This will stich two images.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('left', help='Path of the left image.', type=str)
    parser.add_argument('right', help='Path of the left image.', type=str)
    parser.add_argument('data', help='Output path of the stitching data.', type=str)
    parser.add_argument('transform', help='Type of Transfromation: \n'
                                          '0 - Translation\n'
                                          '1 - EUCLIDEAN\n'
                                          '(2 - SIMILARITY)\n'
                                          '3 - AFFINE\n'
                                          '4 - PROJECTIVE', type=int, choices=[0,1,3,4])
    parser.add_argument('--pano', '-p', nargs=1, help='path of the result image')

    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()
