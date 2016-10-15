import fb_stitcher.core as core
from fb_stitcher.stitcher import Transformation
import argparse
import fb_stitcher.helpers as helpers
import cv2
from argparse import RawTextHelpFormatter
import os

def process_images(args):
    # checks if filenames are valid
    assert helpers.check_filename(args.left) and helpers.check_filename(args.right)

    start_time_l = helpers.get_start_datetime(args.left)
    start_time_r = helpers.get_start_datetime(args.right)
    out_basename = ''.join([str(args.transform), '_', start_time_l, '_ST_', start_time_r])

    if os.path.isdir(args.data):
        data_basename = ''.join([out_basename, '.npz'])
        data_path = os.path.join(args.data, data_basename)
    else:
        data_path = args.data


    camIdx_l = helpers.get_CamIdx(args.left)
    camIdx_r = helpers.get_CamIdx(args.right)
    assert camIdx_l in [0, 1, 2, 3] and camIdx_r in [0, 1, 2, 3]

    img_l =cv2.imread(args.left, -1)
    img_r =cv2.imread(args.right, -1)

    bb_stitcher_fb = core.BB_FeatureBasedStitcher(Transformation(args.transform))
    bb_stitcher_fb((img_l, img_r),(camIdx_l, camIdx_r),(args.left_angle, args.right_angle))

    bb_stitcher_fb.save_data(data_path)
    print('Saved stitching params to: {} '.format(data_path))

    if args.pano is not None:
        if os.path.isdir(args.pano[0]):
            pano_basename = ''.join([out_basename, '.jpg'])
            pano_path = os.path.join(args.pano[0], pano_basename)
        else:
            pano_path = args.pano[0]
        result = bb_stitcher_fb.overlay_images()
        cv2.imwrite(pano_path, result)


def main():
    parser = argparse.ArgumentParser(
        prog = 'BeesBook feature based Stitcher.',
        description = 'This will stitch two images and return the needed data '
                      ' for reproducing the stitching (,also with points).',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('left', help='Path of the left image.', type=str)
    parser.add_argument('right', help='Path of the left image.', type=str)
    parser.add_argument('left_angle', help='Rotation angle of the left image', type=int)
    parser.add_argument('right_angle', help='Rotation angle of the right image', type=int)
    parser.add_argument('transform', help='Type of Transformation: \n'
                                          ' 0 - Translation\n'
                                          ' 1 - EUCLIDEAN\n'
                                          '(2 - SIMILARITY)\n'
                                          ' 3 - AFFINE\n'
                                          ' 4 - PROJECTIVE', type=int, choices=[0,1,3,4])
    parser.add_argument('data', help='Output directory/path of the stitching data.', type=str)
    parser.add_argument('--pano', '-p', nargs=1, help='Path of the panorama.')

    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()
