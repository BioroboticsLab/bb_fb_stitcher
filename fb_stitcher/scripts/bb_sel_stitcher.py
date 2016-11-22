import argparse
from argparse import RawTextHelpFormatter
import cv2
import fb_stitcher.core as core
import fb_stitcher.helpers as helpers
import os


def process_images(args):
    # checks if filenames are valid

    name_l = os.path.splitext(os.path.basename(args.left))[0]
    name_r = os.path.splitext(os.path.basename(args.right))[0]
    out_basename = ''.join(['S_', name_l, '_ST_', name_r])

    if os.path.isdir(args.data):
        data_basename = ''.join([out_basename, '.npz'])
        data_path = os.path.join(args.data, data_basename)
    else:
        data_path = args.data

    camIdx_l = int(name_l.split('_', 2)[1])
    camIdx_r = int(name_r.split('_', 2)[1])
    assert camIdx_l in [0, 1, 2, 3] and camIdx_r in [0, 1, 2, 3]

    img_l = cv2.imread(args.left, 0)
    img_r = cv2.imread(args.right, 0)

    bb_sel_stitcher = core.BB_SelectionStitcher()
    bb_sel_stitcher((img_l, img_r), (camIdx_l, camIdx_r), (args.left_angle, args.right_angle))

    bb_sel_stitcher.save_data(data_path)
    print('Saved stitching params to: {} '.format(data_path))

    if args.pano is not None:
        if os.path.isdir(args.pano[0]):
            pano_basename = ''.join([out_basename, '.jpg'])
            pano_path = os.path.join(args.pano[0], pano_basename)
        else:
            pano_path = args.pano[0]
        result = bb_sel_stitcher.overlay_images()
        cv2.imwrite(pano_path, result)


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook Selction Stitcher.',
        description='This will stitch two images, based on selected rectangle'
                    ' for reproducing the stitching (,also with points).',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('left', help='Path of the left image.', type=str)
    parser.add_argument('right', help='Path of the left image.', type=str)
    parser.add_argument('left_angle', help='Rotation angle of the left image', type=int)
    parser.add_argument('right_angle', help='Rotation angle of the right image', type=int)
    parser.add_argument('data', help='Output directory/path of the stitching data.', type=str)
    parser.add_argument('--pano', '-p', nargs=1, help='Path of the panorama.')

    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()
