import argparse
import cv2
import fb_stitcher.core as core


def process_images(args):
    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    stitcher = core.BB_FeatureBasedStitcher()
    stitcher.load_data(args.data)
    result = stitcher.overlay_images(left_img, right_img)
    cv2.imwrite('pano.jpg', result)


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook video stitcher',
        description='This will stitch two images.'
    )

    parser.add_argument('left', help='Path of the left image.', type=str)
    parser.add_argument('right', help='Path of the right image.', type=str)
    parser.add_argument("data", help="Path of the stitching data.", type=str)

    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()
