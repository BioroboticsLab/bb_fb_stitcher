import affine_stitcher.helpers as helpers
import argparse
import os
import cv2


def process_videos(args):
    # getting the filename without .extension
    basename = os.path.basename(args.video_path)
    filename = os.path.splitext(basename)[0]

    # setting file-type of output image
    ext = '.jpg'

    # getting background image
    cap = cv2.VideoCapture(args.video_path)
    background = helpers.subtract_foreground(cap, show=True)
    cap.release()

    cv2.imwrite(filename + ext, background)


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook foreground subtractor.',
        description='This will remove the moving objects of an video and'
                    'return an image of the background.'
    )

    parser.add_argument('video_path', help='Path of the video.', type=str)
    args = parser.parse_args()
    process_videos(args)

if __name__ == '__main__':
    main()