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

    out_path = os.path.abspath(args.out_dir)
    if os.path.isdir(out_path):
        file_path = os.path.join(out_path, filename + ext)
    else:
        file_path = args.out_dir

    # getting background image
    print(out_path)
    print(file_path)
    cap = cv2.VideoCapture(args.video_path)
    background = helpers.subtract_foreground(cap, args.show)
    cap.release()

    cv2.imwrite(file_path, background)


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook foreground subtractor.',
        description='This will remove the moving objects of an video and'
                    'return an image of the background.'
    )

    parser.add_argument('video_path', help='Path of the video.', type=str)
    parser.add_argument('out_dir', help='Path of the output dir.', type=str)
    parser.add_argument("--show", help="Shows the current background.", action="store_true")

    args = parser.parse_args()
    process_videos(args)

if __name__ == '__main__':
    main()