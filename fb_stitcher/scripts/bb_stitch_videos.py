import argparse
import cv2
import fb_stitcher.core as core


def process_videos(args):
    left_cap = cv2.VideoCapture(args.left)
    right_cap = cv2.VideoCapture(args.right)
    stitcher = core.BB_FeatureBasedStitcher()
    stitcher.load_data(args.data)
    result_writer = cv2.VideoWriter('result2.avi', cv2.VideoWriter_fourcc(*'XVID'), 3, (600, 400))

    try:
        while 1:
            left_ret, left_frame = left_cap.read()
            right_ret, right_frame = right_cap.read()
            result = stitcher.overlay_images(left_frame, right_frame)
            ret = result_writer.write(result)
            print(ret)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img', 800, 600)
            cv2.imshow('img', result)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook video stitcher',
        description='This will stitch two images.'
    )

    parser.add_argument('left', help='Path of the left video.', type=str)
    parser.add_argument('right', help='Path of the right video.', type=str)
    parser.add_argument("data", help="Shows the current background.", type=str)

    args = parser.parse_args()
    process_videos(args)

if __name__ == '__main__':
    main()
