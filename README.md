# bb_affine_stitcher

## Requirements
* [OpenCV3](https://github.com/opencv/opencv)
* [opencv_contrib](https://github.com/opencv/opencv_contrib) (only required for step 2)
* [FFmpeg](https://trac.ffmpeg.org/wiki/CompilationGuide) with H.265/HEVC (only required for step 1)

[Good Instruction](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/) for installing opencv with opencv_contrib

## Usage

Stitching process for points is divided in 3 steps:

1. [Foreground Subtraction](#1-foreground-fubtraction)
2. [Stitching background images](#2-stitching-background-images)
3. [Map Coordinates](#3-mapping-points)
4. [Dockerfile](#4-dockerfile)

### 1. Foreground Subtraction
```bash
$  bb_fg_subtract <video_path> <output_directory>
```
This will subtract the moving foreground (bees) from the video and will
return an image of background (comb without bees).
If you didn't want to install the whole package on the system like on HLRN,
you can use this [script](https://gist.github.com/gitmirgut/3617b94094df918b956662de6d792827).

*Example:*
```bash
$  bb_fg_subtract Cam_0_2016-09-01T14\:20\:38.410765Z--2016-09-01T14\:26\:18.257648Z.mkv test/
```
will return the background image: `test/Cam_0_2016-09-01T14\:20\:38.410765Z--2016-09-01T14\:26\:18.257648Z.jpg`

### 2. Stitching background images
In this step two background images derived in step 1 will be stitched and
the required data for mapping locations (detected in the bb_pipeline) to
the overall coordinate system of one comb side will be saved to a file.

```bash
$ bb_fb_stitcher <path_left_img> <path_right_img> <angle_left_rot> <angle_right_rot> <transformation_type> <data_out> -p <preview_img>
```

* `<path_*_img>` - Path to the input images.
* `<angle_*_rot>` - initial Rotation angle of both images. Rotation ist measured counterclockwise.
* `<transfromation_type>` - defines the transformation used to stitch both images.
* `<data_out>` - path for the stitching data, required for further stitching or mapping points. If its a directory, the filename will be 'stitched' of the both input images (```Cam_0_2016-09-01T14:20:38.410765Z_ST_Cam_1_2016-09-01T14:16:13.311603Z.npz```)
* `-p <preview_img>` - creates an preview of the stitched images. (optional)

(See also `$ bb_fb_stitcher -h`)

*Example:*
```bash
$ bb_fb_stitcher Cam_0_2016-09-01T14\:20\:38.410765Z--2016-09-01T14\:26\:18.257648Z.jpg Cam_1_2016-09-01T14\:16\:13.311603Z--2016-09-01T14\:21\:53.157900Z.jpg 90 -90 3 dir_for_params/ -p preview.jpg
```

### 3. Mapping Points

minimal script for mapping points (x,y)

```python
import numpy as np
import fb_stitcher.core as core

camIdx = 0
pts_left_org = np.array([[[1000, 3000], [353, 400], [369, 2703]]]).astype(np.float64)

# initialize Stitcher and load data from step 2
stitcher = core.BB_FeatureBasedStitcher()
stitcher.load_data('Cam_0_2016-09-01T14:20:38.410765Z_ST_Cam_1_2016-09-01T14:16:13.311603Z.npz')

# 'stitch' points
pts_mapped = stitcher.map_cam_coordinates(camIdx, pts_left_org)

print(pts_mapped)
```
returns `[[[ 2990.34596943  3098.03015643] [  396.12205245  3725.00678215] [ 2694.5516094   3727.65687635]]]`

### 4. Dockerfile
Dockerfile with OpenCV3 and ffmpeg:
```bash
$ docker pull gitmirgut/bb_fb_stitcher
$ docker run -it -v   HOST_DIR:/storage gitmirgut/bb_fb_stitcher
```
