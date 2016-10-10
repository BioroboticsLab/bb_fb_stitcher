# bb_affine_stitcher

## Usage

Stitching process for points is divided in 3 steps:

1. Foreground Subtraction
2. Stitching background images
3. Map Coordinates

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
$ bb_fb_stitcher <path_left_img> <angle_left_rot> <path_right_img> <angle_right_rot> <transformation_type> <data_out> -p <preview_img>
```

`<path_*_img>` - Path to the input images.
`<angle_*_rot>` - initial Rotation angle of both images. Rotation ist measured counterclockwise.
`<transfromation_type>` - defines the transformation used to stitch both images.
`<data_out>` - path for the stitching data, required for further stitching or mapping points.
`-p <preview_img>` - creates an preview of the stitched images. (optional)

(See also `$ bb_fb_stitcher -h`)

*Example:*
```bash
$ bb_fb_stitcher Cam_0_2016-09-01T14\:20\:38.410765Z--2016-09-01T14\:26\:18.257648Z.jpg 90 Cam_1_2016-09-01T14\:16\:13.311603Z--2016-09-01T14\:21\:53.157900Z.jpg -90 3 stitch_params.npz -p preview.jpg
```

### 3. Mapping Points

