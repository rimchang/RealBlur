## Prerequisites for Post-processing
- Matlab (tested with matlab 2016b)
- [Opencv 3.4.0](https://github.com/opencv/opencv/tree/3.4)
- [mexopencv](https://github.com/kyamagu/mexopencv/tree/v3.4.0)
- [libraw](https://www.libraw.org/) for processing raw images

## Step
1. Download [RealBlur_Ori_sample](https://cgdata.postech.ac.kr:5652/sharing/OabkBIk35). Please contact us for full dataset.
2. run extract_libraw_image/extract_libraw_linear_int16.m for processing raw images.
3. run make_srgb_dataset_main_resize.m for JPEG images
4. run make_libraw_using_srgb_dataset_main_resize.m for raw images