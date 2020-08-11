clear all;
addpath('~/installation/mexopencv-3.4.0')
addpath('~/installation/mexopencv-3.4.0/opencv_contrib')
%addpath('./camera_pipeline_simple')
warning('off','all')
rootdir = '../RealBlur_libraw_linear_int16';
outdir = '../RealBlur_Post_processed_temp/ReaBlur-R/ReaBlur-R';
homographydir = '../RealBlur_Post_processed_temp/RealBlur-J/RealBlur-J_ECC_IMCORR_centroid'; % we use homography computed from srgb images

top = 1;
bottom = 3094;
left = 954;
right = 3674;

stereo_params_srgb = load('stereoparams_srgb.mat');
delete(gcp('nocreate'))
p = parpool(6);
scene_dir = dir(rootdir);
scene_dir=scene_dir(~ismember({scene_dir.name},{'.','..'}));



for scene_i = 1:234
    scene = scene_dir(scene_i);
    
    if contains(scene.name, 'scene') == 0
        continue;
    end
    
    tic;
    basedir = fullfile(rootdir, scene.name);
    leftfolder = fullfile(basedir,'gt_linear');
    rightfolder = fullfile(basedir,'blur_linear');
    
    
    leftList = dir(fullfile(leftfolder, '*.tiff'));
    
    tic;
    NumberImages = size(leftList,1);
    ori_leftimg_cell = cell(1 , NumberImages);
    ori_rightimg_cell = cell(1 , NumberImages);
    
    params.resize = 1/4;
    params.undistort = true;
    params.camera_param = stereo_params_srgb;
    params.antialiasing = true;
    identity = [1 0 0; 0 1 0; 0 0 1;];
    
    for i = 1:NumberImages
        leftimg_file = fullfile(leftfolder, sprintf('gt_%d.ARW.tiff', i));
        left_TifLink = Tiff(leftimg_file,'r');
        leftimg = left_TifLink.read();
        leftimg = im2double(leftimg);
        leftimg = leftimg(9:9+3448-1,9:9+5168-1,:);
        leftimg = flip(leftimg,2);
        leftimg = leftimg(top:bottom, left:right, :);
        leftimg = warping_with_resize_undistortion(leftimg, identity, params.camera_param.params.CameraParameters1, params);
        
        
        left_TifLink.close();
        
        rightimg_file = fullfile(rightfolder, sprintf('blur_%d.ARW.tiff', i));
        right_TifLink = Tiff(rightimg_file,'r');
        rightimg = right_TifLink.read();
        rightimg = im2double(rightimg);
        rightimg = rightimg(9:9+3448-1,9:9+5168-1,:);
        rightimg = rightimg(top:bottom, left:right, :);
        rightimg = warping_with_resize_undistortion(rightimg, identity, params.camera_param.params.CameraParameters2, params);
        right_TifLink.close();
        
        ori_leftimg_cell{1, i} = leftimg;
        ori_rightimg_cell{1, i} = rightimg;
    end
    toc;
    
    new_outdir = strcat(outdir,'_BM3D');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [ori_leftimg_cell, ori_rightimg_cell] = processing_scene_bm3d(ori_leftimg_cell, ori_rightimg_cell, outbasedir);
    
    warping_params.resize = 1;
    warping_params.undistort = false;
    warping_params.camera_param = stereo_params_srgb;
    warping_params.antialiasing = false;
    
    
    % warping image from homography and return images
    source_homography = fullfile(homographydir, scene.name);
    [leftimg_cell, rightimg_cell] = processing_scene_warping_from_disk(ori_leftimg_cell, ori_rightimg_cell, source_homography, warping_params);
    
    % intensity alignment using reference image
    new_outdir = strcat(outdir,'_BM3D_ECC_IMCORR_centroid_itensity_ref');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [leftimg_cell, rightimg_cell] = processing_scene_intensity_ref(leftimg_cell, rightimg_cell, outbasedir);
    
    % save uint16 image instead of uint8 image
    new_outdir = strcat(outdir,'_BM3D_ECC_IMCORR_centroid_itensity_ref_unit16');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [leftimg_cell, rightimg_cell] = processing_scene_uint16(leftimg_cell, rightimg_cell, outbasedir);
    
    toc;
    
end

