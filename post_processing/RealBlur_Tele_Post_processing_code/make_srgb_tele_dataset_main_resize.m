clear all;
addpath('~/installation/mexopencv-3.4.0')
addpath('~/installation/mexopencv-3.4.0/opencv_contrib')
%addpath('./camera_pipeline_simple')
warning('off','all')

rootdir = '../RealBlur_Tele_Original';
outdir = '../RealBlur_Tele_Post_processed_temp/RealBlur-J-Tele/RealBlur-J-Tele';

delete(gcp('nocreate'))
p = parpool(6);

stereo_params_srgb = load('stereoparams_srgb.mat'); % not_using

scene_dir = dir(rootdir);
scene_dir=scene_dir(~ismember({scene_dir.name},{'.','..'})); % 15x1 struct

for scene_i = 1:size(scene_dir,1)
    scene = scene_dir(scene_i);
    
    if contains(scene.name, 'scene') == 0
        continue;
    end
    
    tic;
    basedir = fullfile(scene.folder, scene.name);
    leftfolder = fullfile(basedir,'left');
    rightfolder = fullfile(basedir,'right');
    
    
    leftList = dir(fullfile(leftfolder, '*.JPG'));
    rightList = dir(fullfile(rightfolder, '*.JPG'));
    
    leftnameList = extractfield(leftList, 'name');
    rightnameList = extractfield(rightList, 'name');
    
    leftnameList = sort(leftnameList);
    rightnameList = sort(rightnameList);
    
    ori_leftimg_cell = cell(1 , size(leftList, 1));
    ori_rightimg_cell = cell(1 , size(leftList, 1));
    tic;
    for i = 1:size(leftList, 1)
        
        leftimg = imread(fullfile(leftfolder, leftnameList{i}));
        leftimg = im2double(leftimg);
        leftimg = flip(leftimg,2);
        
        rightimg = imread(fullfile(rightfolder, rightnameList{i}));
        rightimg = im2double(rightimg);
        
        %imshow(stereoAnaglyph(leftimg, rightimg));
        ori_leftimg_cell{1, i} = leftimg;
        ori_rightimg_cell{1, i} = rightimg;
    end
    toc
    
    warping_params.resize = 1/8;
    warping_params.undistort = false;
    warping_params.camera_param = stereo_params_srgb;
    warping_params.antialiasing = true;
    
    new_outdir = strcat(outdir,'_ECC');
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    outbasedir = fullfile(new_outdir, scene.name);
    [homography_cell_ECC] = processing_scene_ECC(ori_leftimg_cell, ori_rightimg_cell, outbasedir, warping_params);
        
    new_outdir = strcat(outdir,'_ECC_IMCORR');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [homography_cell_IMCORR] = processing_scene_IMCORR(ori_leftimg_cell, ori_rightimg_cell, homography_cell_ECC, outbasedir, warping_params);
    
    new_outdir = strcat(outdir,'_ECC_IMCORR_centroid');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [homography_cell] = processing_scene_centroid_kernel(ori_leftimg_cell, ori_rightimg_cell, homography_cell_IMCORR, outbasedir, warping_params, 151);
    
    % warping image from homography and return images
    [leftimg_cell, rightimg_cell] = processing_scene_warping_image(ori_leftimg_cell, ori_rightimg_cell, homography_cell, warping_params);
    
    % intensity alignment using reference image
    new_outdir = strcat(outdir,'_ECC_IMCORR_centroid_itensity_ref');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [leftimg_cell, rightimg_cell] = processing_scene_intensity_ref(leftimg_cell, rightimg_cell, outbasedir);
    
    % save uint16 image instead of uint8 image
    new_outdir = strcat(outdir,'_ECC_IMCORR_centroid_itensity_ref_unit16');
    outbasedir = fullfile(new_outdir, scene.name);
    if (~exist(new_outdir, 'dir')); mkdir(new_outdir); end
    [leftimg_cell, rightimg_cell] = processing_scene_uint16(leftimg_cell, rightimg_cell, outbasedir);
    
    
    toc;
    
end

