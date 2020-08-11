function [ new_homography_cell ] = processing_scene_ECC( leftimg_cell, rightimg_cell, outbasedir, warping_params)

addpath('~/installation/mexopencv-3.4.0');
addpath('~/installation/mexopencv-3.4.0/opencv_contrib');

gtout = fullfile(outbasedir,'gt');
blurout = fullfile(outbasedir,'blur');
Anaglyphout = fullfile(outbasedir,'Anaglyph');
gifout = fullfile(outbasedir,'gif');
tformout = fullfile(outbasedir,'tform');

if (~exist(outbasedir, 'dir')); mkdir(outbasedir); end
if (~exist(gtout, 'dir')); mkdir(gtout); end
if (~exist(blurout, 'dir')); mkdir(blurout); end
if (~exist(Anaglyphout, 'dir')); mkdir(Anaglyphout); end
if (~exist(gifout, 'dir')); mkdir(gifout); end
if (~exist(tformout, 'dir')); mkdir(tformout); end

new_homography_cell = cell(1, size(leftimg_cell, 2));

for j = 1:1
    gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
    bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
    Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
    gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));
    tformoutname = fullfile(tformout, sprintf('tform_%d.mat', j));
    AnaglyphRefoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d_ref.jpg', j));
    
    
    ori_fixed = im2double(rightimg_cell{1, j});
    ori_moving = im2double(leftimg_cell{1, j});
    
    
    identity = [1 0 0; 0 1 0; 0 0 1;];
    fixed = warping_with_resize_undistortion(ori_fixed, identity, warping_params.camera_param.params.CameraParameters2, warping_params);
    moving = warping_with_resize_undistortion(ori_moving, identity, warping_params.camera_param.params.CameraParameters1, warping_params);
       
    
    % initial homography
    tformEstimateRef = imregcorr(moving,fixed);
    [top, bottom, left, right] = bboxFromHomography(moving, tformEstimateRef.T);
    if (bottom-top-1) <= (size(moving,1)-150) || (right-left-1) <= (size(moving,2)-150)
        tformEstimateRef = imregcorr(moving,fixed, 'transformtype', 'translation');
        [top, bottom, left, right] = bboxFromHomography(moving, tformEstimateRef.T);
        if (bottom-top-1) <= (size(moving,1)-150) || (right-left-1) <= (size(moving,2)-150)
            tformEstimateRef = projective2d([1 0 0; 0 1 0; 0 0 1;]);
        end
        
    end
    
    % 138 sec for 50 iter (original resolution)
    % 9 sec for 50 iter (1/4 resolution)
    criteria = struct('type','Count+EPS', 'maxCount', 300, 'epsilon', -1);
    H = cv.findTransformECC(rgb2gray(moving), rgb2gray(fixed), ...
        'MotionType', 'Homography', 'InputWarp', tformEstimateRef.T', ...
        'Criteria', criteria);
    
    H = double(H);
    
    tform_ECC = projective2d(H');
    
    
    new_homography_cell{1,j} = tform_ECC.T;
    
    movingReg = warping_with_resize_undistortion(ori_moving, new_homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
    fixedReg = fixed;
    
    [top, bottom, left, right] = bboxFromHomography(movingReg, new_homography_cell{1,j});
    movingReg = movingReg(top:bottom, left:right, :);
    fixedReg = fixedReg(top:bottom, left:right, :);
    
    writeGIF(im2double(movingReg),im2double(fixedReg), gifoutname);
    imwrite(stereoAnaglyph(movingReg,fixedReg), Anaglyphoutname);
    imwrite(stereoAnaglyph(moving, fixed), AnaglyphRefoutname);
    imwrite(movingReg, gtoutname);
    imwrite(fixedReg, bluroutname);
    save_parfor(tformoutname, new_homography_cell{1,j});
end

parfor j = 2:size(leftimg_cell,2)
    gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
    bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
    Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
    AnaglyphRefoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d_ref.jpg', j));
    gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));
    tformoutname = fullfile(tformout, sprintf('tform_%d.mat', j));
    
    ori_latent = im2double(leftimg_cell{1,j});
    ori_blurred = im2double(rightimg_cell{1,j});
    
    
    identity = [1 0 0; 0 1 0; 0 0 1;];
    blurred = warping_with_resize_undistortion(ori_blurred, identity,  warping_params.camera_param.params.CameraParameters2, warping_params);
    latent = warping_with_resize_undistortion(ori_latent, identity, warping_params.camera_param.params.CameraParameters1, warping_params);
    
    % crop invalid region
    [top, bottom, left, right] = bboxFromHomography(latent, identity);
    latent_PRE = latent(top:bottom, left:right, :);
    blurred_PRE = blurred(top:bottom, left:right, :);
    
    new_homography_cell{1,j} = tform_ECC.T;
    
    % apply
    latentReg = warping_with_resize_undistortion(ori_latent, new_homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
    blurredReg = blurred;
    
    [top, bottom, left, right] = bboxFromHomography(latent, new_homography_cell{1,j});
    latentReg = latentReg(top:bottom, left:right, :);
    blurredReg = blurredReg(top:bottom, left:right, :);
    
    writeGIF(latentReg, blurredReg, gifoutname);
    imwrite(stereoAnaglyph(latent_PRE, blurred_PRE), AnaglyphRefoutname);
    imwrite(stereoAnaglyph(latentReg, blurredReg), Anaglyphoutname);
    imwrite(latentReg, gtoutname);
    imwrite(blurredReg, bluroutname);
    save_parfor(tformoutname, new_homography_cell{1,j});
end

end

