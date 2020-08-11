function [ new_leftimg_cell, new_rightimg_cell ] = processing_scene_warping_image( leftimg_cell, rightimg_cell, homography_cell, warping_params)

        
        new_leftimg_cell = cell(1, size(leftimg_cell, 2));
        new_rightimg_cell = cell(1, size(leftimg_cell, 2));
        
        for j = 1:1
            
            ori_fixed = im2double(rightimg_cell{1, j});
            ori_moving = im2double(leftimg_cell{1, j});
                                    
            identity = [1 0 0; 0 1 0; 0 0 1;];
            fixedReg = warping_with_resize_undistortion(ori_fixed, identity, warping_params.camera_param.params.CameraParameters2, warping_params);
            movingReg = warping_with_resize_undistortion(ori_moving, homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
                                        
            [top, bottom, left, right] = bboxFromHomography(movingReg, homography_cell{1,j});         
            movingReg = movingReg(top:bottom, left:right, :);
            fixedReg = fixedReg(top:bottom, left:right, :);
            
            new_leftimg_cell{1, j} = movingReg;
            new_rightimg_cell{1, j} = fixedReg;
            
        end
        
        parfor j = 2:size(leftimg_cell,2)
            
            ori_latent = im2double(leftimg_cell{1,j});
            ori_blurred = im2double(rightimg_cell{1,j});
            
            identity = [1 0 0; 0 1 0; 0 0 1;];
            blurred = warping_with_resize_undistortion(ori_blurred, identity,  warping_params.camera_param.params.CameraParameters2, warping_params);
            latent = warping_with_resize_undistortion(ori_latent, homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
            
            % crop invalid region for estimating kernel
            [top, bottom, left, right] = bboxFromHomography(latent, homography_cell{1,j});
            latentReg = latent(top:bottom, left:right, :);
            blurredReg = blurred(top:bottom, left:right, :);
                        
            new_leftimg_cell{1, j} = latentReg;
            new_rightimg_cell{1, j} = blurredReg;
        end
        
end

